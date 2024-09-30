from typing import List, Optional
import numpy as np
from pydantic import BaseModel
from langchain_core.documents.base import Document
from openai.types import Embedding

import os
import pandas as pd
from openai import OpenAI
from uuid import uuid4  # Generate unique IDs for each chunk
import tiktoken

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import BSHTMLLoader

# Structure the embedded chunk, i.e. a row in the database
class EmbeddedChunks(BaseModel):
    id: str
    content: str
    metadata: Optional[dict] = None
    embedding: List[float]
    embedding_model: Optional[str] = None

class DatabaseBuilder:
    """Builds a database of embedded text chunks for a given subject, given a corpus of PDF and HTML documents."""
    
    def __init__(self, subject: str, embedding_model: str = "text-embedding-3-large", chunk_size: int = 512, overlap_size: int = 64, min_text_length: int = 0, debug: bool = False):
        """Constructor for the DatabaseBuilder class.
        NOTE: We use the default values for chunk_size and overlap_size from here: https://arxiv.org/abs/2405.06681

        Args:
            subject (str): The subject for which the database is being built (e.g., math, physics). This must be the folder name containing the files.
            embedding_model (str, optional): The embedding model used. Defaults to "text-embedding-3-large".
            chunk_size (int, optional): The size of each text chunk (in characters). Defaults to 512.
            overlap_size (int, optional): The size of the overlap between chunks (in characters) to ensure context. Defaults to 64.
            min_text_length (int, optional): The minimum length of text to consider as a chunk (in characters). Defaults to 0.
            debug (bool, optional): Whether to print debug information. Defaults to False.
        """
        possible_subjects = ['math', 'physics', 'chemistry', 'test']    # TODO: Update list with correct subjects
        if subject not in possible_subjects:
            raise ValueError("Invalid subject. Must be one of" + str(possible_subjects))
        self.subject = subject
        self.embedding_model = embedding_model
        if chunk_size < 1 or overlap_size < 1:
            raise ValueError("chunk_size and overlap_size must be greater than 0.")
        self.chunk_size = chunk_size
        self.overlap_size = overlap_size
        self.min_text_length = min_text_length
        self.debug = debug

    def build_database(self, root_folder_path: str):
        """Builds a database of embedded text chunks from PDF and HTML files in a folder.
        NOTE: Only PDF files and HTML files are supported for now.

        Args:
            folder_path (str): The path to the root folder containing the data. The folder must contain subfolders for each subject.
            min_text_length (int, optional): The minimum length of text to consider as a chunk (in characters). Defaults to 50.
        """
        folder_path = os.path.join(root_folder_path, self.subject)
        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"Folder not found: {folder_path}")
        
        docs : List[Document] = []
        for filename in os.listdir(folder_path):
            if filename.lower().endswith('.pdf') or filename.lower().endswith('.html'):
                if self.debug: print(f"Processing file: {filename}")
                file_path = os.path.join(folder_path, filename)
                docs = docs + self.extract_content(file_path)
            else:
                raise ValueError(f"Unsupported file format: {folder_path}/{filename}. Only PDF and HTML files are supported.")

        # Remove newlines and tabs from the text, and encode it to utf-8
        for doc in docs:
            doc.page_content = doc.page_content.replace("\n", " ").replace("\t", " ").replace("\xa0", " ")

        # Chunk the text
        chunks = self.chunk_text(docs)
        if self.debug: print(f"Chunked {len(chunks)} text chunks.")
        
        # Embed the docs and 
        embedded_chunks = self.embed_chunks(chunks)
        if self.debug: print(f"Embedded {len(embedded_chunks)} text chunks.")
        
        # Add embedded chunks to the database
        self.add_to_database(embedded_chunks)
        print(f"Database built successfully for subject: {self.subject}")

    def extract_content(self, path: str) -> List[Document]:
        """Extracts text from a given PDF or HTML file."""
        assert (path.lower().endswith('.pdf') or path.lower().endswith('.html')) and os.path.exists(path), "Invalid file path."
        if path.lower().endswith('.pdf'):
            # Doc: https://python.langchain.com/v0.1/docs/modules/data_connection/document_loaders/pdf/
            loader = PyPDFLoader(file_path=path, extract_images=False)  # Extract text only. Maybe consider images as well?
        elif path.lower().endswith('.html'):
            # Doc: https://python.langchain.com/v0.1/docs/modules/data_connection/document_loaders/html/
            loader = BSHTMLLoader(file_path=path, open_encoding="utf-8")
        if self.debug: print(f"Extracting content from: {path}")
        return loader.load()
        
    def chunk_text(self, docs: List[Document]) -> List[Document]:
        """Chunks the given documents into smaller pieces of a specified size {self.chunk_size} with an overlap {self.overlap_size}.
        Filters out chunks that are too short (less than {self.min_text_length} characters).
        TODO: Semantic chunking should be considered in the future.
        """
        # Doc: https://python.langchain.com/v0.2/docs/how_to/recursive_text_splitter/
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.overlap_size,
            length_function=len,
            is_separator_regex=False,   # use default seperator list ["\n\n", "\n", " ", ""]
        )
        chunked_docs = text_splitter.split_documents(docs)
        # Optional: Filter out chunks that are too short
        chunks_filtered = [chunk for chunk in chunked_docs if len(chunk.page_content) >= self.min_text_length]
        return chunks_filtered
    
    def embed_chunks(self, chunks: List[Document]) -> List[EmbeddedChunks]:
        """Embeds a list of text chunks using the specified embedding model {self.embedding_model}.
        NOTE: For now, only the OpenAI API is supported for embeddings."""
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY")) # Load the OpenAI API key from the .env file
        
        # Embed all chunks at once
        chunks_as_strings = [chunk.page_content for chunk in chunks]

        # Get the number of tokens used for the embeddings
        tokenizer = tiktoken.encoding_for_model(self.embedding_model)
        numTokens = 0
        embedded_chunks = []
        for chunk in chunks_as_strings:
            numTokens += len(tokenizer.encode(chunk))
            if len(tokenizer.encode(chunk)) > 8191: # Reference: https://platform.openai.com/docs/guides/embeddings/embedding-models
                raise ValueError("The total number of tokens for a chunk exceeds the limit of 8191 tokens. Use a smaller chunk size.")
            
        # Split the chunks into batches of 1 million tokens each
        splits = numTokens // 100000
        batched_chunks = np.array_split(chunks_as_strings, splits) if splits > 0 else [chunks_as_strings]

        embedded_chunks : List[Embedding] = []
        for i, batch in enumerate(batched_chunks):
            if self.debug: print(f"Embedding batch {i+1}/{len(batched_chunks)}")
            embedded_chunks.extend(client.embeddings.create(input=batch, model=self.embedding_model).data)
        
        if self.debug: print(f"Total tokens used for Embeddings with {self.embedding_model}: {numTokens}")
        
        # Combine the embedding with content and metadata
        embedded_data : List[EmbeddedChunks] = []
        for chunk, embedding in zip(chunks, embedded_chunks):
            embedded_data.append(EmbeddedChunks(
                id=str(uuid4()),    # Generate a unique ID for each chunk
                content=chunk.page_content,
                source=chunk.metadata,
                embedding=embedding.embedding,
                embedding_model=self.embedding_model
            ))
        return embedded_data

    def add_to_database(self, embedded_data: List[EmbeddedChunks], database_path: str = "embedding_database/"):
        """Adds the given embedded text chunks to the database.
        """
        # TODO: Move this to a database
        # For now: Save embedded chunks to a Parquet file in the embedding_database folder
        data_dicts = [chunk.model_dump() for chunk in embedded_data] # Convert Pydantic objects to dictionaries
        new_df = pd.DataFrame(data_dicts)
        
        # Define the file path
        file_path = f"{database_path}{self.subject}.parquet"

        new_df.to_parquet(file_path, compression='gzip', index=False)
        if self.debug: print(f"Embeddings saved to {file_path}")