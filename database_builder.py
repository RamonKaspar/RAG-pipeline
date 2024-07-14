from typing import List, Optional
from pydantic import BaseModel
from langchain_core.documents.base import Document
from openai.types import Embedding, CreateEmbeddingResponse

import os
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
from uuid import uuid4  # Generate unique IDs for each chunk

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import BSHTMLLoader

# Structure the embedded chunk, i.e. a row in the database
class EmbeddedChunks(BaseModel):
    id: str
    content: str
    source: Optional[str] = None
    embedding: List[float]
    embedding_model: Optional[str] = None

class DatabaseBuilder:
    """Builds a database of embedded text chunks for a given subject, given a corpus of PDF and HTML documents."""
    
    def __init__(self, subject: str, database, embedding_model: str = "text-embedding-3-large", chunk_size: int = 1000, overlap_size: int = 200, min_text_length: int = 50):
        """Constructor for the DatabaseBuilder class.
        TODO: Do the default values for chunk_size, overlap_size amd min_text_length make sense?

        Args:
            subject (str): The subject for which the database is being built (e.g., math, physics). This must be the folder name containing the files.
            database (_type_): The database object to which the text chunks will be added.
            embedding_model (str, optional): The embedding model used. Defaults to "text-embedding-3-large".
            chunk_size (int, optional): The size of each text chunk (in characters). Defaults to 1000.
            overlap_size (int, optional): The size of the overlap between chunks (in characters) to ensure context. Defaults to 200.
            min_text_length (int, optional): The minimum length of text to consider as a chunk (in characters). Defaults to 50.
        """
        possible_subjects = ['math', 'physics', 'chemistry', 'test']    # TODO: Update list with correct subjects
        if subject not in possible_subjects:
            raise ValueError("Invalid subject. Must be one of" + str(possible_subjects))
        self.subject = subject
        self.database = database
        self.embedding_model = embedding_model
        self.chunk_size = chunk_size
        self.overlap_size = overlap_size
        self.min_text_length = min_text_length

    def build_database(self, root_folder_path: str):
        """Builds a database of embedded text chunks from PDF files in a folder.
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
                file_path = os.path.join(folder_path, filename)
                docs = self.extract_content(file_path)
            else:
                print(f"Unsupported file format: {filename}")
        # Chunk the text
        chunks = self.chunk_text(docs)
        # Embed the docs and add them to the database
        embedded_chunks = self.embed_chunks(chunks)
        self.add_to_database(embedded_chunks)
        print(f"Database built successfully for subject: {self.subject}")

    def extract_content(self, path: str) -> List[Document]:
        """Extracts text from a given PDF file and chunks it into smaller pieces of a 
        specified size {self.chunk_size} with an overlap {self.overlap_size}."""
        assert (path.lower().endswith('.pdf') or path.lower().endswith('.html')) and os.path.exists(path), "Invalid file path."
        if path.lower().endswith('.pdf'):
            # Doc: https://python.langchain.com/v0.1/docs/modules/data_connection/document_loaders/pdf/
            loader = PyPDFLoader(file_path=path, extract_images=False)  # Extract text only. Maybe consider images as well?
        elif path.lower().endswith('.html'):
            # Doc: https://python.langchain.com/v0.1/docs/modules/data_connection/document_loaders/html/
            loader = BSHTMLLoader(file_path=path, open_encoding="utf-8")
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
        """Embeds a list of text chunks using the specified embedding model {self.embedding_model}."""
        load_dotenv()   # Load the OpenAI API key from the .env file
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        # Embed all chunks at once
        chunks_as_strings = [chunk.page_content for chunk in chunks]
        embedded_chunks : CreateEmbeddingResponse = client.embeddings.create(input=chunks_as_strings, model=self.embedding_model)
        print(f"Total tokens used for Embeddings with {self.embedding_model}: {embedded_chunks.usage.prompt_tokens}")
        
        # Combine the embedding with content and metadata
        embedded_data : List[EmbeddedChunks] = []
        for chunk, embedding in zip(chunks, embedded_chunks.data):
            embedded_data.append(EmbeddedChunks(
                id=str(uuid4()),    # Generate a unique ID for each chunk
                content=chunk.page_content,
                source=chunk.metadata.get("source", None),
                embedding=embedding.embedding,
                embedding_model=self.embedding_model
            ))
        return embedded_data

    def add_to_database(self, embedded_data: List[EmbeddedChunks]):
        """Adds the given embedded text chunks to the database."""
        # TODO: Move this to a database
        # For now: Save embedded chunks to a CSV file in the embedding_database folder
        data_dicts = [chunk.model_dump() for chunk in embedded_data] # Convert Pydantic objects to dictionaries
        df = pd.DataFrame(data_dicts)
        df.to_csv(f"embedding_database/{self.subject}.csv", index=False)
        print(f"Embeddings saved to embedding_database/{self.subject}.csv")