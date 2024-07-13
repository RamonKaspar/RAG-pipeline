from typing import List
from langchain_core.documents.base import Document

import os

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import UnstructuredHTMLLoader

class DatabaseBuilder:
    """Builds a database of embedded text chunks for a given subject, given a corpus of PDF and HTML documents."""
    
    def __init__(self, subject: str, database, embedding_model: str = "text-embedding-3-large", chunk_size: int = 1000, overlap_size: int = 200, min_text_length: int = 50):
        """Constructor for the DatabaseBuilder class.
        TODO: Do the default values for chunk_size, overlap_size amd min_text_length make sense?

        Args:
            subject (str): The subject for which the database is being built (e.g., math, physics).
            database (_type_): The database object to which the text chunks will be added.
            embedding_model (str, optional): The embedding model used. Defaults to "text-embedding-3-large".
            chunk_size (int, optional): The size of each text chunk (in characters). Defaults to 1000.
            overlap_size (int, optional): The size of the overlap between chunks (in characters) to ensure context. Defaults to 200.
            min_text_length (int, optional): The minimum length of text to consider as a chunk (in characters). Defaults to 50.
        """
        possible_subjects = ['math', 'physics', 'chemistry']    # TODO: Update list with correct subjects
        if subject not in possible_subjects:
            raise ValueError("Invalid subject. Must be one of" + str(possible_subjects))
        self.subject = subject
        self.database = database
        self.embedding_model = embedding_model
        self.chunk_size = chunk_size
        self.overlap_size = overlap_size
        self.min_text_length = min_text_length

    def build_database(self, folder_path: str):
        """Builds a database of embedded text chunks from PDF files in a folder.
        NOTE: Only PDF files and HTML files are supported for now.

        Args:
            folder_path (str): The path to the folder containing PDF files.
            min_text_length (int, optional): The minimum length of text to consider as a chunk (in characters). Defaults to 50.
        """
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
        chunks_embedded = self.embed_chunks(chunks)
        self.add_to_database(chunks_embedded)
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
            loader = UnstructuredHTMLLoader(file_path=path)
        return loader.load()
        
    def chunk_text(self, docs: List[Document]) -> List[Document]:
        """Chunks the given documents into smaller pieces of a specified size {self.chunk_size} with an overlap {self.overlap_size}.
        Filters out chunks that are too short (less than {self.min_text_length} characters)."""
        # Doc: https://python.langchain.com/v0.2/docs/how_to/recursive_text_splitter/
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.overlap_size,
            length_function=len,
            is_separator_regex=False,   # use default seperator list ["\n\n", "\n", " ", ""]
        )
        chunked_docs = text_splitter.split_documents(docs)
        # Filter out chunks that are too short
        chunks_filtered = [chunk for chunk in chunked_docs if len(chunk.page_content) >= self.min_text_length]
        return chunks_filtered
    
    def embed_chunks(self, chunks: List[Document]):
        """Embeds a list of text chunks using the specified embedding model {self.embedding_model}."""
        raise NotImplementedError("This method is not implemented yet.")

    def add_to_database(self, chunks):
        """Adds the given embedded text chunks to the database."""
        raise NotImplementedError("This method is not implemented yet.")
