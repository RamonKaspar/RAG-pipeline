from src.database_builder import DatabaseBuilder
from src.retriever import Retriever
from src.generator import Generator

class RAGPipeline:
    """A pipeline that combines the DatabaseBuilder, Retriever, and Generator classes to provide a complete question-answering system."""
    def __init__(self, db_builder: DatabaseBuilder):
        self.db_builder = db_builder
        self.retriever = None
        self.generator = Generator()
    
    def initialize(self):
        self.retriever = Retriever(self.db_builder.subject, self.db_builder.embedding_model)
    
    def run(self, query, top_k=3, threshold=0.2, max_tokens=500, temperature=0.7):
        retrieved_data = self.retriever.retrieve(query, top_k, threshold)
        response = self.generator.generate_response(self.db_builder.subject, retrieved_data, query, max_tokens, temperature)
        return response