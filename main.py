from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(dotenv_path=".env", override=True)

# Create an embedding service
from util.embedding_service.embedding_service_factory import EmbeddingServiceFactory
embedding_service = EmbeddingServiceFactory().get_embedding_service(provider="openai", model_name="text-embedding-3-large")

""" 
Preprocessing step: This step only needs to be done once for each subject.
Example Usage of DatabaseBuilder class with the subject "test" and the embedding model "text-embedding-3-large". 
The data that is used has to be in the data/test folder.
"""
from src.database_builder import DatabaseBuilder

db_builder = DatabaseBuilder(subject="test", embedding_service=embedding_service, chunk_size=512, overlap_size=64, min_text_length=0, debug=True)
db_builder.build_database(root_folder_path="data/")


"""
Example Usage of Retriever class with the subject "test" and the embedding model "text-embedding-3-large".
The database with the embeddings was created above.
"""
from src.retriever import Retriever

retriever = Retriever(subject="test", embedding_service=embedding_service, debug=True)
retrieved_data, tokens = retriever.retrieve(user_queries=["When is Albert Einstein's Birthday?"], top_k=5, threshold=0.2)


"""
Example Usage of Generator class with the model provider "HuggingFace" and the model 
"meta-llama/Meta-Llama-3-8B-Instruct". The retrieved data is used to generate a response to the user query.
"""
from src.generator import Generator

generator = Generator(model_provider="HuggingFace", model_name="mistralai/Mistral-7B-Instruct-v0.2", temperature=0.1, max_tokens=2048)
response = generator.generate_response(user_query="When is Albert Einstein's Birthday?", retrieved_data=retrieved_data)
print(response)