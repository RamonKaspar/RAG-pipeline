import os
from src.database_builder import DatabaseBuilder
from src.pipeline import RAGPipeline
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(dotenv_path=".env", override=True)

def main():
    # Define parameters for database building
    subject = "math"
    embedding_model = "text-embedding-3-large"
    chunk_size = 1000
    overlap_size = 50
    min_text_length = 0
    
    # Initialize the DatabaseBuilder
    db_builder = DatabaseBuilder(
        subject=subject, 
        database=None, 
        embedding_model=embedding_model, 
        chunk_size=chunk_size, 
        overlap_size=overlap_size, 
        min_text_length=min_text_length
    )
    
    # Uncomment the following line to rebuild the database if necessary
    # db_builder.build_database(root_folder_path="data/")
    
    # Initialize the RAG pipeline
    pipeline = RAGPipeline(db_builder)
    pipeline.initialize()
    
    # Define the user query
    user_query = "Who is Michael?"
    
    # Define parameters for retrieval
    top_k = 3
    threshold = 0.15
    
    # Run the pipeline to get the response
    response = pipeline.run(user_query, top_k=top_k, threshold=threshold, max_tokens=500, temperature=0.7)
    
    # Print the generated response
    print("Generated Response:", response)

if __name__ == "__main__":
    main()
