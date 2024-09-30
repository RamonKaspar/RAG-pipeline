from typing import List, Tuple, Optional
import os
import numpy as np
import pandas as pd
from pydantic import BaseModel

from util.embedding_service.embedding_service import EmbeddingService

class RetrievedDocument(BaseModel):
    content: str
    metadata: Optional[dict] = None
    similarity: float   
    similarity_function: str
    embedding_model: str


class Retriever:
    """
    A class for retrieving suitable text snippets (given a user query) from a pre-built vector database
    of embedded text snippets for a specific schhol subject. We use cosine similarity as the similarity metric.
    """
    
    def __init__(self, subject: str, embedding_service: EmbeddingService, debug: bool = False):
        """Initializes the Retriever class with the given school subject and embedding model.
        NOTE: Use the same embedding model that was used to create the database!
        
        Args:
            subject (str): The school subject for which the pipeline is created.
            embedding_service (EmbeddingService): The embedding service to use for embedding the text snippets.
            debug (bool): Whether to print debug information.
        """
        self.subject = subject
        self.embedding_service = embedding_service
        self.debug = debug
        
        # Load the embeddings and metadata into memory
        self.embedded_data = self.load_embedded_data()  # Load Parquet file
        self.embeddings_array = np.array(self.embedded_data['embedding'].to_list())
        self.contents = self.embedded_data['content'].to_list()
        self.metadatas = self.embedded_data['metadata'].to_list()
        
    def retrieve(self, user_queries: List[str], top_k: int = 5, threshold: float = 0.15) -> Tuple[List[RetrievedDocument], int]:
        """ Retrieve the top-K most relevant documents based on multiple user queries.

        Args:
            user_queries (List[str]): The list of queries for which we search for relevant information.
            top_k (int): The number of snippets to retrieve. Defaults to 5.
            threshold (float): The threshold for the similarity search in the retrieval. Defaults to 0.15.

        Returns:
            List[str]: The list of snippets retrieved by the model.
            int: The number of tokens used by the prompt embedding.
        """
        # Embedd the user queries
        query_embeddings, total_tokens = self.embedding_service.get_embeddings(user_queries)
        query_embeddings = np.array(query_embeddings)
        
        # Calculate the cosine similarity between the query embeddings and the database embeddings.
        # We vectorize the cosine_similarity function to calculate the cosine similarity between each query embedding and all database embeddings.
        # OpenAI embeddings are already normalized (https://platform.openai.com/docs/guides/embeddings/which-distance-function-should-i-use),
        # so cosine similarity is just the dot product
        similarities = np.dot(self.embeddings_array, query_embeddings.T)
        max_similarities = np.max(similarities, axis=1)
        indices = np.where(max_similarities >= threshold)[0]
        top_k_indices = indices[np.argsort(-max_similarities[indices])][:top_k]
        retrieved_docs = [
            RetrievedDocument(
                content=self.contents[idx],
                metadata=self.metadatas[idx],
                similarity=max_similarities[idx],
                similarity_function='cosine_similarity',
                embedding_model=self.embedding_service.model_name
            )
            for idx in top_k_indices
        ]
        return retrieved_docs, total_tokens

    def load_embedded_data(self) -> pd.DataFrame:
        """Loads the embedded data from a Parquet file."""
        # TODO: Load from the vector database in the future
        file_path = f"embedding_database/{self.subject}.parquet"
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Embedded data not found for subject: {self.subject}")
        
        df = pd.read_parquet(file_path)
        return df