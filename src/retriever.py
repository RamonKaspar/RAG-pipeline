from typing import List, Tuple

import os
import numpy as np
from openai import OpenAI
import pandas as pd

from dotenv import load_dotenv

load_dotenv(dotenv_path=".env", override=True)


class Retriever:
    def __init__(self, subject: str, embedding_model: str = "text-embedding-3-large"):
        """Generates a pipeline for retrieving text snippets from resources of a specific school subject.

        Args:
            subject (str): The school subject for which the pipeline is created.
            embedding model (str): The embedding model used for the text data.
        """
        possible_subjects = ["math", "physics", "chemistry"]
        if subject not in possible_subjects:
            raise ValueError(f"Invalid subject. Must be one of{
                             possible_subjects}")
        self.subject = subject
        self.embedding_model = embedding_model

        # pandas DataFrame containing the embedded data
        self.embedded_data = self.load_embedded_data(subject)

    def retrieve(
        self, user_query: str, top_k: int, threshold: float
    ) -> Tuple[List[str], int]:
        """Runs the pipeline for the RAG model. Returns the response retrieved by the model.

        Args:
            user_query (str): The query for which we search for relevant information.
            top_k (int): The number of snippets to retrieve.
            threshold (float): The threshold for the similarity search in the retrieval.

        Returns:
            List[str]: The list of snippets retrieved by the model.
        """

        # Step 1: Retrieve Documents
        retrieved_docs, promptEmbeddingTokenUsage = self.retrieve_documents(
            user_query)

        # Step 2: Select most relevant Documents
        selected_snippets = self.select_documents(
            retrieved_docs, top_k, threshold)

        # Ensure that the number of snippets is less than or equal to the number of snippets requested
        assert (
            len(selected_snippets) <= top_k
        ), "Number of snippets retrieved is greater than top_k."

        print(f"Retrieved {len(selected_snippets)} snippets.")
        return selected_snippets, promptEmbeddingTokenUsage

    def retrieve_documents(self, user_query: str) -> Tuple[pd.DataFrame, int]:
        """Retrieves documents relevant to the user query."""
        results = self.embedded_data.copy()
        query_embedding, promptEmbeddingTokenUsage = self.get_embedding(
            user_query)
        results["similarity"] = self.embedded_data["embedding"].apply(
            lambda x: self.cosine_similarity(x, query_embedding)
        )
        return (results[["content", "similarity"]], promptEmbeddingTokenUsage)

    def select_documents(
        self, retrieved_docs: pd.DataFrame, top_k: int, threshold: float
    ) -> List[str]:
        """Selects the most relevant documents from the retrieved documents."""
        filtered_docs = retrieved_docs[retrieved_docs["similarity"] >= threshold]
        top_docs = filtered_docs.nlargest(top_k, "similarity")
        return top_docs["content"].tolist()

    def load_embedded_data(self, subject: str) -> pd.DataFrame:
        """Loads the embedded data for the given subject."""
        # TODO: In the future, this method should load the embedded data from a database
        if not os.path.exists(f"embedding_database/{subject}.parquet"):
            raise FileNotFoundError(
                f"Embedded data not found for subject: {subject}")
        df = pd.read_parquet(f"embedding_database/{subject}.parquet")
        print(f"Loaded {len(df)} embedded data.")

        return df

    def get_embedding(self, text: str) -> Tuple[np.ndarray, int]:
        """Gets the embedding of the given text using the OpenAI API."""
        client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY")
        )  # Load the OpenAI API key from the .env file
        response = client.embeddings.create(
            input=[text], model=self.embedding_model)
        print("Token Usage for Prompt Embedding: ", response.usage.total_tokens)
        return np.array(response.data[0].embedding), response.usage.total_tokens

    def cosine_similarity(self, a, b):
        """Calculates the cosine similarity between two vectors."""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
