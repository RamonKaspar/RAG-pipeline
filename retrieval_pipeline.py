from typing import List

import os
import pandas as pd

class RetrievalPipeline:
    def __init__(self, subject: str):
        """Generates a pipeline for retrieving text snippets from resources of a specific school subject.

        Args:
            subject (str): The school subject for which the pipeline is created.
            data_folder (str): The folder containing the resources for the subject (in PDF format).
        """
        possible_subjects = ['math', 'physics', 'chemistry']
        if subject not in possible_subjects:
            raise ValueError("Invalid subject. Must be one of" + str(possible_subjects))
        self.subject = subject
        
        self.embedded_data = self.load_embedded_data(subject)   # pandas DataFrame containing the embedded data
        

    def retrieve(self, user_query: str, top_k: int, threshold: float) -> List[str]:
        """Runs the pipeline for the RAG model. Returns the response retrieved by the model.

        Args:
            user_query (str): The query for which we search for relevant information.
            top_k (int): The number of snippets to retrieve.
            threshold (float): The threshold for the similarity search in the retrieval.

        Returns:
            List[str]: The list of snippets retrieved by the model.
        """
        
        # Step 1: Retrieve Documents
        retrieved_docs = self.retrieve_documents(user_query)
        
        # Step 2: Select most relevant Documents
        selected_snippets = self.select_documents(retrieved_docs, top_k, threshold)
        
        # TODO: Whatever remaining steps are needed
        
        assert len(selected_snippets) <= top_k   # Ensure that the number of snippets is less than or equal to the number of snippets requested
        return selected_snippets
    
    def retrieve_documents(self, user_query: str) -> List[str]:
        """Retrieves documents relevant to the user query."""
        raise NotImplementedError("This method is not implemented yet.")

    def select_documents(self, retrieved_docs, num_snippets: int) -> List[str]:
        """Selects the most relevant documents from the retrieved documents."""
        raise NotImplementedError("This method is not implemented yet.")
    
    def load_embedded_data(self, subject: str) -> pd.DataFrame:
        """Loads the embedded data for the given subject."""
        # TODO: In the future, this method should load the embedded data from a database
        if not os.path.exists(f"embedding_database/{subject}.csv"):
            raise FileNotFoundError(f"Embedded data not found for subject: {subject}")
        df = pd.read_csv(f"embedding_database/{subject}.csv")
        return df
