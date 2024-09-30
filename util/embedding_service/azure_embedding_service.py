import os
from openai import AzureOpenAI
import numpy as np
import tiktoken
from typing import List, Tuple

from util.embedding_service.embedding_service import EmbeddingService

class AzureEmbeddingService(EmbeddingService):
    def __init__(self, model_name: str = "text-embedding-3-large"):
        super().__init__(model_name=model_name)
        self.client  = AzureOpenAI(
            api_key = os.getenv("AZURE_OPENAI_API_KEY"),  
            api_version = "2024-06-01",
            azure_endpoint =os.getenv("AZURE_OPENAI_ENDPOINT") 
        )

    def get_embeddings(self, texts: List[str]) -> Tuple[List[np.ndarray], int]:
        response = self.client.embeddings.create(input=texts, model=self.model_name)
        embeddings = [np.array(res.embedding) for res in response.data]
        total_tokens = response.usage.total_tokens
        return embeddings, total_tokens

    def count_tokens(self, text: str) -> int:
        tokenizer = tiktoken.encoding_for_model(self.model_name)
        return len(tokenizer.encode(text))