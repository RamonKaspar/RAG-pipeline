import os
from openai import OpenAI
import numpy as np
import tiktoken
from typing import List, Tuple

from util.embedding_service.embedding_service import EmbeddingService

class OpenAIEmbeddingService(EmbeddingService):
    def __init__(self, model_name: str = "text-embedding-3-large"):
        self.model_name = model_name
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def get_embeddings(self, texts: List[str]) -> Tuple[List[np.ndarray], int]:
        response = self.client.embeddings.create(input=texts, model=self.model_name)
        embeddings = [np.array(res.embedding) for res in response.data]
        total_tokens = response.usage.total_tokens
        return embeddings, total_tokens

    def count_tokens(self, text: str) -> int:
        tokenizer = tiktoken.encoding_for_model(self.model_name)
        return len(tokenizer.encode(text))