from abc import ABC, abstractmethod
from typing import List, Tuple
import numpy as np

class EmbeddingService(ABC):
    """
    Abstract base class defining the interface for embedding services.
    This allows for easy switching between different embedding models and APIs.

    This class outlines the methods that any embedding service must implement.
    Subclasses should provide concrete implementations for generating embeddings
    and counting tokens for text data.

    Methods:
        get_embedding(text: str) -> np.ndarray:
            Generate an embedding vector for a single text string.

        get_embeddings(texts: List[str]) -> Tuple[List[np.ndarray], int]:
            Generate embedding vectors for a list of text strings.

        count_tokens(text: str) -> int:
            Count the number of tokens in a text string using the appropriate tokenizer.
    """
    def __init__(self, model_name: str):
        self.model_name = model_name

    def get_embedding(self, text: str) -> np.ndarray:
        """
        Generate an embedding vector for a single text string.

        Args:
            text (str): The text string to be embedded.

        Returns:
            np.ndarray: The embedding vector represented as a NumPy array.
        """
        return self.get_embeddings(texts=[text])[0]

    @abstractmethod
    def get_embeddings(self, texts: List[str]) -> Tuple[List[np.ndarray], int]:
        """
        Generate embedding vectors for a list of text strings.

        Args:
            texts (List[str]): A list of text strings to be embedded.

        Returns:
            Tuple[List[np.ndarray], int]:
                - List[np.ndarray]: A list of embedding vectors, each represented as a NumPy array.
                - int: The total number of tokens used during the embedding process.
        """
        raise NotImplementedError("Method 'get_embeddings' must be implemented in a subclass.")

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """
        Count the number of tokens in a text string using the appropriate tokenizer.

        Args:
            text (str): The text string for which to count the tokens.

        Returns:
            int: The number of tokens in the text string.
        """
        raise NotImplementedError("Method 'count_tokens' must be implemented in a subclass.")
