from util.embedding_service.embedding_service import EmbeddingService

from util.embedding_service.openai_embedding_service import OpenAIEmbeddingService
from util.embedding_service.azure_embedding_service import AzureEmbeddingService

class EmbeddingServiceFactory:
    """
    Factory class to create instances of EmbeddingService based on the specified provider.

    This class provides a static method `get_embedding_service` that returns an instance
    of a concrete subclass of `EmbeddingService` corresponding to the given provider.
    It encapsulates the instantiation logic, making it easy to switch between different
    embedding service providers without changing the core codebase.

    Supported Providers:
        - 'openai': Uses the OpenAI API for embeddings.
        - 'azure': Uses the Azure OpenAI service for embeddings.

    Methods:
        get_embedding_service(provider: str, **kwargs) -> EmbeddingService:
            Static method to obtain an embedding service instance for the specified provider.
    """
    
    @staticmethod
    def get_embedding_service(provider: str, model_name: str = "text-embedding-3-large") -> EmbeddingService:
        """Get an instance of EmbeddingService for the specified provider."""
        if provider.lower() == "openai":
            return OpenAIEmbeddingService(model_name=model_name)
        elif provider.lower() == "azure":
            return AzureEmbeddingService(model_name=model_name)
        # Add additional providers here as you implement them
        else:
            raise ValueError(f"Unknown provider '{provider}'. Supported providers are 'openai' and 'azure'.")
