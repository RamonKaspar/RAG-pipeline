# Embedding Service

This directory contains the implementation of the Embedding Service used in our RAG pipeline. It allows us to easily switch between different embedding models and providers, facilitating testing and experimentation.

## Overview

We designed the Embedding Service to:

- **Support Multiple Providers**: Currently supports OpenAI and Azure OpenAI.
- **Ease of Switching**: Swap between providers and models without changing core code.
- **Facilitate Testing**: Quickly test different models to find the best fit for our application.

## How It Works

- **Abstract Base Class (`embedding_service.py`)**: Defines the `EmbeddingService` interface with methods like `get_embeddings` and `count_tokens`.
- **Provider Implementations**:
  - `openai_embedding_service.py`: Implements `EmbeddingService` using the OpenAI API.
  - `azure_embedding_service.py`: Implements `EmbeddingService` using the Azure OpenAI service.
- **Factory (`embedding_service_factory.py`)**: Contains `EmbeddingServiceFactory` to instantiate the appropriate service based on the specified provider.

## Usage

1. **Import the Factory**:

   ```python
   from util.embedding_service.embedding_service_factory import EmbeddingServiceFactory
   ```

2. **Create an Embedding Service Instance**:

   - For **OpenAI**:

     ```python
     embedding_service = EmbeddingServiceFactory.get_embedding_service(
         provider='openai',
         model_name='text-embedding-ada-002'  # Replace with your desired model
     )
     ```

   - For **Azure OpenAI**:

     ```python
     embedding_service = EmbeddingServiceFactory.get_embedding_service(
         provider='azure',
         model_name='text-embedding-ada-002'  # Replace with your desired model
     )
     ```

3. **Generate Embeddings**:

   ```python
   texts = ["Your text here"]
   embeddings, total_tokens = embedding_service.get_embeddings(texts)
   ```

## Why We Did It

To support multiple embedding models and providers, allowing us to easily switch between them and test different models without modifying the core codebase.
