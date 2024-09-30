# RAG-pipeline

A pipeline for building a Retrieval-Augmented Generation (RAG) system using document embedding.

We use three stages in the pipeline:

- **(1) Collect Data**: Preprocess a corpus of PDF and HTML documents to build a database of embedded text chunks for a specified subject.
- **(2) Preprocessing**: Extract text from documents and embed text chunks.
- **(3) Retrieval**: Retrieve relevant information from the database using cosine similarity.
- **(4) Generation**: Generate responses using the retrieved information with a large language model (LLM).

## (1) Collect Data: Data Collection for RAG-Pipeline

Manually place the PDF and HTML documents in the `data/{subject}` folder. The folder structure should look like this:

```bash
data/
└── subject/
    ├── document1.pdf
    ├── document2.html
    └── ...
```

Note: Use meaningful document names for the PDF and HTML files, since the document name is used as source information in the RAG system (so we can reference the actual document later).

## (2) Preprocessing: Database Builder for RAG-Pipeline

The module, `database_builder.py`, preprocesses a corpus of PDF and HTML documents to build a database of embedded text chunks for a specified subject. This is part of the initial preprocessing step in the RAG pipeline.

- Extracts text from PDF and HTML documents, located in `data/{subject}` folder.
- Chunks text into manageable pieces with specified size and overlap.
- Embeds text chunks using the specified embedding model (e.g. `text-embedding-3-large` from OpenAI).
- Saves embedded chunks to a Parquet file located in `embedding_database/{subject}.parquet` for further use. Note: This should be moved to an actual vector database in the future.

### Usage

1. Install dependencies: `pip install -r requirements.txt`
2. Ensure the OpenAI API key is set in a `.env` file.

```python
from database_builder import DatabaseBuilder

# Initialize the DatabaseBuilder
db_builder = DatabaseBuilder(
    subject="test", # Replace with actual subject (e.g. "biology")
    embedding_model="text-embedding-3-large",   # or "text-embedding-3-small"
    chunk_size=1000,
    overlap_size=50,
    min_text_length=0
)

# Build the database
db_builder.build_database(root_folder_path="data/")
```

### ToDo's

- [ ] Determine the optimal values for `chunk_size`, `overlap_size`, and `min_text_length`. Many different values are used in research papers.
- [ ] Implement a database to store the embedded text chunks. For now, we save the embedded chunks to a Parquet file.
- [ ] Support other file types in addition to PDF and HTML?
- [ ] Which embedding model to use? We currently use the `text-embedding-3-large` model from OpenAI.
- [ ] Consider semantic chunking stratgies instead of the `RecursiveTextSplitter`.

## (3) Retrieval: Find Relevant Information

In this stage, we retrieve relevant information from the embedded database using the user's query. The retrieval process involves:

- **Embedding the User Query:** The query (or list of queries) is embedded using the same embedding model used to create the database.
- **Computing Similarities:** We compute the cosine similarity between the query embeddings and the document embeddings in the database.
- **Retrieving Top-K Documents:** The documents with the highest similarity scores above a specified threshold are retrieved.
- **Thresholding:** We can set a threshold to filter out documents with low similarity scores.

We support both single queries and lists of queries, which can be useful when you want to capture different aspects or phrasings of a question.

### Usage

```python
from retriever import Retriever

# Initialize the Retriever
retriever = Retriever(
    subject="test",  # Replace with actual subject (e.g., "biology")
    embedding_model="text-embedding-3-large",  # Use the same model used in preprocessing
    debug=False  # Set to True to enable debug information
)

# Query the database
query = "What is the capital of France?"
retrieved_docs, total_tokens = retriever.retrieve(
    user_queries=[query],  # List of user queries
    top_k=5,               # Number of top documents to retrieve
    threshold=0.2          # Cosine similarity threshold
)

# Process the retrieved documents
for idx, doc in enumerate(retrieved_docs):
    print(f"\nDocument {idx + 1}")
    print(f"Content: {doc.content}")
    print(f"Metadata: {doc.metadata}")
    print(f"Similarity: {doc.similarity:.4f}")

```

## (4) Generation: Generate Responses

In this optional final stage, we generate a response to the user's query using the retrieved documents. The Generator class utilizes a language model to produce an answer that is informed by the retrieved chunks of information. We use a generic factory (LLMServiceFactory) to interact with different language models (LLMs) – for more details, see the `README.md` in the `llm_service` folder. \\
This step is optional. If you only need to retrieve relevant chunks, you can omit this part.

### Usage

```python
from generator import Generator
from retriever import Retriever

# Assume we have already retrieved documents using the Retriever
retriever = Retriever(
    subject="test",
    embedding_model="text-embedding-ada-002"
)
user_query = "What is the capital of France?"
retrieved_docs, _ = retriever.retrieve(
    user_queries=[user_query],
    top_k=5,
    threshold=0.2
)

# Initialize the Generator
generator = Generator(
    model_provider="HuggingFace",
    model_name="meta-llama/Meta-Llama-3-8B-Instruct",
    temperature=0.1,
    max_tokens=2048
)

# Generate a response
response = generator.generate_response(
    user_query=user_query,
    retrieved_data=retrieved_docs
)

# Output the generated response
print("Generated Response:")
print(response)
```

The retrieved documents are formatted and included in the prompt to provide context to the language model. You can customize the prompt in the `generate_response` method to better suit your use case or to optimize the model's output.
