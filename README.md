# RAG-pipeline

A pipeline for building a Retrieval-Augmented Generation (RAG) system using document embedding.

We use two stages in the pipeline:

- **(1) Preprocessing**: Extract text from documents and embed text chunks.
- **(2) RAG-pipeline**: Use the embedded text chunks to build a RAG system.

## (1) Preprocessing: Database Builder for RAG-Pipeline

The module, `database_builder.py`, preprocesses a corpus of PDF and HTML documents to build a database of embedded text chunks for a specified subject. This is part of the initial preprocessing step in the RAG pipeline.

- Extracts text from PDF and HTML documents, located in `data/{subject}` folder.
- Chunks text into manageable pieces with specified size and overlap.
- Embeds text chunks using the specified embedding model (e.g. `text-embedding-3-large` from OpenAI).
- Saves embedded chunks to a CSV file located in `embeddings/{subject}.csv` for further use. Note: This should be moved to an actual database in the future.

### Usage

1. Install dependencies: `pip install -r requirements.txt`
2. Ensure the OpenAI API key is set in a `.env` file.

```python
from database_builder import DatabaseBuilder

# Initialize the DatabaseBuilder
db_builder = DatabaseBuilder(
    subject="test", # Replace with actual subject (e.g. "biology")
    database=None,  # Replace with actual database object in the future
    embedding_model="text-embedding-3-large",   # or "text-embedding-3-small"
    chunk_size=1000,
    overlap_size=50,
    min_text_length=0
)

# Build the database
db_builder.build_database(root_folder_path="data/")
```

### ToDo's

- [] Determine the optimal values for `chunk_size`, `overlap_size`, and `min_text_length`.
- [] Implement a database to store the embedded text chunks.
- [] Support other file types in addition to PDF and HTML?

## (2) RAG-pipeline

Blabla
