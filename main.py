from database_builder import DatabaseBuilder

""" 
Example Usage of DatabaseBuilder class with the subject "test" and the embedding model "text-embedding-3-large". 
The data that is used has to be in the data/test folder.
TODO: What is the best value for chunk_size, overlap_size and min_text_length?
"""

db_builder = DatabaseBuilder(subject="test", database=None, embedding_model="text-embedding-3-large", chunk_size=1000, overlap_size=50, min_text_length=0)
db_builder.build_database(root_folder_path="data/")
