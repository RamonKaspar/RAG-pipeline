from util.llm_service.llm_factory import LLMServiceFactory
from src.retriever import RetrievedDocument
from typing import List, Tuple

class Generator:
    """Generator class that generates a response to a user query based on the retrieved data.
    TODO: This generate is only an example, the prompts should be optimized for the specific use case and
    a system prompt should be added."""
    
    def __init__(self, model_provider: str = "HuggingFace", model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct", temperature: float = 0.1, max_tokens: int = 2048):
        """Initializes the Generator class with the given school subject.
        We use the LLMServiceFactory to connect to an LLM API for text generation. Defaults to the Meta-Llama-3-8b model, that
        we can inference for free using the HuggingFace API.
        
        Args:
            model_provider (str): The model provider used for the text generation. Defaults to "HuggingFace".
            model_name (str): The model name used for the text generation. Defaults to "meta-llama/Meta-Llama-3-8B-Instruct".
            debug (bool): Whether to print debug information.
        """
        self.client = LLMServiceFactory().get_service(
            provider=model_provider, 
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens
        )
        self.max_tokens = max_tokens

    def generate_response(self, user_query: str, retrieved_data: List[RetrievedDocument]) -> Tuple[str, int]:
        """Generates a response to a user query based on the retrieved data
        
        Parameters:
            user_query (str): The user query for which we generate a response.
            retrieved_data (List[RetrievedDocument]): The list of retrieved documents.
        """
        prompt = f"""Use the below information chunks to answer the subsequent question. If the answer cannot be found, write 'I don't know.'
    
        DOCUMENTS:
        {self.format_documents(retrieved_data)}
        
        Whenever possible cite the source of the information using the metadata provided (i.e., mention for example the title of the document and on which page).
        
        Question: {user_query}
        """
        
        response = self.client.make_request(messages=[{"role": "user", "content": prompt}])
        return response
    
    def format_documents(self, retrieved_data: List[RetrievedDocument]) -> str:
        """Formats the retrieved documents into a string that can be used as a prompt for the LLM API.
        
        Parameters:
            retrieved_data (List[RetrievedDocument]): The list of retrieved documents.
        """
        formatted_documents = ""
        total_length = 0
        for i, doc in enumerate(retrieved_data):
            metadata_str = doc.metadata or ""
            doc_content = f"Metadata: {metadata_str}\nContent: {doc.content}\n\n"
            total_length += len(doc_content)
            if total_length > self.max_tokens:
                break
            formatted_documents += doc_content
        return formatted_documents