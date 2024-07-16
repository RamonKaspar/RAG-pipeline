import os
from openai import OpenAI

from dotenv import load_dotenv
load_dotenv()


class Generator:

    def generate_response(self, subject, retrieved_data, user_query, max_tokens, temperature):
        """Generates a response to a user query based on the retrieved data."""
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY")) # Load the OpenAI API key from the .env file
        prompt = f"Subject: {subject}\nContext: {retrieved_data}\n\nUser Query: {user_query}\n\nResponse:"
        response = client.chat.completions.create(
            messages=[
                
                {"role": "user", "content": prompt}
            ],
            model="gpt-4",
            max_tokens=max_tokens,
            n=1,
            stop=None,
            temperature=temperature
        )
        
        return response.choices[0].message.content.strip()