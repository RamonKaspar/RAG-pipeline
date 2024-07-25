import os
from openai import OpenAI


class Generator:

    def generate_response(
        self, subject, retrieved_data, user_query, max_tokens, temperature
    ):
        """Generates a response to a user query based on the retrieved data."""

        print("Generating response for user query...")
        # Load the OpenAI API key from the .env file
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        prompt = f"Subject: {subject}\nContext: {retrieved_data}\n\nUser Query: {user_query}\n\nResponse:"
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="gpt-4o",
            max_tokens=max_tokens,
            n=1,
            stop=None,
            temperature=temperature,
        )

        if response is not None and response.usage is not None:
            return response.choices[0].message.content, response.usage.total_tokens
        else:
            return None, 0
