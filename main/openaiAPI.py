import openai
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get API key from environment variable
openai.api_key = os.getenv("OPENAI_API_KEY")

GPT_MODEL = "gpt-3.5-turbo"

# Helper function
def query_gpt(question: str, context: str = None):
    """Generate a single detailed response using OpenAI GPT."""
    messages = [{"role": "user", "content": question}]
    if context:
        messages.insert(0, {"role": "system", "content": context})
    
    response = openai.ChatCompletion.create(
        model=GPT_MODEL,
        messages=messages,
        temperature=0.7,
    )
    return response.choices[0].message.content

# User asks a question
# USER_QUESTION = "Wo ging Hegel zur Schule?"

# # Generate a single detailed answer
# response = query_gpt(
#     USER_QUESTION,
#     context="You are a helpful assistant answering user questions with accurate, concise, and detailed information. always answer in english",
# )

# Display the result
# print("User Question:", USER_QUESTION)
# print("\nResponse:")
# print(response)
