import os
import openai
from openai import OpenAI

# Create client once at module level
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

def get_openai_chat_completion(user_message: str, 
                               model: str = "gpt-3.5-turbo") -> dict:
    """
    Send a chat‐style completion request to OpenAI.
    Returns the full response dict.
    """
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": user_message}]
    )
    return resp

def get_openai_text_completion(prompt: str, 
                               model: str = "text-davinci-003",
                               max_tokens: int = 256) -> dict:
    """
    Send a text‐completion (single prompt) request to OpenAI.
    Returns the full response dict.
    """
    resp = client.completions.create(
        model=model,
        prompt=prompt,
        max_tokens=max_tokens
    )
    return resp

def get_openai_embeddings(text):
    """
    Get embeddings for a text using the OpenAI API.
    
    Args:
        text (str): The text to embed
        
    Returns:
        list: The embedding vector or None if there was an error
    """
    try:
        # Get embeddings using the updated client API
        response = client.embeddings.create(
            model="text-embedding-ada-002",
            input=text
        )
        
        # Extract the embedding vector (note the new structure)
        embedding = response.data[0].embedding
        return embedding
    
    except Exception as e:
        print(f"Error getting embeddings: {str(e)}")
        # Log the text that caused the issue (first 100 chars)
        print(f"Problem text: {text[:100]}... (length: {len(text)})")
        return None