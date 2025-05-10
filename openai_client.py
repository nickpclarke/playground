import os
import openai

# Load your OpenAI key from environment
openai.api_key = os.environ.get("OPENAI_API_KEY")

def get_openai_chat_completion(user_message: str, 
                               model: str = "gpt-3.5-turbo") -> dict:
    """
    Send a chat‐style completion request to OpenAI.
    Returns the full response dict.
    """
    resp = openai.ChatCompletion.create(
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
    resp = openai.Completion.create(
        model=model,
        prompt=prompt,
        max_tokens=max_tokens
    )
    return resp

def get_openai_embeddings(texts: list[str], 
                          model: str = "text-embedding-ada-002") -> list[list[float]]:
    """
    Batch‐embed a list of strings using OpenAI’s embeddings endpoint.
    Uses the new openai-python interface (>=1.0.0).
    """
    resp = openai.embeddings.create(
        model=model,
        input=texts
    )
    # resp.data is a list of CreateEmbeddingResponseItem objects
    return [item.embedding for item in resp.data]