import os
from cerebras.cloud.sdk import Cerebras

# Default model to use across all functions
DEFAULT_MODEL = "Qwen-3-32B"

class CerebrasClient:
    def __init__(self, api_key=None):
        self.client = Cerebras(
            api_key=api_key or os.environ.get("CEREBRAS_API_KEY"),
        )

    def get_chat_completion(self, user_message, model=DEFAULT_MODEL):
        chat_completion = self.client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": user_message,
                }
            ],
            model=model,
        )
        return chat_completion
    
    def get_text_completion(self, user_message, model=DEFAULT_MODEL):
        text_completion = self.client.completions.create(
            prompt=user_message,
            model=model,
            max_tokens=256
        )
        return text_completion

# Simplified convenience wrappers
def get_chat_completion(prompt, model=DEFAULT_MODEL):
    """Get a chat completion from the Cerebras API."""
    client = CerebrasClient()
    return client.get_chat_completion(prompt, model=model)

def get_text_completion(user_message, model=DEFAULT_MODEL):
    """Get a text completion from the Cerebras API."""
    client = CerebrasClient()
    return client.get_text_completion(user_message, model=model)
