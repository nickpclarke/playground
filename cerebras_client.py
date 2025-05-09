import os
from cerebras.cloud.sdk import Cerebras

class CerebrasClient:
    def __init__(self):
        self.client = Cerebras(
            api_key=os.environ.get("CEREBRAS_API_KEY"),
        )

    def get_chat_completion(self, user_message):
        chat_completion = self.client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": user_message,
                }
            ],
            model="llama-4-scout-17b-16e-instruct",
        )
        return chat_completion
    
    def get_text_completion(self, user_message):
        # Fix: The error indicates the API doesn't accept 'best_of' and 'echo'
        # and expects different parameters
        text_completion = self.client.completions.create(
            prompt=user_message,
            model="llama3.1-8b",
            max_tokens=256
            # Remove any default parameters that might be causing issues
            # Let the SDK use its defaults
        )
        return text_completion
    
# convenience wrapper so you can do `from cerebras_client import get_chat_completion`
def get_chat_completion(user_message):
    client = CerebrasClient()
    return client.get_chat_completion(user_message)

def get_text_completion(user_message):
    client = CerebrasClient()
    return client.get_text_completion(user_message)
