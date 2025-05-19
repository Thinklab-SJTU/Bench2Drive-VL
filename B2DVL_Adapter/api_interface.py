import requests
from openai import OpenAI
import time

class LLMApiInterface:
    def __init__(self, api_key, base_url):
        self.api_key = api_key
        self.base_url = base_url

    def ask(self, messages, max_tokens, response_format):
        raise NotImplementedError("To be implemented.")

class DeepseekInterface(LLMApiInterface):
    def __init__(self, api_key, base_url):
        super().__init__(api_key, base_url)

        self.api_keys = api_key if isinstance(api_key, list) else [api_key]
        self.clients = [OpenAI(api_key=key, base_url=self.base_url) for key in self.api_keys]
        self.key_index = 0
        
        self.base_url = base_url
        self.model = 'deepseek-chat'
        self.MAX_RETRIES = 32
        self.WAIT_TIME = 2
    
    def get_chat_response(self, messages, max_tokens=4096, response_format=None):
        retries = 0
        total_keys = len(self.api_keys)
        while retries < self.MAX_RETRIES:
            current_index = (self.key_index + retries) % total_keys
            client = self.clients[current_index]

            try:
                response = client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=max_tokens,
                    response_format=response_format
                )
                self.key_index = (current_index + 1) % total_keys
                return response
            except Exception as e:
                print(f"VLM API error: {e}. Retrying {retries + 1}/{self.MAX_RETRIES}...")
                time.sleep(self.WAIT_TIME)
                retries += 1
            # except Exception as e:
            #     print(f"Unexpected error: {e}")
            #     break
        print("VLM API max retries reached. Aborting this question.")
        return None
    
    def ask(self, messages, max_tokens=4096, response_format=None):
        # print("[debug] sleep before ask")
        time.sleep(0.2)
        # print("[debug] asking llm now")
        try:
            response = self.get_chat_response(
                messages=messages,
                max_tokens=max_tokens,
                response_format=response_format
            )

            return response.choices[0].message.content
        except Exception as e:
            print(f"Error communicating with Deepseek API: {e}")
            return None
