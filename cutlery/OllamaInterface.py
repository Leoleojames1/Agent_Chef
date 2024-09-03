import ollama
import json
from colorama import Fore, Back, Style
from colorama import init
init(autoreset=True)

class OllamaInterface:
    def __init__(self, model):
        self.model = model

    def set_model(self, model):
        self.model = model

    def is_llama_3_1(self):
        return "llama3.1" in self.model.lower()
    
    def chat(self, messages):
        try:
            if self.is_llama_3_1():
                response = ollama.chat(model=self.model, messages=messages)
                print(f"{Fore.YELLOW}Model Response:{Style.RESET_ALL} {response['message']['content']}")
                return response
            else:
                response = ollama.chat(model=self.model, messages=messages)
                content = response['message']['content']
                print(f"{Fore.YELLOW}Model Response:{Style.RESET_ALL} {content}")
                return {"message": {"content": content}}
        except Exception as e:
            print(f"{Fore.RED}Error in Ollama chat: {str(e)}{Style.RESET_ALL}")
            return {"message": {"content": f"Error: {str(e)}"}}

    def chat_json(self, messages):
        try:
            if self.is_llama_3_1():
                response = ollama.chat(model=self.model, messages=messages, format='json')
                return response.json
            else:
                response = ollama.chat(model=self.model, messages=messages)
                try:
                    return json.loads(response['message']['content'])
                except json.JSONDecodeError:
                    print(f"{Fore.RED}Error: Response is not valid JSON{Style.RESET_ALL}")
                    return None
        except Exception as e:
            print(f"{Fore.RED}Error in Ollama chat JSON mode: {str(e)}{Style.RESET_ALL}")
            return None

    def list_models(self):
        models = ollama.list()
        return [model['name'] for model in models['models']]