import ollama
import json

class OllamaInterface:
    def __init__(self, model):
        self.model = model

    def set_model(self, model):
        self.model = model

    def chat(self, messages):
        try:
            response = ollama.chat(model=self.model, messages=messages)
            return response
        except Exception as e:
            print(f"Error in Ollama chat: {str(e)}")
            return {"message": {"content": f"Error: {str(e)}"}}

    def chat_json(self, messages):
        try:
            response = ollama.chat(model=self.model, messages=messages, format='json')
            return response.json
        except Exception as e:
            print(f"Error in Ollama chat JSON mode: {str(e)}")
            return None

    def list_models(self):
        models = ollama.list()
        return [model['name'] for model in models['models']]