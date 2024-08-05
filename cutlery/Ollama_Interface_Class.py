import ollama

class Ollama_Interface_Class:
    def __init__(self, model):
        self.model = model

    def chat(self, messages, tools=None):
        return ollama.chat(model=self.model, messages=messages, tools=tools)