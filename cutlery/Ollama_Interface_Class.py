import ollama
import json

class Ollama_Interface_Class:
    def __init__(self, model):
        self.model = model
        self.tools = [
            {
                'type': 'function',
                'function': {
                    'name': 'generate_random_data',
                    'description': 'Generate random data based on a given prompt',
                    'parameters': {
                        'type': 'object',
                        'properties': {
                            'prompt': {
                                'type': 'string',
                                'description': 'The prompt for generating random data',
                            },
                        },
                        'required': ['prompt'],
                    },
                },
            },
            {
                'type': 'function',
                'function': {
                    'name': 'load_dataset',
                    'description': 'Load a dataset from Hugging Face',
                    'parameters': {
                        'type': 'object',
                        'properties': {
                            'dataset_name': {
                                'type': 'string',
                                'description': 'The name of the dataset on Hugging Face',
                            },
                        },
                        'required': ['dataset_name'],
                    },
                },
            },
        ]

    def chat(self, messages, use_tools=True):
        try:
            if use_tools:
                response = ollama.chat(model=self.model, messages=messages, tools=self.tools)
            else:
                response = ollama.chat(model=self.model, messages=messages)
            
            return response
        except Exception as e:
            print(f"Error in Ollama chat: {str(e)}")
            return {"message": {"content": f"Error: {str(e)}"}}

    def list_models(self):
        models = ollama.list()
        return [model['name'] for model in models['models']]

    def parse_tool_response(self, response):
        if 'tool_calls' in response['message']:
            tool_call = response['message']['tool_calls'][0]
            function_name = tool_call['function']['name']
            try:
                arguments = json.loads(tool_call['function']['arguments'])
            except json.JSONDecodeError:
                arguments = tool_call['function']['arguments']
            return function_name, arguments
        return None, None