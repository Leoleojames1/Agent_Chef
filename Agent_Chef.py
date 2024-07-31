""" Agent_Chef.py

    Agent_Chef is a personal Parquet dataset procedural construction and sythetic generation tool for
    personal dataset expansion.
    
    @Leo Borcherding
    6/30/2024
    ### MIT COPYRIGHT LISCENSE ###
"""

import ollama
import pandas as pd
from tqdm import tqdm
import os
import json
from datasets import load_dataset
from halo import Halo
from colorama import init, Fore, Style

# Initialize colorama
init(autoreset=True)

#TODO
# 1. Garbage Data Collector
# 2. Abliteration
# 3. Reverse Abliteration

class Agent_Chef:
    def __init__(self, dataset_params=None, seed_file=None, mode='custom', user_json=None): 
        
        print(Fore.CYAN + "Welcome to Agent_Chef!")
        
        self.user_model_select = input(Fore.YELLOW + "Which Ollama Model do you want to let cook? \n")
        
        
        self.model = self.user_model_select
        self.base_dir = os.path.join(os.getcwd(), "agent_chef_data")
        self.input_dir = os.path.join(self.base_dir, "ingredients")
        self.output_dir = os.path.join(self.base_dir, "dishes")
        self.dataset_params = dataset_params or {
            'size': 100,
            'fields': [
                {'name': 'ai_technical_phrases', 'prompt': 'Generate a JSON object with 10 key-value pairs, where keys are technical AI phrases and values are their definitions.'},
            ]
        }
        self.seed_file = seed_file
        self.mode = mode
        self.user_json = user_json
        os.makedirs(self.input_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)

        # Define tools for the Ollama model
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

    def load_seed_data(self):
        if self.seed_file and os.path.exists(os.path.join(self.input_dir, self.seed_file)):
            return pd.read_parquet(os.path.join(self.input_dir, self.seed_file))
        else:
            print(f"{Fore.RED}Seed file {self.seed_file} not found in the ingredients directory.")
            return None

    def save_json_to_parquet(self, json_file):
        with open(json_file, 'r') as f:
            data = json.load(f)
        df = pd.DataFrame(data)
        parquet_file = os.path.join(self.output_dir, 'user_data.parquet')
        df.to_parquet(parquet_file, engine='pyarrow')
        return parquet_file

    def clone_huggingface_dataset(self, dataset_name):
        response = ollama.chat(
            model=self.model,
            messages=[
                {'role': 'system', 'content': 'You are a data loading assistant.'},
                {'role': 'user', 'content': f"Load the dataset '{dataset_name}' from Hugging Face"}
            ],
            tools=self.tools
        )
        
        if 'tool_calls' in response['message']:
            tool_call = response['message']['tool_calls'][0]
            if tool_call['function']['name'] == 'load_dataset':
                # Here we're still using the actual load_dataset function
                # In a full implementation, you might want to handle this differently
                dataset = load_dataset(dataset_name)
                return dataset['train'].to_pandas()
        
        # Fallback to the original method if the tool wasn't used
        dataset = load_dataset(dataset_name)
        return dataset['train'].to_pandas()

    def generate_data_point(self, fields):
        data_point = {}
        for field in fields:
            prompt = f"Generate data for {field['name']}: {field['prompt']}"
            response = ollama.chat(
                model=self.model,
                messages=[
                    {'role': 'system', 'content': 'You are a data generation assistant. Respond only with the requested data point as a valid JSON object, nothing else.'},
                    {'role': 'user', 'content': prompt}
                ],
                tools=self.tools
            )
            
            if 'tool_calls' in response['message']:
                tool_call = response['message']['tool_calls'][0]
                if tool_call['function']['name'] == 'generate_random_data':
                    try:
                        data_point[field['name']] = json.loads(tool_call['function']['arguments'])
                    except json.JSONDecodeError:
                        print(f"Failed to parse JSON for {field['name']}. Using raw response.")
                        data_point[field['name']] = tool_call['function']['arguments']
            else:
                try:
                    data_point[field['name']] = json.loads(response['message']['content'].strip())
                except json.JSONDecodeError:
                    print(f"Failed to parse JSON for {field['name']}. Using raw response.")
                    data_point[field['name']] = response['message']['content'].strip()
        return data_point

    def generate_dataset(self):
        dataset = []
        if hasattr(self, 'seed_data') and self.seed_data is not None:
            dataset.extend(self.seed_data.to_dict(orient='records'))
            remaining_size = self.dataset_params['size'] - len(dataset)
        else:
            remaining_size = self.dataset_params['size']

        for _ in tqdm(range(remaining_size), desc="Generating dataset"):
            data_point = self.generate_data_point(self.dataset_params['fields'])
            dataset.append(data_point)
        return dataset

    def generate_synthetic_data(self, original_data):
        synthetic_data = []
        for _, row in tqdm(original_data.iterrows(), desc="Generating synthetic data"):
            for _ in range(10):  # Generate 10 duplicates for each data point
                synthetic_row = {}
                for column in original_data.columns:
                    prompt = f"Generate a similar but different {column} based on: {row[column]}. Maintain the same meaning but use different phrasing."
                    response = ollama.chat(model=self.model, messages=[
                        {'role': 'system', 'content': 'You are a data generation assistant. Respond only with the requested data point, nothing else.'},
                        {'role': 'user', 'content': prompt}
                    ])
                    synthetic_row[column] = response['message']['content'].strip()
                synthetic_data.append(synthetic_row)
        return synthetic_data

    def save_json_to_parquet(self):
        json_path = os.path.join(self.input_dir, self.user_json)
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                data = json.load(f)
            df = pd.DataFrame(data)
            parquet_file = os.path.join(self.output_dir, 'user_data_dish.parquet')
            df.to_parquet(parquet_file, engine='pyarrow')
            return parquet_file
        else:
            print(f"{Fore.RED}JSON file {self.user_json} not found in the ingredients directory.")
            return None
        
    def save_to_parquet(self, dataset, filename='dataset.parquet'):
        file_path = os.path.join(self.output_dir, filename)
        df = pd.DataFrame(dataset)
        df.to_parquet(file_path, engine='pyarrow')
        return file_path

    def build_user_json(self):
        print(Fore.CYAN + "Let's build a JSON file with your custom data!")
        data = []
        while True:
            entry = {}
            print(Fore.YELLOW + "Enter your data (press Enter without typing to finish):")
            formula = input("Formula: ").strip()
            if not formula:
                break
            entry['formula'] = formula
            entry['explanation'] = input("Explanation: ").strip()
            entry['python_code'] = input("Python code (if applicable): ").strip()
            entry['example'] = input("Example: ").strip()
            data.append(entry)
        
        filename = input(Fore.YELLOW + "Enter a name for your JSON file: ").strip()
        if not filename.endswith('.json'):
            filename += '.json'
        file_path = os.path.join(self.input_dir, filename)
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
        print(Fore.GREEN + f"JSON file saved as {file_path}")
        return filename

    def main(self):
        
        print(Fore.RED + "########################################################")
        print(Fore.CYAN + "This tool helps you cook up synthetic datasets using custom user data or Hugging Face datasets.")
        print(Fore.CYAN + "You can choose between four cooking modes:")
        print(Fore.CYAN + "1. Custom: Create a dataset from scratch with custom ingredients.")
        print(Fore.CYAN + "2. Hugging Face: Use a pre-made dataset from Hugging Face and add our special sauce.")
        print(Fore.CYAN + "3. JSON: Transform your JSON ingredients into a gourmet Parquet dish.")
        print(Fore.CYAN + "4. Build JSON: Create a custom JSON file with your own formulas and data.")
        print(Fore.RED + "########################################################")

        self.mode = input(Fore.YELLOW + "Enter cooking mode (custom/huggingface/json/build): ").strip().lower()

        if self.mode == 'build':
            self.user_json = self.build_user_json()
            self.mode = 'json'  # Switch to JSON mode after building
        else:
            self.seed_file = input(Fore.YELLOW + f"Enter seed file name in the ingredients directory ({self.input_dir}), if any: ").strip()
            dataset_name = input(Fore.YELLOW + "Enter Hugging Face dataset name (if using huggingface mode): ").strip()
            self.user_json = input(Fore.YELLOW + f"Enter JSON file name in the ingredients directory ({self.input_dir}), if using json mode: ").strip()

        custom_spinner = {"interval": 80, "frames": ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]}
        spinner = Halo(text='Cooking in progress', spinner=custom_spinner)
        spinner.start()

        if self.mode == 'custom':
            print(Fore.GREEN + "Preparing a fresh synthetic dataset...")
            dataset = self.generate_dataset()
        elif self.mode == 'huggingface':
            print(Fore.GREEN + f"Fetching Hugging Face dataset: {dataset_name}")
            original_data = self.clone_huggingface_dataset(dataset_name)
            print(Fore.GREEN + "Adding our special sauce to create a synthetic dataset...")
            dataset = self.generate_synthetic_data(original_data)
        elif self.mode == 'json':
            print(Fore.GREEN + f"Transforming JSON ingredients into Parquet: {self.user_json}")
            parquet_file = self.save_json_to_parquet()
            if parquet_file:
                print(Fore.GREEN + f"Cooking up a synthetic dataset from the ingredients in {parquet_file}...")
                original_data = pd.read_parquet(parquet_file)
                dataset = self.generate_synthetic_data(original_data)
            else:
                spinner.stop()
                print(Fore.RED + "Failed to process JSON ingredients. Cooking aborted.")
                return

        saved_path = self.save_to_parquet(dataset)
        spinner.stop()
        print(Fore.BLUE + f"A delicious dataset with {len(dataset)} entries has been served in Parquet format at: {saved_path}")

if __name__ == "__main__":
    chef = Agent_Chef()
    chef.main()