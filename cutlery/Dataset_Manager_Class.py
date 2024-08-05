import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
import json

class Dataset_Manager_Class:
    def __init__(self, ollama_interface):
        self.ollama_interface = ollama_interface

    def generate_dataset(self, params, seed_data=None):
        dataset = []
        if seed_data is not None:
            dataset.extend(seed_data.to_dict(orient='records'))
            remaining_size = params['size'] - len(dataset)
        else:
            remaining_size = params['size']

        for _ in tqdm(range(remaining_size), desc="Generating dataset"):
            data_point = self.generate_data_point(params['fields'])
            dataset.append(data_point)
        return dataset

    def generate_data_point(self, fields):
        data_point = {}
        for field in fields:
            prompt = f"Generate data for {field['name']}: {field['prompt']}"
            response = self.ollama_interface.chat(messages=[
                {'role': 'system', 'content': 'You are a data generation assistant. Respond only with the requested data point as a valid JSON object, nothing else.'},
                {'role': 'user', 'content': prompt}
            ])
            data_point[field['name']] = self.parse_response(response, field['name'])
        return data_point

    def parse_response(self, response, field_name):
        if 'tool_calls' in response['message']:
            tool_call = response['message']['tool_calls'][0]
            if tool_call['function']['name'] == 'generate_random_data':
                try:
                    return json.loads(tool_call['function']['arguments'])
                except json.JSONDecodeError:
                    print(f"Failed to parse JSON for {field_name}. Using raw response.")
                    return tool_call['function']['arguments']
        else:
            try:
                return json.loads(response['message']['content'].strip())
            except json.JSONDecodeError:
                print(f"Failed to parse JSON for {field_name}. Using raw response.")
                return response['message']['content'].strip()

    def generate_synthetic_data(self, original_data):
        synthetic_data = []
        for _, row in tqdm(original_data.iterrows(), desc="Generating synthetic data"):
            for _ in range(10):  # Generate 10 duplicates for each data point
                synthetic_row = {}
                for column in original_data.columns:
                    prompt = f"Generate a similar but different {column} based on: {row[column]}. Maintain the same meaning but use different phrasing."
                    response = self.ollama_interface.chat(messages=[
                        {'role': 'system', 'content': 'You are a data generation assistant. Respond only with the requested data point, nothing else.'},
                        {'role': 'user', 'content': prompt}
                    ])
                    synthetic_row[column] = response['message']['content'].strip()
                synthetic_data.append(synthetic_row)
        return synthetic_data

    def clone_huggingface_dataset(self, dataset_name):
        dataset = load_dataset(dataset_name)
        return dataset['train'].to_pandas()

    def convert_json_to_parquet(self, json_file_path, parquet_file_path):
        try:
            df = pd.read_json(json_file_path)
            df.to_parquet(parquet_file_path, engine='pyarrow')
            return parquet_file_path
        except Exception as e:
            print(f"Error converting JSON to Parquet: {str(e)}")
            return None
