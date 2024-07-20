import ollama
import pandas as pd
from tqdm import tqdm
import os

class Agent_Chef:
    def __init__(self, model='qwen2', dataset_params=None, seed_file=None):
        self.model = model
        self.output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'datasets', 'output')
        self.dataset_params = dataset_params or {
            'size': 100,
            'fields': [
                {'name': 'customer_name', 'prompt': 'Generate a random customer name'},
                {'name': 'age', 'prompt': 'Generate a random age between 18 and 80'},
                {'name': 'purchase_amount', 'prompt': 'Generate a random purchase amount between $10 and $1000'},
                {'name': 'product_category', 'prompt': 'Generate a random product category (e.g., electronics, clothing, food)'},
            ]
        }
        self.seed_data = self.load_seed_data(seed_file) if seed_file else None
        os.makedirs(self.output_dir, exist_ok=True)

    def load_seed_data(self, seed_file):
        if os.path.exists(seed_file):
            return pd.read_parquet(seed_file)
        else:
            print(f"Seed file {seed_file} not found.")
            return None

    def generate_data_point(self, fields):
        data_point = {}
        for field in fields:
            prompt = f"Generate data for {field['name']}: {field['prompt']}"
            response = ollama.chat(model=self.model, messages=[
                {'role': 'system', 'content': 'You are a data generation assistant. Respond only with the requested data point, nothing else.'},
                {'role': 'user', 'content': prompt}
            ])
            data_point[field['name']] = response['message']['content'].strip()
        return data_point

    def generate_dataset(self):
        dataset = []
        if self.seed_data is not None:
            dataset.extend(self.seed_data.to_dict(orient='records'))
            remaining_size = self.dataset_params['size'] - len(dataset)
        else:
            remaining_size = self.dataset_params['size']

        for _ in tqdm(range(remaining_size), desc="Generating dataset"):
            data_point = self.generate_data_point(self.dataset_params['fields'])
            dataset.append(data_point)
        return dataset

    def save_to_parquet(self, dataset, filename='dataset.parquet'):
        file_path = os.path.join(self.output_dir, filename)
        df = pd.DataFrame(dataset)
        df.to_parquet(file_path, engine='pyarrow')
        return file_path

if __name__ == "__main__":
    seed_file = 'path/to/seed_file.parquet'  # Change this to your seed file path if needed
    kitchen1 = Agent_Chef(seed_file=seed_file)
    print("Generating synthetic dataset...")
    dataset = kitchen1.generate_dataset()
    saved_path = kitchen1.save_to_parquet(dataset)
    print(f"Dataset with {len(dataset)} entries has been generated and saved as Parquet at: {saved_path}")