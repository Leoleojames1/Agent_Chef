import os
import pandas as pd
import json
from colorama import Fore
import logging

class FileHandler:
    def __init__(self, input_dir, output_dir):
        self.input_dir = input_dir
        self.output_dir = output_dir
        logging.basicConfig(level=logging.INFO)

    def save_to_parquet(self, dataset, filename='dataset.parquet'):
        file_path = os.path.join(self.output_dir, filename)
        dataset.to_parquet(file_path, engine='pyarrow')
        logging.info(f"Saved dataset to Parquet: {file_path}")
        return file_path

    def load_parquet(self, filename):
        file_path = os.path.join(self.input_dir, filename)
        if os.path.exists(file_path):
            return pd.read_parquet(file_path)
        else:
            logging.error(f"Parquet file {filename} not found in the input directory.")
            return None

    def load_seed_data(self, seed_file):
        if seed_file and os.path.exists(os.path.join(self.input_dir, seed_file)):
            return pd.read_parquet(os.path.join(self.input_dir, seed_file))
        else:
            logging.error(f"Seed file {seed_file} not found in the ingredients directory.")
            return None

    def save_json_to_parquet(self, json_file):
        json_path = os.path.join(self.input_dir, json_file)
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                data = json.load(f)
            df = pd.DataFrame(data)
            parquet_file = os.path.join(self.output_dir, 'user_data_dish.parquet')
            df.to_parquet(parquet_file, engine='pyarrow')
            logging.info(f"Saved JSON to Parquet: {parquet_file}")
            return parquet_file
        else:
            logging.error(f"JSON file {json_file} not found in the ingredients directory.")
            return None