import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
import json
import numpy as np
import glob
import os
import logging

class Dataset_Manager_Class:
    def __init__(self, ollama_interface, template_manager):
        self.ollama_interface = ollama_interface
        self.template_manager = template_manager

    def generate_synthetic_data(self, seed_parquet, num_samples=100, system_prompt=None, template=None):
        try:
            logging.info(f"Generating synthetic data with seed_parquet: {seed_parquet}, template: {template}")
            seed_data = pd.read_parquet(seed_parquet)
            logging.info(f"Successfully read seed parquet. Shape: {seed_data.shape}")
            logging.info(f"Seed parquet columns: {seed_data.columns.tolist()}")
            
            if system_prompt is None:
                system_prompt = "You are a data generation assistant. Respond only with the requested data point, nothing else."
            
            if template is None or template == '':
                template = seed_data.columns.tolist()
            elif isinstance(template, str):
                template = self.template_manager.get_template(template)
            
            logging.info(f"Using template: {template}")
            
            synthetic_data = []
            for _ in tqdm(range(num_samples), desc="Generating synthetic data"):
                synthetic_row = {}
                for column in template:
                    if column in seed_data.columns:
                        example = seed_data[column].sample().iloc[0]
                        prompt = f"Generate a similar but different {column} based on this example: {example}. Maintain the same structure and meaning but use different phrasing."
                    else:
                        prompt = f"Generate a plausible value for the column: {column}"
                    
                    try:
                        response = self.ollama_interface.chat(messages=[
                            {'role': 'system', 'content': system_prompt},
                            {'role': 'user', 'content': prompt}
                        ])
                        
                        function_name, arguments = self.ollama_interface.parse_tool_response(response)
                        
                        if function_name == 'generate_random_data':
                            synthetic_row[column] = arguments['prompt']
                        else:
                            synthetic_row[column] = response['message']['content'].strip()
                    except Exception as e:
                        logging.error(f"Error generating synthetic data for column {column}: {str(e)}")
                        synthetic_row[column] = f"Error: {str(e)}"
                synthetic_data.append(synthetic_row)
            
            result_df = pd.DataFrame(synthetic_data)
            logging.info(f"Successfully generated synthetic data. Shape: {result_df.shape}")
            logging.info(f"Generated DataFrame columns: {result_df.columns.tolist()}")
            return result_df
        except Exception as e:
            logging.exception(f"Error in generate_synthetic_data: {str(e)}")
            return pd.DataFrame()

    def combine_parquets(self, seed_parquet_dir):
        try:
            # Get all parquet files in the directory
            parquet_files = glob.glob(os.path.join(seed_parquet_dir, '*.parquet'))
            logging.info(f"Found {len(parquet_files)} parquet files in {seed_parquet_dir}")
            
            # Read and combine all parquet files
            dfs = []
            for file in parquet_files:
                try:
                    df = pd.read_parquet(file)
                    dfs.append(df)
                    logging.info(f"Successfully read {file}. Shape: {df.shape}")
                except Exception as e:
                    logging.error(f"Error reading {file}: {str(e)}")
            
            if not dfs:
                raise ValueError("No valid parquet files found")
            
            combined_df = pd.concat(dfs, ignore_index=True)
            logging.info(f"Successfully combined parquet files. Final shape: {combined_df.shape}")
            
            return combined_df
        except Exception as e:
            logging.exception(f"Error in combine_parquets: {str(e)}")
            return pd.DataFrame()

    def augment_data(self, seed_parquet):
        try:
            logging.info(f"Attempting to augment data from: {seed_parquet}")
            # Load the seed data
            seed_data = pd.read_parquet(seed_parquet)
            logging.info(f"Successfully read seed parquet. Shape: {seed_data.shape}")
            
            # Implement data augmentation logic
            augmented_data = seed_data.copy()
            for column in seed_data.columns:
                if seed_data[column].dtype == 'object':  # Text data
                    augmented_data[column] = augmented_data[column].apply(lambda x: f"{x} (augmented)")
                elif seed_data[column].dtype in ['int64', 'float64']:  # Numeric data
                    augmented_data[column] = augmented_data[column] * (1 + np.random.uniform(-0.1, 0.1, len(seed_data)))
            
            logging.info(f"Successfully augmented data. Shape: {augmented_data.shape}")
            return augmented_data
        except Exception as e:
            logging.exception(f"Error in augment_data: {str(e)}")
            return pd.DataFrame()

    def parse_text_to_parquet(self, text_content, template_name):
        template = self.template_manager.get_template(template_name)
        if not template:
            raise ValueError(f"Template '{template_name}' not found")

        # Use Ollama to parse the text content based on the template
        response = self.ollama_interface.chat(messages=[
            {'role': 'system', 'content': f"You are a data parsing assistant. Parse the given text into the following columns: {', '.join(template)}. Return the result as a JSON object where keys are column names and values are lists of parsed data."},
            {'role': 'user', 'content': text_content}
        ])

        function_name, arguments = self.ollama_interface.parse_tool_response(response)
        
        if function_name == 'generate_random_data':
            parsed_data = json.loads(arguments['prompt'])
        else:
            try:
                parsed_data = json.loads(response['message']['content'])
            except json.JSONDecodeError:
                raise ValueError("Failed to parse Ollama response as JSON")

        # Ensure all template columns are present in the parsed data
        for column in template:
            if column not in parsed_data:
                parsed_data[column] = []

        # Create DataFrame from parsed data
        df = pd.DataFrame(parsed_data)
        return df