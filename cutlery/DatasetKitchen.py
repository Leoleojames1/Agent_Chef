import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
import json
import numpy as np
import glob
import os
import logging
from colorama import Fore
import time

class DatasetManager:
    def __init__(self, ollama_interface, template_manager, input_dir, output_dir):
        self.ollama_interface = ollama_interface
        self.template_manager = template_manager
        self.logger = logging.getLogger(__name__)
        self.input_dir = input_dir
        self.output_dir = output_dir

    def generate_synthetic_data(self, seed_parquet, num_samples=100, system_prompt=None, template=None):
        try:
            logging.info(f"Generating synthetic data with seed_parquet: {seed_parquet}, template: {template}")
            seed_data = pd.read_parquet(seed_parquet)
            logging.info(f"Successfully read seed parquet. Shape: {seed_data.shape}")
            logging.info(f"Seed parquet columns: {seed_data.columns.tolist()}")
            
            if system_prompt is None:
                raise ValueError("System prompt is required for synthetic data generation")
            
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

    def parse_text_to_parquet(self, text_content, template_name, filename):
        template = self.template_manager.get_template(template_name)
        if not template:
            raise ValueError(f"Template '{template_name}' not found")

        self.logger.info(f"Parsing text content using template: {template_name}")
        
        # Save the text content to a txt file
        txt_file = os.path.join(self.input_dir, f"{filename}.txt")
        with open(txt_file, 'w', encoding='utf-8') as f:
            f.write(text_content)
        self.logger.info(f"Saved text content to {txt_file}")

        # Use Ollama's JSON mode to parse the text content based on the template
        json_structure = {column: [] for column in template}
        prompt = f"Parse the following text into JSON with these keys: {', '.join(template)}. Each key should contain a list of relevant parsed data.\n\nText to parse:\n{text_content}"
        
        parsed_data = self.ollama_interface.chat_json(messages=[
            {'role': 'system', 'content': "You are a data parsing assistant. Parse the given text into the specified JSON structure."},
            {'role': 'user', 'content': prompt}
        ])

        if not parsed_data:
            self.logger.warning("AI model failed to generate valid JSON. Falling back to manual structuring.")
            parsed_data = self.fallback_json_structure(text_content, template)

        self.logger.info("Successfully structured content as JSON")
        
        # Ensure all template columns are present in the parsed data
        for column in template:
            if column not in parsed_data:
                parsed_data[column] = []

        # Create DataFrame from parsed data
        df = pd.DataFrame(parsed_data)
        self.logger.info(f"Created DataFrame with shape: {df.shape}")

        # Save as JSON
        json_file = os.path.join(self.input_dir, f"{filename}.json")
        df.to_json(json_file, orient='records', indent=2)
        self.logger.info(f"Saved JSON file: {json_file}")

        # Save as Parquet
        parquet_file = os.path.join(self.input_dir, f"{filename}.parquet")
        df.to_parquet(parquet_file, engine='pyarrow')
        self.logger.info(f"Saved Parquet file: {parquet_file}")

        return df, json_file, parquet_file
    
    def fallback_json_structure(self, text_content, template):
        # Split the content into lines
        lines = text_content.split('\n')
        
        # Initialize the structured data
        structured_data = {column: [] for column in template}
        
        # Assign each line to a column in a round-robin fashion
        for i, line in enumerate(lines):
            if line.strip():  # Skip empty lines
                column = template[i % len(template)]
                structured_data[column].append(line.strip())
        
        return structured_data
    
class TemplateManager:
    def __init__(self, input_dir):
        self.input_dir = input_dir
        self.templates_file = os.path.join(input_dir, 'templates.json')
        print(f"Templates file path: {self.templates_file}")  # Add this line for debugging
        self.templates = self.load_templates()

    def load_templates(self):
        if os.path.exists(self.templates_file):
            with open(self.templates_file, 'r') as f:
                templates = json.load(f)
        else:
            templates = {
                "instruct": ["user", "request", "assistant", "response"],
                "functions": ["function"],
                "functionCall": ["command", "description", "args", "actions"],
                "mathFusion": ["formula", "solution", "python", "example"],
                "formulas": ["formula"],
                "latexSeries": ["formula", "solution"],
                "latexTheory": ["theory", "explanation"],
                "aiConcept": ["concept", "definition", "useCase", "example"],
                "dataStructure": ["name", "description", "timeComplexity", "pythonImplementation"],
                "pythonBase": ["code", "description", "args", "returns"],
                "pythonOllama": ["code", "description", "args", "actions", "chainOfThought","prompts"],
                "ontology1": ["ontology", "description", "application", "actions", "chainOfThought","self_prompts"],
                "ontology2": ["ontology", "description"],
                "ontology3": ["ontology", "description", "chainOfThought"],
            }
            self.save_templates(templates)
        
        # Ensure all template values are lists
        for key, value in templates.items():
            if not isinstance(value, list):
                templates[key] = list(value) if isinstance(value, (tuple, set)) else [str(value)]
        
        return templates

    def get_templates(self):
        return self.templates

    def save_templates(self, templates):
        with open(self.templates_file, 'w') as f:
            json.dump(templates, f, indent=2)

    def create_template(self, template_name, template_fields):
        if template_name in self.templates:
            raise ValueError(f"Template '{template_name}' already exists")
        self.templates[template_name] = template_fields
        self.save_templates(self.templates)
        return self.templates[template_name]

    def get_template(self, template_name):
        return self.templates.get(template_name)

    def add_template(self, template_name, template_fields):
        if template_name in self.templates:
            raise ValueError(f"Template '{template_name}' already exists")
        self.templates[template_name] = template_fields
        self.save_templates(self.templates)
        return self.templates[template_name]