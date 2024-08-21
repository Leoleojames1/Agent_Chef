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
import re

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
        
        # Save the original text content to a txt file
        txt_file = os.path.join(self.input_dir, f"{filename}.txt")
        with open(txt_file, 'w', encoding='utf-8') as f:
            f.write(text_content)
        self.logger.info(f"Saved original text content to {txt_file}")

        # Parse the content using manual formatting
        parsed_data = self.parse_manual_formatting(text_content, template)

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
    
    def parse_dataset(self, content, template_name, mode='manual'):
        template = self.template_manager.get_template(template_name)
        if not template:
            raise ValueError(f"Template '{template_name}' not found")

        if mode == 'manual':
            parsed_data = self.parse_manual_formatting(content, template)
        elif mode == 'automatic':
            formatted_content = self.apply_automatic_formatting(content, template)
            parsed_data = self.parse_manual_formatting(formatted_content, template)
        else:
            raise ValueError(f"Invalid parsing mode: {mode}")

        df = pd.DataFrame(parsed_data)
        return df
    
    def parse_manual_formatting(self, content, template):
        parsed_data = {column: [] for column in template}
        pattern = re.compile(r'\$\("((?:(?!\$\(").|\n)*?)"\)', re.DOTALL)
        matches = pattern.findall(content)

        num_columns = len(template)
        for i, match in enumerate(matches):
            column_index = i % num_columns
            column = template[column_index]
            # Remove any leading/trailing whitespace, but preserve internal formatting
            parsed_data[column].append(match.strip())

        # Ensure all columns have the same number of entries
        max_length = max(len(column_data) for column_data in parsed_data.values())
        for column in parsed_data:
            parsed_data[column] += [''] * (max_length - len(parsed_data[column]))

        return parsed_data
    
    def apply_automatic_formatting(self, content, template):
        prompt = f"""Format the following text using $("") symbols according to these categories: {', '.join(template)}. 
        Follow these rules strictly:
        1. Each $("") group should correspond to a single category in order, cycling through the categories as needed.
        2. Everything between $("") should be treated as a single unit, even if it spans multiple lines.
        3. Do not split content within $("") groups.
        4. Ignore any content that is not enclosed in $("") delimiters.
        5. Preserve all formatting, including newlines, within the $("") groups.

        Text to format:
        {content}"""
        
        response = self.ollama_interface.chat(messages=[
            {'role': 'system', 'content': "You are a data formatting assistant. Format the given text using $('') symbols for the specified categories, following the rules provided strictly."},
            {'role': 'user', 'content': prompt}
        ])

        return response['message']['content']
    
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
                "chat": ["user", "assistant"],
                "instruct": ["request", "response"],
                "aiConcept": ["request", "response", "concept", "definition", "useCase", "example"],                
                "commander": ["request", "response", "commandName", "task", "args", "clarification", "confirmation"],
                "intention": ["request", "response", "commandName", "task", "args", "enumerate", "validate", "describe"],
                "latexSeries": ["request", "response", "formula", "solution"],
                "latexTheory": ["request", "response", "theory", "explanation"],
                "latexMath" : ["request", "response", "formula", "solution", "theory", "explanation"],
                "mathPythonFusion": ["request", "response", "formula", "solution", "python", "example"],
                "pythonBase": ["request", "response", "code", "description", "args", "returns"],
                "pythonOllama": ["request", "response", "code", "description", "args", "returns", "actions", "chainOfThought", "prompts"],
                "ontology1": ["request", "response", "ontology", "description"],
                "ontology2": ["request", "response", "ontology", "description", "chainOfThought"],
                "ontology3": ["request", "response", "ontology", "description", "application", "actions", "chainOfThought","self_prompts"],
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