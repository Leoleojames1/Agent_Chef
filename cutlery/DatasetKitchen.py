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
import random

class EnhancedDatasetGenerator:
    def __init__(self, ollama_interface, template_manager):
        self.ollama_interface = ollama_interface
        self.template_manager = template_manager

    def generate_paraphrase(self, text, row, column_types):
        reference_values = {col: row[col] for col, col_type in column_types.items() if col_type == 'reference'}
        is_question = self.is_question(text)
        paraphrased = self.paraphrase_text_with_references(text, reference_values)
        verified = self.verify_paraphrase(original=text, paraphrased=paraphrased, reference=reference_values, is_question=is_question)
        return verified

    def paraphrase_text_with_references(self, text, reference_values):
        system_prompt = """You are a dataset paraphrasing assistant. Your task is to maintain all of the details of the description given maintaining its original meaning and incorporating the provided reference values. Do not add any explanatory text or meta-information."""
        
        user_prompt = f"""Original text: {text}
        Reference values: {reference_values}
        
        Please maintain all of the details of the description given, maintaining its core meaning and incorporating the reference values where appropriate. Ensure the paraphrased text is coherent and contextually relevant. Provide only the paraphrased text without any additional explanations or formatting.
        Please Do maintain the structure of the sentence, if the sentence is a question the generated paraphrase should be a similarly asked question with the question mark, if the sentence is a statement, the paraphrase maintain its meaning should be a similary stated statement with a period.
        """

        response = self.ollama_interface.chat(messages=[
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_prompt}
        ])
        
        paraphrased_text = response['message']['content'].strip()
        return paraphrased_text
    
    def generate_paraphrase(self, text, row, column_types):
        reference_values = {col: row[col] for col, col_type in column_types.items() if col_type == 'reference'}
        is_question = self.is_question(text)
        paraphrased = self.paraphrase_text_with_references(text, reference_values)
        verified = self.verify_paraphrase(original=text, paraphrased=paraphrased, reference=reference_values, is_question=is_question)
        return verified

    def verify_paraphrase(self, original, paraphrased, reference, is_question):
        system_prompt = """You are a verification assistant for Agent Chef a dataset constructor tool. Your task is to ensure that the paraphrased content maintains the original meaning, format (question or statement), and incorporates the reference values correctly. If the paraphrase is accurate, return it as-is. If not, provide a corrected version."""
        
        user_prompt = f"""Original: {original}
        Paraphrased: {paraphrased}
        Reference values: {reference}
        Is question: {is_question}
        
        Verify that the paraphrased content maintains the original meaning, format (question or statement), and correctly incorporates the reference values. If it does, return the paraphrased content. If not, provide a corrected version that accurately reflects the original meaning, format, and includes the reference values. Do not include any explanatory text or meta-information in your response."""

        response = self.ollama_interface.chat(messages=[
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_prompt}
        ])
        #TODO IMPLEMENT DEMOCRACY MOE FOR PARAHRASE VALIDATION
        verified_text = response['message']['content'].strip()
        
        # Ensure the verified text ends with a question mark if the original was a question
        if is_question and not verified_text.endswith('?'):
            verified_text += '?'
        
        return verified_text
    
    def generate_enhanced_synthetic_data(self, seed_data, num_samples=100, template_name='commander', system_prompt=None):
        template = self.template_manager.get_template(template_name)
        synthetic_data = []

        for _ in tqdm(range(num_samples), desc="Generating synthetic data"):
            synthetic_row = {}
            original_row = seed_data.sample(n=1).iloc[0]

            for column in template:
                if column in seed_data.columns:
                    original_content = original_row[column]

                    if column in ['command_description', 'request', 'response']:
                        synthetic_row[column] = self.generate_paraphrases(original_content, n=1, system_prompt=system_prompt)[0]
                    else:
                        # For other columns (like 'command'), keep the original content
                        synthetic_row[column] = original_content
                else:
                    synthetic_row[column] = f"Placeholder for {column}"

            synthetic_data.append(synthetic_row)

        return pd.DataFrame(synthetic_data)
    
    def is_question(self, text):
        #TODO QUESTION OR REQUEST (Please explain what /swap does.)
        return text.strip().endswith('?') or text.lower().startswith(('what', 'when', 'where', 'who', 'why', 'how', 'can', 'could', 'would', 'should', 'is', 'are', 'do', 'does', 'please', 'explain', 'describe'))

class DatasetManager:
    def __init__(self, ollama_interface, template_manager, input_dir, output_dir):
        self.ollama_interface = ollama_interface
        self.template_manager = template_manager
        self.logger = logging.getLogger(__name__)
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.enhanced_generator = EnhancedDatasetGenerator(ollama_interface, template_manager)

    def parquet_to_txt(self, parquet_file):
        try:
            logging.info(f"Converting parquet file to text: {parquet_file}")
            
            parquet_file_path = os.path.join(self.input_dir, parquet_file)
            if not os.path.exists(parquet_file_path):
                raise FileNotFoundError(f"Parquet file not found: {parquet_file_path}")
            
            df = pd.read_parquet(parquet_file_path)
            
            txt_content = []
            for _, row in df.iterrows():
                for column in df.columns:
                    txt_content.append(f'$("{row[column]}")')
                txt_content.append('')  # Add an empty line between rows
            
            txt_filename = f"{os.path.splitext(parquet_file)[0]}.txt"
            txt_file_path = os.path.join(self.input_dir, txt_filename)
            
            with open(txt_file_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(txt_content))
            
            logging.info(f"Created text file: {txt_filename}")
            return txt_filename
        except Exception as e:
            logging.exception(f"Error in parquet_to_txt: {str(e)}")
            raise
    
    def generate_paraphrased_txt(self, seed_file, num_samples=1, system_prompt=None):
        try:
            logging.info(f"Generating paraphrased text from seed file: {seed_file}")
            
            seed_file_path = os.path.join(self.input_dir, seed_file)
            if not os.path.exists(seed_file_path):
                raise FileNotFoundError(f"Seed file not found: {seed_file_path}")
            
            with open(seed_file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            pattern = re.compile(r'\$\("((?:(?!\$\(").|\n)*?)"\)', re.DOTALL)
            matches = pattern.findall(content)
            
            paraphrased_content = []
            for _ in range(num_samples):
                sample_content = []
                for group in tqdm(matches, desc="Paraphrasing groups"):
                    paraphrased_group = self.enhanced_generator.generate_paraphrases(group, n=1, system_prompt=system_prompt)[0]
                    sample_content.append(f'$("{paraphrased_group}")')
                paraphrased_content.append('\n\n'.join(sample_content))
            
            # Save paraphrased content to new text files
            output_files = []
            for i, sample in enumerate(paraphrased_content):
                output_filename = f"{os.path.splitext(seed_file)[0]}_paraphrased_{i+1}.txt"
                output_path = os.path.join(self.output_dir, output_filename)
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(sample)
                output_files.append(output_filename)
            
            logging.info(f"Generated {len(output_files)} paraphrased text files")
            return output_files
        except Exception as e:
            logging.exception(f"Error in generate_paraphrased_txt: {str(e)}")
            raise

    def txt_to_parquet(self, txt_file):
        try:
            logging.info(f"Converting text file to parquet: {txt_file}")
            
            txt_file_path = os.path.join(self.output_dir, txt_file)
            if not os.path.exists(txt_file_path):
                raise FileNotFoundError(f"Text file not found: {txt_file_path}")
            
            with open(txt_file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            pattern = re.compile(r'\$\("((?:(?!\$\(").|\n)*?)"\)', re.DOTALL)
            matches = pattern.findall(content)
            
            data = []
            for i in range(0, len(matches), 5):  # Assuming 5 columns based on your template
                row = matches[i:i+5]
                if len(row) == 5:
                    data.append(row)
            
            df = pd.DataFrame(data, columns=['task', 'instruction', 'input', 'output', 'command'])
            
            parquet_filename = f"{os.path.splitext(txt_file)[0]}.parquet"
            parquet_path = os.path.join(self.output_dir, parquet_filename)
            df.to_parquet(parquet_path, engine='pyarrow')
            
            logging.info(f"Created parquet file: {parquet_filename}")
            return parquet_filename
        except Exception as e:
            logging.exception(f"Error in txt_to_parquet: {str(e)}")
            raise
        
    def read_data(self, file_path):
        """Read data from parquet, json, or txt file."""
        base_name, ext = os.path.splitext(file_path)
        
        if ext.lower() == '.parquet':
            try:
                return pd.read_parquet(file_path)
            except Exception as e:
                self.logger.warning(f"Failed to read parquet file: {e}")
        
        # Try JSON
        json_path = f"{base_name}.json"
        if os.path.exists(json_path):
            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)
                return pd.DataFrame(data)
            except Exception as e:
                self.logger.warning(f"Failed to read JSON file: {e}")
        
        # Try TXT
        txt_path = f"{base_name}.txt"
        if os.path.exists(txt_path):
            try:
                with open(txt_path, 'r') as f:
                    lines = f.readlines()
                # Assuming tab-separated values, adjust if needed
                data = [line.strip().split('\t') for line in lines]
                return pd.DataFrame(data[1:], columns=data[0])
            except Exception as e:
                self.logger.warning(f"Failed to read TXT file: {e}")
        
        raise ValueError(f"Unable to read data from {file_path} or its JSON/TXT alternatives")
    
    def generate_synthetic_data(self, seed_file, sample_rate, paraphrases_per_sample, column_types, use_all_samples=True, **kwargs):
        try:
            seed_file_path = os.path.join(self.input_dir, seed_file)
            logging.info(f"Looking for seed file at: {seed_file_path}")
            logging.info(f"Column types: {column_types}")
            
            if not os.path.exists(seed_file_path):
                raise FileNotFoundError(f"Seed file not found: {seed_file_path}")
            
            seed_data = pd.read_parquet(seed_file_path)
            num_samples = len(seed_data) if use_all_samples else int(len(seed_data) * (sample_rate / 100))
            
            samples_to_use = seed_data if use_all_samples else seed_data.sample(n=num_samples, replace=False)

            synthetic_data = []
            for _, row in tqdm(samples_to_use.iterrows(), total=len(samples_to_use)):
                for _ in range(paraphrases_per_sample):
                    synthetic_row = {}
                    reference_values = {col: row[col] for col in seed_data.columns if column_types.get(col) == 'reference'}
                    
                    for column in seed_data.columns:
                        col_type = column_types.get(column, 'dynamic')
                        logging.info(f"Processing column: {column}, type: {col_type}")
                        
                        if col_type in ['static', 'reference']:
                            synthetic_row[column] = row[column]
                            logging.info(f"Copied {col_type} column '{column}': {synthetic_row[column]}")
                        elif col_type == 'dynamic':
                            original_content = row[column]
                            paraphrased_content = self.enhanced_generator.generate_paraphrase(original_content, row, column_types)
                            synthetic_row[column] = paraphrased_content
                            logging.info(f"Generated dynamic content for column '{column}': {synthetic_row[column]}")
                        else:
                            raise ValueError(f"Unknown column type '{col_type}' for column '{column}'")

                    synthetic_data.append(synthetic_row)

            result_df = pd.DataFrame(synthetic_data)
            
            # Verify all columns
            for column in seed_data.columns:
                col_type = column_types.get(column, 'dynamic')
                if col_type in ['static', 'reference']:
                    original_values = seed_data[column].repeat(paraphrases_per_sample).reset_index(drop=True)
                    if not result_df[column].equals(original_values):
                        logging.error(f"{col_type.capitalize()} column '{column}' has been modified!")
                        logging.error(f"Original values: {original_values}")
                        logging.error(f"Modified values: {result_df[column]}")
                        raise ValueError(f"{col_type.capitalize()} column '{column}' has been modified during synthetic data generation. This is not allowed.")
                    else:
                        logging.info(f"Verified {col_type} column '{column}' remains unchanged.")
                elif col_type == 'dynamic':
                    if result_df[column].equals(seed_data[column].repeat(paraphrases_per_sample).reset_index(drop=True)):
                        logging.warning(f"Dynamic column '{column}' has not been modified. This is unexpected.")
                    else:
                        logging.info(f"Verified dynamic column '{column}' has been modified as expected.")
            
            if result_df.empty:
                logging.warning("Generated DataFrame is empty")
            else:
                logging.info(f"Generated DataFrame shape: {result_df.shape}")
            
            return result_df
        except Exception as e:
            logging.exception(f"Error in generate_synthetic_data: {str(e)}")
            raise

    def paraphrase_text(self, text):
        if not text.strip():
            return text

        system_prompt = """You are a paraphrasing assistant for Ollama Agent Roll Cage. 
        Generate a diverse paraphrase that maintains the original meaning but varies in structure and wording. 
        Do not include any meta-text or explanations in your response, just provide the paraphrased content."""
        
        user_prompt = f"""Paraphrase the following text, maintaining its core meaning but varying the structure and wording:
        {text}
        Provide only the paraphrased text in your response, without any additional explanations or meta-text."""

        response = self.ollama_interface.chat(messages=[
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_prompt}
        ])
        
        paraphrased_text = response['message']['content'].strip()
        return self.clean_paraphrased_text(paraphrased_text)

    def clean_generated_content(self, text):
        # Remove any explanatory phrases or meta-information
        text = re.sub(r'^(Paraphrased content:|Verified content:|Corrected version:)\s*', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\s*(Verification result:.*|Reference Command:.*|Note:.*|Verified Paraphrase:.*)$', '', text, flags=re.IGNORECASE)
        
        # Remove any remaining placeholder-like patterns
        text = re.sub(r'___[A-Za-z_]+___', '', text)
        
        # Remove any quotes that might have been added
        text = text.strip('"\'')
        
        # Remove any leading/trailing whitespace
        text = text.strip()
        
        # Ensure the text starts with a capital letter
        if text and text[0].islower():
            text = text[0].upper() + text[1:]
        
        # Ensure the text ends with proper punctuation
        if text and not text[-1] in '.!?':
            text += '.'
        
        return text
    
    def clean_generated_content(self, text):
        # Remove any explanatory phrases or meta-information
        text = re.sub(r'^(Paraphrased content:|Verified content:|Corrected version:)\s*', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\s*(Verification result:.*|Reference Command:.*|Note:.*|Verified Paraphrase:.*)$', '', text, flags=re.IGNORECASE)
        
        # Remove any remaining placeholder-like patterns
        text = re.sub(r'___[A-Za-z_]+___', '', text)
        
        # Remove any quotes that might have been added
        text = text.strip('"\'')
        
        # Remove any leading/trailing whitespace
        text = text.strip()
        
        # Ensure the text starts with a capital letter
        if text and text[0].islower():
            text = text[0].upper() + text[1:]
        
        # Ensure the text ends with proper punctuation
        if text and not text[-1] in '.!?':
            text += '.'
        
        return text

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

    def paraphrase_input(self, text, reference_values):
        system_prompt = """You are an assistant for Ollama Agent Roll Cage. Your task is to rephrase the given input as a question about the command specified in the reference values. Ensure the question is clear, concise, and directly related to the command's function."""
        
        user_prompt = f"""Original input: {text}
        Command: {reference_values.get('command', '')}
        
        Please rephrase the input as a clear question about the specified command's function. The question should be concise and directly related to what the command does."""

        response = self.ollama_interface.chat(messages=[
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_prompt}
        ])
        
        paraphrased_text = response['message']['content'].strip()
        return self.clean_paraphrased_text(paraphrased_text)

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
                "functionCall": ["request", "response", "explain", "task",],
                "commander": ["task", "instruction", "input", "output", "command"],   
                "commander_extra": ["request", "response", "commandName", "task", "args", "clarification", "confirmation"],
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