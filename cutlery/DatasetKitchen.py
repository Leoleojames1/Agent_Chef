import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
import numpy as np
from colorama import Fore, Style, init
import os, json, urllib.parse, tarfile, gzip, shutil, requests, random, re, time, logging, glob
from typing import List, Dict, Any, Optional

# Import document loaders (assuming they're in a separate file)
from langchain.document_loaders import (
    WebBaseLoader, PyPDFLoader, TextLoader, Docx2txtLoader,
    UnstructuredPowerPointLoader, WikipediaLoader, ArxivLoader,
    UnstructuredEPubLoader, JSONLoader, CSVLoader
)

# Import external APIs (assuming they're in separate files)
from huggingface_hub import snapshot_download, HfApi, hf_hub_download
from github import Github
from datasets import load_dataset
init(autoreset=True)

#TODO allow arxiv & hugging face links in ui for digestion and dataset construction

class PromptManager:
    def __init__(self):
        self.prompts = {
            'system': "You are a dataset generation assistant. Your task is to generate diverse, high-quality data while maintaining consistency with the provided context and reference values.",
            'paraphrase': {
                'system': "You are a paraphrasing assistant. Your task is to rephrase the given text while maintaining its original meaning and incorporating any provided reference values.",
                'user': "Original text: {text}\nReference values: {reference_values}\n\nPlease rephrase the text, maintaining its core meaning and incorporating the reference values where appropriate. Ensure the rephrased text is coherent and contextually relevant."
            },
            'verify': {
                'system': "You are a verification assistant. Your task is to ensure that the generated content maintains the original meaning, format, and incorporates any reference values correctly.",
                'user': "Original: {original}\nGenerated: {generated}\nReference values: {reference}\nIs question: {is_question}\n\nVerify that the generated content maintains the original meaning, format, and correctly incorporates the reference values. If it does, return the generated content. If not, provide a corrected version."
            },
            'dynamicColumns': {}
        }

    def get_prompt(self, prompt_type, sub_type=None, column=None):
        if column and prompt_type == 'dynamicColumns':
            return self.prompts['dynamicColumns'].get(column, {}).get(sub_type, "")
        elif sub_type:
            return self.prompts.get(prompt_type, {}).get(sub_type, "")
        return self.prompts.get(prompt_type, "")

    def set_prompt(self, prompt_type, prompt_text, sub_type=None, column=None):
        if column and prompt_type == 'dynamicColumns':
            if column not in self.prompts['dynamicColumns']:
                self.prompts['dynamicColumns'][column] = {}
            self.prompts['dynamicColumns'][column][sub_type] = prompt_text
        elif sub_type:
            if prompt_type not in self.prompts:
                self.prompts[prompt_type] = {}
            self.prompts[prompt_type][sub_type] = prompt_text
        else:
            self.prompts[prompt_type] = prompt_text

    def get_all_prompts(self):
        return self.prompts

class EnhancedDatasetGenerator:
    def __init__(self, ollama_interface, template_manager):
        self.ollama_interface = ollama_interface
        self.template_manager = template_manager
        self.prompt_manager = PromptManager()

    def generate_content(self, column, text, row, column_types, custom_prompts):
        print(f"{Fore.CYAN}Generating content for column: {column}{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}Custom prompts: {json.dumps(custom_prompts, indent=2)}{Style.RESET_ALL}")

        reference_values = {col: row[col] for col, col_type in column_types.items() if col_type == 'reference'}
        is_question = self.is_question(text)

        system_prompt = custom_prompts.get('system') or self.prompt_manager.get_prompt('system')
        column_prompts = custom_prompts.get('dynamicColumns', {}).get(column, {})
        user_prompt = column_prompts.get('user') or self.prompt_manager.get_prompt('dynamicColumns', 'user', column)

        print(f"{Fore.MAGENTA}System prompt: {system_prompt}{Style.RESET_ALL}")
        print(f"{Fore.MAGENTA}User prompt: {user_prompt}{Style.RESET_ALL}")

        if not user_prompt:
            if column == 'input':
                user_prompt = """
                Original text: {text}
                Reference values: {reference_values}
                Is question: {is_question}

                Generate a rephrased input question that maintains the original meaning and incorporates the reference values. If the original is not a question, convert it into one. Ensure the question starts with an appropriate question word (What, When, Where, Who, Why, How, Can, Could, Would, Should, Is, Are, Do, Does) and ends with a question mark. Do not provide any explanations or additional information.
                """
            elif column == 'output':
                user_prompt = """
                Original text: {text}
                Reference values: {reference_values}
                Is question: {is_question}

                Generate a rephrased output statement that maintains the original meaning and incorporates the reference values. If the original is a question, convert it into a statement. Ensure the statement is clear, concise, and ends with a period. Do not provide any explanations or additional information.
                """
            else:
                user_prompt = """
                Original text: {text}
                Reference values: {reference_values}
                Is question: {is_question}

                Generate a suitable response for the '{column}' column, maintaining its core meaning and incorporating the reference values where appropriate. Ensure the response is coherent and contextually relevant.
                """

        formatted_user_prompt = user_prompt.format(
            text=text,
            reference_values=reference_values,
            is_question=is_question,
            column=column
        )

        print(f"{Fore.BLUE}Formatted user prompt: {formatted_user_prompt}{Style.RESET_ALL}")

        response = self.ollama_interface.chat(messages=[
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': formatted_user_prompt}
        ])
        
        generated_content = response['message']['content'].strip()
        print(f"{Fore.GREEN}Generated content: {generated_content}{Style.RESET_ALL}")

        cleaned_content = self.clean_generated_content(generated_content, is_question)
        print(f"{Fore.LIGHTGREEN_EX}Cleaned content: {cleaned_content}{Style.RESET_ALL}")

        return cleaned_content
        
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
        Please Do maintain the structure of the sentence, if the sentence is a question the generated paraphrase should be a similarly asked question with the question mark, if the sentence is a statement, the paraphrase maintain its meaning should be a similary stated statement with a period. DO NOT COPY OUTPUT.
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

    def clean_generated_content(self, text, is_question):
        # Remove any explanatory phrases or meta-information
        text = re.sub(r'^(Generated content:|Verified content:|Corrected version:)\s*', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\s*(Verification result:.*|Reference Command:.*|Note:.*|Verified Response:.*)$', '', text, flags=re.IGNORECASE)
        
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
        if is_question and not text.endswith('?'):
            text += '?'
        elif not is_question and not text[-1] in '.!?':
            text += '.'
        
        return text
    
    def verify_paraphrase(self, original, paraphrased, reference, is_question):
        system_prompt = """You are a verification assistant for Agent Chef a dataset constructor tool. Your task is to ensure that the paraphrased content maintains the original meaning, format (question or statement), and incorporates the reference values correctly. If the paraphrase is accurate, return it as-is. If not, provide a corrected version."""
        
        user_prompt = f"""Original: {original}
        Paraphrased: {paraphrased}
        Reference values: {reference}
        Is question: {is_question}
        
        Verify that the paraphrased content maintains the original meaning, format (question or statement), and correctly incorporates the reference values. If it does, return the paraphrased content. If not, provide a corrected version that accurately reflects the original meaning, format, and includes the reference values.
        Do not include create any explanatory text or meta-information in your response, instead just utilize the existing meaning."""

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

    def verify_content(self, column, original, generated, reference, is_question, custom_prompts):
        verify_prompts = custom_prompts.get('verify', {})
        system_prompt = verify_prompts.get('system', '')
        user_prompt = verify_prompts.get('user', '')

        if not user_prompt:
            user_prompt = """
            Original: {original}
            Generated: {generated}
            Reference values: {reference}
            Is question: {is_question}

            Verify that the generated content maintains the original meaning, format, and correctly incorporates the reference values. If it does, return the generated content. If not, provide a corrected version that accurately reflects the original meaning, format, and includes the reference values. Do not include any explanations or meta-information in your response.
            """

        formatted_user_prompt = user_prompt.format(
            original=original,
            generated=generated,
            reference=reference,
            is_question=is_question
        )

        response = self.ollama_interface.chat(messages=[
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': formatted_user_prompt}
        ])
        
        verified_text = response['message']['content'].strip()
        
        # Ensure the verified text ends with a question mark if the original was a question
        if is_question and not verified_text.endswith('?'):
            verified_text += '?'
        elif not is_question and verified_text.endswith('?'):
            verified_text = verified_text[:-1] + '.'
        
        return verified_text

    def generate_enhanced_synthetic_data(self, seed_data, num_samples, column_types, custom_prompts):
        synthetic_data = []
        samples_per_original = num_samples // len(seed_data)
        remaining_samples = num_samples % len(seed_data)

        for _, original_row in tqdm(seed_data.iterrows(), total=len(seed_data), desc="Generating synthetic data"):
            samples_for_this_row = samples_per_original + (1 if remaining_samples > 0 else 0)
            remaining_samples = max(0, remaining_samples - 1)

            for _ in range(samples_for_this_row):
                synthetic_row = {}
                for column, col_type in column_types.items():
                    if col_type in ['static', 'reference']:
                        synthetic_row[column] = original_row[column]
                    elif col_type == 'dynamic':
                        original_content = original_row[column]
                        synthetic_row[column] = self.generate_content(column, original_content, original_row, column_types, custom_prompts)
                    else:
                        raise ValueError(f"Unknown column type '{col_type}' for column '{column}'")

                synthetic_data.append(synthetic_row)

        result_df = pd.DataFrame(synthetic_data)

        # Verify all columns
        for column, col_type in column_types.items():
            if col_type in ['static', 'reference']:
                original_values = seed_data[column].repeat(samples_per_original).reset_index(drop=True)
                if remaining_samples > 0:
                    original_values = pd.concat([original_values, seed_data[column][:remaining_samples]], ignore_index=True)
                if not result_df[column].equals(original_values):
                    logging.error(f"{col_type.capitalize()} column '{column}' has been modified!")
                    logging.error(f"Original values: {original_values}")
                    logging.error(f"Modified values: {result_df[column]}")
                    raise ValueError(f"{col_type.capitalize()} column '{column}' has been modified during synthetic data generation. This is not allowed.")
                else:
                    logging.info(f"Verified {col_type} column '{column}' remains unchanged.")
            elif col_type == 'dynamic':
                if result_df[column].equals(seed_data[column].repeat(samples_per_original).reset_index(drop=True)):
                    logging.warning(f"Dynamic column '{column}' has not been modified. This is unexpected.")
                else:
                    logging.info(f"Verified dynamic column '{column}' has been modified as expected.")

        return result_df
    
    def is_question(self, text):
        return text.strip().endswith('?') or text.lower().startswith(('what', 'when', 'where', 'who', 'why', 'how', 'can', 'could', 'would', 'should', 'is', 'are', 'do', 'does'))
    
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
    
    def generate_synthetic_data(self, seed_file, sample_rate, paraphrases_per_sample, column_types, use_all_samples=True, custom_prompts={}, **kwargs):
        try:
            seed_file_path = os.path.join(self.input_dir, seed_file)
            logging.info(f"Looking for seed file at: {seed_file_path}")
            logging.info(f"Column types: {column_types}")
            
            if not os.path.exists(seed_file_path):
                raise FileNotFoundError(f"Seed file not found: {seed_file_path}")
            
            seed_data = pd.read_parquet(seed_file_path)
            num_samples = len(seed_data) if use_all_samples else int(len(seed_data) * (sample_rate / 100))
            
            samples_to_use = seed_data if use_all_samples else seed_data.sample(n=num_samples, replace=False)

            total_samples = num_samples * paraphrases_per_sample
            
            print(f"{Fore.CYAN}Starting synthetic data generation:{Style.RESET_ALL}")
            print(f"Seed file: {seed_file}")
            print(f"Total samples to generate: {total_samples}")
            print(f"Sample rate: {sample_rate}%")
            print(f"Paraphrases per sample: {paraphrases_per_sample}")
            print(f"Use all samples: {use_all_samples}")
            
            result_df = self.enhanced_generator.generate_enhanced_synthetic_data(
                samples_to_use, 
                total_samples, 
                column_types, 
                custom_prompts
            )
            
            print(f"{Fore.GREEN}Synthetic data generation completed successfully{Style.RESET_ALL}")
            return result_df
        except Exception as e:
            error_msg = f"Error in generate_synthetic_data: {str(e)}"
            print(f"{Fore.RED}{error_msg}{Style.RESET_ALL}")
            logging.exception(error_msg)
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
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Templates file path: {self.templates_file}")
        self.templates = self.load_templates()

    def load_templates(self):
        try:
            if os.path.exists(self.templates_file):
                with open(self.templates_file, 'r') as f:
                    templates = json.load(f)
                self.logger.info(f"Successfully loaded templates from {self.templates_file}")
            else:
                self.logger.warning(f"Templates file not found at {self.templates_file}. Creating default templates.")
                templates = self.create_default_templates()
                self.save_templates(templates)
            
            # Ensure all template values are lists
            for key, value in templates.items():
                if not isinstance(value, list):
                    templates[key] = list(value) if isinstance(value, (tuple, set)) else [str(value)]
            
            return templates
        except Exception as e:
            self.logger.error(f"Error loading templates: {str(e)}")
            return self.create_default_templates()

    def create_default_templates(self):
        return {
            "chat": ["instruction", "input", "output"],
            "chat_task": ["task", "instruction", "input", "output"],
            "chat_modeltag": ["instruction", "input", "output"],
            "instruct": ["task", "instruction", "input", "output", "generationModel"],
            "instruct_modeltag": ["task", "instruction", "input", "output"],
            "commander": ["task", "instruction", "input", "output", "command"],
            "commander_modeltag": ["task", "instruction", "input", "output", "command", "generationModel"],                
            "commander_testset": ["task", "instruction", "input", "command"],
            "commander_testset_modeltag": ["task", "instruction", "input", "command", "generationModel"],
            "aiConcept_instruct": ["task", "instruction", "input", "output", "concept", "definition", "useCase", "example"],   
            "latexSeries": ["task", "instruction", "input", "output", "formula", "solution"],
            "latexTheory": ["task", "instruction", "input", "output", "theory", "explanation"],
            "latexMath" : ["task", "instruction", "input", "output", "formula", "solution", "theory", "explanation"],
            "ontology1": ["task", "instruction", "input", "output", "ontology", "description"],
            "ontology2": ["task", "instruction", "input", "output", "ontology", "description", "chainOfThought"],
            "ontology3": ["task", "instruction", "input", "output", "ontology", "description", "application", "actions", "chainOfThought","self_prompts"],
            "mathPythonFusion": ["task", "instruction", "input", "output", "formula", "solution", "python", "example"],
            "pythonBase": ["task", "instruction", "input", "output", "code", "description", "args", "returns"],
            "pythonOllama": ["task", "instruction", "input", "output", "code", "description", "args", "returns", "actions", "chainOfThought", "prompts"],   
        }

    def save_templates(self, templates):
        try:
            os.makedirs(os.path.dirname(self.templates_file), exist_ok=True)
            with open(self.templates_file, 'w') as f:
                json.dump(templates, f, indent=2)
            self.logger.info(f"Successfully saved templates to {self.templates_file}")
        except Exception as e:
            self.logger.error(f"Error saving templates: {str(e)}")

    def get_templates(self):
        return self.templates

    def create_template(self, template_name, template_fields):
        if template_name in self.templates:
            raise ValueError(f"Template '{template_name}' already exists")
        self.templates[template_name] = template_fields
        self.save_templates(self.templates)
        return self.templates[template_name]

    def get_template(self, template_name):
        template = self.templates.get(template_name)
        if template is None:
            self.logger.warning(f"Template '{template_name}' not found")
        return template

    def add_template(self, template_name, template_fields):
        if template_name in self.templates:
            raise ValueError(f"Template '{template_name}' already exists")
        self.templates[template_name] = template_fields
        self.save_templates(self.templates)
        return self.templates[template_name]

class DocumentLoader:
    def __init__(self, github_access_token=None):
        self.GITHUB_ACCESS_TOKEN = github_access_token

    # URL
    def is_url(self, string):
        try:
            result = urllib.parse.urlparse(string)
            return all([result.scheme, result.netloc])
        except ValueError:
            return False

    def is_huggingface_url(self, url):
        return 'huggingface.co/' in url.lower()

    def is_wikipedia_url(self, url):
        parsed_url = urllib.parse.urlparse(url)
        return parsed_url.netloc.endswith('wikipedia.org') and '/wiki/' in parsed_url.path

    def is_arxiv_url(self, url):
        return 'arxiv.org' in url.lower()

    def is_github_url(self, url):
        return 'github.com' in url.lower()

    def load_web_page(self, url):
        loader = WebBaseLoader(url)
        return loader.load()

    # WIKIPEDIA   
    def extract_wikipedia_title(self, url):
        parsed_url = urllib.parse.urlparse(url)
        title = parsed_url.path.split('/wiki/', 1)[-1]
        return urllib.parse.unquote(title).replace('_', ' ')

    def load_wikipedia(self, query, load_max_docs=5):
        loader = WikipediaLoader(query, load_max_docs=load_max_docs)
        return loader.load()

    # ARXIV
    def extract_arxiv_id(self, url):
        arxiv_id = url.rstrip('/').split('/')[-1]
        return arxiv_id

    def _download_latex_source(self, arxiv_id, output_dir):
        try:
            source_url = f"https://arxiv.org/e-print/{arxiv_id}"
            filename = f"{arxiv_id.split('v')[0]}_source"
            filepath = os.path.join(output_dir, filename)
            response = requests.get(source_url)
            response.raise_for_status()

            with open(filepath, 'wb') as f:
                f.write(response.content)

            if tarfile.is_tarfile(filepath):
                with tarfile.open(filepath, 'r') as tar:
                    tar.extractall(path=filepath + '_extracted')
                os.remove(filepath)
                return f"LaTeX source extracted to: {filepath}_extracted"
            elif filepath.endswith('.gz'):
                with gzip.open(filepath, 'rb') as f_in:
                    with open(filepath[:-3], 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                os.remove(filepath)
                return f"LaTeX source extracted to: {filepath[:-3]}"
            else:
                return f"Source file downloaded to: {filepath}"
        except Exception as e:
            return f"Error downloading LaTeX source: {str(e)}"

    def load_arxiv(self, query, load_max_docs=5):
        loader = ArxivLoader(query, load_max_docs=load_max_docs)
        return loader.load()

    # DOCUMENTS
    def load_pdf(self, file_path):
        loader = PyPDFLoader(file_path)
        return loader.load()

    def load_text(self, file_path):
        loader = TextLoader(file_path)
        return loader.load()

    def load_docx(self, file_path):
        loader = Docx2txtLoader(file_path)
        return loader.load()

    def load_ppt(self, file_path):
        loader = UnstructuredPowerPointLoader(file_path)
        return loader.load()

    def load_epub(self, file_path):
        loader = UnstructuredEPubLoader(file_path)
        return loader.load()

    def load_csv(self, file_path):
        loader = CSVLoader(file_path)
        return loader.load()

    def load_json(self, file_path):
        loader = JSONLoader(
            file_path=file_path,
            jq_schema='.',
            text_content=False
        )
        return loader.load()

    # GITHUB
    def extract_github_repo_info(self, url):
        url = url.lower()
        repo_index = url.find('github.com/')    
        if repo_index != -1:
            repo_part = url[repo_index + len('github.com/'):].rstrip('/')
            parts = repo_part.split('/')        
            if len(parts) >= 2:
                repo_name = f"{parts[0]}/{parts[1]}"
            else:
                return None, None
            if 'tree/' in url:
                branch_index = url.find('tree/')
                branch_name = url[branch_index + len('tree/'):].split('/')[0]
            else:
                branch_name = 'main'
            return repo_name, branch_name
        return None, None

    def load_github(self, repo: str, branch: str) -> List[dict]:
        g = Github(self.GITHUB_ACCESS_TOKEN)
        repo = g.get_repo(repo)
        contents = repo.get_contents("", ref=branch)
        
        files = []
        while contents:
            file_content = contents.pop(0)
            if file_content.type == "dir":
                contents.extend(repo.get_contents(file_content.path, ref=branch))
            else:
                files.append({
                    "name": file_content.name,
                    "path": file_content.path,
                    "download_url": file_content.download_url
                })
        
        return files

    def select_github_files(self, files: List[dict]) -> List[dict]:
        print("Available files in the repository:")
        for file in files:
            print(f"{file['path']}")
        
        selected_files = []
        while True:
            file_path = input("Enter the path of a file you want to download (or press Enter to finish): ").strip()
            if not file_path:
                break
            matching_files = [f for f in files if f['path'] == file_path]
            if matching_files:
                selected_files.extend(matching_files)
                print(f"Added: {file_path}")
            else:
                print(f"File not found: {file_path}")
        
        return selected_files

    # HUGGINGFACE
    def hf_repo(self, repo_id):
        loader = snapshot_download(repo_id=repo_id)
        return loader.load()

    def extract_huggingface_repo_id(self, url: str) -> str:
        parts = url.split('huggingface.co/')
        if len(parts) > 1:
            return parts[1].strip('/')
        return ""

    def list_huggingface_repo_files(self, repo_id: str) -> List[str]:
        api = HfApi()
        files = api.list_repo_files(repo_id)
        return files

    def select_huggingface_files(self, files: List[str]) -> List[str]:
        print("Available files in the repository:")
        for file in files:
            print(f"{file}")
        
        selected_files = []
        while True:
            file_path = input("Enter the path of a file you want to download (or press Enter to finish): ").strip()
            if not file_path:
                break
            if file_path in files:
                selected_files.append(file_path)
                print(f"Added: {file_path}")
            else:
                print(f"File not found: {file_path}")
        
        return selected_files

    def download_huggingface_files(self, repo_id: str, files: List[str], output_dir: str) -> List[str]:
        downloaded_files = []
        for file in files:
            try:
                local_file = hf_hub_download(repo_id=repo_id, filename=file, local_dir=output_dir)
                downloaded_files.append(local_file)
                print(f"Downloaded: {local_file}")
            except Exception as e:
                print(f"Error downloading {file}: {str(e)}")
        return downloaded_files

    def process_huggingface_repo(self, url: str, output_dir: str) -> List[str]:
        repo_id = self.extract_huggingface_repo_id(url)
        if not repo_id:
            print("Invalid Hugging Face repository URL")
            return []

        files = self.list_huggingface_repo_files(repo_id)
        selected_files = self.select_huggingface_files(files)
        return self.download_huggingface_files(repo_id, selected_files, output_dir)
    
    def load_hf_dataset(self, dataset_name):
        try:
            dataset = load_dataset(dataset_name)
            documents = {}
            for split in dataset.keys():
                documents[f'default_{split}'] = [item for item in dataset[split]]
            return documents
        except ValueError as e:
            if "Config name is missing" in str(e):
                configs = str(e).split("among the available configs: ")[1].split("]")[0].replace("[", "").replace("'", "").split(", ")
                
                documents = {}
                for config in configs:
                    try:
                        dataset = load_dataset(dataset_name, config)
                        for split in dataset.keys():
                            documents[f'{config}_{split}'] = [item for item in dataset[split]]
                    except Exception as config_error:
                        print(f"Error loading config '{config}': {str(config_error)}")
                
                if documents:
                    return documents
            raise
        except Exception as e:
            print(f"Error loading dataset: {str(e)}")
            return {}
    
    def is_huggingface_url(self, url):
        return 'huggingface.co/' in url.lower() and 'datasets/' not in url.lower()

    def is_huggingface_dataset_url(self, url):
        return 'huggingface.co/datasets/' in url.lower()

    def extract_huggingface_dataset_name(self, url):
        # Find the index of 'datasets/' in the URL
        dataset_index = url.lower().find('huggingface.co/datasets/')
        if dataset_index != -1:
            # Return everything after 'datasets/'
            dataset_name = url[dataset_index + len('huggingface.co/datasets/'):].rstrip('/')
        return dataset_name

    def serialize_dataset_item(self, item):
        """
        Serialize a dataset item into a string representation.
        """
        return json.dumps(item, ensure_ascii=False, indent=2)

    def ingest_document(self, file_path_or_url: str, output_dir: str = "./output") -> str:
        """
        Ingests a single document or web page, extracting text based on its format and saves it as a txt file.

        Args:
            file_path_or_url (str): Path to the document file or URL.
            output_dir (str): Directory to save the output txt file.

        Returns:
            str: Path to the saved txt file.
        """
        # Check if it's a URL
        url_flag = self.is_url(file_path_or_url)
        
        # If it's not a URL, check if it could be a URL without the protocol
        if not url_flag and '/' in file_path_or_url and not os.path.exists(file_path_or_url):
            file_path_or_url = 'http://' + file_path_or_url
            url_flag = True

        wikipedia_flag = False
        arxiv_flag = False
        huggingface_dataset_flag = False
        huggingface_repo_flag = False
        github_flag = False
        if url_flag:
            wikipedia_flag = self.is_wikipedia_url(file_path_or_url)
            arxiv_flag = self.is_arxiv_url(file_path_or_url)
            huggingface_dataset_flag = self.is_huggingface_dataset_url(file_path_or_url)
            huggingface_repo_flag = self.is_huggingface_url(file_path_or_url) and not huggingface_dataset_flag
            github_flag = self.is_github_url(file_path_or_url)
            if wikipedia_flag:
                file_name = self.extract_wikipedia_title(file_path_or_url)
            elif arxiv_flag:
                arxiv_id = self.extract_arxiv_id(file_path_or_url)
                file_name = f"arxiv_{arxiv_id}"
                documents = self.load_arxiv(arxiv_id, load_max_docs=1)
                latex_source_result = self._download_latex_source(arxiv_id, output_dir)
            elif huggingface_dataset_flag :
                file_name = self.extract_huggingface_dataset_name(file_path_or_url).replace('/', '_')
            elif github_flag:
                repo_name, branch_name = self.extract_github_repo_info(file_path_or_url)
                file_name = f"{repo_name.replace('/', '_')}_{branch_name}"
            else:
                file_name = urllib.parse.urlparse(file_path_or_url).path.split('/')[-1]
            if not file_name:
                file_name = "webpage"
            output_file_name = f"{file_name}_extracted.txt"
        else:
            file_name = os.path.basename(file_path_or_url)
            output_file_name = f"{os.path.splitext(file_name)[0]}_extracted.txt"
        
        # Create the output directory if it doesn't exist
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        else:
            output_dir = os.getcwd()  # Use current working directory if no output_dir is specified
        
        output_path = os.path.join(output_dir, output_file_name)
        
        try:
            if wikipedia_flag:
                documents = self.load_wikipedia(file_name)
            elif arxiv_flag:
                # We've already loaded the documents for arXiv, so we don't need to do it again
                pass
            elif huggingface_repo_flag:
                downloaded_files = self.process_huggingface_repo(file_path_or_url, output_dir)
                if downloaded_files:
                    print(f"Downloaded files from Hugging Face repository: {', '.join(downloaded_files)}")
                    return output_dir  # Return the directory containing all downloaded files
                else:
                    print("No files were downloaded from the Hugging Face repository.")
                    return ""
            elif huggingface_dataset_flag :
                dataset_name = self.extract_huggingface_dataset_name(file_path_or_url)
                documents = self.load_hf_dataset(dataset_name)
                # Handle multiple configs and splits
                for config_split, data in documents.items():
                    config_split_output_file_name = f"{file_name}_{config_split}_extracted.txt"
                    config_split_output_path = os.path.join(output_dir, config_split_output_file_name)
                    content = "\n\n".join([self.serialize_dataset_item(doc) for doc in data])
                    with open(config_split_output_path, 'w', encoding='utf-8') as f:
                        f.write(content)
                    print(f"Extracted content for '{config_split}' saved to: {config_split_output_path}")
                return output_dir  # Return the directory containing all config files
            elif github_flag:
                repo_name, branch_name = self.extract_github_repo_info(file_path_or_url)
                all_files = self.load_github(repo_name, branch_name)
                selected_files = self.select_github_files(all_files)
                
                documents = []
                for file in selected_files:
                    file_content = requests.get(file['download_url']).text
                    documents.append({"page_content": file_content, "metadata": {"source": file['path']}})
                
                # Create a directory for GitHub files
                github_output_dir = os.path.join(output_dir, f"{repo_name.replace('/', '_')}_{branch_name}")
                os.makedirs(github_output_dir, exist_ok=True)
                
                for doc in documents:
                    file_path = os.path.join(github_output_dir, doc['metadata']['source'])
                    os.makedirs(os.path.dirname(file_path), exist_ok=True)
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(doc['page_content'])
                    print(f"Saved: {file_path}")
                
                return github_output_dir
            elif url_flag:
                documents = self.load_web_page(file_path_or_url)
            else:
                file_extension = os.path.splitext(file_path_or_url)[1].lower()
                if file_extension == '.pdf':
                    documents = self.load_pdf(file_path_or_url)
                elif file_extension in ['.doc', '.docx']:
                    documents = self.load_docx(file_path_or_url)
                elif file_extension == '.csv':
                    documents = self.load_csv(file_path_or_url)
                elif file_extension == '.json':
                    documents = self.load_json(file_path_or_url)
                elif file_extension == '.txt':
                    documents = self.load_text(file_path_or_url)
                elif file_extension == '.epub':
                    documents = self.load_epub(file_path_or_url)
                elif file_extension in ['.ppt', '.pptx']:
                    documents = self.load_ppt(file_path_or_url)
                else:
                    # If it's not a recognized file type, assume it's a Wikipedia query
                    documents = self.load_wikipedia(file_path_or_url)
            
            # Handle non-Hugging Face and non-GitHub cases
            if not huggingface_dataset_flag  and not github_flag:
                content = "\n\n".join([doc.page_content if hasattr(doc, 'page_content') else str(doc) for doc in documents])
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                print(f"Extracted content saved to: {output_path}")
                return output_path

        except Exception as e:
            print(f"Error processing {file_path_or_url}: {str(e)}")
            return ""
# example usage
# loader = DocumentLoader()
# file_path =r"https://huggingface.co/datasets/SkunkworksAI/reasoning-0.01"
# ingested_documents = loader.ingest_document(file_path)