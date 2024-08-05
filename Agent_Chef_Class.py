import os
import pandas as pd
import json
import logging
from colorama import init, Fore
from cutlery.Dataset_Manager_Class import Dataset_Manager_Class
from cutlery.Template_Manager_Class import Template_Manager_Class
from cutlery.Prompt_Manager_Class import Prompt_Manager_Class
from cutlery.Ollama_Interface_Class import Ollama_Interface_Class
from cutlery.File_Handler_Class import File_Handler_Class
from cutlery.User_Interface_Class import User_Interface_Class

# Initialize colorama
init(autoreset=True)

class Agent_Chef_Class:
    def __init__(self):
        self.base_dir = os.path.join(os.path.dirname(__file__), 'agent_chef_data')
        self.input_dir = os.path.join(self.base_dir, "ingredients")
        self.output_dir = os.path.join(self.base_dir, "dishes")
        self.latex_library_dir = os.path.join(self.base_dir, "latex_library")
        self.ui = User_Interface_Class()
        self.model = None
        self.ollama_interface = None
        self.dataset_manager = None
        self.template_manager = Template_Manager_Class(self.input_dir)
        self.prompt_manager = Prompt_Manager_Class(self.input_dir)
        self.file_handler = File_Handler_Class(self.input_dir, self.output_dir)
        self.mode = None
        self.user_json = None
        self.dataset_name = None
        self.template = None
        self.system_prompt = None
        logging.basicConfig(level=logging.INFO)

    def main(self):
        self.ui.display_message("Welcome to Agent_Chef!", Fore.CYAN)
        self.ui.display_message("This tool helps you cook up synthetic datasets using custom user data or Hugging Face datasets.", Fore.CYAN)
        self.ui.display_message("You can choose between four cooking modes:", Fore.CYAN)
        self.ui.display_message("1. Custom: Create a dataset from scratch with custom ingredients.", Fore.CYAN)
        self.ui.display_message("2. Hugging Face: Use a pre-made dataset from Hugging Face and add our special sauce.", Fore.CYAN)
        self.ui.display_message("3. JSON: Transform your JSON ingredients into a gourmet Parquet dish.", Fore.CYAN)
        self.ui.display_message("4. Build JSON: Create a custom JSON file with your own formulas and data.", Fore.CYAN)

        if not self.model:
            self.ui.display_message("Model not specified. Exiting.", Fore.RED)
            return

        self.ollama_interface = Ollama_Interface_Class(self.model)
        self.dataset_manager = Dataset_Manager_Class(self.ollama_interface)

        if self.mode == 'build':
            self.user_json = self.build_user_json()
            if not self.user_json:
                self.ui.display_message("JSON building was aborted. Exiting.", Fore.RED)
                return
            self.mode = 'json'  # Switch to JSON mode after building
        elif self.mode in ['custom', 'huggingface', 'json']:
            if self.mode != 'custom' and not self.user_json:
                self.ui.display_message("Seed file not specified. Exiting.", Fore.RED)
                return
            if self.mode == 'huggingface' and not self.dataset_name:
                self.ui.display_message("Dataset name not specified. Exiting.", Fore.RED)
                return
        else:
            self.ui.display_message("Invalid mode selected. Exiting.", Fore.RED)
            return

        self.template = self.template_manager.load_template()
        self.system_prompt = self.prompt_manager.load_system_prompt()

        try:
            if self.mode == 'custom':
                self.ui.display_message("Preparing a fresh synthetic dataset...", Fore.GREEN)
                dataset = self.dataset_manager.generate_dataset(self.dataset_params)
                self.file_handler.save_to_parquet(dataset, 'custom_dish.parquet')
            elif self.mode == 'huggingface':
                self.ui.display_message(f"Fetching Hugging Face dataset: {self.dataset_name}", Fore.GREEN)
                original_data = self.dataset_manager.clone_huggingface_dataset(self.dataset_name)
                self.ui.display_message("Adding our special sauce...", Fore.GREEN)
                dataset = self.dataset_manager.generate_synthetic_data(original_data)
                self.file_handler.save_to_parquet(dataset, 'huggingface_dish.parquet')
            elif self.mode == 'json':
                self.ui.display_message("Transforming JSON ingredients into a gourmet Parquet dish...", Fore.GREEN)
                self.file_handler.save_json_to_parquet(self.user_json)
                dataset = pd.read_parquet(os.path.join(self.output_dir, self.user_json.replace('.json', '.parquet')))
            else:
                self.ui.display_message("Invalid mode selected. Exiting.", Fore.RED)
                return

            self.ui.display_message("Dataset preparation complete!", Fore.GREEN)
            logging.info("Dataset preparation complete!")
            return dataset
        except Exception as e:
            self.ui.display_message(f"Error: {str(e)}", Fore.RED)
            logging.error(f"Error: {str(e)}")
            return None