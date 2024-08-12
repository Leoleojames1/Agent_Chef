import os
import logging
from colorama import init
from cutlery.Dataset_Manager_Class import Dataset_Manager_Class
from cutlery.Template_Manager_Class import Template_Manager_Class
from cutlery.Prompt_Manager_Class import Prompt_Manager_Class
from cutlery.Ollama_Interface_Class import Ollama_Interface_Class
from cutlery.File_Handler_Class import File_Handler_Class
import time
import glob

# Initialize colorama
init(autoreset=True)

class Agent_Chef_Class:
    def __init__(self):
        self.base_dir = os.path.join(os.path.dirname(__file__), 'agent_chef_data')
        self.input_dir = os.path.join(self.base_dir, "ingredients")
        self.output_dir = os.path.join(self.base_dir, "dishes")
        self.latex_library_dir = os.path.join(self.base_dir, "latex_library")
        self.construction_zone_dir = os.path.join(self.base_dir, "construction_zone")  # Add this line
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
        self.dataset_params = None
        logging.basicConfig(level=logging.INFO)

        # Ensure all directories exist
        for dir_path in [self.input_dir, self.output_dir, self.latex_library_dir, self.construction_zone_dir]:
            os.makedirs(dir_path, exist_ok=True)

    def initialize(self, model):
        self.model = model
        self.ollama_interface = Ollama_Interface_Class(self.model)
        self.dataset_manager = Dataset_Manager_Class(self.ollama_interface)

    def run(self, mode, seed_parquet, synthetic_technique=None, template=None, system_prompt=None, num_samples=100):
        self.mode = mode
        self.seed_parquet = seed_parquet
        self.synthetic_technique = synthetic_technique
        self.template = template
        self.system_prompt = system_prompt

        logging.info(f"Running Agent_Chef with mode: {mode}, seed_parquet: {seed_parquet}")

        if not self.seed_parquet:
            error_msg = "No seed parquet file specified"
            logging.error(error_msg)
            return {'error': error_msg}

        self.initialize(self.model)

        try:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            base_filename = os.path.splitext(os.path.basename(self.seed_parquet))[0]

            seed_parquet_path = os.path.join(self.input_dir, self.seed_parquet)
            if not os.path.exists(seed_parquet_path):
                error_msg = f"Seed parquet file not found: {seed_parquet_path}"
                logging.error(error_msg)
                return {'error': error_msg}

            if self.mode == 'custom':
                dataset = self.dataset_manager.generate_synthetic_data(
                    seed_parquet_path, 
                    num_samples=num_samples, 
                    system_prompt=self.system_prompt,
                    template=self.template
                )
                if dataset.empty:
                    error_msg = "Failed to generate synthetic data. Check the logs for details."
                    logging.error(error_msg)
                    return {'error': error_msg}
                output_file = os.path.join(self.output_dir, f'{base_filename}_synthetic_{timestamp}.parquet')
                self.file_handler.save_to_parquet(dataset, output_file)
                return {'message': "Custom synthetic dataset generated successfully", 'file': os.path.basename(output_file)}

            elif self.mode == 'parquet':
                if self.synthetic_technique == 'combine':
                    combined_dataset = self.dataset_manager.combine_parquets(self.input_dir)
                    if combined_dataset.empty:
                        return {'error': "Failed to combine parquet files. Check the logs for details."}
                    output_file = os.path.join(self.output_dir, f'combined_{timestamp}.parquet')
                    self.file_handler.save_to_parquet(combined_dataset, output_file)
                    return {'message': "Parquet files combined successfully", 'file': os.path.basename(output_file)}
                elif self.synthetic_technique == 'augment':
                    augmented_dataset = self.dataset_manager.augment_data(seed_parquet_path)
                    if augmented_dataset.empty:
                        return {'error': "Failed to augment data. Check the logs for details."}
                    output_file = os.path.join(self.output_dir, f'{base_filename}_augmented_{timestamp}.parquet')
                    self.file_handler.save_to_parquet(augmented_dataset, output_file)
                    return {'message': "Data augmented successfully", 'file': os.path.basename(output_file)}
                else:
                    return {'error': "Invalid synthetic technique specified"}

            else:
                return {'error': "Invalid mode selected"}

        except Exception as e:
            error_msg = f"Error in Agent_Chef_Class.run: {str(e)}"
            logging.exception(error_msg)
            return {'error': error_msg}
        
    def process_all_parquets(self, mode, synthetic_technique=None, template=None, system_prompt=None, num_samples=100):
        parquet_files = glob.glob(os.path.join(self.output_dir, '*.parquet'))
        results = []
        for parquet_file in parquet_files:
            result = self.run(
                mode=mode,
                seed_parquet=parquet_file,
                synthetic_technique=synthetic_technique,
                template=template,
                system_prompt=system_prompt,
                num_samples=num_samples
            )
            results.append(result)
        return results