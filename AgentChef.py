import logging
import os
import time
from cutlery.DatasetKitchen import DatasetManager, TemplateManager
from cutlery.OllamaInterface import OllamaInterface
from cutlery.FileHandler import FileHandler

class AgentChef:
    def __init__(self, base_dir):
        self.base_dir = base_dir
        self.input_dir = os.path.join(self.base_dir, "ingredients")
        self.output_dir = os.path.join(self.base_dir, "dishes")
        self.latex_library_dir = os.path.join(self.base_dir, "latex_library")
        self.construction_zone_dir = os.path.join(self.base_dir, "construction_zone")

        self.ollama_interface = OllamaInterface(None)
        self.file_handler = FileHandler(self.input_dir, self.output_dir)
        self.template_manager = TemplateManager(self.input_dir)
        self.dataset_manager = DatasetManager(
            self.ollama_interface, 
            self.template_manager,
            self.input_dir,
            self.output_dir
        )

        logging.basicConfig(level=logging.INFO)

        for dir_path in [self.input_dir, self.output_dir, self.latex_library_dir, self.construction_zone_dir]:
            os.makedirs(dir_path, exist_ok=True)

    def initialize(self, model):
        self.ollama_interface.set_model(model)

    def run(self, mode, seed_parquet, **kwargs):
        logging.info(f"Running AgentChef with mode: {mode}, seed_parquet: {seed_parquet}")

        if not seed_parquet:
            return {'error': "No seed parquet file specified"}

        try:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            base_filename = os.path.splitext(os.path.basename(seed_parquet))[0]
            seed_parquet_path = os.path.join(self.input_dir, seed_parquet)

            if not os.path.exists(seed_parquet_path):
                return {'error': f"Seed parquet file not found: {seed_parquet_path}"}

            if mode == 'custom':
                system_prompt = kwargs.get('system_prompt')
                if not system_prompt:
                    return {'error': "System prompt is required for custom mode"}

                dataset = self.dataset_manager.generate_synthetic_data(
                    seed_parquet_path,
                    num_samples=kwargs.get('num_samples', 100),
                    system_prompt=system_prompt,
                    template=kwargs.get('template')
                )
                if dataset.empty:
                    return {'error': "Failed to generate synthetic data. Check the logs for details."}
                output_file = os.path.join(self.output_dir, f'{base_filename}_synthetic_{timestamp}.parquet')
                self.file_handler.save_to_parquet(dataset, output_file)
                return {'message': "Custom synthetic dataset generated successfully", 'file': os.path.basename(output_file)}
            
            else:
                return {'error': "Invalid mode selected"}

        except Exception as e:
            error_msg = f"Error in AgentChef.run: {str(e)}"
            logging.exception(error_msg)
            return {'error': error_msg}