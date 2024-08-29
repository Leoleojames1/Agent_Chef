import logging
import os
import time
import traceback 
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

    def run(self, mode, seed_file, **kwargs):
        logging.info(f"Running AgentChef with mode: {mode}, seed_file: {seed_file}")

        if not seed_file:
            return {'error': "No seed file specified"}

        try:
            if mode == 'custom':
                sample_rate = kwargs.get('sample_rate', 100)
                paraphrases_per_sample = kwargs.get('paraphrases_per_sample', 1)
                column_types = kwargs.get('column_types', {})
                use_all_samples = kwargs.get('use_all_samples', True)
                custom_prompts = kwargs.get('custom_prompts', {})

                seed_file_path = os.path.join(self.input_dir, seed_file)
                if not os.path.exists(seed_file_path):
                    return {'error': f"Seed file not found: {seed_file_path}"}

                result_df = self.dataset_manager.generate_synthetic_data(
                    seed_file,
                    sample_rate=sample_rate,
                    paraphrases_per_sample=paraphrases_per_sample,
                    column_types=column_types,
                    use_all_samples=use_all_samples,
                    custom_prompts=custom_prompts
                )

                if result_df.empty:
                    return {'error': "Generated dataset is empty. Check the logs for details."}

                # Generate output filename based on input filename
                input_name = os.path.splitext(seed_file)[0]
                timestamp = int(time.time())
                output_filename = f'{input_name}_synthetic_{timestamp}.parquet'
                output_file = os.path.join(self.output_dir, output_filename)
                result_df.to_parquet(output_file)

                return {
                    'message': "Custom synthetic dataset generated successfully",
                    'file': output_filename
                }
            else:
                return {'error': "Invalid mode selected"}

        except Exception as e:
            error_msg = f"Error in AgentChef.run: {str(e)}\n{traceback.format_exc()}"
            logging.exception(error_msg)
            return {'error': error_msg}