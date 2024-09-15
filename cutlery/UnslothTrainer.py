import os
import subprocess
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer
from unsloth import FastLanguageModel
import shutil

class UnslothTrainer:
    def __init__(self, base_dir, input_dir, output_dir):
        self.project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        self.cutlery_dir = os.path.join(self.project_dir, 'cutlery')
        self.base_dir = base_dir
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.logger = logging.getLogger(__name__)
        self.unsloth_script_path = self._find_unsloth_script()
        self.llama_cpp_dir = os.path.expanduser("~/llama.cpp")

    def _find_unsloth_script(self):
        script_path = os.path.join(self.cutlery_dir, 'unsloth-cli-2.py')
        if not os.path.exists(script_path):
            raise FileNotFoundError(f"unsloth-cli-2.py not found at {script_path}")
        return script_path

    def train(self, model_name, train_dataset, validation_dataset=None, test_dataset=None, output_dir="unsloth_model", **kwargs):
        self.logger.info("Starting Unsloth training")
        self.logger.info(f"Model name: {model_name}")
        self.logger.info(f"Training dataset: {train_dataset}")
        self.logger.info(f"Validation dataset: {validation_dataset}")
        self.logger.info(f"Test dataset: {test_dataset}")
        self.logger.info(f"Output directory: {output_dir}")
        
        cli_args = [
            "python",
            self.unsloth_script_path,
            "train",
            "--model_name", model_name,
            "--train_dataset", train_dataset,
            "--output_dir", output_dir,
        ]

        if validation_dataset:
            cli_args.extend(["--validation_dataset", validation_dataset])
        
        if test_dataset:
            cli_args.extend(["--test_dataset", test_dataset])

        # Handle precision option
        if kwargs.get('load_in_16bit'):
            cli_args.append("--load_in_16bit")
        elif kwargs.get('load_in_4bit'):
            cli_args.append("--load_in_4bit")

        # Add any additional kwargs as CLI arguments
        for key, value in kwargs.items():
            if key not in ['load_in_16bit', 'load_in_4bit']:
                if isinstance(value, bool):
                    if value:
                        cli_args.append(f"--{key}")
                elif value is not None:
                    cli_args.extend([f"--{key}", str(value)])

        self.logger.info(f"Running command: {' '.join(cli_args)}")

        # Run the command and capture output in real-time
        process = subprocess.Popen(
            cli_args,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True
        )

        output = []
        while True:
            line = process.stdout.readline()
            if not line:
                break
            self.logger.info(line.strip())
            output.append(line.strip())

        process.wait()

        if process.returncode != 0:
            self.logger.error("Training failed")
            return {"error": "Training failed", "output": "\n".join(output)}
        else:
            self.logger.info("Training completed successfully")
            if kwargs.get('convert_to_gguf'):
                self.convert_to_gguf(output_dir, os.path.basename(output_dir))
            return {"message": "Training completed successfully", "output": "\n".join(output)}

    def merge_adapter(self, base_model_path, adapter_path, output_path, convert_to_gguf=True, dequantize='no'):
        self.logger.info(f"Merging adapter from {adapter_path} into base model {base_model_path}")
        
        # Create a new directory for the merged model
        merged_dir = os.path.join(self.output_dir, "merged_models")
        os.makedirs(merged_dir, exist_ok=True)
        
        # Parse the original output path
        original_output_name = os.path.basename(output_path)
        
        # If dequantizing, create a separate directory
        if dequantize != 'no':
            dequantized_dir = os.path.join(merged_dir, f"dequantized_{dequantize}")
            os.makedirs(dequantized_dir, exist_ok=True)
            final_output_path = os.path.join(dequantized_dir, original_output_name)
        else:
            final_output_path = os.path.join(merged_dir, original_output_name)

        # Ensure the final output directory exists
        os.makedirs(os.path.dirname(final_output_path), exist_ok=True)

        cli_args = [
            "python",
            self.unsloth_script_path,
            "merge",
            "--base_model_path", base_model_path,
            "--adapter_path", adapter_path,
            "--output_path", final_output_path,
            "--dequantize", dequantize
        ]

        self.logger.info(f"Running command: {' '.join(cli_args)}")

        process = subprocess.Popen(
            cli_args,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True
        )

        output = []
        while True:
            line = process.stdout.readline()
            if not line:
                break
            self.logger.info(line.strip())
            output.append(line.strip())

        process.wait()

        if process.returncode != 0:
            self.logger.error("Merging failed")
            return {"error": "Merging failed", "output": "\n".join(output)}
        else:
            self.logger.info("Merging completed successfully")
            
            if convert_to_gguf:
                self.logger.info("Starting GGUF conversion")
                gguf_result = self.convert_to_gguf(final_output_path, original_output_name)
                if gguf_result:
                    return {"message": "Merging completed successfully, and Converted to GGUF", "output": "\n".join(output), "merged_path": final_output_path}
                else:
                    return {"message": "Merging completed successfully, but GGUF conversion failed", "output": "\n".join(output), "merged_path": final_output_path}
            return {"message": "Merging completed successfully", "output": "\n".join(output), "merged_path": final_output_path}
    
    def convert_to_gguf(self, input_path, model_name):
        self.logger.info(f"Converting model to GGUF format: {input_path}")
        llama_cpp_dir = os.path.expanduser("~/llama.cpp")
        convert_script = os.path.join(llama_cpp_dir, "convert_hf_to_gguf.py")
        
        if not os.path.exists(convert_script):
            self.logger.error(f"Error: convert_hf_to_gguf.py not found at {convert_script}")
            return False
        
        gguf_dir = os.path.join(self.output_dir, "gguf_models")
        os.makedirs(gguf_dir, exist_ok=True)
        output_file = os.path.join(gguf_dir, f"{model_name}.gguf")

        command = [
            "python",
            convert_script,
            "--outtype", "f16",  # Use f16 to avoid quantization
            "--outfile", output_file,
            input_path
        ]

        self.logger.info(f"Running command: {' '.join(command)}")

        try:
            result = subprocess.run(command, check=True, capture_output=True, text=True)
            self.logger.info(f"Conversion to GGUF completed. Output saved in {output_file}")
            self.logger.info(f"Conversion output: {result.stdout}")
            return True
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Error during GGUF conversion: {e}")
            self.logger.error(f"Command output: {e.stdout}")
            self.logger.error(f"Command error: {e.stderr}")
            return False
        
    def cleanup_merged_models(self, keep_latest=5):
        merged_dir = os.path.join(self.output_dir, "merged_models")
        if not os.path.exists(merged_dir):
            return

        # List all subdirectories in the merged_models directory
        subdirs = [d for d in os.listdir(merged_dir) if os.path.isdir(os.path.join(merged_dir, d))]
        
        # Sort subdirectories by modification time (newest first)
        subdirs.sort(key=lambda x: os.path.getmtime(os.path.join(merged_dir, x)), reverse=True)

        # Keep the latest 'keep_latest' directories and remove the rest
        for subdir in subdirs[keep_latest:]:
            dir_to_remove = os.path.join(merged_dir, subdir)
            self.logger.info(f"Removing old merged model: {dir_to_remove}")
            shutil.rmtree(dir_to_remove)

    def cleanup_gguf_models(self, keep_latest=5):
        gguf_dir = os.path.join(self.output_dir, "gguf_models")
        if not os.path.exists(gguf_dir):
            return

        # List all GGUF files in the gguf_models directory
        gguf_files = [f for f in os.listdir(gguf_dir) if f.endswith('.gguf')]
        
        # Sort GGUF files by modification time (newest first)
        gguf_files.sort(key=lambda x: os.path.getmtime(os.path.join(gguf_dir, x)), reverse=True)

        # Keep the latest 'keep_latest' files and remove the rest
        for gguf_file in gguf_files[keep_latest:]:
            file_to_remove = os.path.join(gguf_dir, gguf_file)
            self.logger.info(f"Removing old GGUF model: {file_to_remove}")
            os.remove(file_to_remove)