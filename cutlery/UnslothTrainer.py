import os
import subprocess
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer
import shutil

class UnslothTrainer:
    def __init__(self, base_dir, input_dir, output_dir):
        self.project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        self.cutlery_dir = os.path.join(self.project_dir, 'cutlery')
        self.base_dir = base_dir
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.oven_dir = os.path.join(self.output_dir, "oven")
        self.gguf_dir = os.path.join(self.oven_dir, "gguf_models")
        self.logger = logging.getLogger(__name__)
        self.unsloth_script_path = self._find_unsloth_script()
        self.llama_cpp_dir = os.path.expanduser("~/llama.cpp")

        os.makedirs(self.gguf_dir, exist_ok=True)


        for dir_path in [self.models_dir, self.adapters_dir, self.merged_dir, self.gguf_dir]:
            os.makedirs(dir_path, exist_ok=True)

    def get_latest_checkpoint(self, model_dir):
        checkpoints = [d for d in os.listdir(model_dir) if d.startswith('checkpoint-')]
        if not checkpoints:
            return None
        latest_checkpoint = max(checkpoints, key=lambda x: int(x.split('-')[1]))
        return os.path.join(model_dir, latest_checkpoint)
    
    def get_merged_model_path(self, model_name):
        model_dir = os.path.join(self.oven_dir, model_name)
        if not os.path.exists(model_dir):
            return None
        numeric_dirs = [d for d in os.listdir(model_dir) if d.isdigit()]
        if not numeric_dirs:
            return None
        latest_dir = max(numeric_dirs, key=int)
        return os.path.join(model_dir, latest_dir)
    
    def _find_unsloth_script(self):
        script_path = os.path.join(self.cutlery_dir, 'unsloth-cli-2.py')
        if not os.path.exists(script_path):
            raise FileNotFoundError(f"unsloth-cli-2.py not found at {script_path}")
        return script_path

    def train(self, model_name, train_dataset, validation_dataset=None, test_dataset=None, output_name="unsloth_model", **kwargs):
        self.logger.info("Starting Unsloth training")
        output_dir = os.path.join(self.oven_dir, output_name)
        
        cli_args = [
            "python", self.unsloth_script_path, "train",
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
            # Move the adapter to the adapters directory
            adapter_path = os.path.join(output_dir, "adapter_model.bin")
            if os.path.exists(adapter_path):
                shutil.move(adapter_path, os.path.join(self.adapters_dir, f"{output_name}_adapter.bin"))
            return {"message": "Training completed successfully", "output": "\n".join(output), "model_path": output_dir}
        
    def merge_adapter(self, base_model_path, adapter_path, output_name, convert_to_gguf=True, dequantize='no'):
        self.logger.info(f"Merging adapter from {adapter_path} into base model {base_model_path}")
        
        base_model_dir = os.path.join(self.oven_dir, base_model_path)
        adapter_dir = os.path.join(self.oven_dir, adapter_path)
        
        latest_checkpoint = self.get_latest_checkpoint(adapter_dir)
        if not latest_checkpoint:
            raise ValueError(f"No checkpoint found in {adapter_dir}")
        
        adapter_model_path = os.path.join(latest_checkpoint, "adapter_model.safetensors")
        if not os.path.exists(adapter_model_path):
            raise FileNotFoundError(f"Adapter model not found: {adapter_model_path}")

        final_output_path = os.path.join(self.oven_dir, output_name)
        os.makedirs(final_output_path, exist_ok=True)

        cli_args = [
            "python", self.unsloth_script_path, "merge",
            "--base_model_path", base_model_dir,
            "--adapter_path", adapter_model_path,
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
                gguf_result = self.convert_to_gguf(final_output_path, output_name)
                if gguf_result['success']:
                    return {"message": "Merging completed successfully, and Converted to GGUF", "output": "\n".join(output), "merged_path": final_output_path, "gguf_path": gguf_result['output_file']}
                else:
                    return {"message": "Merging completed successfully, but GGUF conversion failed", "output": "\n".join(output), "merged_path": final_output_path}
            return {"message": "Merging completed successfully", "output": "\n".join(output), "merged_path": final_output_path}
    
    def convert_to_gguf(self, input_path, output_name):
        self.logger.info(f"Converting model to GGUF format: {input_path}")
        convert_script = os.path.join(self.llama_cpp_dir, "convert_hf_to_gguf.py")
        
        if not os.path.exists(convert_script):
            self.logger.error(f"Error: convert_hf_to_gguf.py not found at {convert_script}")
            return {"success": False, "error": "GGUF conversion script not found"}
        
        output_file = os.path.join(self.gguf_dir, f"{output_name}.gguf")

        command = [
            "python", convert_script,
            "--outtype", "f16",
            "--outfile", output_file,
            input_path
        ]

        self.logger.info(f"Running command: {' '.join(command)}")

        try:
            result = subprocess.run(command, check=True, capture_output=True, text=True)
            self.logger.info(f"Conversion to GGUF completed. Output saved in {output_file}")
            self.logger.info(f"Conversion output: {result.stdout}")
            return {"success": True, "message": f"GGUF conversion completed successfully", "output_file": output_file}
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Error during GGUF conversion: {e}")
            self.logger.error(f"Command output: {e.stdout}")
            self.logger.error(f"Command error: {e.stderr}")
            return {"success": False, "error": str(e), "details": e.stderr}
        
    def cleanup(self, keep_latest=5):
        for dir_path in [self.models_dir, self.adapters_dir, self.merged_dir, self.gguf_dir]:
            self._cleanup_directory(dir_path, keep_latest)

    def _cleanup_directory(self, directory, keep_latest):
        items = [os.path.join(directory, d) for d in os.listdir(directory)]
        items.sort(key=os.path.getmtime, reverse=True)

        for item in items[keep_latest:]:
            if os.path.isdir(item):
                shutil.rmtree(item)
            else:
                os.remove(item)
            self.logger.info(f"Removed old item: {item}")
            
    def get_models(self):
        return self._get_directory_contents(self.models_dir)

    def get_adapters(self):
        return self._get_directory_contents(self.adapters_dir)

    def get_merged_models(self):
        return self._get_directory_contents(self.merged_dir)

    def get_gguf_models(self):
        return self._get_directory_contents(self.gguf_dir)

    def _get_directory_contents(self, directory):
        return [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
    
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