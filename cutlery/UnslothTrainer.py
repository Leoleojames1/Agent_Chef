import os
import subprocess
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer
from unsloth import FastLanguageModel

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

    def merge_adapter(self, base_model_path, adapter_path, output_path, convert_to_gguf=False):
        self.logger.info(f"Merging adapter from {adapter_path} into base model {base_model_path}")
        
        cli_args = [
            "python",
            self.unsloth_script_path,
            "merge",
            "--base_model_path", base_model_path,
            "--adapter_path", adapter_path,
            "--output_path", output_path
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
                self.convert_to_gguf(output_path, os.path.basename(output_path))
                return {"message": "Merging completed successfully, and Converted to GGUF", "output": "\n".join(output)}
            return {"message": "Merging completed successfully", "output": "\n".join(output)}

    def convert_to_gguf(self, input_path, model_name):
        self.logger.info(f"Converting model to GGUF format: {input_path}")
        llama_cpp_dir = os.path.expanduser("~/llama.cpp")
        convert_script = os.path.join(llama_cpp_dir, "convert_hf_to_gguf.py")
        
        if not os.path.exists(convert_script):
            self.logger.error(f"Error: convert.py not found at {convert_script}")
            return
        
        gguf_dir = os.path.join(input_path, "gguf")
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
            subprocess.run(command, check=True)
            self.logger.info(f"Conversion to GGUF completed. Output saved in {output_file}")
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Error during GGUF conversion: {e}")
            self.logger.error(f"Command output: {e.output}")