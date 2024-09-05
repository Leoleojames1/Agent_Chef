import os
import subprocess
import logging
import sys

class UnslothTrainer:
    def __init__(self, base_dir, input_dir, output_dir):
        self.project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        self.cutlery_dir = os.path.join(self.project_dir, 'cutlery')
        self.base_dir = base_dir
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.logger = logging.getLogger(__name__)
        self.unsloth_script_path = self._find_unsloth_script()
        self.unsloth_path = self._find_unsloth_path()

    def _find_unsloth_script(self):
        script_path = os.path.join(self.cutlery_dir, 'unsloth_train_script.py')
        if not os.path.exists(script_path):
            raise FileNotFoundError(f"unsloth_train_script.py not found at {script_path}")
        return script_path

    def _find_unsloth_path(self):
        unsloth_path = os.path.expanduser('~/unsloth')
        if not os.path.exists(unsloth_path):
            raise FileNotFoundError(f"Local Unsloth directory not found at {unsloth_path}")
        return unsloth_path

    def train(self, model_name, train_dataset, validation_dataset, test_dataset=None, output_dir="unsloth_model", **kwargs):
        self.logger.info("Starting Unsloth training")
        
        cli_args = [
            "python",
            self.unsloth_script_path,
            "--model_name", model_name,
            "--dataset", train_dataset,
            "--output_dir", output_dir,
        ]

        if validation_dataset:
            cli_args.extend(["--validation_dataset", validation_dataset])
        if test_dataset:
            cli_args.extend(["--test_dataset", test_dataset])

        # Add any additional kwargs as CLI arguments
        for key, value in kwargs.items():
            if isinstance(value, bool):
                if value:
                    cli_args.append(f"--{key}")
            elif value is not None:
                cli_args.extend([f"--{key}", str(value)])

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
            
            # You can add logic here to parse the output and extract progress information
            if line.startswith("Progress:"):
                progress = int(line.split(":")[1].strip().rstrip('%'))
                # You can store this progress information or use it as needed

        process.wait()

        if process.returncode != 0:
            return {"error": "Training failed", "output": "\n".join(output)}
        else:
            return {"message": "Training completed successfully", "output": "\n".join(output)}

    def save_gguf(self, model_path, output_dir, quantization_method="q4_k_m"):
        self.logger.info(f"Saving GGUF model to {output_dir} with quantization {quantization_method}")
        
        cli_args = [
            "python",
            self.unsloth_script_path,
            "--model_name", model_path,
            "--save_gguf",
            "--output_dir", output_dir,
            "--quantization", quantization_method
        ]

        return self._run_command(cli_args)

    def push_to_hub(self, model_path, repo_id, token):
        self.logger.info(f"Pushing model to Hugging Face Hub: {repo_id}")
        
        cli_args = [
            "python",
            self.unsloth_script_path,
            "--model_name", model_path,
            "--push_to_hub",
            "--hub_model_id", repo_id,
            "--hub_token", token
        ]

        return self._run_command(cli_args)

    def _run_command(self, cli_args):
        process = subprocess.Popen(
            cli_args,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True
        )

        output = []
        for line in process.stdout:
            self.logger.info(line.strip())
            output.append(line.strip())

        process.wait()

        if process.returncode != 0:
            return {"error": "Command failed", "output": "\n".join(output)}
        else:
            return {"message": "Command completed successfully", "output": "\n".join(output)}