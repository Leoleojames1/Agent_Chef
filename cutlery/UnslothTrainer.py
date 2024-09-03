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
        unsloth_path = os.path.abspath(os.path.join(self.project_dir, '..', 'unsloth'))
        if not os.path.exists(unsloth_path):
            raise FileNotFoundError(f"Local Unsloth directory not found at {unsloth_path}")
        return unsloth_path

    def train(self, model_name, train_dataset, validation_dataset, test_dataset=None, output_dir="unsloth_model", **kwargs):
        self.logger.info("Starting Unsloth training")
        
        # Convert Windows paths to WSL paths
        wsl_train_dataset = self._to_wsl_path(train_dataset)
        wsl_validation_dataset = self._to_wsl_path(validation_dataset) if validation_dataset else None
        wsl_test_dataset = self._to_wsl_path(test_dataset) if test_dataset else None
        wsl_output_dir = self._to_wsl_path(os.path.join(self.output_dir, output_dir))
        
        cli_args = [
            "python",
            self._to_wsl_path(self.unsloth_script_path),
            "--model_name", model_name,
            "--dataset", wsl_train_dataset,
            "--output_dir", wsl_output_dir,
        ]

        if wsl_validation_dataset:
            cli_args.extend(["--validation_dataset", wsl_validation_dataset])
        if wsl_test_dataset:
            cli_args.extend(["--test_dataset", wsl_test_dataset])

        # Add any additional kwargs as CLI arguments
        for key, value in kwargs.items():
            if isinstance(value, bool):
                if value:
                    cli_args.append(f"--{key}")
            elif value is not None:
                cli_args.extend([f"--{key}", str(value)])

        # Construct the WSL command
        wsl_command = ["wsl", "bash", "/mnt/c/path/to/your/unsloth_run.sh"] + cli_args

        # Run the command and capture output in real-time
        process = subprocess.Popen(
            wsl_command,
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

    def _to_wsl_path(self, windows_path):
        # Convert Windows path to WSL path
        drive, path = os.path.splitdrive(windows_path)
        return f"/mnt/{drive.lower().rstrip(':')}{path.replace(os.sep, '/')}"

    def save_gguf(self, model_path, output_dir, quantization_method="q4_k_m"):
        self.logger.info(f"Saving GGUF model to {output_dir} with quantization {quantization_method}")
        
        cli_args = [
            "--model_name", model_path,
            "--save_gguf",
            "--save_path", output_dir,
            "--quantization", quantization_method
        ]

        return self._run_unsloth_cli(cli_args)

    def push_to_hub(self, model_path, repo_id, token):
        self.logger.info(f"Pushing model to Hugging Face Hub: {repo_id}")
        
        cli_args = [
            "--model_name", model_path,
            "--push_model",
            "--hub_path", repo_id,
            "--hub_token", token
        ]

        return self._run_unsloth_cli(cli_args)