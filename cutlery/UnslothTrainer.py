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

    def _find_unsloth_script(self):
        script_path = os.path.join(self.cutlery_dir, 'unsloth-cli.py')
        if not os.path.exists(script_path):
            raise FileNotFoundError(f"unsloth-cli.py not found at {script_path}")
        return script_path

    def train(self, model_name, train_dataset, output_dir="unsloth_model", **kwargs):
        self.logger.info("Starting Unsloth training")
        
        cli_args = [
            "python",
            self.unsloth_script_path,
            "--model_name", model_name,
            "--dataset", train_dataset,
            "--output_dir", output_dir,
            "--load_in_4bit",
            "--save_model",
        ]

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