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
        self.unsloth_dir, self.unsloth_cli_path = self._find_unsloth()

    def _find_unsloth(self):
        # Look for the unsloth directory at the same level as Agent_Chef
        agent_chef_parent = os.path.dirname(self.project_dir)
        unsloth_dir = os.path.join(agent_chef_parent, 'unsloth')
        unsloth_cli_path = os.path.join(unsloth_dir, 'unsloth-cli.py')
        
        if not os.path.exists(unsloth_cli_path):
            raise FileNotFoundError(f"unsloth-cli.py not found at {unsloth_cli_path}")
        
        return unsloth_dir, unsloth_cli_path

    def train(self, model_name, train_dataset, validation_dataset, test_dataset=None, output_dir="unsloth_model", **kwargs):
        self.logger.info("Starting Unsloth training")
        
        # Ensure we're using the Python from the current environment
        python_executable = sys.executable
        
        cli_args = [
            python_executable,
            self.unsloth_cli_path,
            "--model_name", model_name,
            "--dataset", train_dataset,
            "--output_dir", output_dir,
            "--load_in_4bit",  # Add this if you're using 4-bit quantization
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

        # Set up the environment to use the local unsloth installation
        env = os.environ.copy()
        env["PYTHONPATH"] = f"{self.unsloth_dir}:{env.get('PYTHONPATH', '')}"

        # Run the command and capture output in real-time
        process = subprocess.Popen(
            cli_args,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            env=env
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