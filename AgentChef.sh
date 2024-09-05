#!/bin/bash

set -e  # Exit immediately if a command exits with a non-zero status.

CONDA_PATH="$HOME/miniconda3/bin/conda"
CONDA_ACTIVATE="$HOME/miniconda3/bin/activate"

echo "Starting AgentChef.sh script"

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check AgentChef environment
check_agentchef_env() {
    if [ ! -d "$HOME/miniconda3" ] || \
       ! $CONDA_PATH info --envs | grep -q "AgentChef" || \
       ! $CONDA_PATH run -n AgentChef python --version | grep -q "Python 3.10"; then
        return 1
    fi
    return 0
}

echo "Checking Miniconda..."
$CONDA_PATH --version

# Initialize conda for bash
echo "Initializing conda..."
source $HOME/miniconda3/etc/profile.d/conda.sh

# Check and install Ollama if necessary
if ! command_exists ollama; then
    echo "Ollama not found. Installing Ollama..."
    curl https://ollama.ai/install.sh | sh
fi

# Set up AgentChef environment if necessary
if ! check_agentchef_env; then
    echo "AgentChef environment not found or incomplete. Setting up AgentChef..."
    $CONDA_PATH create --name AgentChef python=3.10 -y
    source $CONDA_ACTIVATE AgentChef
    $CONDA_PATH install pytorch pytorch-cuda=11.8 -c pytorch -c nvidia -y
    pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
    pip install xformers==0.0.20
    pip install --no-deps trl peft accelerate bitsandbytes
    echo "AgentChef environment setup complete."
else
    echo "AgentChef environment found. Activating..."
    source $CONDA_ACTIVATE AgentChef
fi

# Set Ollama environment variables
export OLLAMA_NUM_PARALLEL=2
export OLLAMA_MAX_LOADED_MODELS=2
export OLLAMA_FLASH_ATTENTION=1

# Start the application components
echo "Starting Ollama server..."
ollama serve &

echo "Starting Python app..."
python app.py &

echo "Starting React app..."
cd ./react-app && npm start &

# Wait for all background processes to finish
wait

echo "Script completed."