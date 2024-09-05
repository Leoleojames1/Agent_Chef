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

# Check and install Node.js and npm if necessary
if ! command_exists npm; then
    echo "Node.js and npm not found. Installing..."
    curl -fsSL https://deb.nodesource.com/setup_lts.x | sudo -E bash -
    sudo apt-get install -y nodejs
fi

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
    pip install flask  # Ensure Flask is installed
    echo "AgentChef environment setup complete."
else
    echo "AgentChef environment found. Activating..."
    source $CONDA_ACTIVATE AgentChef
fi

# Install Python requirements
echo "Installing Python requirements..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
else
    echo "requirements.txt not found. Installing essential packages..."
    pip install flask  # Ensure Flask is installed even if requirements.txt is missing
fi

# Install npm packages
echo "Installing npm packages..."
if [ -d "./react-app" ]; then
    cd ./react-app
    if [ -f "package.json" ]; then
        npm install
    else
        echo "package.json not found in react-app directory. Skipping npm package installation."
    fi
    cd ..
else
    echo "react-app directory not found. Skipping npm package installation."
fi

# Set Ollama environment variables
export OLLAMA_NUM_PARALLEL=2
export OLLAMA_MAX_LOADED_MODELS=2
export OLLAMA_FLASH_ATTENTION=1

# Function to check if port 11434 is in use
is_port_in_use() {
    netstat -tuln | grep -q :11434
}

# Start the application components
echo "Checking Ollama service status..."
if sudo systemctl is-active --quiet ollama; then
    echo "Ollama service is already running. Stopping it..."
    sudo systemctl stop ollama
    sleep 2
fi

if is_port_in_use; then
    echo "Port 11434 is still in use. Attempting to free it..."
    sudo fuser -k 11434/tcp
    sleep 2
fi

echo "Starting Ollama service..."
sudo systemctl start ollama
sleep 2  # Give Ollama some time to start

echo "Starting Python app..."
python app.py &

echo "Starting React app..."
if command_exists npm; then
    cd ./react-app && npm start &
else
    echo "npm not found. Please install Node.js and npm to run the React app."
fi

# Wait for all background processes to finish
wait

echo "Script completed."