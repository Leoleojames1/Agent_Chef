#!/bin/bash

set -e  # Exit immediately if a command exits with a non-zero status.

CONDA_PATH="$HOME/miniconda3/bin/conda"
CONDA_ACTIVATE="$HOME/miniconda3/bin/activate"

echo "Starting AgentChef installation script"

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check if tmux is installed
if ! command_exists tmux; then
    echo "tmux is not installed. Installing tmux..."
    sudo apt-get update && sudo apt-get install -y tmux
fi

# Function to check AgentChef environment
check_agentchef_env() {
    if [ ! -d "$HOME/miniconda3" ] || \
       ! $CONDA_PATH info --envs | grep -q "AgentChef" || \
       ! $CONDA_PATH run -n AgentChef python --version | grep -q "Python 3.11"; then
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
    $CONDA_PATH create --name AgentChef python=3.11 -y
    source $CONDA_ACTIVATE AgentChef
    $CONDA_PATH install pytorch torchvision torchaudio cpuonly -c pytorch -y
    pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
    pip install xformers
    pip install --no-deps trl peft accelerate bitsandbytes
    pip install flask flask-cors
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
    pip install flask flask-cors
fi

# Install npm packages
echo "Installing npm packages..."
if [ -d "./react-app" ]; then
    cd ./react-app
    if [ -f "package.json" ]; then
        echo "Ensuring correct permissions for npm..."
        sudo chown -R $(whoami) .
        sudo chown -R $(whoami) "$HOME/.npm"
        
        echo "Running npm install..."
        npm install
        
        if [ $? -eq 0 ]; then
            echo "Running npm audit fix..."
            npm audit fix
        else
            echo "npm install failed. Please try running the following commands manually:"
            echo "cd ~/Agent_Chef/react-app"
            echo "npm install"
            echo "npm audit fix"
            echo "If issues persist, you may need to run: npm cache clean --force"
        fi
    else
        echo "package.json not found in react-app directory. Skipping npm package installation."
    fi
    cd ..
else
    echo "react-app directory not found. Skipping npm package installation."
fi

echo "AgentChef installation completed."