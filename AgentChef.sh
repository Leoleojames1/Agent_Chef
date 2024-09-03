#!/bin/bash

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check Unsloth environment
check_unsloth_env() {
    if [ ! -d "$HOME/miniconda3" ] || ! conda info --envs | grep -q "unsloth_env" || ! conda run -n unsloth_env pip list | grep -q "unsloth"; then
        return 1
    fi
    return 0
}

# Check and install Ollama if necessary
if ! command_exists ollama; then
    echo "Ollama not found. Installing Ollama..."
    curl https://ollama.ai/install.sh | sh
fi

# Check and install Miniconda if necessary
if [ ! -d "$HOME/miniconda3" ]; then
    echo "Miniconda not found. Installing Miniconda..."
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
    bash miniconda.sh -b -p $HOME/miniconda3
    rm miniconda.sh
    echo 'export PATH="$HOME/miniconda3/bin:$PATH"' >> ~/.bashrc
    source ~/.bashrc
fi

# Set up Unsloth environment if necessary
if ! check_unsloth_env; then
    echo "Unsloth environment not found or incomplete. Setting up Unsloth..."
    source $HOME/miniconda3/bin/activate
    conda create --name unsloth_env python=3.11 pytorch-cuda=12.1 pytorch cudatoolkit xformers -c pytorch -c nvidia -c xformers -y
    conda activate unsloth_env
    pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
    pip install --no-deps trl peft accelerate bitsandbytes
    echo "Unsloth environment setup complete."
fi

# Set Ollama environment variables
export OLLAMA_NUM_PARALLEL=2
export OLLAMA_MAX_LOADED_MODELS=2
export OLLAMA_FLASH_ATTENTION=1

# Activate Conda environment
source $HOME/miniconda3/bin/activate raglocal

# Start Windows Terminal with multiple panes
wt.exe --maximized \
    -p "Agent Chef" bash -c "ollama serve" \; \
    split-pane -d "." bash -c "python app.py" \; \
    split-pane -d "./react-app" bash -c "npm start"