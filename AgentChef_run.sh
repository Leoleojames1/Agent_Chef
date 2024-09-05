#!/bin/bash

set -e  # Exit immediately if a command exits with a non-zero status.

CONDA_PATH="$HOME/miniconda3/bin/conda"
CONDA_ACTIVATE="$HOME/miniconda3/bin/activate"

echo "Starting AgentChef runner script"

# Set Ollama environment variables
export OLLAMA_NUM_PARALLEL=2
export OLLAMA_MAX_LOADED_MODELS=2
export OLLAMA_FLASH_ATTENTION=1

# Function to check if Ollama is running
is_ollama_running() {
    if systemctl is-active --quiet ollama; then
        return 0
    else
        return 1
    fi
}

# Check if Ollama is running, start if not
if is_ollama_running; then
    echo "Ollama is already running."
else
    echo "Starting Ollama service..."
    sudo systemctl start ollama
    sleep 2  # Give Ollama some time to start
fi

# Ensure port 3000 is available
echo "Ensuring port 3000 is available for React app..."
sudo fuser -k 3000/tcp || true

# Run the other processes in the background, with output redirected to /dev/null
source $CONDA_ACTIVATE AgentChef && echo "Ollama is already running. No need to start it here." >/dev/null 2>&1 &
source $CONDA_ACTIVATE AgentChef && cd ./react-app && PORT=3000 npm start >/dev/null 2>&1 &

# Run app.py in the foreground so you can see its output
source $CONDA_ACTIVATE AgentChef && python app.py

echo "AgentChef runner script completed."
