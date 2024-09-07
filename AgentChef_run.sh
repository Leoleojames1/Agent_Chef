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

# Ensure port 5000 is available
echo "Ensuring port 5000 is available for Flask app..."
sudo fuser -k 5000/tcp || true

# Start the Flask app in the background
source $CONDA_ACTIVATE AgentChef
python app.py &
FLASK_PID=$!

# Start the React app
cd ./react-app
npm start

# If the React app exits, kill the Flask app
kill $FLASK_PID

echo "AgentChef runner script completed."