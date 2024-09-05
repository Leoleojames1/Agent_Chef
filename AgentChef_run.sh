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

# Check if tmux session exists and kill it if it does
if tmux has-session -t AgentChef 2>/dev/null; then
    echo "Existing AgentChef tmux session found. Killing it..."
    tmux kill-session -t AgentChef
fi

# Create a new tmux session
tmux new-session -d -s AgentChef

# Split the window into three panes
tmux split-window -h
tmux split-window -v

# Send commands to each pane
tmux send-keys -t 0 "source $CONDA_ACTIVATE AgentChef && echo 'Ollama is already running. No need to start it here.'" C-m
tmux send-keys -t 1 "source $CONDA_ACTIVATE AgentChef && python app.py" C-m
tmux send-keys -t 2 "source $CONDA_ACTIVATE AgentChef && cd ./react-app && PORT=3000 npm start" C-m

# Attach to the tmux session
tmux attach-session -t AgentChef

echo "AgentChef runner script completed."