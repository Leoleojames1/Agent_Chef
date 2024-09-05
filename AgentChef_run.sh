#!/bin/bash

set -e  # Exit immediately if a command exits with a non-zero status.

CONDA_PATH="$HOME/miniconda3/bin/conda"
CONDA_ACTIVATE="$HOME/miniconda3/bin/activate"

echo "Starting AgentChef runner script"

# Set Ollama environment variables
export OLLAMA_NUM_PARALLEL=2
export OLLAMA_MAX_LOADED_MODELS=2
export OLLAMA_FLASH_ATTENTION=1

# Function to check if port 11434 is in use
is_port_in_use() {
    ss -tuln | grep -q :11434
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
tmux send-keys -t 0 "source $CONDA_ACTIVATE AgentChef && ollama serve" C-m
tmux send-keys -t 1 "source $CONDA_ACTIVATE AgentChef && python app.py" C-m
tmux send-keys -t 2 "source $CONDA_ACTIVATE AgentChef && cd ./react-app && npm start" C-m

# Attach to the tmux session
tmux attach-session -t AgentChef

echo "AgentChef runner script completed."