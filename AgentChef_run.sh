#!/bin/bash

set -e  # Exit immediately if a command exits with a non-zero status.

CONDA_PATH="$HOME/miniconda3/bin/conda"
CONDA_ACTIVATE="$HOME/miniconda3/bin/activate"

echo "Starting AgentChef runner script"

# Set Ollama environment variables
export OLLAMA_NUM_PARALLEL=2
export OLLAMA_MAX_LOADED_MODELS=2
export OLLAMA_FLASH_ATTENTION=1

# Function to check if a port is in use
is_port_in_use() {
    ss -tuln | grep -q :$1
}

# Function to kill tmux session
kill_tmux_session() {
    session_name="AgentChef"
    
    # Try pkill first
    if pkill -f "tmux.*$session_name" 2>/dev/null; then
        echo "Killed tmux session using pkill."
    else
        # If pkill fails, try kill -9
        tmux_pids=$(pgrep -f "tmux.*$session_name")
        if [ -n "$tmux_pids" ]; then
            if sudo kill -9 $tmux_pids 2>/dev/null; then
                echo "Killed tmux session using kill -9."
            else
                echo "Failed to kill tmux session. You may need to manually terminate it."
            fi
        else
            echo "No existing tmux session found."
        fi
    fi
}

# Stop Ollama service and free up port 11434
echo "Stopping Ollama service and freeing up port 11434..."
sudo systemctl stop ollama || echo "Failed to stop Ollama service. It may not be running."
sudo fuser -k 11434/tcp || echo "No process using port 11434."

# Free up port 3000 for React app
echo "Freeing up port 3000 for React app..."
sudo fuser -k 3000/tcp || echo "No process using port 3000."

# Start Ollama service
echo "Starting Ollama service..."
sudo systemctl restart ollama
sleep 2  # Give Ollama some time to start

# Kill existing tmux session
kill_tmux_session

# Create a new tmux session
tmux new-session -d -s AgentChef

# Split the window into three panes
tmux split-window -h
tmux split-window -v

# Send commands to each pane
tmux send-keys -t 0 "source $CONDA_ACTIVATE AgentChef && ollama serve" C-m
tmux send-keys -t 1 "source $CONDA_ACTIVATE AgentChef && python app.py" C-m
tmux send-keys -t 2 "source $CONDA_ACTIVATE AgentChef && cd ./react-app && PORT=3001 npm start" C-m

# Attach to the tmux session
tmux attach-session -t AgentChef

echo "AgentChef runner script completed."