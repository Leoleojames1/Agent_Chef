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
    echo "Ollama service started."
fi

# Ensure port 3000 is available
echo "Ensuring port 3000 is available for React app..."
sudo fuser -k 3000/tcp || true
echo "Port 3000 is now available."

# Ensure port 5000 is available
echo "Ensuring port 5000 is available for Flask app..."
sudo fuser -k 5000/tcp || true
echo "Port 5000 is now available."

# Start the Flask app in a new terminal window
echo "Starting Flask app..."
gnome-terminal --title="Flask App" -- bash -c "
source $CONDA_ACTIVATE AgentChef
echo 'Flask app starting...'
python app.py 2>&1 | tee flask_output.log
echo 'Flask app stopped.'
read -p 'Press Enter to close this window...'
"

# Start the React app in a new terminal window
echo "Starting React app..."
gnome-terminal --title="React App" -- bash -c "
cd react-app
unset HOST
echo 'React app starting...'
npm start 2>&1 | tee react_output.log
echo 'React app stopped.'
read -p 'Press Enter to close this window...'
"

echo "AgentChef components started in separate windows."
echo "Please check the individual terminal windows for detailed output."
echo "Flask app logs are being saved to flask_output.log"
echo "React app logs are being saved to react_output.log"

# Wait for user input before exiting
read -p "Press Enter to stop all components and exit..."

# Kill Flask and React processes
echo "Stopping Flask and React apps..."
pkill -f "python app.py"
pkill -f "npm start"

echo "AgentChef runner script completed."