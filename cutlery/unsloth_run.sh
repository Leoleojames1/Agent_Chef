#!/bin/bash

# Set the path to your project directory
PROJECT_DIR="$HOME/Agent_Chef"

# Check if AgentChef environment is set up
check_agentchef_env() {
    if [ ! -d "$HOME/miniconda3" ] || \
       ! $HOME/miniconda3/bin/conda info --envs | grep -q "AgentChef" || \
       ! $HOME/miniconda3/bin/conda run -n AgentChef python --version | grep -q "Python 3.11"; then
        return 1
    fi
    return 0
}

# Run the installer if the environment is not set up
if ! check_agentchef_env; then
    echo "AgentChef environment not found or incomplete. Running installer..."
    bash "$PROJECT_DIR/unsloth_install.sh"
    if [ $? -ne 0 ]; then
        echo "Installation failed. Please check the error messages and try again."
        exit 1
    fi
fi

# Activate the AgentChef environment
source "$HOME/miniconda3/bin/activate" AgentChef

# Run the Unsloth training script
echo "Running Unsloth training..."
python "$PROJECT_DIR/cutlery/unsloth_train_script.py" "$@"