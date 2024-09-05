#!/bin/bash

# Check if AgentChef environment is set up
check_agentchef_env() {
    if [ ! -d "$HOME/miniconda3" ] || \
       ! conda info --envs | grep -q "AgentChef" || \
       ! conda run -n AgentChef python --version | grep -q "Python 3.11"; then
        return 1
    fi
    return 0
}

# Run the installer if the environment is not set up
if ! check_agentchef_env; then
    echo "AgentChef environment not found or incomplete. Running installer..."
    bash /path/to/unsloth_install.sh
    if [ $? -ne 0 ]; then
        echo "Installation failed. Please check the error messages and try again."
        exit 1
    fi
fi

# Activate the AgentChef environment
source $HOME/miniconda3/bin/activate AgentChef

# Run the Unsloth training script
echo "Running Unsloth training..."
python /mnt/c/path/to/your/unsloth_train_script.py "$@"