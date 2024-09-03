#!/bin/bash

# Check if Unsloth environment is set up
check_unsloth_env() {
    if [ ! -d "$HOME/miniconda3" ] || \
       ! conda info --envs | grep -q "unsloth_env" || \
       ! conda run -n unsloth_env pip list | grep -q "unsloth"; then
        return 1
    fi
    return 0
}

# Run the installer if the environment is not set up
if ! check_unsloth_env; then
    echo "Unsloth environment not found or incomplete. Running installer..."
    bash /path/to/unsloth_install.sh
    if [ $? -ne 0 ]; then
        echo "Installation failed. Please check the error messages and try again."
        exit 1
    fi
fi

# Activate the Unsloth environment
source $HOME/miniconda3/bin/activate unsloth_env

# Run the Unsloth training script
echo "Running Unsloth training..."
python /mnt/c/path/to/your/unsloth_train_script.py "$@"