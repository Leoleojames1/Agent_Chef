#!/bin/bash

# Set the target directory for Hugging Face models
HF_DIR="$HOME/Agent_Chef/agent_chef_data/huggingface_models"

# Ensure the directory exists
mkdir -p "$HF_DIR"

# Change to the Hugging Face models directory
cd "$HF_DIR"

# Install Git LFS globally (only needs to be done once)
git lfs install

MODELS=(
    "https://huggingface.co/unsloth/Meta-Llama-3.1-8B-bnb-4bit"
    "https://huggingface.co/mlabonne/Meta-Llama-3.1-8B-Instruct-abliterated"
    "https://huggingface.co/unsloth/llama-3-8b-bnb-4bit"
    "https://huggingface.co/unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"
)

for model in "${MODELS[@]}"; do
    model_name=$(basename "$model")
    echo "Cloning $model_name..."
    if [ -d "$model_name" ]; then
        echo "Directory $model_name already exists. Updating..."
        cd "$model_name"
        git pull
        git lfs pull
        cd ..
    else
        GIT_LFS_SKIP_SMUDGE=1 git clone "$model"
        cd "$model_name"
        git lfs pull
        cd ..
    fi
    echo "Finished processing $model_name"
    echo
done

echo "All models have been cloned/updated in $HF_DIR"