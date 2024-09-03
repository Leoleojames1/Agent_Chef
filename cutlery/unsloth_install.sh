#!/bin/bash

# Setup Conda environment for Unsloth
if [ ! -d "$HOME/miniconda3" ]; then
    echo "Installing Miniconda..."
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
    bash miniconda.sh -b -p $HOME/miniconda3
    rm miniconda.sh
fi

source $HOME/miniconda3/bin/activate

if ! conda info --envs | grep -q "unsloth_env"; then
    echo "Creating Unsloth environment..."
    conda create --name unsloth_env python=3.11 pytorch-cuda=12.1 pytorch cudatoolkit xformers -c pytorch -c nvidia -c xformers -y
fi

conda activate unsloth_env

if ! pip list | grep -q "unsloth"; then
    echo "Installing Unsloth and dependencies..."
    pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
    pip install --no-deps trl peft accelerate bitsandbytes
fi

echo "Unsloth environment setup complete."