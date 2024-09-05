#!/bin/bash

# Setup Conda environment for AgentChef
if [ ! -d "$HOME/miniconda3" ]; then
    echo "Installing Miniconda..."
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
    bash miniconda.sh -b -p $HOME/miniconda3
    rm miniconda.sh
fi

source $HOME/miniconda3/bin/activate

if ! conda info --envs | grep -q "AgentChef"; then
    echo "Creating AgentChef environment..."
    conda create --name AgentChef python=3.11 -y
fi

conda activate AgentChef

echo "Installing PyTorch and other dependencies..."
conda install pytorch pytorch-cuda=12.1 cudatoolkit -c pytorch -c nvidia -y

echo "Cloning Unsloth repository..."
git clone https://github.com/unslothai/unsloth.git $HOME/unsloth

echo "Installing Unsloth and dependencies..."
pip install -e $HOME/unsloth
pip install xformers
pip install --no-deps trl peft accelerate bitsandbytes

echo "AgentChef environment setup complete."