#!/bin/bash

# Start LLaMA server
ollama serve &

# Wait for 1 second to let the server start
sleep 1

# Activate Conda environment
source $HOME/miniconda3/bin/activate $HOME/miniconda3/envs/raglocal

# Set environment variables
export OLLAMA_NUM_PARALLEL=2
export OLLAMA_MAX_LOADED_MODELS=2
export OLLAMA_FLASH_ATTENTION=1
# export PYTHONPATH=$PYTHONPATH:/path/to/your/python-p2p-network

# Set PostgreSQL password
export PGPASSWORD=admin

# Run Python script
python Agent_Chef.py