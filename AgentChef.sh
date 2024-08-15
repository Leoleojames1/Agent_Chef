#!/bin/bash

# Start LLaMA server
ollama serve &

# Wait for 1 second to let the server start
sleep 1

# Activate Conda environment
source ~/miniconda3/bin/activate raglocal
export OLLAMA_NUM_PARALLEL=2
export OLLAMA_MAX_LOADED_MODELS=2
export OLLAMA_FLASH_ATTENTION=1

# Run Flask app
python app.py &

# Run React development server (assuming it's in a directory named 'react-app')
cd react-app
npm start &