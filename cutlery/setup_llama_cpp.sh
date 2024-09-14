#!/bin/bash

# Set up error handling
set -e

# Define the directory where llama.cpp will be installed
LLAMA_CPP_DIR="$HOME/llama.cpp"

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check and install dependencies
echo "Checking and installing dependencies..."
sudo apt update
sudo apt install -y build-essential cmake git

# Check if CUDA is available
if command_exists nvcc; then
    echo "CUDA found. Will build with CUDA support."
    MAKE_ARGS="LLAMA_CUBLAS=1"
else
    echo "CUDA not found. Building without CUDA support."
    MAKE_ARGS=""
fi

# Clone or update llama.cpp
if [ -d "$LLAMA_CPP_DIR" ]; then
    echo "Updating existing llama.cpp repository..."
    cd "$LLAMA_CPP_DIR"
    git pull
else
    echo "Cloning llama.cpp repository..."
    git clone https://github.com/ggerganov/llama.cpp.git "$LLAMA_CPP_DIR"
    cd "$LLAMA_CPP_DIR"
fi

# Build llama.cpp
echo "Building llama.cpp..."
make clean
make $MAKE_ARGS

# Add llama.cpp directory to PATH
echo "Adding llama.cpp to PATH..."
echo "export PATH=\"$LLAMA_CPP_DIR:\$PATH\"" >> "$HOME/.bashrc"

echo "llama.cpp has been set up successfully!"
echo "Please run 'source ~/.bashrc' or start a new terminal session to update your PATH."