#!/bin/bash

# Activate the Conda environment
source $HOME/miniconda3/bin/activate AgentChef

# Freeze Python requirements
echo "Freezing Python requirements..."
pip freeze > requirements.txt

echo "Python requirements frozen to requirements.txt"

# Freeze npm requirements
echo "Freezing npm requirements..."
if [ -d "./react-app" ]; then
    cd react-app
    if [ -f "package.json" ]; then
        npm list --depth=0 > npm-requirements.txt
        echo "npm requirements frozen to react-app/npm-requirements.txt"
    else
        echo "package.json not found in react-app directory"
    fi
    cd ..
else
    echo "react-app directory not found"
fi

echo "Freezing process completed."