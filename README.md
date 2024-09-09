<p align="center">
  <img src="docs/icons/Agent_Chef_logo.png" alt="OARC LOGO" width="250"/>
</p>
<p align="center">
  <a href="https://ko-fi.com/theborch"><img src="docs/icons/buy me a coffee button.png" height="48"></a>
  <a href="https://discord.gg/dAzSYcnpdF"><img src="docs/icons/Discord button.png" height="48"></a>
</p>

# 🍲Agent Chef (AC) V0.1.16🥘
***[🦾Borch's AI Development Guide🦿](https://share.note.sx/c3topc9y#iaFb281+b0x66J+2lWIhWp4PV+wwoKsd5GqoXYg1i4I)***   ***[🦙 Ollama Discord Server 🦙](https://discord.gg/ollama)***   ***[🤖 OARC V0.28 VIDEO GUIDE 🧙](https://www.youtube.com/watch?v=W7TusPTnNXA)***

🍲Agent Chef is a powerful tool designed for dataset refinement, structuring, and generation. It empowers users to create high-quality, domain-specific datasets for fine-tuning AI models.🥘

## Features

- 🥕**Dataset Refinement**🥩:
  - Clean and refine your existing datasets
- 🥣**Synthetic Data Generation**🥣:
  - Create procedural and synthetic datasets
- 🔪**Data Poisoning Elimination**🔪:
  - Identify and remove low-quality or malicious data
- 🍛**Specialized Dataset Construction**🍛:
  Generate datasets for specific use cases, including:
  - Function-calling
  - Programming: Python, React, C++
  - Mathematics: LaTeX, Python
  - Languages, Physics, Biology, Chemistry, Law, Cooking, wikipedias, history, context, and more!

<img
src="docs/agent_chef_poster.jpeg"
  style="display: inline-block; margin: 0 auto; max-width: 50px">


## Setup
Start by selecting your ollama model, system prompt, number of duplicates, and the parquet column template format.
<img
src="docs/icons/agent_chef_ui_4.png"
  style="display: inline-block; margin: 0 auto; max-width: 50px">

## Procedural construction
After setup, manually construct your dataset group formatting by wrapping each data point in $("data") wrappers.

<img
src="docs/icons/OARC_commander.png"
  style="display: inline-block; margin: 0 auto; max-width: 50px">
  
These group tags allow agent chef to splice the data out into the correct parquet cells for dataset construction. The automatic formatting tool is currently experimental however it attempts to generate the $("data") tags from the template formatting and inference. Only use this feature if you are daring. 
  [Insert Image]
  
## Synthetic Generation
After procedural dataset construction is complete, you can move on to generating the synthetic dataset from the procedural parquet.
<img
src="docs/icons/OARC_commander_synth.png"
  style="display: inline-block; margin: 0 auto; max-width: 50px">

## Why Agent Chef?

Agent Chef aims to revolutionize home-brewed AI by providing tools and frameworks that enable users to create high-quality, domain-specific datasets. Whether you're looking to improve an existing dataset or generate new data from scratch, Agent Chef has you covered.

Agent Chef Installation Guide

Prerequisites:

Windows Users:
1. Install Windows Subsystem for Linux (WSL):
   - Follow the official Microsoft guide to install WSL:
     https://learn.microsoft.com/en-us/windows/wsl/install
   - After installation, open a WSL terminal for the next steps.

Linux Users:
- Skip the WSL installation step.

Installation Steps:

1. Install Miniconda
   Open a terminal and run the following commands:
   
   ```
   cd ~
   mkdir -p ~/miniconda3
   wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
   bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
   rm ~/miniconda3/miniconda.sh
   ```
   
   Add Miniconda to your PATH:
   
   ```
   echo 'export PATH="$HOME/miniconda3/bin:$PATH"' >> ~/.bashrc
   source ~/.bashrc
   ```

2. Create and Activate Conda Environment
   
   ```
   conda create -n AgentChef python=3.11 -y
   conda activate AgentChef
   ```

3. Clone the Repository
   
   ```
   cd ~
   git clone https://github.com/Leoleojames1/Agent_Chef.git
   cd Agent_Chef
   ```

4. Install Dependencies
   
   ```
   bash AgentChef_install.sh
   pip install -r requirements.txt
   ```

5. Install Node.js Dependencies
   
   ```
   cd react-app
   npm install
   cd ..
   ```

6. Setup Hugging Face and C++ Compiler
   
   ```
   pip install huggingface_hub
   huggingface-cli login --token YOUR_TOKEN_HERE
   conda install -c conda-forge gcc_linux-64 gxx_linux-64 -y
   gcc --version
   ```
   
   Replace YOUR_TOKEN_HERE with your actual Hugging Face token.

Usage:

To run Agent Chef:

1. Navigate to the Agent Chef directory:
   
   ```
   cd ~/Agent_Chef
   ```

2. Activate the conda environment:
   
   ```
   conda activate AgentChef
   ```

3. Run the application:
   
   ```
   bash AgentChef_run.sh
   ```

Troubleshooting:

- If you encounter any issues during installation, please check the project's GitHub issues or create a new issue for support:
  https://github.com/Leoleojames1/Agent_Chef/issues
- Ensure all prerequisites are correctly installed and your system meets the minimum requirements.

For more detailed information and advanced usage, please refer to the project's documentation.
