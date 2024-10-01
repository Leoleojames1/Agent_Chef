@echo off
setlocal enabledelayedexpansion

echo Starting AgentChef installation script

:: Set paths
set "CONDA_PATH=%USERPROFILE%\miniconda3\Scripts\conda.exe"
set "CONDA_ACTIVATE=%USERPROFILE%\miniconda3\Scripts\activate.bat"

:: Function to check if a command exists
:command_exists
where %1 >nul 2>nul
if %errorlevel% neq 0 (
    exit /b 1
)
exit /b 0

:: Check if Miniconda is installed
call :command_exists conda
if %errorlevel% neq 0 (
    echo Miniconda is not installed. Please install Miniconda and run this script again.
    exit /b 1
)

:: Initialize conda for cmd
call %CONDA_PATH% init cmd

:: Check and install Node.js and npm if necessary
call :command_exists npm
if %errorlevel% neq 0 (
    echo Node.js and npm not found. Please install Node.js from https://nodejs.org/
    exit /b 1
)

:: Check and install Ollama if necessary
call :command_exists ollama
if %errorlevel% neq 0 (
    echo Ollama not found. Please install Ollama from https://ollama.ai/
    exit /b 1
)

:: Function to check AgentChef environment
:check_agentchef_env
%CONDA_PATH% info --envs | findstr /C:"AgentChef" >nul
if %errorlevel% neq 0 (
    exit /b 1
)
%CONDA_PATH% run -n AgentChef python --version | findstr /C:"Python 3.11" >nul
if %errorlevel% neq 0 (
    exit /b 1
)
exit /b 0

:: Set up AgentChef environment if necessary
call :check_agentchef_env
if %errorlevel% neq 0 (
    echo AgentChef environment not found or incomplete. Setting up AgentChef...
    %CONDA_PATH% create --name AgentChef python=3.11 -y
    call %CONDA_ACTIVATE% AgentChef
    %CONDA_PATH% install pytorch torchvision torchaudio cpuonly -c pytorch -y
    pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
    pip install xformers
    pip install --no-deps trl peft accelerate bitsandbytes
    pip install flask flask-cors
    echo AgentChef environment setup complete.
) else (
    echo AgentChef environment found. Activating...
    call %CONDA_ACTIVATE% AgentChef
)

:: Install Python requirements
echo Installing Python requirements...
if exist requirements.txt (
    pip install -r requirements.txt
) else (
    echo requirements.txt not found. Installing essential packages...
    pip install flask flask-cors
)

:: Install npm packages
echo Installing npm packages...
if exist react-app (
    cd react-app
    if exist package.json (
        echo Running npm install...
        npm install
        if %errorlevel% equ 0 (
            echo Running npm audit fix...
            npm audit fix
        ) else (
            echo npm install failed. Please try running the following commands manually:
            echo cd %CD%
            echo npm install
            echo npm audit fix
            echo If issues persist, you may need to run: npm cache clean --force
        )
    ) else (
        echo package.json not found in react-app directory. Skipping npm package installation.
    )
    cd ..
) else (
    echo react-app directory not found. Skipping npm package installation.
)

echo AgentChef installation completed.
endlocal
