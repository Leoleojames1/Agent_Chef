@echo off
:: Start LLaMA server
start cmd.exe /c "ollama serve"
:: Wait for 1 second to let the server start
ping localhost -n 2 >nul
:: Activate Conda environment
call C:\Users\%USERNAME%\miniconda3\Scripts\activate.bat C:\Users\%USERNAME%\miniconda3\envs\raglocal
set OLLAMA_NUM_PARALLEL=2
set OLLAMA_MAX_LOADED_MODELS=2
set OLLAMA_FLASH_ATTENTION=1
:: Run Flask app
start cmd.exe /k "python app.py"
:: Run React development server (assuming it's in a directory named 'react-app')
cd react-app
start cmd.exe /k "npm start"