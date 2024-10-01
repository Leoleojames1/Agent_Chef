@echo off

set OLLAMA_NUM_PARALLEL=2
set OLLAMA_MAX_LOADED_MODELS=2
set OLLAMA_FLASH_ATTENTION=1
:: Activate Conda environment
call C:\Users\%USERNAME%\miniconda3\Scripts\activate.bat C:\Users\%USERNAME%\miniconda3\envs\raglocal

wt --maximized -p "Agent Chef" ollama serve; split-pane -d "." python app.py ; split-pane -d "./react-app" "npm.cmd" start

@REM :: Start LLaMA server
@REM start cmd.exe /c "ollama serve"
@REM :: Wait for 1 second to let the server start
@REM ping localhost -n 2 >nul
@REM :: Activate Conda environment
@REM call C:\Users\%USERNAME%\miniconda3\Scripts\activate.bat C:\Users\%USERNAME%\miniconda3\envs\raglocal
@REM set OLLAMA_NUM_PARALLEL=2
@REM set OLLAMA_MAX_LOADED_MODELS=2
@REM set OLLAMA_FLASH_ATTENTION=1
@REM :: Run Flask app
@REM start cmd.exe /k "python app.py"
@REM :: Run React development server (assuming it's in a directory named 'react-app')
@REM cd react-app
@REM start cmd.exe /k "npm start"