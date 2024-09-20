@echo off
setlocal enabledelayedexpansion

set CONDA_PATH=%USERPROFILE%\miniconda3\Scripts\conda.exe
set CONDA_ACTIVATE=%USERPROFILE%\miniconda3\Scripts\activate.bat

echo Starting AgentChef runner script

:: Set Ollama environment variables
set OLLAMA_NUM_PARALLEL=2
set OLLAMA_MAX_LOADED_MODELS=2
set OLLAMA_FLASH_ATTENTION=1

:: Function to check if Ollama is running
:is_ollama_running
tasklist /FI "IMAGENAME eq ollama.exe" 2>NUL | find /I /N "ollama.exe">NUL
if "%ERRORLEVEL%"=="0" (
    echo Ollama is already running.
) else (
    echo Starting Ollama service...
    start "" ollama.exe
    timeout /t 2 >NUL
)

:: Ensure port 3000 is available
echo Ensuring port 3000 is available for React app...
for /f "tokens=5" %%a in ('netstat -aon ^| find ":3000"') do (
    taskkill /F /PID %%a
)

:: Ensure port 5000 is available
echo Ensuring port 5000 is available for Flask app...
for /f "tokens=5" %%a in ('netstat -aon ^| find ":5000"') do (
    taskkill /F /PID %%a
)

:: Start the Flask app in the background
call %CONDA_ACTIVATE% AgentChef
start /B "" python app.py

:: Start the React app
echo Starting React app...
cd react-app
set HOST=
start /B "" npm start

echo AgentChef runner script completed.
pause
