@echo off
echo Starting OARC Commander Llama-3 Fine-tuning Process
echo ===================================================

:: Activate your Python environment if you're using one
:: Uncomment and modify the next line if needed
:: call C:\path\to\your\environment\Scripts\activate.bat

:: Set the path to your Python script
set PYTHON_SCRIPT=D:\CodingGit_StorageHDD\Ollama_Custom_Mods\Agent_Chef\cutlery\unsloth\ollama_llama3_finetuning.py

:: Run the Python script
python "%PYTHON_SCRIPT%"

:: Check if the script ran successfully
if %ERRORLEVEL% EQU 0 (
    echo ===================================================
    echo Fine-tuning process completed successfully!
) else (
    echo ===================================================
    echo An error occurred during the fine-tuning process.
    echo Please check the output above for more details.
)

:: Deactivate the Python environment if you activated one
:: Uncomment the next line if needed
:: call deactivate

echo.
pause