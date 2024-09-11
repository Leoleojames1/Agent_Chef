@echo off
setlocal enabledelayedexpansion

REM Activate the Conda environment
call %USERPROFILE%\miniconda3\Scripts\activate.bat AgentChef

REM Freeze Python requirements
echo Freezing Python requirements...
pip freeze > requirements.txt

echo Python requirements frozen to requirements.txt

REM Freeze npm requirements
echo Freezing npm requirements...
if exist "react-app" (
    cd react-app
    if exist "package.json" (
        npm list --depth=0 > npm-requirements.txt
        echo npm requirements frozen to react-app\npm-requirements.txt
    ) else (
        echo package.json not found in react-app directory
    )
    cd ..
) else (
    echo react-app directory not found
)

echo Freezing process completed.

endlocal