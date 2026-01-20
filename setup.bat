@echo off
REM AI Career Recommendation System - Quick Start Script for Windows

echo ==================================================
echo  AI Career Recommendation System - Setup
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo  Python is not installed. Please install Python 3.8 or higher.
    pause
    exit /b 1
)

echo  Python found
python --version
echo.

REM Check if CSV exists
if not exist "AI_Career_Recommendation_8000.csv" (
    echo   Warning: AI_Career_Recommendation_8000.csv not found
    echo    Please ensure your dataset is in the project root folder.
    echo.
)

REM Create virtual environment
echo  Creating virtual environment...
python -m venv venv
if errorlevel 1 (
    echo  Failed to create virtual environment
    pause
    exit /b 1
)
echo  Virtual environment created
echo.

REM Activate virtual environment
echo  Activating virtual environment...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo  Failed to activate virtual environment
    pause
    exit /b 1
)
echo  Virtual environment activated
echo.

REM Install requirements
echo  Installing dependencies (this may take a few minutes)...
python -m pip install --upgrade pip
pip install -r requirements.txt
if errorlevel 1 (
    echo  Failed to install dependencies
    pause
    exit /b 1
)
echo  All dependencies installed
echo.

REM Train model
echo ==================================================
echo  Training Ensemble Model
echo.
if exist "AI_Career_Recommendation_8000.csv" (
    python train_model.py
    if errorlevel 1 (
        echo  Model training failed
        pause
        exit /b 1
    )
    echo.
    echo  Model training complete!
) else (
    echo   Skipping model training - dataset not found
    echo    Place AI_Career_Recommendation_8000.csv in the project folder
    echo    Then run: python train_model.py
)

echo.
echo ==================================================
echo Setup Complete!
echo ==================================================
echo.
echo To run the application:
echo   1. Make sure virtual environment is active
echo   2. Run: streamlit run app.py
echo.
echo To activate virtual environment later:
echo   venv\Scripts\activate
echo.
echo ==================================================
pause
