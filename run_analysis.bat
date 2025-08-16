@echo off
echo ======================================
echo  NLTK Bigram Analyzer - Windows
echo ======================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python from https://python.org
    pause
    exit /b 1
)

echo Installing required packages...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo ERROR: Failed to install packages
    pause
    exit /b 1
)

echo.
echo Downloading NLTK data...
python -c "import nltk; nltk.download(['punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger', 'omw-1.4'], quiet=True); print('NLTK data downloaded successfully')"

echo.
echo ======================================
echo  Running Quick Analysis
echo ======================================
echo.

python examples\quick_start.py

echo.
echo ======================================
echo  Analysis Complete!
echo ======================================
echo.
echo Results saved to: quick_analysis_results.csv
echo.
echo To analyze your own text file:
echo   python examples\quick_start.py --input your_file.txt
echo.
pause
