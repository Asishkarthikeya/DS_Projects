@echo off
REM Code Crawler CLI - Quick Setup Script for Windows

echo Code Crawler CLI - Quick Setup
echo ==================================
echo.

REM Check Python
echo Checking Python version...
python --version
if %errorlevel% neq 0 (
    echo Error: Python not found
    echo Please install Python 3.9 or higher
    pause
    exit /b 1
)
echo.

REM Install the CLI
echo Installing Code Crawler CLI...
pip install -e .
if %errorlevel% neq 0 (
    echo Error: Installation failed
    pause
    exit /b 1
)
echo Installation complete
echo.

REM Check API key
if "%GOOGLE_API_KEY%"=="" if "%GROQ_API_KEY%"=="" (
    echo Warning: No API key found!
    echo.
    echo Please set your API key:
    echo   For Gemini: set GOOGLE_API_KEY=your-key-here
    echo   For Groq:   set GROQ_API_KEY=your-key-here
    echo.
    echo Or configure it with:
    echo   code-crawler config set GOOGLE_API_KEY your-key-here
    echo.
) else (
    echo API key found
    echo.
)

REM Verify installation
echo Verifying installation...
code-crawler version
if %errorlevel% neq 0 (
    echo Error: Installation verification failed
    pause
    exit /b 1
)
echo.

echo Setup complete!
echo.
echo Next steps:
echo   1. Index a codebase: code-crawler index ./your-project
echo   2. Start chatting:   code-crawler chat
echo   3. Get help:         code-crawler --help
echo.
pause
