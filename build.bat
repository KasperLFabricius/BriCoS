@echo off
REM ============================================================
REM  Build BriCoS as a standalone Windows app (one folder).
REM  Double-click this file, or run it from a terminal.
REM  Full instructions: see PACKAGING.md
REM ============================================================
cd /d "%~dp0"

echo.
echo [1/3] Installing the app's dependencies...
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
if errorlevel 1 goto :error

echo.
echo [2/3] Installing the packaging tool (PyInstaller)...
python -m pip install pyinstaller
if errorlevel 1 goto :error

echo.
echo [3/3] Building (this takes a few minutes the first time)...
python -m PyInstaller --noconfirm --clean bricos.spec
if errorlevel 1 goto :error

echo.
echo ============================================================
echo  DONE. Your app is in:  dist\BriCoS\
echo  Run it by double-clicking:  dist\BriCoS\BriCoS.exe
echo  To share it, zip the whole dist\BriCoS folder.
echo ============================================================
pause
exit /b 0

:error
echo.
echo *** BUILD FAILED. Scroll up for the error, or see PACKAGING.md. ***
pause
exit /b 1
