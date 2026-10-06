@echo off
rem Windows: double-click to start Reveal Cam.
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" start.py
  goto :eof
)
for %%v in (3.12 3.11 3.13) do (
  py -%%v -c "" >nul 2>&1 && (
    py -%%v start.py
    goto :eof
  )
)
python -c "import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)" >nul 2>&1 && (
  python start.py
  goto :eof
)
echo Reveal Cam needs Python 3.11 or newer.
echo Install it from https://www.python.org/downloads/ and tick "Add python.exe to PATH",
echo then double-click Reveal Cam again.
pause
