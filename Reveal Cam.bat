@echo off
rem Windows: double-click to start Reveal Cam.
cd /d "%~dp0"
set "OKPY=import sys; sys.exit(0 if (3, 11) <= sys.version_info[:2] <= (3, 14) else 1)"
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" -c "%OKPY%" >nul 2>&1 && (
    ".venv\Scripts\python.exe" start.py
    goto :eof
  )
)
for %%v in (3.13 3.12 3.14 3.11) do (
  py -%%v -c "" >nul 2>&1 && (
    py -%%v start.py
    goto :eof
  )
)
python -c "%OKPY%" >nul 2>&1 && (
  python start.py
  goto :eof
)
echo Reveal Cam needs Python 3.11, 3.12, 3.13 or 3.14.
echo Install Python 3.13 from https://www.python.org/downloads/ and tick "Add python.exe to PATH",
echo then double-click Reveal Cam again.
pause
