@echo off
REM Smart Modeling Agent - Windows baslatici
setlocal
cd /d "%~dp0"

if not exist .venv (
  echo Sanal ortam olusturuluyor...
  python -m venv .venv
  if errorlevel 1 (
    echo Python bulunamadi. Lutfen Python 3.10+ yukleyin: https://www.python.org/downloads/
    pause
    exit /b 1
  )
)

call .venv\Scripts\activate.bat

echo Bagimliliklar kontrol ediliyor...
python -m pip install --upgrade pip >nul
pip install -r requirements.txt
if errorlevel 1 (
  echo Bagimliliklar yuklenemedi.
  pause
  exit /b 1
)

echo Smart Modeling Agent baslatiliyor...
streamlit run app.py

endlocal
