#!/usr/bin/env bash
# Smart Modeling Agent — Linux/macOS başlatıcısı
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

PYTHON_BIN="${PYTHON:-python3}"

if [ ! -d ".venv" ]; then
  echo "🐍 Sanal ortam oluşturuluyor..."
  "$PYTHON_BIN" -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate

echo "📦 Bağımlılıklar kontrol ediliyor..."
pip install --upgrade pip >/dev/null
pip install -r requirements.txt

echo "🚀 Smart Modeling Agent başlatılıyor..."
streamlit run app.py
