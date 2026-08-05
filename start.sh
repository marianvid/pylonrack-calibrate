#!/bin/zsh
cd "$(dirname "$0")"

# Prefer an explicitly supplied shared interpreter. Marian's rack uses the
# shared conda environment below; other installations remain self-contained.
PYBIN="${PYLONRACK_PYTHON:-/opt/homebrew/anaconda3/envs/pylonrack/bin/python3}"

if [ ! -x "$PYBIN" ]; then
    if [ ! -x ".venv/bin/python3" ]; then
        echo "First run: creating virtual environment..."
        python3 -m venv .venv || exit 1
        .venv/bin/pip install -r requirements.txt -q || exit 1
        echo "Dependencies installed."
    fi
    PYBIN=".venv/bin/python3"
fi

exec "$PYBIN" server.py
