#!/bin/bash
set -e
if [ ! -d ".env" ]; then
    virtualenv .env
fi
. .env/bin/activate
pip install -U pip
pip install -r pip-requirements.txt
python src/scripts/predict.py
