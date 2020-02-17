#!/bin/bash
set -e
. .env/bin/activate
python src/scripts/predict.py "$@"
