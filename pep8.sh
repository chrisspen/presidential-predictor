#!/bin/bash
set -e
. .env/bin/activate
pylint --rcfile=pylint.rc src/scripts/predict.py
