#!/bin/bash
set -e
if [ -d ".env" ]; then
    rm -Rf .env
fi
virtualenv -p python3.7 .env
. .env/bin/activate
pip install -U pip setuptools
pip install -r pip-requirements.txt
