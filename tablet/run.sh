#!/bin/bash

export TEST_FLAG=0
export CACHE_DIR="/root/autodl-tmp/models/"
export HF_ENDPOINT='https://hf-mirror.com'

echo "Running Python scripts..."
python data_generate.py
echo "dataset_generate finish ====================================="
python baseline.py
echo "baseline finish ====================================="
python CURE.py
echo "CURE.py finish ====================================="
python tabletallinone.py
echo "tabletallinone.py finish ====================================="