#!/bin/bash

export TEST_FLAG=0
export CACHE_DIR="/root/autodl-tmp/models/"
export HF_ENDPOINT='https://hf-mirror.com'

echo "Running Python scripts..."
python dataset_generate.py
echo "dataset_generate finish ====================================="
python baseline.py
echo "baseline finish ====================================="
python ambgious.py
echo "ambgious finish ====================================="
python uncertainty_model_train_RND2_low_gpu.py
echo "uncertainty_model_train_RND2_low_gpu.py finish ====================================="
python result_generate.py
echo "result_generate.py finish ====================================="
python introplan_result_generate.py
echo "introplan_result_generate.py finish ====================================="
python plotandexcel.py
echo "plotandexcel.py finish ====================================="