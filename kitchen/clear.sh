#!/bin/bash

# 获取当前工作目录的basename
current_dir=$(basename "$(pwd)")

# 检查当前目录是否为kitchen
if [ "$current_dir" != "kitchen" ]; then
    echo "当前不在 kitchen 文件夹中，脚本退出。"
    exit 1
fi

rm cache/encode/*
rm cache/encode_cls_output/*
rm cache/data*

rm content/task_data_llama/*

rm dataset_generate/task/task_action2.json
rm dataset_generate/task_data2/* -r

rm img/*

rm introplan/introplan_result_with_confidence.pkl
rm introplan/introplan_success_rate_conditioned_on_confidence.pkl

rm pickle/*