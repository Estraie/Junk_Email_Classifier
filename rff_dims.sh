#!/bin/bash

# 定义要循环运行的命令
command_base="python test_en.py --model"
models=("svc")  # 添加其他模型
dims=(512 1024 2048 4096 8192 16384)  # 添加其他分词器

# 循环运行命令
for model in "${models[@]}"; do
  for dim in "${dims[@]}"; do
    # 构建完整的命令
    command="$command_base $model --rff True --dim $dim"
    
    # 打印并执行命令
    echo "Running command: $command"
    $command
  done
done