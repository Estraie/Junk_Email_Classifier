#!/bin/bash

# 定义要循环运行的命令
command_base="python test_en.py --model"
models=("svc" "logi" "tree" "forest")  # 添加其他模型
tokenizers=("nltk")  # 添加其他分词器

# 循环运行命令
for model in "${models[@]}"; do
  for tokenizer in "${tokenizers[@]}"; do
    # 构建完整的命令
    command="$command_base $model --tokenizer $tokenizer"
    
    # 打印并执行命令
    echo "Running command: $command"
    $command
  done
done