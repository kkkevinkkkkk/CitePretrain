#!/bin/bash


CONFIG_PATH="configures/ft_Qwen2.5-7B-ActiveIndex-Instruct.yml"

export LOCAL_DIR="${LOCAL_DIR:-/usr/xtmp/yh386/CitePretrain}"

CONFIG_NAME=$(basename "$CONFIG_PATH")
if [[ $CONFIG_NAME == pretrain* ]]; then
    file_name="pretrain.py"
else
    file_name="fine_tune.py"
fi



num_gpus=$(echo $CUDA_VISIBLE_DEVICES | awk -F',' '{print NF}')

echo "Using idle port: $IDLE_PORT"
echo $CONFIG_PATH
echo $file_name


echo "using $num_gpus GPUs"
torchrun  --standalone --nnodes 1 --nproc_per_node=$num_gpus  "$file_name" --config_path "$CONFIG_PATH"
