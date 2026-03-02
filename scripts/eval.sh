#!/bin/bash


CONFIG_PATH="configures/eval_repliqa.yml"
CONFIG_PATH="configures/eval_asqa.yml"

export LOCAL_DIR="${LOCAL_DIR:-/usr/xtmp/yh386/CitePretrain}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn

echo $CONFIG_PATH
python run.py --config_path "$CONFIG_PATH"