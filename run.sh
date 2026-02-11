#!/usr/bin/bash

# python3 kernel_sample.py --cmd="python3 bert_medium.py" --name="bert" 
# python3 kernel_sample.py --cmd="python3 resnet50.py" --name="resnet50"

# CUDA_INJECTION64_PATH=/fast_data/echung67/nvbit_release/tools/main/main.so \
#   CUDA_VISIBLE_DEVICES=0 \
#   TRACE_PATH=./bert/ \
#   DEBUG_TRACE=0 \
#   OVERWRITE=1 \
#   SAMPLED_KERNEL_INFO=./bert_sampled_kernels_info_small.txt \
#   python3 bert_medium.py

# CUDA_INJECTION64_PATH=/fast_data/echung67/nvbit_release/tools/main/main.so \
#   CUDA_VISIBLE_DEVICES=1 \
#   TRACE_PATH=./resnet50/ \
#   DEBUG_TRACE=0 \
#   OVERWRITE=1 \
#   SAMPLED_KERNEL_INFO=./resnet_sampled_kernels_info_small.txt \
#   python3 resnet50.py

CUDA_INJECTION64_PATH=/fast_data/echung67/nvbit_release/tools/main/main.so \
  CUDA_VISIBLE_DEVICES=0 \
  COMPRESSOR_PATH=/fast_data/echung67/nvbit_release/tools/main/compress \
  TRACE_PATH=./temp/ \
  DEBUG_TRACE=1 \
  OVERWRITE=1 \
  ./test-apps/vectoradd/vectoradd