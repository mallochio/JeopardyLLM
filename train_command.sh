#!/bin/bash

# Standard training command
python src/main.py \
  --data-dir /Users/sid/Projects/code/JeopardyLLM/data \
  --output-dir /Users/sid/Projects/code/JeopardyLLM/models \
  --model-name google/gemma-3-1b-it \
  --iters 1000 \
  --batch-size 4 \
  --learning-rate 1e-4

# Advanced training command with all options
# python src/main.py \
#   --data-dir /Users/sid/Projects/code/JeopardyLLM/data \
#   --output-dir /Users/sid/Projects/code/JeopardyLLM/models \
#   --model-name google/gemma-3-1b-it \
#   --fine-tune-type lora \
#   --optimizer adamw \
#   --iters 1000 \
#   --batch-size 4 \
#   --learning-rate 1e-4 \
#   --mask-prompt \
#   --num-layers 16 \
#   --steps-per-report 10 \
#   --steps-per-eval 100 \
#   --save-every 200 \
#   --max-seq-length 2048 \
#   --grad-checkpoint \
#   --seed 42 \
#   --run-test \
#   --use-rag \
#   --build-rag-index

# Memory-optimized training for low-RAM systems
# python src/main.py \
#   --data-dir /Users/sid/Projects/code/JeopardyLLM/data \
#   --output-dir /Users/sid/Projects/code/JeopardyLLM/models \
#   --model-name google/gemma-3-1b-it \
#   --batch-size 1 \
#   --iters 1500 \
#   --learning-rate 5e-5 \
#   --grad-checkpoint \
#   --num-layers 8 \
#   --max-seq-length 1024
