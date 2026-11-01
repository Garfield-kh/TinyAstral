#!/bin/bash
# Fine-tune a TinyAstral student with ms-swift (full fine-tuning, ViT + aligner frozen, ZeRO-2).
#
#   bash train/train.sh MODEL DATA_DIR OUT_DIR [extra swift args...]
#
#   MODEL     base model: Qwen/Qwen3.5-4B | Qwen/Qwen3.5-2B | Qwen/Qwen3.5-0.8B (hub id or local dir)
#   DATA_DIR  directory with train.jsonl and val.jsonl in the format of docs/training.md
#             (the released set: Hugging Face dataset kehong/TinyAstral-SFT; image paths are relative to DATA_DIR)
#   OUT_DIR   output directory (ms-swift adds a versioned sub-directory)
#
# Hyper-parameters reproduce the released students: lr 1e-5 cosine, warmup 0.05, weight decay 0.01, grad clip 1.0,
# 2 epochs, max_length 12288 (longer samples are dropped), per-device batch 1, global batch 32, loss on the last
# assistant turn only, bf16, flash-attention, gradient checkpointing, seed 42.
# The gradient accumulation is derived from the world size so that the global batch stays 32:
#   4B  : 2 nodes x 8 GPUs x batch 1 x GA 2      (the released 4B)
#   2B  : 1 node  x 8 GPUs x batch 1 x GA 4      (the released 2B)
#   0.8B: 1 node  x 8 GPUs x batch 1 x GA 4      (the released 0.8B)
# A 1-node 4B run (8 GPUs, GA 4) has the same global batch.
#
# Env: NPROC (GPUs per node, default 8), CUDA_VISIBLE_DEVICES, GLOBAL_BATCH (default 32), BS (per-device, default 1),
#      GA (override the derived value), multi-node: NNODES, NODE_RANK, MASTER_ADDR, MASTER_PORT (read by swift).
set -eu
MODEL=${1:?usage: train.sh MODEL DATA_DIR OUT_DIR [extra args]}
DATA=${2:?usage: train.sh MODEL DATA_DIR OUT_DIR [extra args]}
OUT=${3:?usage: train.sh MODEL DATA_DIR OUT_DIR [extra args]}
shift 3

export NPROC_PER_NODE=${NPROC:-8}
NN=${NNODES:-1}
BS=${BS:-1}
GB=${GLOBAL_BATCH:-32}
if [ -z "${GA:-}" ]; then
  W=$(( NPROC_PER_NODE * NN * BS ))
  [ $(( GB % W )) -eq 0 ] || { echo "global batch $GB is not divisible by world_size*BS=$W; set GA explicitly"; exit 1; }
  GA=$(( GB / W ))
fi
echo "world=$(( NPROC_PER_NODE * NN )) per_device_batch=$BS grad_accum=$GA -> global batch $(( NPROC_PER_NODE * NN * BS * GA ))"

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8} TOKENIZERS_PARALLELISM=false
export IMAGE_MAX_TOKEN_NUM=512 MAX_PIXELS=262144          # image budget per image (Qwen3.5-VL processor)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "$DATA"                                                 # image paths in the jsonl are relative to DATA_DIR
exec swift sft --model "$MODEL" --tuner_type full --torch_dtype bfloat16 \
  --dataset "$DATA/train.jsonl" --val_dataset "$DATA/val.jsonl" --split_dataset_ratio 0 \
  --freeze_vit true --freeze_aligner true --loss_scale last_round --max_length 12288 --truncation_strategy delete \
  --per_device_train_batch_size "$BS" --gradient_accumulation_steps "$GA" --learning_rate 1e-5 --lr_scheduler_type cosine --warmup_ratio 0.05 \
  --weight_decay 0.01 --max_grad_norm 1.0 --num_train_epochs 2 --gradient_checkpointing true --attn_impl flash_attn \
  --deepspeed zero2 --save_strategy epoch --save_total_limit 2 --eval_steps 200 --logging_steps 10 --dataloader_num_workers 8 \
  --dataset_num_proc 32 --seed 42 --output_dir "$OUT" --report_to tensorboard "$@"
