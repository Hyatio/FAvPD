#!/usr/bin/env bash

DEVICES=$1
TASK_NAME="FewNERD"
DATA_DIR="../DATA/dataset/FewNERD"
MODEL_TYPE="bert"
MODEL_PATH="../DATA/PLMs/FewNERD-K-030-10/checkpoint-3200"
OUTPUT_DIR="../DATA/output_tuning/K-3200"
EVAL_SET="test"
WANDB_PROJECT=""
WANDB_NAME=""

SEED=102
TRAIN_EPOCH=3
TRAIN_BATCH_SIZE=16
EVAL_BATCH_SIZE=16
GRADIENT_ACCUMULATION_STEPS=1
LEARNING_RATE=5e-5
ADAM_BETA_1=0.9
ADAM_BETA_2=0.999
ADAM_EPSILON=1e-8
WARMUP_PROPORTION=0
WEIGHT_DECAY=0
MAX_GRAD_NORM=1.0
MAX_SEQ_LENGTH=128
LOGGING_STEPS=2000
EARLYSTOP_PATIENCE=100

echo "Start Running"

CUDA_VISIBLE_DEVICES=${DEVICES} python tuning/run_ner.py \
    --task_name=${TASK_NAME} \
    --data_dir=${DATA_DIR} \
    --model_type=${MODEL_TYPE} \
    --model_path=${MODEL_PATH} \
    --output_dir=${OUTPUT_DIR} \
    --eval_set=${EVAL_SET} \
    --wandb_project=${WANDB_PROJECT} \
    --wandb_name=${WANDB_NAME} \
    --seed=${SEED} \
    --train_epoch=${TRAIN_EPOCH} \
    --train_batch_size=${TRAIN_BATCH_SIZE} \
    --eval_batch_size=${EVAL_BATCH_SIZE} \
    --gradient_accumulation_steps=${GRADIENT_ACCUMULATION_STEPS} \
    --learning_rate=${LEARNING_RATE} \
    --adam_beta_1=${ADAM_BETA_1} \
    --adam_beta_2=${ADAM_BETA_2} \
    --adam_epsilon=${ADAM_EPSILON} \
    --warmup_proportion=${WARMUP_PROPORTION} \
    --weight_decay=${WEIGHT_DECAY} \
    --max_grad_norm=${MAX_GRAD_NORM} \
    --max_seq_length=${MAX_SEQ_LENGTH} \
    --logging_steps=${LOGGING_STEPS} \
    --earlystop_patience=${EARLYSTOP_PATIENCE} \
    --do_lower_case
    # --do_lower_case \
    # --save_checkpoint \

rm -rf ../DATA/output_tuning/T