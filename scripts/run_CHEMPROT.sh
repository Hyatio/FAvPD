#!/usr/bin/env bash

DEVICES=$1
TASK_NAME="CHEMPROT"
DATA_DIR="../DATA/dataset/CHEMPROT"
MODEL_TYPE="roberta"
MODEL_PATH="../DATA/PLMs/roberta-base"
OUTPUT_DIR="../DATA/output_tuning/CHEMPROT-3"
EVAL_SET="test"
WANDB_PROJECT=""
WANDB_NAME=""

SEED=121
TRAIN_EPOCH=8
TRAIN_BATCH_SIZE=16
EVAL_BATCH_SIZE=16
GRADIENT_ACCUMULATION_STEPS=1
LEARNING_RATE=1e-5
ADAM_BETA_1=0.9
ADAM_BETA_2=0.98
ADAM_EPSILON=1e-6
WARMUP_PROPORTION=0.1
WEIGHT_DECAY=0.01
MAX_GRAD_NORM=0.0
MAX_SEQ_LENGTH=128
LOGGING_STEPS=100
EARLYSTOP_PATIENCE=100

echo "Start Running"

CUDA_VISIBLE_DEVICES=${DEVICES} python tuning/run_re.py \
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
    --add_entity_embedding
    # --add_entity_embedding \
    # --do_lower_case \
    # --save_checkpoint \

rm -rf ../DATA/output_tuning/T
