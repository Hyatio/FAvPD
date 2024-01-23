#!/usr/bin/env bash

DEVICES=$1
PROMPT_TYPE="KLPrompt"
TASK_NAME="TACRED"
DATA_DIR="../DATA/dataset/ReTACRED"
MODEL_TYPE="bert"
MODEL_PATH="../DATA/PLMs/bert-base-uncased"
OUTPUT_DIR="../DATA/PLMs/ReT-BBU-KLP"
EVAL_SET="test"
WANDB_PROJECT=""
WANDB_NAME=""

SEED=200
TRAIN_EPOCH=10
TRAIN_BATCH_SIZE=16
EVAL_BATCH_SIZE=16
GRADIENT_ACCUMULATION_STEPS=1
LEARNING_RATE=1e-5
ADAM_BETA_1=0.9
ADAM_BETA_2=0.999
ADAM_EPSILON=1e-8
WARMUP_PROPORTION=0.1
WEIGHT_DECAY=0.01
MAX_GRAD_NORM=0.0
MAX_SEQ_LENGTH=128
LOGGING_STEPS=200
EARLYSTOP_PATIENCE=15
MLM_PROBABILITY=0.15
K_MLM_PROBABILITY=0.3
T_LOSS_WEIGHT=0.2
K_LOSS_WEIGHT=0.4
L_LOSS_WEIGHT=0.4

echo "Start Running"

CUDA_VISIBLE_DEVICES=${DEVICES} python adapting/run_adapting.py \
    --prompt_type=${PROMPT_TYPE} \
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
    --mlm_probability=${MLM_PROBABILITY} \
    --k_mlm_probability=${K_MLM_PROBABILITY} \
    --t_loss_weight=${T_LOSS_WEIGHT} \
    --k_loss_weight=${K_LOSS_WEIGHT} \
    --l_loss_weight=${L_LOSS_WEIGHT} \
    --dynamic_mask \
    --add_entity_embedding \
    --do_lower_case \
    --save_checkpoint
    # --dynamic_mask \
    # --add_entity_embedding \
    # --do_lower_case \
    # --save_checkpoint \

rm -rf ../DATA/PLMs/T
