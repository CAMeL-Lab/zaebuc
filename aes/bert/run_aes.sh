#!/bin/bash
#SBATCH -p nvidia
# use gpus
#SBATCH --gres=gpu:v100:1
# Walltime format hh:mm:ss
#SBATCH --time=47:59:00
# Output and error files
#SBATCH -o job.%J.out
#SBATCH -e job.%J.err

nvidia-smi
module purge


####################################
# AES FINE-TUNING SCRIPT
####################################

# Arabic
export DATA_DIR=/scratch/ba63/zaebuc-lrec-2026/expriments/written/aes/bert/data/mix/ar
export BERT_MODEL=/scratch/ba63/BERT_models/bert-base-arabic-camelbert-msa
export OUTPUT_DIR=/scratch/ba63/zaebuc-lrec-2026/expriments/written/aes/bert/models/mix/ar

# English
# export DATA_DIR=/scratch/ba63/zaebuc-lrec-2026/expriments/written/aes/bert/data/mix/en
# export BERT_MODEL=/scratch/ba63/BERT_models/bert-base-uncased
# export OUTPUT_DIR=/scratch/ba63/zaebuc-lrec-2026/expriments/written/aes/bert/models/mix/en

export BATCH_SIZE=32
export NUM_EPOCHS=5
export SAVE_STEPS=500
export SEED=42


python aes_classification.py \
    --data_dir $DATA_DIR \
    --optim adamw_torch \
    --labels_path $DATA_DIR/labels.txt \
    --model_name_or_path $BERT_MODEL \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs $NUM_EPOCHS \
    --per_device_train_batch_size $BATCH_SIZE \
    --save_steps $SAVE_STEPS \
    --seed $SEED \
    --do_train \
    --overwrite_output_dir

