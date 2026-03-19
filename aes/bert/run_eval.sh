#!/bin/bash
#SBATCH -p nvidia
#SBATCH -q nlp
# use gpus
#SBATCH --gres=gpu:v100:1
# Walltime format hh:mm:ss
#SBATCH --time=47:59:00
#SBATCH --mem=50GB
# Output and error files
#SBATCH -o job.%J.out
#SBATCH -e job.%J.err



####################################
# AES Eval Script
####################################

for lang in ar en
do
    for dataset in zaebuc-w1 zaebuc-w2
    do
        DATA_DIR=/scratch/ba63/zaebuc-lrec-2026/expriments/written/aes/bert/data/mix/$lang
        TEST_FILE=/scratch/ba63/zaebuc-lrec-2026/expriments/written/aes/bert/data/$dataset/$lang/dev.txt
        MODEL_DIR=/scratch/ba63/zaebuc-lrec-2026/expriments/written/aes/bert/models/mix/${lang}
        BATCH_SIZE=32
        PRED_MODE=${dataset}_dev

        python aes_classification.py \
            --data_dir $DATA_DIR \
            --test_data_file $TEST_FILE \
            --model_name_or_path $MODEL_DIR \
            --labels_path $DATA_DIR/labels.txt \
            --output_dir $MODEL_DIR \
            --per_device_eval_batch_size $BATCH_SIZE \
            --pred_mode $PRED_MODE \
            --do_predict
    done
done

