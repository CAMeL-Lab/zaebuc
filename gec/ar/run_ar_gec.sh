#!/usr/bin/env bash
#SBATCH -p nvidia
# SBATCH -q nlp
# use gpus
#SBATCH --gres=gpu:v100:1
# memory
#SBATCH --mem=200GB
# Walltime format hh:mm:ss
#SBATCH --time=40:00:00
# Output and error files
#SBATCH -o job.%J.out
#SBATCH -e job.%J.err

# make sure to run conda activate gec

# Zaebuc-1 inference
for split in dev test
do
    python run_ar_gec.py \
        --input_file /scratch/ba63/zaebuc-lrec-2026/public-release/written/zaebuc-w1/ar/gec/$split.raw.tok \
        --ged_model CAMeL-Lab/camelbert-msa-zaebuc-ged-13 \
        --gec_model CAMeL-Lab/arabart-zaebuc-gec-ged-13 \
        --output_file /scratch/ba63/zaebuc-lrec-2026/expriments/written/gec/models/seq2seq/zaebuc-1/ar/$split.txt
done


# Zaebuc-2 inference
for split in train dev test
do
    python run_ar_gec.py \
        --input_file /scratch/ba63/zaebuc-lrec-2026/public-release/written/zaebuc-w2/ar/gec/$split.raw.tok \
        --ged_model CAMeL-Lab/camelbert-msa-zaebuc-ged-13 \
        --gec_model CAMeL-Lab/arabart-zaebuc-gec-ged-13 \
        --output_file /scratch/ba63/zaebuc-lrec-2026/expriments/written/gec/models/seq2seq/zaebuc-2/ar/$split.txt
done

