#!/bin/bash
#SBATCH -p nvidia
#SBATCH -q nlp
# use gpus
#SBATCH --gres=gpu:a100:1
#SBATCH --nodelist=cn010
# Walltime format hh:mm:ss
#SBATCH --time=47:59:00
# Output and error files
#SBATCH --mem=150GB
#SBATCH -o job.%J.out
#SBATCH -e job.%J.err

data_lang=en
llm=jais
dataset=zaebuc-w2
split=test

if [[ $data_lang == "ar" ]]; then
    input_file=$split.raw.tok
else
    input_file=$split.raw
fi

python run_jais.py \
    --prompt_lang ar \
    --data_lang $data_lang \
    --examples /home/ba63/zaebuc-lrec-2026/aes/llms/few-shot-examples/few-shot-examples-${data_lang}.json \
    --input_data /scratch/ba63/zaebuc-lrec-2026/public-release/written/$dataset/$data_lang/aes/$input_file \
    --output /scratch/ba63/zaebuc-lrec-2026/expriments/written/aes/llms-outputs/$llm/$dataset/$data_lang/$split.txt