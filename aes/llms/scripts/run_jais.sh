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
    input_file=/scratch/ba63/zaebuc-lrec-2026/public-release/written/$dataset/$data_lang/aes/$split.raw.tok
else
    input_file=/scratch/ba63/zaebuc-lrec-2026/public-release/written/$dataset/$data_lang/aes/$split.raw
fi

# removing the header from the input file
input_tmp=$(mktemp)
tail -n +2 "$input_file" > "$input_tmp"


python run_jais.py \
    --prompt_lang ar \
    --data_lang $data_lang \
    --examples /home/ba63/zaebuc-lrec-2026/aes/llms/few-shot-examples/few-shot-examples-${data_lang}.json \
    --input_data $input_tmp \
    --output /scratch/ba63/zaebuc-lrec-2026/expriments/written/aes/llms-outputs/$llm/$dataset/$data_lang/$split.txt

rm "$input_tmp"