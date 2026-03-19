#!/bin/bash
# Set number of tasks to run
# SBATCH -q nlp
# SBATCH --ntasks=1
# Set number of cores per task (default is 1)
#SBATCH --cpus-per-task=1
# Walltime format hh:mm:ss
#SBATCH --time=47:59:00
# Output and error files
#SBATCH -o job.%J.out
#SBATCH -e job.%J.err


data_lang=en
llm=Fanar-C-1-8.7B
dataset=zaebuc-w2
split=test

if [[ $data_lang == "ar" ]]; then
    input_file=$split.raw.tok
else
    input_file=$split.raw
fi

python run_fanar.py \
    --model $llm \
    --prompt_lang ar \
    --data_lang $data_lang \
    --examples /home/ba63/zaebuc-lrec-2026/aes/llms/few-shot-examples/few-shot-examples-${data_lang}.json \
    --input_data /scratch/ba63/zaebuc-lrec-2026/public-release/written/$dataset/$data_lang/aes/$input_file \
    --output /scratch/ba63/zaebuc-lrec-2026/expriments/written/aes/llms-outputs/$llm/$dataset/$data_lang/$split.txt
   