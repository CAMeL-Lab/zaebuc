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
llm=gpt-5
dataset=zaebuc-w2
split=test

if [[ $data_lang == "ar" ]]; then
    input_file=/scratch/ba63/zaebuc-lrec-2026/public-release/written/$dataset/$data_lang/gec/$split.raw.tok
else
    input_file=/scratch/ba63/zaebuc-lrec-2026/public-release/written/$dataset/$data_lang/gec/$split.raw
fi


# removing the header from the input file
input_tmp=$(mktemp)
tail -n +2 "$input_file" > "$input_tmp"


python run_gpt.py \
    --model $llm \
    --prompt_lang en \
    --data_lang $data_lang \
    --examples /home/ba63/zaebuc-lrec-2026/gec/llms/4-shot-examples-${data_lang}.json \
    --input_data $input_tmp \
    --output /scratch/ba63/zaebuc-lrec-2026/expriments/written/gec/models/llms-outputs/$llm/$dataset/$data_lang/$split.txt

rm "$input_tmp"