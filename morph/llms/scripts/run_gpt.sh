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


data_lang=ar
llm=gpt-5
dataset=zaebuc-w2
split=test

# Note that for csw we don't provide few shot examples

# written
python run_gpt.py \
    --model $llm \
    --prompt_lang en \
    --data_lang $data_lang \
    --examples /home/ba63/zaebuc-lrec-2026/morph/llms/few-shot-examples/few-shot-examples-${data_lang}.json \
    --input_data /scratch/ba63/zaebuc-lrec-2026/expriments/written/morph/data/$dataset/$data_lang/${split}.txt \
    --output /scratch/ba63/zaebuc-lrec-2026/expriments/written/morph/llms-outputs/$llm/$dataset/$data_lang/$split.txt

# spoken
# python run_gpt.py \
#     --model $llm \
#     --prompt_lang en \
#     --data_lang $data_lang \
#     --examples /home/ba63/zaebuc-lrec-2026/morph/llms/few-shot-examples/few-shot-examples-${data_lang}.json \
#     --input_data /scratch/ba63/zaebuc-lrec-2026/expriments/spoken/morph/data/all_recordings_transcriptions_tok_lang_csw.no_ids \
#     --output /scratch/ba63/zaebuc-lrec-2026/expriments/spoken/morph/llms-outputs/$llm/csw.no_ids
