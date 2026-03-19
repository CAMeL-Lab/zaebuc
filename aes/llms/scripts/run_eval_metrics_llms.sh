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


for llm in gpt-4o gpt-5 jais Fanar-C-1-8.7B
do
    for dataset in zaebuc-w1 zaebuc-w2
    do
        for lang in ar en
        do
            for split in dev test
            do
                gold_file=/scratch/ba63/zaebuc-lrec-2026/public-release/written/$dataset/$lang/aes/$split.cefr
                tmp_gold=$(mktemp)
                # removing the header from the gold file
                tail -n +2 "$gold_file" > "$tmp_gold"

                python evaluate.py \
                    --pred_file /scratch/ba63/zaebuc-lrec-2026/expriments/written/aes/llms-outputs/$llm/$dataset/$lang/$split.txt \
                    --gold_file "$tmp_gold"

                rm "$tmp_gold"
            done
        done
    done
done