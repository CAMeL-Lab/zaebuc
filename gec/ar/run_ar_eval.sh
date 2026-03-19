#!/bin/bash
# Set number of tasks to run
#SBATCH -q nlp
# SBATCH --ntasks=1
# Set number of cores per task (default is 1)
#SBATCH --cpus-per-task=1
# Walltime format hh:mm:ss
#SBATCH --time=47:59:00
# Output and error files
#SBATCH -o job.%J.out
#SBATCH -e job.%J.err


#zaebuc-w1
for split in dev test
    do  
        printf "zaebuc-w1 $split\n"

        output=/scratch/ba63/zaebuc-lrec-2026/expriments/written/gec/models/seq2seq/zaebuc-w1/ar/$split.txt
        m2_edits=/scratch/ba63/zaebuc-lrec-2026/public-release/written/zaebuc-w1/ar/gec/$split.m2

        python evaluate.py \
            --system_output $output \
            --m2_file $m2_edits
done

#zaebuc-w2
for split in dev test
    do  
        printf "zaebuc-w2 $split\n"

        output=/scratch/ba63/zaebuc-lrec-2026/expriments/written/gec/models/seq2seq/zaebuc-w2/ar/$split.txt
        m2_edits=/scratch/ba63/zaebuc-lrec-2026/public-release/written/zaebuc-w2/ar/gec/$split.m2

        python evaluate.py \
            --system_output $output \
            --m2_file $m2_edits

done


# LLMs
llm=gpt-5
for dataset in zaebuc-w1 zaebuc-w2
do
    for split in dev test
        do
            output=/scratch/ba63/zaebuc-lrec-2026/gec/llms-outputs/$llm/$dataset/ar/$split.txt
            m2_edits=/scratch/ba63/zaebuc-lrec-2026/public-release/written/$dataset/ar/gec/$split.m2

            python evaluate.py \
                --system_output $output \
                --m2_file $m2_edits

    done
done
