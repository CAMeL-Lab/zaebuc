#!/bin/bash
#SBATCH -p nvidia
#SBATCH -q nlp
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=120GB
#SBATCH --time=47:59:00
#SBATCH -o job.%J.out
#SBATCH -e job.%J.err


# Arabic
for dataset in zaebuc-w1 zaebuc-w2
do
    for split in train dev test
    do
        data_dir=/scratch/ba63/zaebuc-lrec-2026/public-release/written/$dataset/ar/gec

        src=$data_dir/$split.raw.tok
        tgt=$data_dir/$split.cor.tok

        # removing the headers
        tmp_src=$(mktemp)
        tmp_tgt=$(mktemp)
        tail -n +2 "$src" > "$tmp_src"
        tail -n +2 "$tgt" > "$tmp_tgt"

        output=/scratch/ba63/zaebuc-lrec-2026/expriments/written/gec/data/$dataset/ar/alignment/$split.txt

        python aligner.py --src $tmp_src \
                          --tgt $tmp_tgt \
                          --output $output
        
      rm "$tmp_src"
      rm "$tmp_tgt"

    done
done



# English
for dataset in zaebuc-w1 zaebuc-w2
do
    for split in train dev test
    do
        data_dir=/scratch/ba63/zaebuc-lrec-2026/public-release/written/$dataset/ar/en

        src=$data_dir/$split/$split.raw.tok
        tgt=$data_dir/$split/$split.cor.tok

        # removing the headers
        tmp_src=$(mktemp)
        tmp_tgt=$(mktemp)
        tail -n +2 "$src" > "$tmp_src"
        tail -n +2 "$tgt" > "$tmp_tgt"

        output=/scratch/ba63/zaebuc-lrec-2026/expriments/written/gec/data/$dataset/en/alignment/$split.txt

        python aligner.py --src $src \
                          --tgt $tgt \
                          --output $output

        rm "$tmp_src"
        rm "$tmp_tgt"

    done
done