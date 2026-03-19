
for dataset in zaebuc-w1 zaebuc-w2
do
      for split in train dev test
      do
      
      src=/scratch/ba63/zaebuc-lrec-2026/public-release/written/$dataset/ar/gec/$split.raw.tok
      tgt=/scratch/ba63/zaebuc-lrec-2026/public-release/written/$dataset/ar/gec/$split.cor.tok
      
      # removing the headers
      tmp_src=$(mktemp)
      tmp_tgt=$(mktemp)
      tail -n +2 "$src" > "$tmp_src"
      tail -n +2 "$tgt" > "$tmp_tgt"

      align_file=/scratch/ba63/zaebuc-lrec-2026/expriments/written/gec/data/$dataset/ar/alignment/$split.txt
      output=/scratch/ba63/zaebuc-lrec-2026/public-release/written/$dataset/ar/gec/$split.m2.check

      python create_m2_file.py \
            --src $tmp_src \
            --tgt $tmp_tgt \
            --align $align_file \
            --output $output

      rm "$tmp_src"
      rm "$tmp_tgt"
      done
done
