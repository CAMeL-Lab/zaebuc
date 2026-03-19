for dataset in zaebuc-w1 zaebuc-w2
do
    for split in train dev test
    do
        src=/scratch/ba63/zaebuc-lrec-2026/public-release/written/$dataset/en/gec/$split.raw.tok
        tgt=/scratch/ba63/zaebuc-lrec-2026/public-release/written/$dataset/en/gec/$split.cor.tok

        # removing the headers
        tmp_src=$(mktemp)
        tmp_tgt=$(mktemp)
        tail -n +2 "$src" > "$tmp_src"
        tail -n +2 "$tgt" > "$tmp_tgt"

        out=/scratch/ba63/zaebuc-lrec-2026/public-release/written/$dataset/en/gec/$split.m2

        errant_parallel -orig $tmp_src -cor $tmp_tgt -out $out

      rm "$tmp_src"
      rm "$tmp_tgt"
    done

done