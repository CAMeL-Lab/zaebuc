
dir_public=/scratch/ba63/zaebuc-lrec-2026/public-release
dir_exp=/scratch/ba63/zaebuc-lrec-2026/expriments/

for dataset in zaebuc-w1 zaebuc-w2; do
  python run_camelTools.py \
      --input_data ${dir_public}/written/${dataset}/ar/morph/all.cor.tok \
      --output ${dir_exp}/written/morph/models/camelTools/$dataset/all.morph.tsv
done

for dataset in zaebuc-w1 zaebuc-w2; do
  for split in dev test; do
    python run_camelTools.py \
        --input_data ${dir_public}/written/${dataset}/ar/morph/${split}.cor.tok \
        --output ${dir_exp}/written/morph/models/camelTools/$dataset/${split}.morph.tsv
  done
done
