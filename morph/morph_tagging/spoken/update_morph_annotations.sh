dir=/Users/injy.hamed/Documents/Projects/Zaebuc/camera_ready/final/release
#dir=/scratch/ba63/zaebuc-lrec-2026

python update_morph_annotations.py \
    --input_transcriptions ${dir}/experiments/spoken/morph/data/text_tok.all.txt \
    --prev_morph_annotations ${dir}/public/spoken/morph/previous_morph_annotations \
    --output ${dir}/experiments/spoken/morph/models/camelTools+stanza/all.tsv

  
