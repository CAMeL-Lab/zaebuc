dir_public=/scratch/ba63/zaebuc-lrec-2026/public-release
dir_exp=/scratch/ba63/zaebuc-lrec-2026/expriments/

python update_morph_annotations.py \
    --input_transcriptions ${dir_public}/spoken/zaebuc-s/morph/all.transcriptions.tok \
    --prev_morph_annotations ${dir_exp}/spoken/morph/data/previous_morph_annotations \
    --output ${dir_exp}/spoken/morph/models/camelTools+stanza/all.input_redacted.morph.tsv

  
