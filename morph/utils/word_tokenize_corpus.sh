
dir_public=/scratch/ba63/zaebuc-lrec-2026/public-release
dir_exp=/scratch/ba63/zaebuc-lrec-2026/expriments/

python word_tokenize_corpus.py \
    --input_transcriptions ${dir_public}/spoken/zaebuc-s/transcriptions/text \
    --manual_morph_annotations ${dir_public}/spoken/zaebuc-s/morph/all.morph.tsv \
    --output ${dir_exp}/spoken/morph/data/all.transcriptions.tok

