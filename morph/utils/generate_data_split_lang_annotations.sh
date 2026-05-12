

dir_public=/scratch/ba63/zaebuc-lrec-2026/public-release
dir_exp=/scratch/ba63/zaebuc-lrec-2026/expriments/

#AR CamelTools + Stanza
python generate_data_split_lang_annotations.py \
    --input_data ${dir_exp}/spoken/morph/models/camelTools+stanza/all.morph.tsv \
    --lang_ids ${dir_public}/spoken/zaebuc-s/lang_annotations/utt_LID \
    --lang AR \
    --output_file ${dir_exp}/spoken/morph/models/camelTools+stanza/ar.morph.tsv

#EN CamelTools + Stanza
python generate_data_split_lang_annotations.py \
    --input_data ${dir_exp}/spoken/morph/models/camelTools+stanza/all.morph.tsv \
    --lang_ids ${dir_public}/spoken/zaebuc-s/lang_annotations/utt_LID \
    --lang EN \
    --output_file ${dir_exp}/spoken/morph/models/camelTools+stanza/en.morph.tsv

#CSW CamelTools + Stanza
python generate_data_split_lang_annotations.py \
    --input_data ${dir_exp}/spoken/morph/models/camelTools+stanza/all.morph.tsv \
    --lang_ids ${dir_public}/spoken/zaebuc-s/lang_annotations/utt_LID \
    --lang CSW_AR-EN \
    --output_file ${dir_exp}/spoken/morph/models/camelTools+stanza/csw.morph.tsv
