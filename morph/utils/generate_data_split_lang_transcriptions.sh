
dir_public=/scratch/ba63/zaebuc-lrec-2026/public-release
dir_exp=/scratch/ba63/zaebuc-lrec-2026/expriments/

#AR tokenized transcriptions
python generate_data_split_lang_transcriptions.py \
    --input_data ${dir_public}/spoken/zaebuc-s/morph/all.transcriptions.tok \
    --lang_ids ${dir_public}/spoken/zaebuc-s/lang_annotations/utt_LID \
    --lang AR \
    --output_file ${dir_exp}/spoken/morph/data/ar.transcriptions.tok

#EN tokenized transcriptions
python generate_data_split_lang_transcriptions.py \
    --input_data ${dir_public}/spoken/zaebuc-s/morph/all.transcriptions.tok \
    --lang_ids ${dir_public}/spoken/zaebuc-s/lang_annotations/utt_LID \
    --lang EN \
    --output_file ${dir_exp}/spoken/morph/data/en.transcriptions.tok

#CSW tokenized transcriptions
python generate_data_split_lang_transcriptions.py \
    --input_data ${dir_public}/spoken/zaebuc-s/morph/all.transcriptions.tok \
    --lang_ids ${dir_public}/spoken/zaebuc-s/lang_annotations/utt_LID \
    --lang CSW_AR-EN \
    --output_file ${dir_exp}/spoken/morph/data/csw.transcriptions.tok

