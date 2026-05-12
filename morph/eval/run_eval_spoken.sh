

dir_public=/Users/injy.hamed/Documents/Projects/Zaebuc/camera_ready/final/release/public
dir_exp=/Users/injy.hamed/Documents/Projects/Zaebuc/camera_ready/final/release/experiments
#dir_public=/scratch/ba63/zaebuc-lrec-2026/public-release
#dir_exp=/scratch/ba63/zaebuc-lrec-2026/expriments/

# CAMeL Tools + Stanza (C&S)
echo ${dataset}
python evaluate_spoken.py \
  --input_data_auto ${dir_exp}/spoken/morph/models/camelTools+stanza/all.morph.tsv \
  --input_data_gold ${dir_public}/spoken/zaebuc-s/morph/all.morph.tsv  \
  --lang_ids ${dir_public}/spoken/zaebuc-s/lang_annotations/utt_LID

# LLMs
for model in gpt-4o gpt-5; do
  echo ${model}
  python evaluate_spoken.py \
    --input_data_auto ${dir_exp}/spoken/morph/models/llms-outputs/${model}/all.parsed.fixed.tsv \
    --input_data_gold ${dir_public}/spoken/zaebuc-s/morph/all.morph.tsv  \
    --lang_ids ${dir_public}/spoken/zaebuc-s/lang_annotations/utt_LID
done
