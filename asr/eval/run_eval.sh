# Reporting WER for multichannel (combined) and single channel (separate)

dir_public=/scratch/ba63/zaebuc-lrec-2026/public-release
dir_exp=/scratch/ba63/zaebuc-lrec-2026/expriments/

#Without Language Flag
for channel in multichannel singlechannel; do
  for data_split in dev test; do
    echo ${channel}-${data_split}
    python run_eval.py \
      --transcriptions ${dir_public}/spoken/zaebuc-s/transcriptions/text \
      --hypotheses ${dir_exp}/spoken/ASR/whisper/hyp.${channel}.withoutLangFlag \
      --lang_ids ${dir_public}/spoken/zaebuc-s/lang_annotations/utt_LID \
      --data_split ${data_split}
  done
done

#Without Language Flag
for channel in multichannel singlechannel; do
  for data_split in dev test; do
    echo ${channel}-${data_split}
    python run_eval.py \
      --transcriptions ${dir_public}/spoken/zaebuc-s/transcriptions/text \
      --hypotheses ${dir_exp}/spoken/ASR/whisper/hyp.${channel}.withLangFlag \
      --lang_ids ${dir_public}/spoken/zaebuc-s/lang_annotations/utt_LID \
      --data_split ${data_split}
  done
done
