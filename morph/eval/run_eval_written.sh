
dir_public=/Users/injy.hamed/Documents/Projects/Zaebuc/camera_ready/final/release/public
dir_exp=/Users/injy.hamed/Documents/Projects/Zaebuc/camera_ready/final/release/experiments
#dir_public=/scratch/ba63/zaebuc-lrec-2026/public-release
#dir_exp=/scratch/ba63/zaebuc-lrec-2026/expriments/

# CAMeL Tools
for dataset in zaebuc-w1 zaebuc-w2; do
  for split in dev test; do
    echo ${dataset}-${split}
    python evaluate_written.py \
      --input_data_auto ${dir_exp}/written/morph/models/camelTools/${dataset}/${split}.morph.tsv \
      --input_data_gold ${dir_public}/written/${dataset}/ar/morph/${split}.morph.tsv 
  done
done

# Stanza
for dataset in zaebuc-w1 zaebuc-w2; do
  for split in dev test; do
    echo ${dataset}-${split}
    python evaluate_written.py \
      --input_data_auto ${dir_exp}/written/morph/models/stanza/${dataset}/${split}.morph.tsv \
      --input_data_gold ${dir_public}/written/${dataset}/en/morph/${split}.morph.tsv
  done
done

# LLMs
for model in gpt-4o gpt-5; do
  for dataset in zaebuc-w1 zaebuc-w2; do
    for lang in ar en; do
      for split in dev test; do
        echo ${model}-${dataset}-${lang}-${split}
        python evaluate_written.py \
          --input_data_auto ${dir_exp}/written/morph/models/llms-outputs/${model}/${dataset}/${lang}/${split}.parsed.fixed.txt \
          --input_data_gold ${dir_public}/written/${dataset}/${lang}/morph/${split}.morph.tsv
      done
    done
  done
done
