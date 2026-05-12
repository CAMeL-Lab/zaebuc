
dir_public=/scratch/ba63/zaebuc-lrec-2026/public-release
dir_exp=/scratch/ba63/zaebuc-lrec-2026/expriments/

for model in gpt-4o gpt-5; do
  python process_llm_annotation_files.py \
    --input_manual_annotations ${dir_public}/spoken/zaebuc-s/morph/all.morph.tsv \
    --input_llm_annotation_ar ${dir_exp}/spoken/morph/models/llms-outputs/${model}/ar.with_ids.tsv \
    --input_llm_annotation_en ${dir_exp}/spoken/morph/models/llms-outputs/${model}/en.with_ids.tsv \
    --input_llm_annotation_en_fixed ${dir_exp}/spoken/morph/models/llms-outputs/${model}/en.eval.final.tsv \
    --input_llm_annotation_csw ${dir_exp}/spoken/morph/models/llms-outputs/${model}/csw.with_ids.tsv \
    --output_llm_annotation_ar ${dir_exp}/spoken/morph/models/llms-outputs/${model}/ar.parsed.fixed.tsv \
    --output_llm_annotation_en ${dir_exp}/spoken/morph/models/llms-outputs/${model}/en.parsed.fixed.tsv \
    --output_llm_annotation_csw ${dir_exp}/spoken/morph/models/llms-outputs/${model}/csw.parsed.fixed.tsv \
    --output_llm_annotation_all ${dir_exp}/spoken/morph/models/llms-outputs/${model}/all.parsed.fixed.tsv
done
