# download gector: https://github.com/grammarly/gector
for dataset in zaebuc-w1 zaebuc-w2
do
    for split in dev test
    do
        printf "$dataset $split\n"
        python /home/bashar.alhafni/gector/predict.py \
            --model_path /home/bashar.alhafni/gector_models/roberta_1_gectorv2.th \
            --vocab_path /home/bashar.alhafni/gector/data/output_vocabulary \
            --input_file /scratch/ba63/zaebuc-lrec-2026/expriments/written/gec/data/$dataset/en/$split.sent.raw.clean.tok.seg \
            --output_file /scratch/ba63/zaebuc-lrec-2026/expriments/written/gec/models/gector/$dataset/en/$split.seg.txt

        python stitch_segments.py \
            --input_file /scratch/ba63/zaebuc-lrec-2026/expriments/written/gec/models/gector/$dataset/en/$split.seg.txt \
            --ids /scratch/ba63/zaebuc-lrec-2026/expriments/written/gec/data/$dataset/en/$split.seg.ids \
            --output_file /scratch/ba63/zaebuc-lrec-2026/expriments/written/gec/models/gector/$dataset/en/$split.seg.unseg.txt
    done
done
