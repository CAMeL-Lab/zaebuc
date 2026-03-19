
exp_data=/scratch/ba63/zaebuc-lrec-2026/expriments/written/aes/bert/data

for lang in ar en
do
    for dataset in zaebuc-w1 zaebuc-w2
    do
        for split in train dev test
            do
                if [ "$lang" = "ar" ]; then
                    paste /scratch/ba63/zaebuc-lrec-2026/public-release/written/$dataset/$lang/aes/$split.raw.tok \
                    /scratch/ba63/zaebuc-lrec-2026/public-release/written/$dataset/$lang/aes/$split.cefr > $exp_data/$dataset/$lang/$split.txt.check
                else
                    paste /scratch/ba63/zaebuc-lrec-2026/public-release/written/$dataset/$lang/aes/$split.raw \
                    /scratch/ba63/zaebuc-lrec-2026/public-release/written/$dataset/$lang/aes/$split.cefr > $exp_data/$dataset/$lang/$split.txt.check
                fi

                # getting the labels based on the avg CEFR label without the Unassessable label
                if [ "$split" = "train" ]; then
                    cat $dataset/$lang/train.txt | cut -f5 | sort | uniq | grep -v "Unassessable" > $exp_data/$dataset/$lang/labels.txt.check
                fi
            done
    done
done

# contactenating the training data for zaebuc arabic and english
for lang in ar en
do
    cat $exp_data/zaebuc-w1/$lang/train.txt $exp_data/zaebuc-w2/$lang/train.txt > mix/$lang/train.txt.check
    cat mix/$lang/train.txt | cut -f5 | sort | uniq | grep -v "Unassessable" > mix/$lang/labels.txt.check
done
