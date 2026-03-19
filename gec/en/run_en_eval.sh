# gector eval
for dataset in zaebuc-w1 zaebuc-w2
do
    for split in dev test
    do
        src=/scratch/ba63/zaebuc-lrec-2026/public-release/written/$dataset/en/gec/$split.raw.tok
        # removing the header from src
        tmp_src=$(mktemp)
        tail -n +2 "$src" > "$tmp_src"

        tgt=/scratch/ba63/zaebuc-lrec-2026/expriments/written/gec/models/gector/$dataset/en/$split.seg.unseg.txt
        hyp_edits=/scratch/ba63/zaebuc-lrec-2026/expriments/written/gec/models/gector/$dataset/$split.seg.unseg.m2
        
        # creating the hyp m2 edits
        errant_parallel -orig $tmp_src -cor $tgt -out $hyp_edits

        gold_edits=/scratch/ba63/zaebuc-lrec-2026/public-release/written/$dataset/en/gec/$split.m2

        # running the eval
        errant_compare -hyp $hyp_edits -ref $gold_edits \
            > /scratch/ba63/zaebuc-lrec-2026/expriments/written/gec/models/gector/$dataset/$split.eval
        
        rm $tmp_src
    done

done


#LLMs evaluation
for llm in gpt-4o gpt-5 jais Fanar-C-1-8.7B
do

    for dataset in zaebuc-w1 zaebuc-w2
    do
        for split in dev test
        do
            # tokenizing the llm output for eval
            llm_ouput=/scratch/ba63/zaebuc-lrec-2026/expriments/written/gec/models/llms-outputs/$llm/$dataset/en/$split.txt
            python tokenize_en.py \
                --input_file  \
                --output_file $llm_ouput.tok

            src=/scratch/ba63/zaebuc-lrec-2026/public-release/written/$dataset/en/gec/$split.raw.tok
            # removing the header from src
            tmp_src=$(mktemp)
            tail -n +2 "$src" > "$tmp_src"

            hyp_edits=$llm_ouput.tok.m2
            
            # creating the hyp m2 edits
            errant_parallel -orig $tmp_src -cor $llm_ouput.tok -out $hyp_edits

            gold_edits=/scratch/ba63/zaebuc-lrec-2026/public-release/written/$dataset/en/gec/$split.m2

            # running the eval
            errant_compare -hyp $hyp_edits -ref $gold_edits \
                > /scratch/ba63/zaebuc-lrec-2026/expriments/written/gec/models/llms-outputs/$llm/$dataset/en/$split.txt.tok.eval

            rm $tmp_src
        done

    done
done

