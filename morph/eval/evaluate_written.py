import argparse
import itertools as itr
from camel_tools.utils.dediac import dediac_ar

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_data_auto", help="File with automatic annotations.")
    parser.add_argument("--input_data_gold", help="File with ground truth annotations.")
    args = parser.parse_args()

    input_manual_lines=open(args.input_data_gold).readlines()
    input_auto_lines=open(args.input_data_auto).readlines()

    list_tok_acc=[]
    list_pos_acc=[]
    list_lemma_acc=[]

    for manual_line,auto_line in zip(input_manual_lines, input_auto_lines):
        manual_line_info=manual_line.strip().split("\t")
        auto_line_info=auto_line.strip().split("\t")
        list_tok_acc.append(int(dediac_ar(manual_line_info[2])==dediac_ar(auto_line_info[2].replace(" ","").replace("_+","+").replace("+_","+"))))
        auto_pos=auto_line_info[3].replace(" ","")
        list_pos_acc.append(int(manual_line_info[3]==auto_pos))
        list_lemma_acc.append(int(manual_line_info[4]==auto_line_info[4]))
    print("tok:",sum(list_tok_acc)/len(list_tok_acc))
    print("pos:",sum(list_pos_acc)/len(list_pos_acc))
    print("lemma:",sum(list_lemma_acc)/len(list_lemma_acc))
