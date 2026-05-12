import argparse
import re
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt_tab')

contraction_set=set()
def tokenize(line):
    for word in line.split():
        word_without_punc=word
        for punc in ",.?!؟،{}%*:-();":
            word_without_punc=word_without_punc.replace(punc,"")
        if word_without_punc.strip()!=" ".join(word_tokenize(word_without_punc)).strip():
            contraction_set.add(word_without_punc.lower())
    line=line.replace("{","{REMSPACE")
    line=line.replace("}","REMSPACE}")
    line=line.replace("<","<REMSPACE")
    line=line.replace(">","REMSPACE>")
    line=line.replace("%","%REMSPACE")
    line=line.replace(":","REMSPACE:")
    line=line.replace("و<","وREMSPACE<")
    line=line.replace("؟"," ؟ ")
    line=line.replace("،"," ، ")
    line= re.sub(r"(?<!\S)(--)(.+?)(?<!\b)","--REMSPACE"+r"\2",line)
    line= re.sub(r"(?<!\S)(.+?)(--)(?<!\b)",r"\1"+"REMSPACE--",line)
    line=line.replace("((--","(( --REMSPACE")
    line=line.replace("~"," ~ ")
    line=line.replace("="," = ")
    line=line.replace("؛"," ؛ ")
    line=re.sub('\s{2,}', ' ', line).strip()
    words_tok=" ".join(word_tokenize(line)).replace("[ /typing ]","[/typing]").replace("[ typing/ ]","[typing/]").replace("[ /noise ]","[/noise]").replace("[ noise/ ]","[noise/]").replace("[ /reading ]","[/reading]").replace("[ reading/ ]","[reading/]").replace("[ /audioCut/ ]","[/audioCut/]").split()

    L_words_merged=[]
    #detokenizing contractions
    for word in words_tok:
        if len(L_words_merged)>0 and word in ["n't","'m","'re","'s","'d", "'ll","'ve"]:
            L_words_merged[len(L_words_merged)-1]=L_words_merged[len(L_words_merged)-1]+word
        elif len(L_words_merged)>0 and L_words_merged[len(L_words_merged)-1]=="gon" and word=="na":
            L_words_merged[len(L_words_merged)-1]=L_words_merged[len(L_words_merged)-1]+word
        elif len(L_words_merged)>0 and L_words_merged[len(L_words_merged)-1]=="wan" and word=="na":
            L_words_merged[len(L_words_merged)-1]=L_words_merged[len(L_words_merged)-1]+word
        else:
            L_words_merged.append(word)
        #print(L_words_merged)
    line=" ".join(L_words_merged)
    line=re.sub('\s{2,}', ' ', line).strip()
    line=line.replace("REMSPACE ","").replace(" REMSPACE","")
    line=line.replace("( (","((").replace(") )","))")
    line=line.replace("(( [","(([").replace("] ))","]))")
    #separate cases:
    line=line.replace("citizens ' %aa attention","citizens' %aa attention")
    line=line.replace("parents ' fears","parents' fear")
    line=line.replace("parents ' approval","parents' approval")
    line=line.replace("locals '","locals'")
    line=line.replace("can not","cannot")
    line=line.replace("ال+ = Gulf =","ال+=Gulf=")
    line=line.replace("> -- ",">-- ")
    return line

def get_new_ID_list(input_manual_annotations_lines):
    list_ids=[]
    for line in input_manual_annotations_lines[1:]:
        utt_id="_".join(line.split("\t")[0].split("_")[:-1]) #removing token id
        if utt_id not in list_ids:
            list_ids.append(utt_id)
    return list_ids

def get_dict_transcriptions(input_transcription_lines):
    dict_transcriptions={}
    for line in input_transcription_lines:
        utt_id=line.split("\t")[0]
        text=line.split("\t")[1]
        dict_transcriptions[utt_id]=text.strip()
    return dict_transcriptions

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_transcriptions", help="Input data file path.")
    parser.add_argument("--manual_morph_annotations", help="File containing manual annotations.")
    parser.add_argument("--output", help="Output file containing tokenized transcriptions.")

    args = parser.parse_args()
    input_transcription_lines=open(args.input_transcriptions).readlines()
    input_manual_annotations_lines=open(args.manual_morph_annotations).readlines()
    output_tok_file=open(args.output,'w')
    
    utt_ids=get_new_ID_list(input_manual_annotations_lines)
    dict_transcriptions=get_dict_transcriptions(input_transcription_lines)
    
    for utt_id in utt_ids:
        transcription=dict_transcriptions[utt_id]
        line_tok=tokenize(transcription)
        output_tok_file.write(utt_id+"\t"+line_tok+"\n")
