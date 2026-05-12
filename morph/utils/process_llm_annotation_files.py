"""
This script is used to:
(1) Fix the English annotations as the en.tsv file has a few issues, so this script uses en.eval.final.tsv instead
(2) Modify the automatic annotations of corpus-related tags, to be consistent with the evaluation of camelTools+stanza output
(3) Combine ar, en, and csw annotations into one file with the same order of token IDs as the manual annotation file
"""

import re
import csv
import argparse

BACKGROUND=["[/typing]","[typing/]","[/noise]","[noise/]","[/reading]","[reading/]","[/audioCut/]"]
NON_SPEECH_TAGS=["{laugh}","{cough}","{sneeze}","{breath}","{noise}","{lipsmack}","{pause}","{typing}","{laughs}","{types}","{writes}","{hem}","{gasp}","{shush}","{ehhe}"]


def get_dict_auto_annotations(dict_annotation_files,annotations_en_fixed):
    dict_auto_annotations={}
    for lang in dict_annotation_files:
        annotation_file=dict_annotation_files[lang]
        tsvfile=open(annotation_file,newline='')
        tsv_reader = csv.DictReader(tsvfile, delimiter='\t')        
        line_index=0
        for row in tsv_reader:
            token_id=row['Token_id']
            word=row['Word']
            tok=row['Tok']
            pos=row['POS']
            lemma=row['Lemma']
            if tok is None or len(tok.strip())==0: tok="<ERROR>"
            if pos is None or len(pos.strip())==0: pos="<ERROR>"
            if lemma is None or len(lemma.strip())==0: lemma="<ERROR>"

            if lang=='en': #the en.tsv file has a few issues, using en.eval.final.tsv instead
                new_fixed_line_info=annotations_en_fixed[line_index].strip().split("\t")
                tok,pos,lemma="<ERROR>","<ERROR>","<ERROR>"
                if len(new_fixed_line_info)>5:
                    tok=new_fixed_line_info[5]
                if len(new_fixed_line_info)>6:
                    pos=new_fixed_line_info[6]
                if len(new_fixed_line_info)>7:
                    lemma=new_fixed_line_info[7]
            dict_auto_annotations[token_id]={"Token_id":token_id,"Word":word,"Tok":tok,"POS":pos,"Lemma":lemma,"lang":lang}
            line_index+=1
    return dict_auto_annotations    

def insert_plus_around_redact_tokens(text):
    # add + before <tag> if preceded by a non-space character
    text = re.sub(r'(\S)(<\w+>)', r'\1+\2', text)
    # add + after <tag> if followed by a non-space character
    text = re.sub(r'(<\w+>)(\S)', r'\1+\2', text)
    return text

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_manual_annotations", help="Input file containing the manual annotations.")
    parser.add_argument("--input_llm_annotation_ar", help="Input file containing the Arabic llm annotations.")
    parser.add_argument("--input_llm_annotation_en", help="Input file containing the English llm annotations.")
    parser.add_argument("--input_llm_annotation_en_fixed", help="Input file containing the fixed English llm annotations.")
    parser.add_argument("--input_llm_annotation_csw", help="Input file containing the Code-switched llm annotations.")
    parser.add_argument("--output_llm_annotation_ar", help="Output file containing the processed Arabic llm annotations.")
    parser.add_argument("--output_llm_annotation_en", help="Output file containing the processed English llm annotations.")
    parser.add_argument("--output_llm_annotation_csw", help="Output file containing the processed Code-switched llm annotations.")
    parser.add_argument("--output_llm_annotation_all", help="Output file containing the processed conbined llm annotations.")

    args = parser.parse_args()
    
    manual_annotations=open(args.input_manual_annotations).readlines()
    en_annotations_fixed=open(args.input_llm_annotation_en_fixed).readlines()
    output_file_all=open(args.output_llm_annotation_all,'w')
    output_file_ar=open(args.output_llm_annotation_ar,'w')
    output_file_en=open(args.output_llm_annotation_en,'w')
    output_file_csw=open(args.output_llm_annotation_csw,'w')

    dict_annotation_files={'ar':args.input_llm_annotation_ar,'en':args.input_llm_annotation_en,'csw':args.input_llm_annotation_csw}
    
    output_file_all.write("\t".join(["Token_id","Word","Tok","POS","Lemma"])+"\n")
    output_file_ar.write("\t".join(["Token_id","Word","Tok","POS","Lemma"])+"\n")
    output_file_en.write("\t".join(["Token_id","Word","Tok","POS","Lemma"])+"\n")
    output_file_csw.write("\t".join(["Token_id","Word","Tok","POS","Lemma"])+"\n")

    dict_auto_annotations=get_dict_auto_annotations(dict_annotation_files,en_annotations_fixed)
    
    for line in manual_annotations[1:]: #ignoring header
        
        line_info=line.strip().split()
        token_id=line_info[0]
        word=line_info[1]

        # Modifying the automatic annotations of corpus-related tags, to be consistent with the evaluation of camelTools+stanza output
        if ("<Moderator" in word or "<Interlocutor" in word or "<Student" in word) and "--" not in word:
            #auto_tok,auto_lemma=word,word #redact
            auto_tok=insert_plus_around_redact_tokens(word)
            auto_lemma=re.search(r'<\w+>', word).group(0)
            if token_id in dict_auto_annotations:
                auto_pos=dict_auto_annotations[token_id]["POS"]
            else:#redacts in utterances that are annotations-only
                auto_pos=line_info[3]
        elif word in ['،', '؟']:
            auto_tok,auto_pos,auto_lemma=word,"PUNCT",word
        elif word in ["..","~","*","=","))","((","(([","]))"]:
            auto_tok,auto_pos,auto_lemma=word,"<ANNOT>",word
        elif word in NON_SPEECH_TAGS:
            auto_tok,auto_pos,auto_lemma=word,"<NONSPEECH>",word
        elif word in BACKGROUND:
            auto_tok,auto_pos,auto_lemma=word,"<BACKGROUND>",word    
        elif word.startswith("%"): #interjection
            auto_tok,auto_pos,auto_lemma=word,"<INTJ>",word
        elif word.startswith("--") or word.endswith("--"):
            auto_tok,auto_pos,auto_lemma=word,"<PW>",word
        elif token_id in dict_auto_annotations:
            auto_tok,auto_pos,auto_lemma=dict_auto_annotations[token_id]["Tok"],dict_auto_annotations[token_id]["POS"],dict_auto_annotations[token_id]["Lemma"]
        else:
            auto_tok,auto_pos,auto_lemma="-","-","-" #annotation-only utterance
        
        #writing output into files
        if token_id in dict_auto_annotations: #missing token_ids are the utterances that only contain annotations
            lang=dict_auto_annotations[token_id]["lang"]
            if lang=='ar':
                output_file_ar.write("\t".join([token_id,word,auto_tok,auto_pos,auto_lemma])+"\n")
            elif lang=='en':
                output_file_en.write("\t".join([token_id,word,auto_tok,auto_pos,auto_lemma])+"\n")
            elif lang=='csw':
                output_file_csw.write("\t".join([token_id,word,auto_tok,auto_pos,auto_lemma])+"\n")
        output_file_all.write("\t".join([token_id,word,auto_tok,auto_pos,auto_lemma])+"\n")