import argparse
from camel_tools.utils.charsets import AR_LETTERS_CHARSET
from camel_tools.utils.dediac import dediac_ar

dev_recordings=["R01","R02","R07","R08","R09","R11","R12","R13","R15"]
test_recordings=["R03","R04","R05","R06","R10","R14"]

def stringContainsEnglish(word):
    for c in word:
        if 'a' <= c <= "z" or 'A' <= c <= "Z":
            return True
    return False

def stringContainsArabic(word):
    for c in word:
        if c in AR_LETTERS_CHARSET:
            return True
    return False

def fillInDict(dict,set,tok_correct,pos_correct,lemma_correct):
    dict[set]["tok"].append(tok_correct)
    dict[set]["pos"].append(pos_correct)
    dict[set]["lemma"].append(lemma_correct)

def calcAccuracy(l):
    correct_count=0
    for e in l:
        if e==True:
            correct_count+=1
    if correct_count==0:
        return 0
    return correct_count/len(l)

def calcAccuracy_perlang(dict):
    print("tok: "+str(calcAccuracy(dict["tok"])))
    print("pos: "+str(calcAccuracy(dict["pos"])))
    print("lemma: "+str(calcAccuracy(dict["lemma"])))

def getDict_utt_lang(lang_file_lines):
    map_utt_lang={}
    for lang_file_line in lang_file_lines:
        utt_id,lang_id=lang_file_line.strip().split()
        map_utt_lang[utt_id]=lang_id
    return map_utt_lang

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_data_auto", help="Input data for automatic annotations file path.")
    parser.add_argument("--input_data_gold", help="Input data for manual annotations file path.")
    parser.add_argument("--lang_ids", help="File containing the language ID for each utterance.")

    args = parser.parse_args()

    count_tokens,num_tokens_AR_utt,num_tokens_EN_utt,num_tokens_CSW_utt=0,0,0,0
    count_correct_tok,count_correct_pos,count_correct_lemma=0,0,0

    dict_tokens_count={"dev":{"AR":0,"EN":0,"CSW_AR-EN":0,"annotationsOnly":0},"test":{"AR":0,"EN":0,"CSW_AR-EN":0,"annotationsOnly":0}}

    manual_lines=open(args.input_data_gold).readlines()[2:]
    auto_lines=open(args.input_data_auto).readlines()[2:]
    lang_file_lines=open(args.lang_ids).readlines()

    overall_ar_utt= {"dev":{"tok":[],"pos":[],"lemma":[]}, "test":{"tok":[],"pos":[],"lemma":[]}}
    overall_en_utt= {"dev":{"tok":[],"pos":[],"lemma":[]}, "test":{"tok":[],"pos":[],"lemma":[]}}
    overall_csw_utt={"dev":{"tok":[],"pos":[],"lemma":[]}, "test":{"tok":[],"pos":[],"lemma":[]}}
    overall_all_utt={"overall":{"tok":[],"pos":[],"lemma":[]},"dev":{"tok":[],"pos":[],"lemma":[]}, "test":{"tok":[],"pos":[],"lemma":[]}}

    map_utt_lang=getDict_utt_lang(lang_file_lines)

    for  manual_line, auto_line in zip(manual_lines, auto_lines):
        if "<TRANSCRIPTION>" not in manual_line and len(manual_line.strip())>0:
            count_tokens+=1

            auto_token_id,auto_word,autoTok,autoPOS,autolemma=auto_line.strip().split()[:5]
            manual_token_id,manual_word,manualTok,manualPOS,manualLemma=manual_line.strip().split()[:5]

            utt_id="_".join(auto_token_id.split("_")[:-1])
            rec_id=auto_token_id.split("_")[0].split("-")[2]
            
            utt_lang=map_utt_lang[utt_id]
            if utt_lang=="AR": num_tokens_AR_utt+=1
            elif utt_lang=="EN": num_tokens_EN_utt+=1
            elif utt_lang=="CSW_AR-EN": num_tokens_CSW_utt+=1
            
            tok_correct=dediac_ar(autoTok)==dediac_ar(manualTok)
            pos_correct=autoPOS==manualPOS
            lemma_correct=autolemma==manualLemma

            if tok_correct: count_correct_tok+=1
            if pos_correct: count_correct_pos+=1
            if lemma_correct: count_correct_lemma+=1
            
            fillInDict(overall_all_utt,"overall",tok_correct,pos_correct,lemma_correct)

            if rec_id in dev_recordings:
                fillInDict(overall_all_utt,"dev",tok_correct,pos_correct,lemma_correct)
                if utt_lang=="AR": fillInDict(overall_ar_utt,"dev",tok_correct,pos_correct,lemma_correct)
                elif utt_lang=="EN": fillInDict(overall_en_utt,"dev",tok_correct,pos_correct,lemma_correct)
                elif utt_lang=="CSW_AR-EN": fillInDict(overall_csw_utt,"dev",tok_correct,pos_correct,lemma_correct)
                dict_tokens_count["dev"][utt_lang]=dict_tokens_count["dev"][utt_lang]+1

            elif rec_id in test_recordings:
                fillInDict(overall_all_utt,"test",tok_correct,pos_correct,lemma_correct)
                if utt_lang=="AR": fillInDict(overall_ar_utt,"test",tok_correct,pos_correct,lemma_correct)
                elif utt_lang=="EN": fillInDict(overall_en_utt,"test",tok_correct,pos_correct,lemma_correct)
                elif utt_lang=="CSW_AR-EN": fillInDict(overall_csw_utt,"test",tok_correct,pos_correct,lemma_correct)
                dict_tokens_count["test"][utt_lang]=dict_tokens_count["test"][utt_lang]+1

            if auto_token_id!=manual_token_id or auto_word!=manual_word:
                print("Unmatched entries")
                print(auto_token_id,manual_token_id)
                print(auto_word,manual_word)

    print("\nDev")
    print("\nOverall Arabic utterances:")
    calcAccuracy_perlang(overall_ar_utt["dev"])
    print("\nOverall English utterances:")
    calcAccuracy_perlang(overall_en_utt["dev"])
    print("\nOverall CSW utterances:")
    calcAccuracy_perlang(overall_csw_utt["dev"])

    print("\nTest")
    #print("Overall All utterances:")
    #calcAccuracy_perlang(overall_all_utt["test"])
    print("\nOverall Arabic utterances:")
    calcAccuracy_perlang(overall_ar_utt["test"])
    print("\nOverall English utterances:")
    calcAccuracy_perlang(overall_en_utt["test"])
    print("\nOverall CSW utterances:")
    calcAccuracy_perlang(overall_csw_utt["test"])
