import argparse
import jiwer


NON_SPEECH_TAGS=["{laugh}","{cough}","{sneeze}","{breath}","{lipsmack}","{pause}","{hem}","{shush}","{gasp}"]
BACKGROUND_TAGS=["[/typing]","[typing/]","[/noise]","[noise/]","[/reading]","[reading/]","[/audioCut/]"]
interjection_correct_set=["%oh","%أوه","%aa","%أأ","%um","%إم","%أم","%ام","%mm","%مم","%hm","%هم","%ah","%uh","%آه","%aha","%أها","%ehm","%إهم","%nn","%إن", "%ha","%ها","%إي","%إي","%تت","%tt","%mhm","%er","%ops"]
speaker_annotations=["<Student1>","<Student2>","<Moderator>","<Interlocutor>","<Student1_1>","<Student2_1>","<Manager_1>","<Interlocutor_1>","<Student1_2>","<Student2_2>","<Manager_2>","<Interlocutor_2>","<Student2_3>","<Student2_4>"]

transforms = jiwer.Compose(
    [
        jiwer.ToLowerCase(),
        jiwer.RemoveMultipleSpaces(),
        jiwer.Strip(),
        jiwer.RemovePunctuation(),
        jiwer.ReduceToListOfListOfWords(),
    ]
)

def remove_punctuations_annotations(transcription):
    for tag in NON_SPEECH_TAGS:
        transcription=transcription.replace(tag,"")
    for tag in BACKGROUND_TAGS:
        transcription=transcription.replace(tag,"")
    for tag in interjection_correct_set:
        transcription=transcription.replace(tag,"")
    for tag in speaker_annotations:
        transcription=transcription.replace(tag,"")
    #for symbol in ['.', '،', ',', '?',':','؟','!',':',";","/","؛","~","*","=","--","((","))","[","]","_"]:
    for symbol in ["~","*","=","--","((","))","[","]","_","+"]:
        transcription=transcription.replace(symbol,"")
    return transcription

def get_manual_transcriptions_dict():
    map_manual_transcriptions={}
    for manual_transcriptions_line in manual_transcriptions_lines:
        utt_id=manual_transcriptions_line.strip().split("\t")[0]
        manual_transcription=manual_transcriptions_line.strip().split("\t")[1]
        manual_transcription_withoutAnnotations=remove_punctuations_annotations(manual_transcription)
        map_manual_transcriptions[utt_id]=manual_transcription_withoutAnnotations
    return map_manual_transcriptions

def get_lang_dict(lang_file_lines):
    map_utt_lang={}
    for lang_file_line in lang_file_lines:
        utt_id,lang_id=lang_file_line.strip().split()
        map_utt_lang[utt_id]=lang_id
    return map_utt_lang
        
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--transcriptions", help="Input file having the transcriptions.")
    parser.add_argument("--hypotheses", help="Input file having the ASR hypotheses.")
    parser.add_argument("--lang_ids", help="File containing the language ID for each utterance.")
    parser.add_argument("--data_split", help="Data split being evaluated")

    args = parser.parse_args()

    manual_transcriptions_lines=open(args.transcriptions).readlines()
    hyp_lines=open(args.hypotheses).readlines()
    lang_file_lines=open(args.lang_ids).readlines()
    data_split=args.data_split

    if data_split=="dev":
        recordings=["R01","R02","R07","R08","R09","R11","R12","R13","R15"]
    else:
        recordings=["R03","R04","R05","R06","R10","R14"]

    ref_list_ar,hyp_list_ar=[],[]
    ref_list_en,hyp_list_en=[],[]
    ref_list_csw,hyp_list_csw=[],[]
    ref_list_all,hyp_list_all=[],[]
    map_manual_transcriptions=get_manual_transcriptions_dict()
    map_utt_lang=get_lang_dict(lang_file_lines)
    for hyp_line in hyp_lines:
        hyp_line_info=hyp_line.strip().split("\t")
        hyp_utt_id=hyp_line_info[0]
        recording=hyp_utt_id.split("_")[0].split("-")[2]
        ref=map_manual_transcriptions[hyp_utt_id]
        hyp=""
        if len(hyp_line_info)==2:
            hyp=remove_punctuations_annotations(hyp_line_info[1])
        utt_lang=map_utt_lang[hyp_utt_id]
        if recording in recordings:
            if utt_lang!="annotationsOnly":
                ref_list_all.append(ref)
                hyp_list_all.append(hyp)
            if utt_lang=="AR":
                ref_list_ar.append(ref)
                hyp_list_ar.append(hyp)
            elif utt_lang=="EN":
                ref_list_en.append(ref)
                hyp_list_en.append(hyp)
            elif utt_lang=="CSW_AR-EN":
                ref_list_csw.append(ref)
                hyp_list_csw.append(hyp)
    wer_ar = jiwer.wer(ref_list_ar, hyp_list_ar, truth_transform=transforms, hypothesis_transform=transforms)
    wer_en = jiwer.wer(ref_list_en, hyp_list_en, truth_transform=transforms, hypothesis_transform=transforms)
    wer_csw = jiwer.wer(ref_list_csw, hyp_list_csw, truth_transform=transforms, hypothesis_transform=transforms)
    wer_all = jiwer.wer(ref_list_all, hyp_list_all, truth_transform=transforms, hypothesis_transform=transforms)
    print("wer_ar",wer_ar)
    print("wer_en",wer_en)
    print("wer_csw",wer_csw)
    print("wer_all",wer_all)
