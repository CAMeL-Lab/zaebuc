import argparse
import pickle
import re

from camel_tools.disambig.bert import BERTUnfactoredDisambiguator
from camel_tools.morphology.database import MorphologyDB
from camel_tools.morphology.analyzer import Analyzer
from camel_tools.utils.charmap import CharMapper
from camel_tools.utils.dediac import dediac_ar
from camel_tools.utils.charsets import AR_LETTERS_CHARSET
bw2ar = CharMapper.builtin_mapper('bw2ar')
ar2bw = CharMapper.builtin_mapper('ar2bw')
db_dir="path_to_camel_tools_databases"

unfactored_glf = BERTUnfactoredDisambiguator.pretrained(model_name='glf')
unfactored_msa = BERTUnfactoredDisambiguator.pretrained(model_name='msa')
db = MorphologyDB(db_dir+"/MSA/calima-msa-s31_0.4.2.utf8.db", 'a')

analyzer = Analyzer(db, 'ADD_PROP', cache_size=100000)
unfactored_msa._analyzer = analyzer
with open(db_dir+'/MSA/disambig_ranking_cache/calima-msa-s31/default_cache.pickle', 'rb') as f:
    unfactored_msa._ranking_cache = pickle.load(f)

dict_mapping={
    "noun":"NOUN", "noun_prop":"PROPN", "noun_num":"NUM", "noun_quant":"NOUN", "adj":"ADJ", "adj_comp":"ADJ", 
    "adj_num":"ADJ", "adv":"ADV", "adv_interrog":"ADV", "adv_rel":"ADV", "pron":"PRON", "pron_dem":"DET",
    "pron_exclam":"PRON", "pron_interrog":"PRON", "pron_rel":"PRON", "verb":"VERB", "verb_pseudo":"CCONJ",
    "verb_nom":"VERB", "part":"PART", "part_dem":"DET", "part_det":"DET", "part_focus":"PART", "part_fut":"AUX",
    "part_interrog":"PART", "part_neg":"PART", "part_restrict":"PART", "part_verb":"AUX", "part_voc":"PART",
    "prep":"ADP", "abbrev":"NOUN", "punc":"PUNCT", "conj":"CCONJ", "conj_sub":"SCONJ", "interj":"INTJ",
    "digit":"NUM", "latin":"X", "foreign":"X"
}
dict_mapping_bw={
  "DEM_PRON":"PRON", "INTERROG_ADV":"ADV", "INTERROG_PRON":"PRON", "POSS_PRON":"PRON", "REL_ADV":"ADV",
  "REL_PRON":"PRON", "CONNEC_PART":"PART", "DET":"DET", "FOCUS_PART":"PART", "FUT_PART":"AUX",
  "INTERROG_PART":"PART", "JUS_PART":"PART", "NEG_PART":"PART", "PSEUDO_VERB":"CCONJ", "RC_PART":"PART",
  "RESTRIC_PART":"PART", "SUB_CONJ":"SCONJ", "VERB_PART":"AUX", "VOC_PART":"PART", "IV":"VERB",
  "IV_PASS":"VERB", "PV":"VERB", "CV":"VERB", "EXCLAM_PRON":"PART", "EMPHATIC_PART":"PART"
}
nominal_list= ['>amAm_1', '<ivora_1', '<izA_1', 'baEod_1', 'bayona_1', 'tujAh_1', 'taHota_1', 'tilow_1', 'Ha*ow_1', 'Hawol_1',
               'HawAlay_1', 'Hiyn_1', 'xalof_1', 'Dimon_1', 'Eaqib_1', 'Eabor_1', 'maE_1', 'Einod_1q', 'fawor_1', 'fawoq_1',
               'qabol_1', 'qubayol_1', 'qubAlap_1', 'qurob_1', 'mivol_1', '>avonA\'a_1', 'Did~_1', 'TawAl_1', 'EiwaD_1', 'Hasab_1',
               'wifoq_1', 'gayor_1', 'siwaY_1', '>abad_1', 'maE_1', 'bat~ap_1', 'mivol_1', '$abah_1', 'naHow_3', 'duwn_1', 'ladaY_1',
               'xilAl_1', 'warA\'_1', 'HiyAl_1', 'jar~A\'_1', 'wasoT_1', 'ragom_1', 'dAxil_1', 'xArij_1', 'baEodamA_1']

def stringContainsArabic(word):
    for c in word:
        if c in AR_LETTERS_CHARSET:
            return True
    return False

def stringContainsEnglish(word):
    for c in word:
        if 'a' <= c <= 'z' or 'A' <= c <= 'Z':
            return True
    return False

def get_ud_pos_bw_tag(bw_pos):
  ud_pos=bw_pos
  if bw_pos.split(":")[0].endswith("SUFF_DO"):
    ud_pos="PRON"
  elif bw_pos.startswith("POSS_PRON") or bw_pos.startswith("PRON"):
    ud_pos="PRON"
  elif bw_pos.startswith("DEM_PRON"):
    ud_pos="PRON"
  elif bw_pos.lower() in dict_mapping:
    ud_pos=dict_mapping[bw_pos.lower()]
  elif bw_pos in dict_mapping_bw:
    ud_pos=dict_mapping_bw[bw_pos]
  return ud_pos

def is_valid_suffix(bw_pos):
  conditions=[]
  conditions.append(bw_pos.startswith("IVSUFF_SUBJ"))
  conditions.append(bw_pos.startswith("CVSUFF_SUBJ"))
  conditions.append(bw_pos.startswith("CASE_"))
  conditions.append(bw_pos.startswith("IVSUFF_MOOD"))
  conditions.append(bool(re.match("(IV|PV|CV)SUFF_SUBJ*",bw_pos)))
  conditions.append(bool(re.match("NSUFF_(FEM_SG|FEM_DU|FEM_PL|MASC_DU|MASC_PL)*",bw_pos)))
  return any(conditions)

def get_best_guess_from_bw(bw):
  bw_morphemes=bw.split("+")
  if len(bw_morphemes)>1:
    best_guess_tok=[]
    best_guess_pos=[]
    for bw_morpheme in bw_morphemes:
      bw_morpheme_info=bw_morpheme.split("/")
      morph=bw_morpheme_info[0]
      if morph=="(null)":
        morph=""
      pos=bw_morpheme_info[1]
      #identifying affixes
      if bw_morpheme=="اً/CASE_INDEF_ACC":
        best_guess_tok.append("_ا")
      elif pos=='DET' or bool(re.match("IV[123][MF]?[SDP]",pos)):
        best_guess_tok.append(dediac_ar(morph)+"_")
      elif is_valid_suffix(pos):
        best_guess_tok.append("_"+dediac_ar(morph))
      else:
        best_guess_tok.append(dediac_ar(morph))
        best_guess_pos.append(get_ud_pos_bw_tag(pos))
    best_guess_tok_string1="+".join(best_guess_tok)
    best_guess_tok_string=best_guess_tok_string1.replace("+_","").replace("_+","")
    best_guess_pos_string="+".join(best_guess_pos)
    return best_guess_tok_string,best_guess_pos_string
  else:#no tokenization of word
    bw_morpheme_info=bw.split("/")
    word=dediac_ar(bw_morpheme_info[0])
    pos=get_ud_pos_bw_tag(bw_morpheme_info[1])
    return word,pos

def get_bw_POS_list(bw):
  bw_morphemes=bw.split("+")
  L_morph=[bw_morphemes[i].split("/")[0] for i in range(len(bw_morphemes))]
  L_pos=[bw_morphemes[i].split("/")[1] for i in range(len(bw_morphemes))]
  L_pos_final=[]
  for i in range(len(L_pos)):
    morph=L_morph[i]
    pos=L_pos[i]
    if morph!="(null)" and pos!='DET' and (not bool(re.match("IV[123][MF]?[SDP]",pos))) and (not is_valid_suffix(pos)):
      L_pos_final.append(pos)
  return L_pos_final

def get_corrected_UD(ud,pos,bw):
  correct_ud=ud
  #case 1: handling demonstrative pronouns, assigning PRON instead of DET
  if pos=="pron_dem":
    return ud.replace("DET","PRON")
  #case 2: بأن, assign ADP+SCONJ instead of SCONJ+PRON
  if bw=="بِ/PREP+أَنَّ/SUB_CONJ":
    return "ADP+SCONJ"
  #case 3: س future particle following CCONJ, assigning AUX instead of ADP
  if "FUT_PART" in bw:
    bw_pos_list=get_bw_POS_list(bw)
    ud_list=ud.split("+")
    if len(bw_pos_list)==len(ud_list):
      L_correct_ud=[]
      for i in range(len(bw_pos_list)):
        if bw_pos_list[i]=="FUT_PART" and ud_list[i]=="ADP":
          L_correct_ud.append("AUX")
        else:
          L_correct_ud.append(ud_list[i])
      correct_ud="+".join(L_correct_ud)
      if correct_ud!=ud:
        return correct_ud
  return ud

def get_dict_annotations(prev_annotations):
  dict_prev_annotations={}
  for line in prev_annotations[2:]:
    if not line.startswith("<TRANSCRIPTION>"):
      line_info=line.strip().split("\t")
      token_id=line_info[0]
      token=line_info[1]
      tok=line_info[2]
      pos=line_info[3]
      lemma=line_info[4]    
      dict_prev_annotations[token_id]={"token":token,"tok":tok,"pos":pos,"lemma":lemma}
  return dict_prev_annotations

def insert_plus_around_redact_tokens(text):
    # add + before <tag> if preceded by a non-space character
    text = re.sub(r'(\S)(<\w+>)', r'\1+\2', text)
    # add + after <tag> if followed by a non-space character
    text = re.sub(r'(<\w+>)(\S)', r'\1+\2', text)
    return text

if __name__ == "__main__":

  parser = argparse.ArgumentParser()
  parser.add_argument("--input_transcriptions", help="Input data file path.")
  parser.add_argument("--prev_morph_annotations", help="Previous annotation file from ZAEBUC-Spoken release.")
  parser.add_argument("--output", help="Output annotation file.")

  args = parser.parse_args()
  input_lines=open(args.input_transcriptions).readlines()
  prev_annotations=open(args.prev_morph_annotations).readlines()
  output_file=open(args.output,'w')

  count_corrected=0
  dict_prev_annot=get_dict_annotations(prev_annotations)
  
  output_file.write("\t".join(["Token_id","Word","Tok","POS","Lemma"])+"\n")
  line_count=1
  for line in input_lines:
    line_info=line.strip().split("\t")
    utt_id=line_info[0]
    words=line_info[1].split()
    disambig_MSA = unfactored_msa.disambiguate(words)
    atb_MSA_all = [d.analyses[0].analysis['atbtok'] for d in disambig_MSA]
    lemmas_MSA_all = [d.analyses[0].analysis['lex'] for d in disambig_MSA]
    glosses_MSA_all = [d.analyses[0].analysis['gloss'] for d in disambig_MSA]
    pos_ud_MSA_all = [d.analyses[0].analysis['ud'] for d in disambig_MSA]
    pos_mada_MSA_all = [d.analyses[0].analysis['pos'] for d in disambig_MSA]
    bw_MSA_all = [d.analyses[0].analysis['bw'] for d in disambig_MSA]

    disambig_GULF = unfactored_glf.disambiguate(words)
    lemmas_GULF_all = [d.analyses[0].analysis.get('lex',"-no lex-") for d in disambig_GULF]
    glosses_GULF_all = [d.analyses[0].analysis.get('gloss',"-no gloss-") for d in disambig_GULF]
    bw_GULF_all = [d.analyses[0].analysis.get('bw',"-no bw-") for d in disambig_GULF]

    for i in range (0,len(words)):
        word=words[i]
        token_id=utt_id+"_token"+str(i+1)
        if stringContainsArabic(word) and not stringContainsEnglish(word) and "%" not in word and "--" not in word and not "<" in dict_prev_annot[token_id]["token"]:#to avoid English, MCS words, and puntuation/annotations
          ud_pos_MSA=pos_ud_MSA_all[i]
          lem_MSA=lemmas_MSA_all[i]
          atbtok_MSA=dediac_ar(atb_MSA_all[i].replace("_",""))  
          ud_pos_MSA_corrected=get_corrected_UD(ud_pos_MSA,pos_mada_MSA_all[i],bw_MSA_all[i])
          if not utt_id.startswith("speakerS"): #MSA speaker
            line_output=[utt_id+"_token"+str(i+1),word,atbtok_MSA,ud_pos_MSA_corrected,lem_MSA,glosses_MSA_all[i]]
          else: #DA speaker
            lem_GULF=lemmas_GULF_all[i]
            bw_tok_GULF=bw_GULF_all[i]
            atbtok_GULF,ud_pos_GULF="",""
            if lem_GULF=="-no lex-": lem_GULF=lem_MSA
            if bw_tok_GULF!="-no bw-": #use the bw analysis from Gulf disambiguator to get UD pos tags
              atbtok_GULF,ud_pos_GULF=get_best_guess_from_bw(bw_tok_GULF)
            else: #back off to MSA disambiguator UD pos tags
              atbtok_GULF,ud_pos_GULF=atbtok_MSA,ud_pos_MSA_corrected
            line_output=[utt_id+"_token"+str(i+1),word,atbtok_GULF,ud_pos_GULF,lem_GULF,glosses_GULF_all[i]]
        elif ("<Student" in dict_prev_annot[token_id]["token"] or "<Moderator" in dict_prev_annot[token_id]["token"] 
              or "<Interlocutor" in dict_prev_annot[token_id]["token"]) and dict_prev_annot[token_id]["pos"]!="<PW>":
          tok_redact=insert_plus_around_redact_tokens(dict_prev_annot[token_id]["token"])
          lemma_redact=re.search(r'<\w+>', dict_prev_annot[token_id]["token"]).group(0)
          line_output=[utt_id+"_token"+str(i+1),dict_prev_annot[token_id]["token"],tok_redact,dict_prev_annot[token_id]["pos"],lemma_redact]
        else:
          line_output=[utt_id+"_token"+str(i+1),dict_prev_annot[token_id]["token"],dict_prev_annot[token_id]["tok"],dict_prev_annot[token_id]["pos"],dict_prev_annot[token_id]["lemma"]]
        output_file.write("\t".join(line_output)+"\n")
    print("Progress: "+str(line_count)+"/"+str(len(input_lines)))
    line_count+=1
  print("count_corrected",count_corrected)
