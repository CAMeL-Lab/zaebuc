import argparse
import pickle
import csv
import re
from camel_tools.disambig.bert import BERTUnfactoredDisambiguator
from camel_tools.morphology.database import MorphologyDB
from camel_tools.morphology.analyzer import Analyzer
from camel_tools.utils.charmap import CharMapper
bw2ar = CharMapper.builtin_mapper('bw2ar')
db_dir="/Users/injy.hamed/Documents/Tools/camel_tools/databases"
unfactored_msa = BERTUnfactoredDisambiguator.pretrained(model_name='msa')
db = MorphologyDB(db_dir+'/MSA/calima-msa-s31_0.4.2.utf8.db', 'a')

analyzer = Analyzer(db, 'ADD_PROP', cache_size=100000)
unfactored_msa._analyzer = analyzer
with open(db_dir+'/MSA/disambig_ranking_cache/calima-msa-s31/default_cache.pickle', 'rb') as f:
    unfactored_msa._ranking_cache = pickle.load(f)

def is_valid_suffix(bw_pos):
  conditions=[]
  conditions.append(bw_pos.startswith("IVSUFF_SUBJ"))
  conditions.append(bw_pos.startswith("CVSUFF_SUBJ"))
  conditions.append(bw_pos.startswith("CASE_"))
  conditions.append(bw_pos.startswith("IVSUFF_MOOD"))
  conditions.append(bool(re.match("(IV|PV|CV)SUFF_SUBJ*",bw_pos)))
  conditions.append(bool(re.match("NSUFF_(FEM_SG|FEM_DU|FEM_PL|MASC_DU|MASC_PL)*",bw_pos)))
  return any(conditions)

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

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--input_data", help="Input data file path.")
  parser.add_argument("--output", help="Output directory.")

  args = parser.parse_args()

  input_lines=open(args.input_data).readlines()
  output_file=open(args.output, 'w', newline='', encoding='utf-8')
  fieldnames = ['Doc_id', 'Word', 'Tok','POS','Lemma']
  #writer = csv.DictWriter(output_file, fieldnames=fieldnames, delimiter='\t')
  writer = csv.DictWriter(output_file, fieldnames=fieldnames, delimiter='\t', quotechar='|', quoting=csv.QUOTE_MINIMAL)
  writer.writeheader()

  for line in input_lines:
    [doc_id,text]=line.strip().split("\t")
    if doc_id.strip()!="AR-030-85828":
      words=text.split()
      disambig = unfactored_msa.disambiguate(words)
      ud_tags = [d.analyses[0].analysis.get('ud',"-no ud-") for d in disambig]
      lemmas = [d.analyses[0].analysis.get('lex',"-no lex-") for d in disambig]
      atb_toks = [d.analyses[0].analysis.get('atbtok',"-no atbtok-") for d in disambig]
      pos_tags = [d.analyses[0].analysis.get('pos',"-no pos-") for d in disambig]
      bw = [d.analyses[0].analysis.get('bw',"-no bw-") for d in disambig]
      for i in range (len(words)):
          word=words[i]
          POS_UD=ud_tags[i]
          LEM=lemmas[i]
          POS=pos_tags[i]
          BW=bw[i]
          POS_mapped=get_corrected_UD(POS_UD,POS,BW)
          ATBTOK=atb_toks[i]
          LEM_mapped=LEM.replace("ٱ","ا").replace("اً","ًا")
          writer.writerow({'Doc_id': doc_id, 'Word': word, 'Tok': ATBTOK, 'POS': POS_mapped, 'Lemma': LEM_mapped})
