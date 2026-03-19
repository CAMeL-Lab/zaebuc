import spacy
import argparse

nlp = spacy.load("en_core_web_sm")

parser = argparse.ArgumentParser()
parser.add_argument('--input_file')
parser.add_argument('--output_file')
args = parser.parse_args()

with open(args.input_file) as f1, open(args.output_file, mode='w') as f2:
 for line in f1.readlines():
     tokens = nlp(line.strip())
     tokenized_line = ' '.join([token.text for token in tokens])
     f2.write(tokenized_line)
     f2.write('\n')

