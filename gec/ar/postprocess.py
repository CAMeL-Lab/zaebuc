from camel_tools.utils.charsets import UNICODE_PUNCT_SYMBOL_CHARSET
import string
import re
import argparse
from camel_tools.utils.dediac import dediac_ar

puncs = string.punctuation + ''.join(list(UNICODE_PUNCT_SYMBOL_CHARSET)) + '&amp;'


def read_data(path):
    with open(path) as f:
        return [x.strip() for x in f.readlines()]

def write_data(path, data):
    with open(path, mode='w') as f:
        for line in data:
            f.write(line)
            f.write('\n')

def pnx_tokenize(data):
    pnx_re = re.compile(r'([' + re.escape(puncs) + '])(?!\d)')
    space_re = re.compile(' +')

    pnx_tokenized = []
    for line in data:
        line = line.strip()
        line = pnx_re.sub(r' \1 ', line)
        line = space_re.sub(' ', line)
        line = line.strip()
        line = dediac_ar(line)
        pnx_tokenized.append(line.strip())

    return pnx_tokenized


parser = argparse.ArgumentParser()
parser.add_argument('--input_file')
parser.add_argument('--output_file')
args = parser.parse_args()

data = read_data(args.input_file)
clean_data = pnx_tokenize(data)
write_data(args.output_file, clean_data)
