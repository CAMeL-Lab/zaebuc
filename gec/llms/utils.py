from camel_tools.utils.charsets import UNICODE_PUNCT_SYMBOL_CHARSET
from camel_tools.utils.dediac import dediac_ar

import string
import re

puncs = string.punctuation + ''.join(list(UNICODE_PUNCT_SYMBOL_CHARSET)) + '&amp;'

def pnx_tokenize_dediac(data, dediac=False):
    pnx_re = re.compile(r'([' + re.escape(puncs) + '])(?!\d)')
    space_re = re.compile(' +')

    pnx_tokenized = []
    for line in data:
        line = line.strip()
        line = pnx_re.sub(r' \1 ', line)
        line = space_re.sub(' ', line)
        line = line.strip()
        if dediac:
            line = dediac_ar(line)
        pnx_tokenized.append(line)

    return pnx_tokenized