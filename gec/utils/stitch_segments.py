import argparse
from collections import defaultdict


def read_data(path):
    with open(path) as f:
        return [x.strip() for x in f.readlines()]

def stitch_data(ids, sents, path):
    assert len(ids) == len(sents)
    data = defaultdict(list)

    for id, sent in zip(ids, sents):
        data[id].append(sent)

    with open(path, mode='w') as f:
        for id in data:
            sents = ' '.join(data[id])
            f.write(sents)
            f.write('\n')


parser = argparse.ArgumentParser()
parser.add_argument('--input_file')
parser.add_argument('--ids')
parser.add_argument('--output_file')
args = parser.parse_args()

sents = read_data(args.input_file)
ids = read_data(args.ids)

stitch_data(ids, sents, args.output_file)
