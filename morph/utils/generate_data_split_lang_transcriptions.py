import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_data", help="Input data annotation file path.")
    parser.add_argument("--lang_ids", help="File containing the language ID for each utterance.")
    parser.add_argument("--lang", help="language for data split.")
    parser.add_argument("--output_file", default=None, help="Output file path.")

    args = parser.parse_args()

    input_file_lines=open(args.input_data).readlines()
    lang_file_lines=open(args.lang_ids).readlines()
    output_file=open(args.output_file,'w')

    map_utt_lang={}
    for lang_file_line in lang_file_lines:
        utt_id,lang_id=lang_file_line.strip().split()
        map_utt_lang[utt_id]=lang_id

    for  line in input_file_lines:
        utt_id=line.split("\t")[0]
        utt_lang=map_utt_lang[utt_id]
        if utt_lang==args.lang:
            output_file.write(line)