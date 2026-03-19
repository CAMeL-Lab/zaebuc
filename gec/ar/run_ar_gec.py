from transformers import AutoTokenizer, BertForTokenClassification, MBartForConditionalGeneration
from camel_tools.disambig.bert import BERTUnfactoredDisambiguator
from camel_tools.morphology.analyzer import Analyzer
from camel_tools.morphology.database import MorphologyDB
from camel_tools.utils.dediac import dediac_ar
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from datasets import Dataset
from transformers import DataCollatorForTokenClassification, DataCollatorForSeq2SeqGEC
import torch.nn as nn
import argparse
import numpy as np


def read_data(src_path):
    data = {'src': []}
    with open(src_path) as f1:
        for line in f1.readlines():
            data['src'].append(line.strip())
    return Dataset.from_dict(data)


def run_disambig(data):
    db = MorphologyDB('/scratch/ba63/backup/gender-rewriting/models/calima-msa-s31_0.4.2.db')
    analyzer = Analyzer(db)
    bert_disambig = BERTUnfactoredDisambiguator.pretrained()
    bert_disambig._analyzer = analyzer

    texts = [text.split() for text in data['src']]

    # morph processing the input text
    text_disambigs = bert_disambig.tag_sentences(texts)

    morph_pp_texts = []

    for text_disambig in text_disambigs:
        morph_pp_text = [dediac_ar(w_disambig['diac']) for w_disambig in text_disambig]
        morph_pp_text = ' '.join(morph_pp_text)
        morph_pp_texts.append(morph_pp_text)
    
    return Dataset.from_dict({'src': morph_pp_texts})


class TokenClassificationDataset(torch.utils.data.Dataset):
    """A wrapper class for prediction dataset."""
    def __init__(self, examples, label_map, tokenizer):
        self.tokenizer = tokenizer
        self.features = self.process_examples(examples, label_map,
                                             pad_token_label_id=-100)

    def process_examples(self, examples, label_map, pad_token_label_id=-100):
        texts = [ex.split() for ex in examples['src']]

        featurized_inputs = []

        for ex_id, text in enumerate(texts):
            tokens = []
            label_ids = []

            for word in text:
                word_tokens = self.tokenizer.tokenize(word)

                if len(word_tokens) > 0:
                    tokens.append(word_tokens)
                    # assign a 'UC' (place holder) for the first 
                    # subword and -100 to the remaining subwords
                    label_ids.append([label_map['UC']] +
                                    [pad_token_label_id] *
                                    (len(word_tokens) - 1))

            token_segments = []
            token_segment = []
            label_ids_segments = []
            label_ids_segment = []
            num_word_pieces = 0
            seg_seq_length = self.tokenizer.model_max_length - 2

            for idx, word_pieces in enumerate(tokens):
                if num_word_pieces + len(word_pieces) > seg_seq_length:
                    # convert to ids and add special tokens

                    input_ids = self.tokenizer.convert_tokens_to_ids(token_segment)
                    input_ids = [self.tokenizer.cls_token_id] + input_ids + [self.tokenizer.sep_token_id]

                    label_ids_segment = [pad_token_label_id] + label_ids_segment + [pad_token_label_id]


                    features = {'input_ids': input_ids,
                                'attention_mask': [1] * len(input_ids),
                                'token_type_ids': [0] * len(input_ids),
                                'labels': label_ids_segment,
                                'sent_id': ex_id
                                }

                    featurized_inputs.append(features)

                    token_segments.append(token_segment)
                    label_ids_segments.append(label_ids_segment)
                    token_segment = list(word_pieces)
                    label_ids_segment = list(label_ids[idx])
                    num_word_pieces = len(word_pieces)
                else:
                    token_segment.extend(word_pieces)
                    label_ids_segment.extend(label_ids[idx])
                    num_word_pieces += len(word_pieces)

            if len(token_segment) > 0:
                input_ids = self.tokenizer.convert_tokens_to_ids(token_segment)
                input_ids = [self.tokenizer.cls_token_id] + input_ids + [self.tokenizer.sep_token_id]


                label_ids_segment = [pad_token_label_id] + label_ids_segment + [pad_token_label_id]

                features = {'input_ids': input_ids,
                            'attention_mask': [1] * len(input_ids),
                            'token_type_ids': [0] * len(input_ids),
                            'labels': label_ids_segment,
                            'sent_id': ex_id
                            }

                featurized_inputs.append(features)

                token_segments.append(token_segment)
                label_ids_segments.append(label_ids_segment)

        return featurized_inputs

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx]


def run_ged(model, tokenizer, dataset):
    ged_dataset = TokenClassificationDataset(examples=dataset,
                                            label_map=model.config.label2id,
                                            tokenizer=tokenizer)

    data_collator = DataCollatorForTokenClassification(tokenizer, padding=True)
    data_loader = DataLoader(ged_dataset, batch_size=16, shuffle=False,
                             drop_last=False,
                             collate_fn=data_collator)

    sent_ids = None
    model.eval()

    preds = []

    with torch.no_grad():
        for batch in data_loader:
            inputs = {'input_ids': batch['input_ids'],
                      'token_type_ids': batch['token_type_ids'],
                      'attention_mask': batch['attention_mask']}

            label_ids = batch['labels']
            sent_ids = (batch['sent_id'] if sent_ids is None
                        else torch.cat((sent_ids, batch['sent_id'])))

            logits = model(**inputs)[0]

            predictions = _align_predictions(logits.cpu().numpy(),
                                             label_ids.cpu().numpy(),
                                             model.config.id2label)

            preds.extend(predictions)

    # Collating the predicted labels based on the sentence ids
    sent_ids = sent_ids.cpu().numpy()
    final_preds_list = [[] for _ in range(len(set(sent_ids)))]
    for i, id in enumerate(sent_ids):
        final_preds_list[id].extend(preds[i])

    return final_preds_list


def _align_predictions(predictions, label_ids, label_map):
    preds = np.argmax(predictions, axis=2)
    batch_size, seq_len = preds.shape
    preds_list = [[] for _ in range(batch_size)]

    for i in range(batch_size):
        for j in range(seq_len):
            if label_ids[i, j] != nn.CrossEntropyLoss().ignore_index:
                preds_list[i].append(label_map[preds[i][j]])

    return preds_list

def resolve_merge_delete(words, labels):
    """
    Process words by solving merge and delete errors
    """

    new_words = []
    new_labels = []

    i = 0
    while i < len(words):
        word = words[i]
        label = labels[i]

        if label == 'MERGE-B':
            new_word = []
            new_word.append(word)
            i += 1

            while  i < len(labels) and 'MERGE-I' in labels[i]:
                new_word.append(words[i])
                i += 1

            new_word = ''.join(new_word)
            new_words.append(new_word)
            new_labels.append('UC')

        elif label == 'DELETE':
            i += 1
            continue

        else:
            new_words.append(word)
            new_labels.append(label)
            i += 1

    assert len(new_words) == len(new_labels)

    return new_words, new_labels



def process_gec(examples, ged_labels, tokenizer, label_map):
    texts = [text for text in examples['src']]
    assert len(texts) == len(ged_labels)

    features = {'input_ids': [], 'ged_tags': [], 'attention_mask': []}

    
    for i, (text, ex_labels) in enumerate(zip(texts, ged_labels)):
        words = text.split()

        assert len(words) == len(ex_labels)

        tokenized_text = []
        ex_ged_labels = []

        # resolving deletes and merges
        words, ex_labels = resolve_merge_delete(words, ex_labels)

        for word, ged_label in zip(words, ex_labels):
            tokenized_word = tokenizer.tokenize(word)
            tokenized_text.extend(tokenized_word)
            ex_ged_labels.extend([ged_label] * len(tokenized_word))


        input_ids = tokenizer.convert_tokens_to_ids(tokenized_text)
        # converting the labels to label ids
        # for labels that we have not seen in training, they get pad_id
        label_ids = [label_map.get(label, label_map['<pad>']) for label in ex_ged_labels]

        # Adding bos and eos tokens to the input ids
        input_ids = [tokenizer.bos_token_id] + input_ids + [tokenizer.eos_token_id]
        label_ids = [label_map['UC']] + label_ids + [label_map['UC']]

        assert len(label_ids) == len(input_ids)

        attention_mask = [1 for _ in range(len(input_ids))]

        features['input_ids'].append(input_ids)
        features['ged_tags'].append(label_ids)
        features['attention_mask'].append(attention_mask)
    
    
    return features


def run_gec(model, tokenizer, dataset, ged_preds):
    gec_dataset = dataset.map(process_gec, fn_kwargs={"tokenizer": tokenizer,
                                                      "ged_labels": ged_preds,
                                                      "label_map": model.config.ged_label2id}, batched=True, 
                            load_from_cache_file=False,
                            desc="Running GEC tokenizer on dataset").remove_columns(['src'])

    gec_collator = DataCollatorForSeq2SeqGEC(tokenizer=tokenizer, model=model,
                                            pad_to_multiple_of=None)

    data_loader = DataLoader(gec_dataset, batch_size=16, shuffle=False, collate_fn=gec_collator)

    model.eval()
    preds = []

    with torch.no_grad():
        for batch in data_loader:
            gen_kwargs = {'num_beams': 5, 'max_length': 1024, 'num_return_sequences': 1, 
                        'no_repeat_ngram_size': 0, 'early_stopping': False, 
                        'ged_tags': batch['ged_tags'], 
                        'attention_mask': batch['attention_mask']}

            # GEC generation
            generated = model.generate(batch['input_ids'], **gen_kwargs)

            generated_text = tokenizer.batch_decode(generated, skip_special_tokens=True,
                                                    clean_up_tokenization_spaces=False)
            preds.extend(generated_text)
    
    return preds


def write_data(data, path):
    with open(path, mode='w') as f:
        for example in data:
            f.write(example)
            f.write('\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file')
    parser.add_argument('--ged_model')
    parser.add_argument('--gec_model')
    parser.add_argument('--output_file')

    args = parser.parse_args()

    dataset = read_data(args.input_file)

    # running morph preproccesing
    print('Running morph dismabig...', flush=True)
    morph_pp_dataset = run_disambig(data=dataset)

    # running ged
    print('Running GED...', flush=True)
    ged_tokenizer = AutoTokenizer.from_pretrained(args.ged_model)
    ged_model = BertForTokenClassification.from_pretrained(args.ged_model)
    ged_preds = run_ged(model=ged_model, tokenizer=ged_tokenizer, dataset=morph_pp_dataset)

    # running gec
    print('Running GEC...', flush=True)
    gec_tokenizer = AutoTokenizer.from_pretrained(args.gec_model)
    gec_model = MBartForConditionalGeneration.from_pretrained(args.gec_model)
    gec_preds = run_gec(model=gec_model, tokenizer=gec_tokenizer, dataset=morph_pp_dataset,
                        ged_preds=ged_preds)

    print('Done!', flush=True)
    write_data(gec_preds, path=args.output_file)


if __name__ == '__main__':
    main()