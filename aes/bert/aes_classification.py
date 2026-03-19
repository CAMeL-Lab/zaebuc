import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoConfig,
    AutoTokenizer,
    DataCollatorWithPadding,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)
from datasets import Dataset
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    classification_report,
    cohen_kappa_score
)
import pandas as pd
logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are
    going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from "
                          "huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if "
                                        "not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if "
                                        "not the same as model_name"}
    )
    use_fast: bool = field(default=False, metadata={"help": "Set this flag to "
                                                            "use fast "
                                                            "tokenization."})

    # If you want to tweak more attributes on your tokenizer, you should do it
    # in a distinct script, or just modify its tokenizer_config.json.

    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the "
                                        "pretrained models downloaded from s3"}
    )

    add_class_weights: bool = field(
        default=False, metadata={"help": "Whether to weigh classes during "
                                        "training or not."}
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for
    training and eval.
    """

    data_dir: str = field(
        metadata={"help": "The input data dir. Should contain the .txt files "
                          "for a CoNLL-2003-formatted task."}
    )
    labels_path: str = field(
        metadata={"help": "Path to labels"}
    )
    test_data_file: Optional[str] = field(
        default=None, metadata={"help": "Path to test file"}
    )
    pred_mode: Optional[str] = field(
        default=None, metadata={"help": "Prediction mode to get the actual "
                                        "token predictions on dev or test."}
    )
    pred_output_file: Optional[str] = field(
        default=None, metadata={"help": "Predictions output file."}
    )


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments,
                               DataTrainingArguments,
                               TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a
        # json file, let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(
                                                    json_file=os.path.abspath(
                                                                 sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists "
            "and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=(logging.INFO if training_args.local_rank in [-1, 0]
               else logging.WARN),
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, "
        "16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)

    set_seed(training_args.seed)

    datasets = load_data(data_args.data_dir)
    # datasets = load_data_merged(data_args.data_dir)
    
    labels = get_labels(data_args.labels_path)


    label_map: Dict[int, str] = {i: label for i, label in enumerate(labels)}
    label2id: Dict[str, int] = {label: i for i, label in enumerate(labels)}
    num_labels = len(labels)

    config = AutoConfig.from_pretrained(
        (model_args.config_name if model_args.config_name
            else model_args.model_name_or_path),
        num_labels=num_labels,
        id2label=label_map,
        label2id=label2id,
        cache_dir=model_args.cache_dir,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        (model_args.tokenizer_name if model_args.tokenizer_name
            else model_args.model_name_or_path),
        cache_dir=model_args.cache_dir,
        use_fast=True,
        model_max_length=512
    )

    if training_args.do_train:
        train_dataset = datasets['train'].map(process,
                    fn_kwargs={"label_map": label2id, "tokenizer": tokenizer},
                    batched=True,
                    load_from_cache_file=False,
                    desc="Running tokenizer on train dataset"
                    ).remove_columns(['text', 'labels'])

                    # ).remove_columns(['text', 'labels', 'refs'])

    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True)

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        data_collator=data_collator
    )

    # Training
    if training_args.do_train:
        trainer.train(
            model_path=(model_args.model_name_or_path
                        if os.path.isdir(model_args.model_name_or_path)
                        else None)
        )
        trainer.save_model()
        # For convenience, we also re-save the tokenizer to the same directory,
        # so that you can share your model easily on huggingface.co/models =)
        if trainer.is_world_process_zero():
            tokenizer.save_pretrained(training_args.output_dir)


    # Predict
    if training_args.do_predict:
        pred_mode = data_args.pred_mode
        test_dataset = load_data_file(data_args.test_data_file)
        # test_dataset = load_data_file_merged(data_args.test_data_file)

        test_dataset = test_dataset.map(process,
                    fn_kwargs={"label_map": None, "tokenizer": tokenizer},
                    batched=True,
                    load_from_cache_file=False,
                    desc=f"Running tokenizer on {pred_mode} dataset"
                    )
        multi_reference_labels = test_dataset['refs']
        predictions, label_ids, _ = trainer.predict(test_dataset.remove_columns(['text', 'labels', 'refs']))

        predictions, label_ids, _ = trainer.predict(test_dataset.remove_columns(['text', 'labels']))

        preds = np.argmax(predictions, axis=1)
        preds = [label_map[label] for label in preds]

        metrics = compute_metrics(preds, test_dataset['labels'], label2id.keys())

        multi_ref_metrics = compute_metrics_mulit_ref(preds, gold_multi=multi_reference_labels,
                                                      gold_single=test_dataset['labels'])


        output_preds_file = os.path.join(training_args.output_dir,
                                        pred_mode+"_predictions.txt")
        metrics_file = os.path.join(training_args.output_dir,
                                    pred_mode+"_metrics.txt")
        metrics_file_multi = os.path.join(training_args.output_dir,
                                          pred_mode+"_metrics.mutli_ref.txt")

        if trainer.is_world_process_zero():
            with open(output_preds_file, "w") as writer:
                for label in preds:
                    writer.write(label)
                    writer.write('\n')

            # writing metrics
            with open(metrics_file, "w") as writer:
                for metric in metrics:
                    if metric != 'report':
                        writer.write(f'{metric}\t{metrics[metric]}')
                        writer.write('\n')

            report = metrics['report']

            df = pd.DataFrame(report).transpose()
            df.to_csv(metrics_file, mode='a', sep="\t")

            with open(metrics_file_multi, "w") as writer:
                for metric in multi_ref_metrics:
                    writer.write(f'{metric}\t{multi_ref_metrics[metric]}')
                    writer.write('\n')


def load_data_merged(data_dir):
    data = dict()

    for split in ['train', 'dev', 'test']:
        examples = {'labels': [], 'text': []}

        if os.path.exists(os.path.join(data_dir, f'{split}.txt')):
            with open(os.path.join(data_dir, f'{split}.txt')) as f:
                for line in f.readlines():
                    text, label = line.strip().split('\t')

                     # we ignore Unassesseble training examples
                    if split == 'train' and label == 'Unassessable':
                        continue

                    examples['text'].append(text.strip())
                    examples['labels'].append(label)

            data[split] = examples

    for split in data:
        data[split] = Dataset.from_dict(data[split])

    return data




def load_data(data_dir):
    data = dict()

    for split in ['train', 'dev', 'test']:
        examples = {'labels': [], 'refs': [], 'text': []}

        if os.path.exists(os.path.join(data_dir, f'{split}.txt')):
            with open(os.path.join(data_dir, f'{split}.txt')) as f:
                for line in f.readlines():
                    text, label_1, label_2, label_3, label_avg = line.strip().split('\t')

                    # we ignore Unassesseble training examples
                    if split == 'train' and label_avg == 'Unassessable':
                        continue

                    examples['text'].append(text.strip())
                    examples['refs'].append([label_1, label_2, label_3])
                    examples['labels'].append(label_avg.strip())

            data[split] = examples

    for split in data:
        data[split] = Dataset.from_dict(data[split])

    return data


def load_data_file_merged(file_path):
    examples = {'labels': [], 'text': []}

    with open(file_path) as f:
        for line in f.readlines():
            text, label = line.strip().split('\t')
            # if label_avg == 'Unassessable':
            #     continue
            examples['text'].append(text.strip())
            examples['labels'].append(label.strip())

    return Dataset.from_dict(examples)


def load_data_file(file_path):
    examples = {'labels': [], 'refs': [], 'text': []}

    with open(file_path) as f:
        for line in f.readlines():
            text, label_1, label_2, label_3, label_avg = line.strip().split('\t')
            # if label_avg == 'Unassessable':
            #     continue
            examples['text'].append(text.strip())
            examples['refs'].append([label_1, label_2, label_3])
            examples['labels'].append(label_avg.strip())

    return Dataset.from_dict(examples)


def get_labels(labels_path):
    with open(labels_path) as f:
        return [x.strip() for x in f.readlines()]


def process(examples, label_map, tokenizer):
    tokenized_inputs = tokenizer(examples["text"], truncation=True)

    if label_map is not None:
        label_ids = [label_map[label] for label in examples["labels"]]
        tokenized_inputs["label"] = label_ids
    
    return tokenized_inputs


def compute_metrics(preds, gold, target_names):
    target_names = sorted(list(set(gold)))

    accuracy = accuracy_score(y_true=gold, y_pred=preds)
    f1_macro = f1_score(y_true=gold, y_pred=preds, average='macro')
    f1_micro = f1_score(y_true=gold, y_pred=preds, average='micro')

    p_macro = precision_score(y_true=gold, y_pred=preds, average='macro')
    p_micro = precision_score(y_true=gold, y_pred=preds, average='micro')

    r_macro = recall_score(y_true=gold, y_pred=preds, average='macro')
    r_micro = recall_score(y_true=gold, y_pred=preds, average='micro')

    report = classification_report(y_true=gold, y_pred=preds,
                                   target_names=target_names,
                                   zero_division=0,
                                   output_dict=True)

    qwk = cohen_kappa_score(gold, preds, weights="quadratic", labels=target_names)

    return {'qwk': qwk,
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'p_macro': p_macro,
            'r_macro': r_macro,
            'f1_micro': f1_micro,
            'p_micro': p_micro,
            'r_micro': r_micro,
            'report': report
            }

def compute_metrics_mulit_ref(preds, gold_multi, gold_single):
    assert len(preds) == len(gold_multi) == len(gold_single)
    gold_ = []

    for p, g_multi, g_avg in zip(preds, gold_multi, gold_single):
        if p in g_multi:
            gold_.append(p)
        else:
            gold_.append(g_avg)

    assert len(gold_) == len(preds)

    accuracy = accuracy_score(y_true=gold_, y_pred=preds)
    f1_macro = f1_score(y_true=gold_, y_pred=preds, average='macro')
    f1_micro = f1_score(y_true=gold_, y_pred=preds, average='micro')

    p_macro = precision_score(y_true=gold_, y_pred=preds, average='macro')
    p_micro = precision_score(y_true=gold_, y_pred=preds, average='micro')

    r_macro = recall_score(y_true=gold_, y_pred=preds, average='macro')
    r_micro = recall_score(y_true=gold_, y_pred=preds, average='micro')

    qwk = cohen_kappa_score(gold_, preds, weights="quadratic")


    return {'qwk': qwk,
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'p_macro': p_macro,
            'r_macro': r_macro,
            'f1_micro': f1_micro,
            'p_micro': p_micro,
            'r_micro': r_micro
            }


if __name__ == "__main__":
    main()
