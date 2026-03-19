from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    classification_report,
    cohen_kappa_score
)
from argparse import ArgumentParser
import pandas as pd



def compute_metrics(preds, gold):
    target_names = sorted(list(set(gold) | set(preds)))

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

def load_preds(path):
    with open(path) as f:
        return [x.strip() for x in f.readlines()]

def load_gold(file_path):
    examples = {'avg_label': [], 'multi_label': []}

    with open(file_path) as f:
        for line in f.readlines():
            label_1, label_2, label_3, avg_label = line.strip().split('\t')
            examples['avg_label'].append(avg_label)
            examples['multi_label'].append([label_1, label_2, label_3])
    return examples


def write_metrics(metrics, path):
    with open(path, "w") as writer:
        for metric in metrics:
            if metric != 'report':
                writer.write(f'{metric}\t{metrics[metric]}')
                writer.write('\n')

        report = metrics['report']

        df = pd.DataFrame(report).transpose()
        df.to_csv(path, mode='a', sep="\t")


def write_metrics_mutli_ref(multi_ref_metrics, path):
    with open(path, "w") as writer:
        with open(path, "w") as writer:
            for metric in multi_ref_metrics:
                writer.write(f'{metric}\t{multi_ref_metrics[metric]}')
                writer.write('\n')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--pred_file')
    parser.add_argument('--gold_file')
    args = parser.parse_args()

    preds = load_preds(args.pred_file)
    gold = load_gold(args.gold_file)

    metrics = compute_metrics(preds, gold['avg_label'])
    write_metrics(metrics, f'{args.pred_file}.metrics.txt.check')

    multi_ref_metrics = compute_metrics_mulit_ref(preds, gold_multi=gold['multi_label'], gold_single=gold['avg_label'])
    write_metrics_mutli_ref(multi_ref_metrics, f'{args.pred_file}.metrics.multi_ref.txt.check')