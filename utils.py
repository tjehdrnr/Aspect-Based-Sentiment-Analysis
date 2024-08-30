import yaml
import json
import torch
from sklearn.metrics import recall_score, precision_score, f1_score


def get_config(base_model):
    config_fn = "config.yaml"
    with open(config_fn, 'r') as f:
        config = yaml.safe_load(f)
    
    if base_model not in config["models"]:
        raise ValueError(f"Unknown base model name: {base_model}")
    
    model_config = config["common"].copy()
    model_config.update(config["models"][base_model])

    return model_config


def read_jsonl(dataset_path):
    with open(dataset_path, 'r', encoding='utf-8') as file:
        samples = [json.loads(line) for line in file]
    
    return samples


def compute_metrics(eval_logits, eval_labels):
    
    eval_preds = torch.argmax(eval_logits, dim=-1)
    correct_cnt = torch.sum(eval_preds == eval_labels)

    accuracy = correct_cnt.item() / len(eval_labels)

    eval_preds = eval_preds.cpu().numpy()
    eval_labels = eval_labels.cpu().numpy()

    precision_macro = precision_score(eval_labels, eval_preds, average='macro')
    precision_micro = precision_score(eval_labels, eval_preds, average='micro')

    recall_macro = recall_score(eval_labels, eval_preds, average='macro')
    recall_micro = recall_score(eval_labels, eval_preds, average='micro')

    f1 = f1_score(eval_labels, eval_preds, average=None)
    f1_macro = f1_score(eval_labels, eval_preds, average='macro')
    f1_micro = f1_score(eval_labels, eval_preds, average='micro')

    return {
        'accuracy': accuracy,
        'precision_macro': precision_macro,
        'precision_micro': precision_micro,
        'recall_macro': recall_macro,
        'recall_micro': recall_micro,
        'f1': f1,
        'f1_macro': f1_macro,
        'f1_micro': f1_micro
    }


