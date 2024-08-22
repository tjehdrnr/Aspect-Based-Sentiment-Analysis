import json
import yaml


def read_jsonl(jsonl_fn):
    with open(jsonl_fn, 'r', encoding='utf-8') as file:
        samples = [json.loads(sample) for sample in file]
    
    return samples


def get_config(base_model_name):
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    
    if base_model_name not in config['models']:
        raise ValueError(f"Unknown base model: {base_model_name}")

    model_config = config['common'].copy()
    model_config.update(config['models'][base_model_name])
    
    return model_config