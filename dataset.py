from utils import read_jsonl
from transformers import PreTrainedTokenizerBase
from typing import Optional, Any, Union
from datasets import Dataset, DatasetDict
from dataclasses import dataclass


seed = 42

@dataclass
class CustomDataCollator:
    tokenizer: PreTrainedTokenizerBase
    model: Optional[Any] = None
    padding: bool = True
    truncation: bool = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"


    def __call__(self, features):
        encoding = self.tokenizer(
            samples,
            padding=self.padding,
            truncation=self.truncation,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensor=self.return_tensors,
        )


def extract_data(dataset):
    texts, categories, sentiments = [], [], []
    for sample in dataset:
        texts.append(sample['sentence_form'])
        categories.append(sample['annotation'][0][0])
        sentiments.append(sample['annotation'][0][-1])
    
    return texts, categories, sentiments


def label_to_index(train_labels, valid_labels):
    unique_labels = sorted(set(train_labels + valid_labels))

    label_to_index, index_to_label = {}, {}
    for i, label in enumerate(unique_labels):
        label_to_index[label] = i
        index_to_label[i] = label
    
    return label_to_index, index_to_label


def get_datasets(config):
    train_fn = config['train_fn']
    valid_fn = config['valid_fn']

    train_dataset = read_jsonl(train_fn)
    valid_dataset = read_jsonl(valid_fn)

    train_texts, train_categories, train_sentiments = extract_data(train_dataset)
    valid_texts, valid_categories, valid_sentiments = extract_data(valid_dataset)
    
    category_to_index, index_to_category = label_to_index(train_categories, valid_categories)
    sentiment_to_index, index_to_sentiment = label_to_index(train_sentiments, valid_sentiments)

    # Convert label texts to integer value.
    train_categories = list(map(category_to_index.get, train_categories))
    train_sentiments = list(map(sentiment_to_index.get, train_sentiments))

    valid_categories = list(map(category_to_index.get, valid_categories))
    valid_sentiments = list(map(sentiment_to_index.get, valid_sentiments))

    train_dataset = Dataset.from_dict(
        {
            "text": train_texts,
            "category": train_categories,
            "sentiment": train_sentiments,
        }
    ).shuffle(seed=seed)
    valid_dataset = Dataset.from_dict(
        {
            "text": valid_texts,
            "category": valid_categories,
            "sentiment": valid_sentiments,
        }
    ).shuffle(seed=seed)

    datasets = DatasetDict(
        {
            'train': train_dataset,
            'valid': valid_dataset,
        }
    )

    return datasets, index_to_category, index_to_sentiment

