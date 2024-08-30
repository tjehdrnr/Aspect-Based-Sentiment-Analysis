
import torch
from typing import Dict, Optional, List
from torch.utils.data import DataLoader, Dataset
from utils import get_config, read_jsonl
from collections import OrderedDict

from transformers import AutoTokenizer


property_pair = [
    '제품 전체#품질', '패키지/구성품#디자인', '본품#일반', '제품 전체#편의성', '본품#다양성', '제품 전체#디자인',
    '패키지/구성품#가격', '본품#품질', '브랜드#인지도', '제품 전체#일반', '브랜드#일반', '패키지/구성품#다양성',
    '패키지/구성품#일반', '본품#인지도', '제품 전체#가격', '본품#편의성', '패키지/구성품#편의성', '본품#디자인',
    '브랜드#디자인', '본품#가격', '브랜드#품질', '제품 전체#인지도', '패키지/구성품#품질', '제품 전체#다양성', '브랜드#가격'
    ]

tf_to_index = {"False": 0, "True": 1}
index_to_tf = {i: tf for tf, i in tf_to_index.items()}

sentiment_to_index = {"positive": 0, "negative": 1, "neutral": 2}
index_to_sentiment = {i: sentiment for sentiment, i in sentiment_to_index.items()}


def get_datasets(dataset_path):

    samples = read_jsonl(dataset_path)

    property_texts, property_pairs, property_labels = [], [], []
    sentiment_texts, sentiment_pairs, sentiment_labels = [], [], []

    for sample in samples:
        text = sample['sentence_form']
        property = sample['annotation'][0][0]
        sentiment = sample['annotation'][0][-1]

        for pair in property_pair:
            if property == pair:
                property_texts.append(text)
                property_pairs.append(pair)
                property_labels.append(tf_to_index['True'])
                
                sentiment_texts.append(text)
                sentiment_pairs.append(pair)
                sentiment_labels.append(sentiment_to_index[sentiment])
            else:
                property_texts.append(text)
                property_pairs.append(pair)
                property_labels.append(tf_to_index['False'])
                
    property_dataset = AspectSentimentDataset(property_texts, property_labels, property_pairs)
    sentiment_dataset = AspectSentimentDataset(sentiment_texts, sentiment_labels, sentiment_pairs)

    return property_dataset, sentiment_dataset


def get_loaders(property_dataset, sentiment_dataset, tokenizer, config):

    property_loader = DataLoader(
        property_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        collate_fn=AspectSentimentDataCollator(
            tokenizer,
            max_length=config['max_length'],
            with_text=False
        )
    )
    sentiment_loader = DataLoader(
        sentiment_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        collate_fn=AspectSentimentDataCollator(
            tokenizer,
            max_length=config['max_length'],
            with_text=False
        )
    )

    return property_loader, sentiment_loader


class AspectSentimentDataCollator:

    def __init__(self, tokenizer, max_length, with_text=False):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.with_text = with_text

    def __call__(self, batch):
        texts, labels, pairs = zip(*batch)

        encodings = self.tokenizer(
            texts,
            pairs,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        outputs = {
            "input_ids": encodings["input_ids"],
            "attention_mask": encodings["attention_mask"],
            "labels": torch.tensor(labels, dtype=torch.long),
        }

        if self.with_text:
            outputs["texts"] = texts
        
        return outputs


class AspectSentimentDataset(Dataset):

    def __init__(self, texts, labels, pairs):
        self.texts = texts
        self.labels = labels
        self.pairs = pairs

    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        pair = self.pairs[idx]

        return text, label, pair
        
    


if __name__ == "__main__":
    config = get_config('roberta')
    train_data_path = config['train_data_path']

    property_train_dataset, sentiment_train_dataset = get_datasets(train_data_path)

    tokenizer = AutoTokenizer.from_pretrained(config['base_model'])
    collator = AspectSentimentDataCollator(tokenizer, max_length=config['max_length'])

    property_train_loader = DataLoader(
        property_train_dataset,
        batch_size=config['batch_size'],
        collate_fn=collator,
        shuffle=True,
    )

    for batch in property_train_loader:
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                print(key, value.shape)
                print(key, value)
            else:
                print(key, value)
        break
