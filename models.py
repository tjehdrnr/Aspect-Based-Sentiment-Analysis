import torch
from torch import nn
from transformers import XLMRobertaModel


class RobertaBaseClassifier(nn.Module):
    
    def __init__(self, config, n_category_classes, n_sentiment_classes, len_tokenizer):
        super().__init__()

        self.model = XLMRobertaModel.from_pretrained(config['model'])
        self.model.resize_token_embeddings(len_tokenizer)
        self.category_classifier = nn.Linear(config['hidden_size'], n_category_classes)
        self.sentiment_classifier = nn.Linear(config['hidden_size'], n_sentiment_classes)


    def forward(self, input_ids, attention_mask, labels: dict = None):

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=None,
        )

        cls_output = outputs.last_hidden_state[:, 0, :]

        category_logits = self.category_classifier(cls_output) # (batch_size, 1, 23)
        sentiment_logits = self.sentiment_classifier(cls_output)

        return category_logits, sentiment_logits


