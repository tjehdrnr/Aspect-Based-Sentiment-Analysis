import torch
import torch.nn as nn
from transformers import XLMRobertaModel


class RoBertaBaseClassifier(nn.Module):

    def __init__(self, config, num_labels, len_tokenizer):
        super(RoBertaBaseClassifier, self).__init__()

        self.config = config
        
        self.model = XLMRobertaModel.from_pretrained(config['base_model'])
        self.model.resize_token_embeddings(len_tokenizer)

        self.classifier = nn.Sequential(
            nn.Dropout(config["dropout_p"]),
            nn.Linear(config["hidden_size"], config["hidden_size"]),
            nn.Tanh(),
            nn.Dropout(config["dropout_p"]),
            nn.Linear(config["hidden_size"], num_labels)
        )
    
    def forward(self, input_ids, attention_mask):
        outputs = self.model(
            input_ids,
            attention_mask,
            token_type_ids=None
        )

        pooled_output = outputs.last_hidden_state[:, 0, :]
        # |pooled_output| = (batch_size, hidden_size)
        logits = self.classifier(pooled_output)
        # |logits| = (batch_size, num_labels)

        return logits



