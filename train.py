import torch
from utils import get_config
from dataset import get_datasets, get_loaders
from trainer import get_optimizer, get_scheduler, Trainer
from xlm_roberta import RoBertaBaseClassifier
from transformers import AutoTokenizer

import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


tf_to_index = {"False": 0, "True": 1}
index_to_tf = {i: tf for tf, i in tf_to_index.items()}

sentiment_to_index = {"positive": 0, "negative": 1, "neutral": 2}
index_to_sentiment = {i: sentiment for sentiment, i in sentiment_to_index.items()}

special_tokens_dict = {
'additional_special_tokens': [
    '&name&', '&affiliation&', '&social-security-num&', '&tel-num&', 
    '&card-num&', '&bank-account&', '&num&', '&online-account&'
    ]
}

def main(config):
    train_data_path = config['train_data_path']
    eval_data_path = config['eval_data_path']

    logger.info("Load tokenizer..")
    tokenizer = AutoTokenizer.from_pretrained(config["base_model"])
    tokenizer.add_special_tokens(special_tokens_dict)

    logger.info("Load datasets..")
    property_train_dataset, sentiment_train_dataset = get_datasets(train_data_path)
    property_eval_dataset, sentiment_eval_dataset = get_datasets(eval_data_path)
    
    logger.info(f"[Property] |Train|={len(property_train_dataset)}, |Eval|={len(property_eval_dataset)}")
    logger.info(f"[Sentiment] |Train|={len(sentiment_train_dataset)}, |Eval|={len(sentiment_eval_dataset)}")

    logger.info("Load dataloaders..")
    # Get train data loaders
    property_train_loader, sentiment_train_loader = get_loaders(
        property_train_dataset, sentiment_train_dataset, tokenizer, config
    )
    # Get evaluation data loaders
    property_eval_loader, sentiment_eval_loader = get_loaders(
        property_eval_dataset, sentiment_eval_dataset, tokenizer, config
    )

    logger.info("Load models..")
    property_model = RoBertaBaseClassifier(config, len(tf_to_index), len(tokenizer))
    sentiment_model = RoBertaBaseClassifier(config, len(sentiment_to_index), len(tokenizer))
    print(property_model)
    print(sentiment_model)

    criterion = torch.nn.CrossEntropyLoss()

    logger.info("Initialize optimizers and schedulers..")
    property_optimizer = get_optimizer(property_model, config)
    property_scheduler = get_scheduler(
        property_optimizer,
        len(property_train_loader),
        config
    )
    sentiment_optimizer = get_optimizer(sentiment_model, config)
    sentiment_scheduler = get_scheduler(
        sentiment_optimizer,
        len(sentiment_train_loader),
        config
    )

    logger.info("Define Trainers..")
    property_trainer = Trainer(
        property_model,
        criterion,
        property_optimizer,
        property_scheduler,
        config
    )
    sentiment_trainer = Trainer(
        sentiment_model,
        criterion,
        sentiment_optimizer,
        sentiment_scheduler,
        config
    )

    logger.info("Start Aspect Property Training..")
    property_trainer.train(
        property_train_loader,
        property_eval_loader,
        config,
        save_model_name="roberta_property"
    )

    logger.info("Start Sentiment Training..")
    sentiment_trainer.train(
        sentiment_train_loader,
        sentiment_eval_loader,
        config,
        save_model_name="roberta_sentiment"
    )



if __name__ == "__main__":
    config = get_config('roberta')
    main(config)