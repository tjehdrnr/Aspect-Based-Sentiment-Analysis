from utils import get_config
from dataset import get_datasets
from models import RobertaBaseClassifier
from transformers import AutoTokenizer
from transformers.training_args import TrainingArguments



def main():
    config = get_config('roberta')

    datasets, idx_to_category, idx_to_sentiment = get_datasets(config)
    n_category_classes = len(idx_to_category)
    n_sentiment_classes = len(idx_to_sentiment)

    special_tokens_dict = {
    'additional_special_tokens': ['&name&', '&affiliation&', '&social-security-num&', '&tel-num&', '&card-num&', '&bank-account&', '&num&', '&online-account&']
    }
    
    tokenizer = AutoTokenizer.from_pretrained(config['model'])
    tokenizer.add_special_tokens(special_tokens_dict)
    len_tokenizer = len(tokenizer)

    model = RobertaBaseClassifier(config, n_category_classes, n_sentiment_classes, len_tokenizer)
    
    training_args = TrainingArguments(

    )
    # print(datasets)    




if __name__ == "__main__":
    main()