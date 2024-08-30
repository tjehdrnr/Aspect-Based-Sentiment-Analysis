from tqdm import tqdm
import torch
import os
import torch.optim as optim
from transformers import get_linear_schedule_with_warmup
from copy import deepcopy
from utils import compute_metrics

def get_optimizer(model, config):
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            'weight_decay': 0.01
        },
        {
            'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0
        }
    ]
    optimizer = optim.AdamW(
        optimizer_grouped_parameters,
        lr=config['learning_rate'],
        eps=config['adam_eps']
    )

    return optimizer


def get_scheduler(optimizer, num_batches, config):
    num_total_iterations = num_batches * config['num_epochs']
    num_warmup_steps = int(num_total_iterations * config['warmup_ratio'])

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps,
        num_total_iterations,
    )

    return scheduler



class Trainer:

    def __init__(self, model, criterion, optimizer, scheduler, config):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config

        self.device = torch.device("gpu" if torch.cuda.is_available() else "cpu")


    def _train(self, train_loader):
        # Set model to train mode
        self.model.train()

        epoch_loss = 0

        for batch in tqdm(train_loader):
            # Initialize gradient
            self.optimizer.zero_grad()

            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)

            logits = self.model(input_ids, attention_mask)

            loss = self.criterion(
                logits.view(-1, logits.size(-1)), labels.view(-1)
            )
            
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config['max_grad_norm']
            )

            self.optimizer.step()
            self.scheduler.step()

            epoch_loss += loss.item()
        
        return epoch_loss / len(train_loader)
    

    def _evaluate(self, eval_loader):
        # Set model to eval
        self.model.eval()

        epoch_loss = 0

        with torch.no_grad():
            for batch in tqdm(eval_loader):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                logits = self.model(
                    input_ids,
                    attention_mask,
                    token_type_ids=None,
                )

                loss = self.criterion(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1)
                )

                epoch_loss += loss.item()

                result = compute_metrics(logits, labels)
        
        eval_loss = epoch_loss / len(eval_loader)

        return eval_loss, result


    def train(self, train_loader, eval_loader, config, save_model_name):
        best_eval_loss = float('inf')
        best_model = None

        for epoch in range(self.config['num_epochs']):
            train_loss = self._train(train_loader)
            eval_loss, result = self._evaluate(eval_loader)

            print(f"Epoch {epoch + 1} | Train Loss: {train_loss:.3f} | Eval Loss: {eval_loss:.3f}")
            print(f"Eval Accuracy: {result['accuracy']:.3f} | F1-Score: {result['f1']:.3f}")
            print(f"[Macro] Precision: {result['precision_macro']:.3f} | Recall: {result['recall_macro']:.3f} | F1-Score: {result['f1_macro']:.3f}")
            print(f"[Micro] Precision: {result['precision_micro']:.3f} | Recall: {result['recall_micro']:.3f} | F1-Score: {result['f1_micro']:.3f}")

            # Save best eval loss model, if best_eval_loss is updated
            if eval_loss <= best_eval_loss:
                best_eval_loss = eval_loss
                best_model = deepcopy(self.model.state_dict())
                print(
                    f"The best_eval_loss({best_eval_loss:.3f}) has been updated to the {eval_loss:.3f}"
                )

        # Save the best model
        if not os.path.exists(config['output_dir']):
            os.makedirs(config['output_dir'])

        torch.save(
            best_model,
            os.path.join(config['output_dir'], f"{save_model_name}.pt")
        )