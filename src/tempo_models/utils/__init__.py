import torch
import numpy as np
from transformers import Trainer, PreTrainedTokenizer, AutoTokenizer, EvalPrediction
from datasets import Dataset

import tqdm
import math
import inspect

class NonShuffledTrainer(Trainer):
    """A trainer that does not shuffle the batches, offering more flexibility in batch ordering."""
    def _get_train_sampler(self):
        return None

#############################
# TOKENIZER UTILS           #
#############################

def fetch_tokenizer(model: str, time_token: str, n_contexts: int) -> PreTrainedTokenizer:
    if model == "bert":
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    elif model == "t5":
        tokenizer = AutoTokenizer.from_pretrained('t5-base', model_max_length=512)
    
    if time_token == "special":
        special_tokens = [f"timestamp: {t} text: " for t in range(n_contexts)]
        tokenizer.add_tokens(special_tokens)
    
    return tokenizer

#############################
# TENSOR UTILS              #
#############################

def to_list(tensor_or_iterable):
    if isinstance(tensor_or_iterable, torch.Tensor):
        return tensor_or_iterable.tolist()
    return list(tensor_or_iterable)

def to_tensor(tensor_or_iterable):
    if isinstance(tensor_or_iterable, torch.Tensor):
        return tensor_or_iterable
    return torch.Tensor(tensor_or_iterable)

#############################
# DATASET UTILS             #
#############################

def remove_unused_columns(dataset: Dataset, model_arch: torch.nn.Module):
    """Removes from the dataset columns that are not used by the model's forward call."""
    signature = inspect.signature(model_arch.forward)
    signature_columns = list(signature.parameters.keys()) + ["label", "label_ids"]
    ignored_columns = list(set(dataset.column_names) - set(signature_columns))
    return dataset.remove_columns(ignored_columns)

def shuffle_batched(dataset, batch_size):
    """Shuffles a dataset while keeping batches intact."""

    def add_timestamp(examples):
        """Converts a datasets "timestamps" column into a singular "timestamp" column."""
        try:
            timestamps = torch.tensor(examples['timestamps'])
            examples['timestamp'] = timestamps[:, 0]
        except ValueError:
            examples['timestamp'] = [t[0] for t in examples['timestamps']]
        return examples
    
    dataset = dataset.map(add_timestamp, batched=True)
    dataset = dataset.sort('timestamp')
    dataset = dataset.remove_columns('timestamp')

    num_batches = len(dataset) // batch_size
    idxs = torch.randperm(num_batches) * batch_size
    idxs = idxs.reshape(-1, 1) + torch.arange(batch_size)
    idxs = torch.concat([idxs.flatten(), torch.arange(num_batches * batch_size, len(dataset))])
    return dataset.select(idxs)

def add_special_time_tokens(dataset, tokenizer_len, special_tokens):
    # Inserts special time tokens into the dataset and resizes tokenizer.
    def insert_special_token_batched(examples):
        for k in examples.keys():
            examples[k] = torch.tensor(examples[k])
        for k in examples.keys():
            if k == "input_ids":
                ids = examples['input_ids']
                ts = examples['timestamps']
                examples["input_ids"] = torch.hstack((ids[:, 0:1], (tokenizer_len + ts[:, 0:1]), ids[:, 1:]))
            elif len(examples[k].shape) > 1:
                examples[k] = torch.hstack((examples[k][:, 0:1], examples[k]))
        return examples
    
    def insert_special_token(example):
        for k in example.keys():
            if k == "input_ids":
                example['input_ids'] = (example['input_ids'][0:1] + [tokenizer_len + example['timestamps'][0]] + example['input_ids'][1:])
            else:
                try:
                    example[k] = example[k][0:1] + example[k]
                except TypeError:
                    pass
        return example
    
    if special_tokens == "string":
        raise NotImplementedError("Gotta do this!")
    elif special_tokens == "special":
        try:
            dataset = dataset.map(insert_special_token_batched, batched=True)
        except ValueError:
            dataset = dataset.map(insert_special_token)
    return dataset

#############################
# EVALUATION UTILS          #
#############################

def make_batch_iterator(dataset, batch_size=32, shuffle=False):
    num_examples = len(dataset)
    idxs = torch.arange(num_examples)
    if shuffle:
        idxs = idxs[torch.randperm(num_examples)]
    
    for start_index in range(0, num_examples, batch_size):
        idx = idxs[start_index: start_index + batch_size]
        yield [dataset[int(i)] for i in idx]

def evaluate_mlm(model, dataset, data_collator, 
             no_cuda=False, batch_size=16, pad_id=-100):
    """
    Compute token-level perplexity, accuracy, and MRR metrics.
    Note that the perplexity here is over subwords, not words.
    """
    if not no_cuda:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    model.eval().to(device)

    mean_cross_entropy = 0
    total_mrr = 0.0
    total_predictions = 0
    correct_predictions = 0
    total_ranks = torch.tensor([], dtype=int)
    num_iterations = math.ceil(len(dataset) / batch_size)

    with torch.no_grad():
        for data in tqdm.tqdm(make_batch_iterator(dataset, batch_size), total=num_iterations):
            input = data_collator(data).to(device)
            labels = input["labels"]

            out = model(**input)

            logits = out['logits']

            num_predictions = (labels != pad_id).sum().item()
            total_predictions += num_predictions
            pct_new = num_predictions / total_predictions

            mean_cross_entropy = (mean_cross_entropy * (1 - pct_new)) + out["loss"].item() * pct_new

            batch_correct_predictions = (
              (labels != pad_id) &
              (labels == logits.argmax(2))).sum().item()
            correct_predictions += batch_correct_predictions
            
            idx = torch.nonzero(labels != pad_id)
            labels = labels[idx[:, 0], idx[:, 1]].to(device) ## is a list of length n
            logits_masked = logits[idx[:, 0], idx[:, 1]].to(device) ## should now be of shape n x n_tokens
            logits_values = logits[idx[:, 0], idx[:, 1], labels] ## list of length n

            ranks = (logits_masked > logits_values.reshape(-1, 1)).sum(axis=1) + 1
            total_ranks = torch.cat((total_ranks, ranks.cpu()))
            total_mrr += (1/ranks).sum().item()
    
    perplexity = math.exp(mean_cross_entropy)
    accuracy = 100 * correct_predictions / total_predictions
    mrr = total_mrr / total_predictions
    return {
        'perplexity': perplexity, 
        'accuracy': accuracy, 
        'mrr': mrr,
    }

def evaluate_ssm(model, dataset, data_collator, 
             no_cuda=False, batch_size=16, pad_id=-100):
    """
    Compute token-level F1 metrics.
    """
    if not no_cuda:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    model.eval().to(device)

    total_f1 = 0
    total_predictions = 0
    correct_predictions = 0
    num_iterations = math.ceil(len(dataset) / batch_size)

    with torch.no_grad():
        for data in tqdm.tqdm(make_batch_iterator(dataset, batch_size), total=num_iterations):
            input = data_collator(data).to(device)
            labels = input["labels"]

            out = model(**input)
            logits = out['logits']
            idx_labels = torch.nonzero(labels != pad_id)
            labels = labels[idx[:, 0], idx[:, 1]].to(device) ## is a list of length n
            logits_masked = logits[idx[:, 0], idx[:, 1]].to(device) ## should now be of shape n x n_tokens
            logits_values = logits[idx[:, 0], idx[:, 1], labels] ## list of length n

            ranks = (logits_masked > logits_values.reshape(-1, 1)).sum(axis=1) + 1
            total_ranks = torch.cat((total_ranks, ranks.cpu()))
            total_mrr += (1/ranks).sum().item()
    
    accuracy = 100 * correct_predictions / total_predictions
    mrr = total_mrr / total_predictions
    return {
        'perplexity': perplexity, 
        'accuracy': accuracy, 
        'mrr': mrr,
    }

def trainer_get_predictions_from_logits(eval_prediction: EvalPrediction) -> dict:
    predictions = eval_prediction.predictions
    label_ids = eval_prediction.label_ids
    mask = label_ids != 100
    correct = np.logical_and((predictions == label_ids), mask)

    total_tokens = np.sum(mask, axis=1)
    total_correct = np.sum(correct, axis=1)
    return (total_tokens / total_correct).mean()


def trainer_token_accuracy_from_predictions(logits: torch.nn.Tensor, labels: torch.nn.Tensor) -> torch.nn.Tensor:
    return torch.argmax(logits, dim=-1)