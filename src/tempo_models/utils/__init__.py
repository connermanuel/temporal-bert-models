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


def get_tokenizer(model: str) -> PreTrainedTokenizer:
    if model == "bert":
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    elif model == "t5":
        tokenizer = AutoTokenizer.from_pretrained("t5-base", model_max_length=512)

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
            timestamps = torch.tensor(examples["timestamps"])
            examples["timestamp"] = timestamps[:, 0]
        except ValueError:
            examples["timestamp"] = [t[0] for t in examples["timestamps"]]
        return examples

    dataset = dataset.map(add_timestamp, batched=True)
    dataset = dataset.sort("timestamp")
    dataset = dataset.remove_columns("timestamp")

    num_batches = len(dataset) // batch_size
    idxs = torch.randperm(num_batches) * batch_size
    idxs = idxs.reshape(-1, 1) + torch.arange(batch_size)
    idxs = torch.concat(
        [idxs.flatten(), torch.arange(num_batches * batch_size, len(dataset))]
    )
    return dataset.select(idxs)


def add_special_time_tokens(
    dataset, tokenizer, model_type, token_type, n_contexts, start_year=2010
):
    # Inserts special time tokens into the dataset. Assumes that the tokenizer was already resized.
    tokenizer_len = tokenizer.vocab_size

    def insert_tokens(examples):
        token_sequences = [None]
        if token_type == "string":
            token_sequences = [
                tokenizer(f"year: {start_year + i} text: ")["input_ids"][:-1]
                for i in range(n_contexts)
            ]
        elif token_type == "special":
            token_sequences = [[tokenizer_len + i] for i in range(n_contexts)]

        insert_lengths_per_sequence = [
            len(token_sequences[ts[0]]) for ts in examples["timestamps"]
        ]

        prefix_len = {"t5": 0, "bert": 1}[model_type]

        examples["input_ids"] = [
            (ids[0:prefix_len] + token_sequences[ts[0]] + ids[prefix_len:])
            for ts, ids in zip(examples["timestamps"], examples["input_ids"])
        ]
        for k in ["attention_mask", "special_tokens_mask", "timestamps"]:
            if k in examples.keys():
                examples[k] = [
                    (x[0:1] * insert_lengths_per_sequence[i]) + x
                    for i, x in enumerate(examples[k])
                ]
        if "span_ids" in examples.keys():
            examples["span_ids"] = [
                [
                    [
                        pair[0] + insert_lengths_per_sequence[i],
                        pair[1] + insert_lengths_per_sequence[i],
                    ]
                    for pair in l
                ]
                for i, l in enumerate(examples["span_ids"])
            ]

        return examples

    return dataset.map(insert_tokens, batched=True)


def prepare_time_tokens(
    token_type,
    dataset,
    tokenizer,
    model,
    model_type,
    n_contexts,
    start_year,
    resize_model=True,
):
    if token_type == "string" or token_type == "special":
        dataset = add_special_time_tokens(
            dataset, tokenizer, model_type, token_type, n_contexts, start_year
        )

    if token_type == "special":
        special_tokens = [f"timestamp: {t} text: " for t in range(n_contexts)]
        tokenizer.add_tokens(special_tokens)

        if resize_model:
            model.resize_token_embeddings(
                model.config.vocab_size + n_contexts, pad_to_multiple_of=16
            )

    return dataset, model, tokenizer


#############################
# EVALUATION UTILS          #
#############################


def make_batch_iterator(dataset, batch_size=32, shuffle=False):
    num_examples = len(dataset)
    idxs = torch.arange(num_examples)
    if shuffle:
        idxs = idxs[torch.randperm(num_examples)]

    for start_index in range(0, num_examples, batch_size):
        idx = idxs[start_index : start_index + batch_size]
        yield [dataset[int(i)] for i in idx]


def evaluate_mlm(
    model, dataset, data_collator, no_cuda=False, batch_size=16, pad_id=-100
):
    """
    Compute token-level perplexity, accuracy, and MRR metrics.
    Note that the perplexity here is over subwords, not words.
    """
    if not no_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    model.eval().to(device)

    mean_cross_entropy = 0
    total_mrr = 0.0
    total_predictions = 0
    correct_predictions = 0
    total_ranks = torch.tensor([], dtype=int)
    num_iterations = math.ceil(len(dataset) / batch_size)

    with torch.no_grad():
        for data in tqdm.tqdm(
            make_batch_iterator(dataset, batch_size), total=num_iterations
        ):
            input = data_collator(data).to(device)
            labels = input["labels"]

            out = model(**input)

            logits = out["logits"]

            num_predictions = (labels != pad_id).sum().item()
            total_predictions += num_predictions
            pct_new = num_predictions / total_predictions

            mean_cross_entropy = (mean_cross_entropy * (1 - pct_new)) + out[
                "loss"
            ].item() * pct_new

            batch_correct_predictions = (
                ((labels != pad_id) & (labels == logits.argmax(2))).sum().item()
            )
            correct_predictions += batch_correct_predictions

            idx = torch.nonzero(labels != pad_id)
            labels = labels[idx[:, 0], idx[:, 1]].to(device)  ## is a list of length n
            logits_masked = logits[idx[:, 0], idx[:, 1]].to(
                device
            )  ## should now be of shape n x n_tokens
            logits_values = logits[idx[:, 0], idx[:, 1], labels]  ## list of length n

            ranks = (logits_masked > logits_values.reshape(-1, 1)).sum(axis=1) + 1
            total_ranks = torch.cat((total_ranks, ranks.cpu()))
            total_mrr += (1 / ranks).sum().item()

    perplexity = math.exp(mean_cross_entropy)
    accuracy = 100 * correct_predictions / total_predictions
    mrr = total_mrr / total_predictions
    return {
        "perplexity": perplexity,
        "accuracy": accuracy,
        "mrr": mrr,
    }
