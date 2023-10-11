import torch
import numpy as np
from dataclasses import dataclass
from transformers import PreTrainedTokenizerBase, BatchEncoding
from typing import Optional, List, Union, Any, Dict, Mapping, Tuple, Iterable
from tempo_models.utils import to_list, to_tensor
from tempo_models.models.bert.orthogonal_bert import TIMESTAMP_PAD
import random


def pad_column(seq_len: int, column: Iterable, pad_value: Union[int, float]):
    return torch.tensor(
        [to_list(value) + [pad_value] * (seq_len - len(value)) for value in column],
        dtype=torch.long,
    )


@dataclass
class CollatorMLM:
    """
    Data collator used for timestamps-augmented language modeling.
    Inputs are dynamically padded to the maximum length of a batch if they are not all of the same length.
    """

    tokenizer: PreTrainedTokenizerBase
    mlm: bool = True
    mlm_probability: float = 0.15
    pad_to_multiple_of: Optional[int] = None
    timestamp_pad_value: int = -TIMESTAMP_PAD
    timestamp_mask_value: int = -1

    def __post_init__(self):
        if self.mlm and self.tokenizer.mask_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary for masked language modeling. "
                "You should pass `mlm=False` to train on causal language modeling instead."
            )

    def __call__(
        self, examples: List[Union[List[int], Any, Dict[str, Any]]]
    ) -> Dict[str, Any]:
        timestamps = (
            [example["timestamps"] for example in examples]
            if "timestamps" in examples[0].keys()
            else None
        )
        examples_no_ts = [
            {k: v for k, v in example.items() if k != "timestamps"}
            for example in examples
        ]

        if isinstance(examples[0], Mapping):
            batch = self.tokenizer.pad(
                examples_no_ts,
                return_tensors="pt",
                pad_to_multiple_of=self.pad_to_multiple_of,
            )
        else:
            raise AssertionError(
                "Dataset in the wrong format. Each entry must be a mapping."
            )

        if timestamps is not None:
            timestamps = pad_column(
                batch["input_ids"].shape[1], timestamps, self.timestamp_pad_value
            )

        # If special token mask has been preprocessed, pop it from the dict.
        special_tokens_mask = batch.pop("special_tokens_mask", None)
        if self.mlm:
            batch["input_ids"], batch["labels"], timestamps = self.mask_tokens(
                batch["input_ids"], timestamps, special_tokens_mask=special_tokens_mask
            )
        else:
            labels = batch["input_ids"].clone()
            if self.tokenizer.pad_token_id is not None:
                labels[labels == self.tokenizer.pad_token_id] = -100
            batch["labels"] = labels

        if timestamps is not None:
            batch["timestamps"] = timestamps

        return batch

    def mask_tokens(
        self, inputs: Any, timestamps: Any, special_tokens_mask: Optional[Any] = None
    ) -> Tuple[Any, Any]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """

        labels = inputs.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(
                    val, already_has_special_tokens=True
                )
                for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = (
            torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        )
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(
            self.tokenizer.mask_token
        )

        if timestamps is not None:
            timestamps[indices_replaced] = self.timestamp_mask_value

        # 10% of the time, we replace masked input tokens with random word
        indices_random = (
            torch.bernoulli(torch.full(labels.shape, 0.5)).bool()
            & masked_indices
            & ~indices_replaced
        )
        random_words = torch.randint(
            len(self.tokenizer), labels.shape, dtype=torch.long
        )
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels, timestamps


@dataclass
class CollatorCLS:
    tokenizer: PreTrainedTokenizerBase
    timestamp_pad_value: int = -TIMESTAMP_PAD

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        timestamps = (
            [example["timestamps"] for example in examples]
            if "timestamps" in examples[0].keys()
            else None
        )
        examples_no_ts = [
            {k: v for k, v in example.items() if k != "timestamps"}
            for example in examples
        ]
        batch = self.tokenizer.pad(examples_no_ts, return_tensors="pt")
        if timestamps is not None:
            batch["timestamps"] = pad_column(
                batch["input_ids"].shape[1], timestamps, self.timestamp_pad_value
            )
        return batch


@dataclass
class CollatorSSM:
    """
    Data collator used for timestamps-augmented salient span masking.
    Inputs must contain the following columns:
    * span_ids -- a list of [start, end] index pairs
    * timestamps -- a list containing any integer from 0 to num_timestamps - 1, repeated len(input_ids) times
    * input_ids -- result of input_ids from tokenizer
    Inputs are dynamically padded to the maximum length of a batch if they are not all of the same length.
    """

    tokenizer: PreTrainedTokenizerBase
    pad_to_multiple_of: Optional[int] = None
    timestamp_pad_value: int = -TIMESTAMP_PAD
    timestamp_mask_value: int = -1
    label_pad_value: int = -100
    special_token_0: int = 32099
    special_token_1: int = 32098
    eos_token: int = 1

    def __call__(
        self, examples: List[Union[List[int], Any, Dict[str, Any]]]
    ) -> Dict[str, Any]:
        timestamps = (
            [example["timestamps"] for example in examples]
            if "timestamps" in examples[0].keys()
            else None
        )
        input_ids = [example["input_ids"] for example in examples]
        span_ids = [example["span_ids"] for example in examples]

        input_ids, labels, timestamps, label_timestamps = self.mask_tokens(
            input_ids, span_ids, timestamps
        )

        batch = self.tokenizer.pad(
            {"input_ids": input_ids},
            return_tensors="pt",
            pad_to_multiple_of=self.pad_to_multiple_of,
        )
        
        batch["labels"] = pad_column(
            max([len(l) for l in labels]), labels, self.label_pad_value
        )

        if timestamps is not None:
            batch["timestamps"]  = pad_column(
                batch["input_ids"].shape[1], timestamps, self.timestamp_pad_value
            )
            batch["label_timestamps"] = pad_column(
                batch["labels"].shape[1], label_timestamps, self.timestamp_pad_value
            )

        return batch

    def mask_tokens(
        self,
        inputs: list[list[int]],
        span_ids: list[list[list[int]]],
        timestamps: list[list[int]],
    ) -> Tuple[Any, Any]:
        """
        Prepare masked tokens inputs/labels for salient span masking.
        Retrieve one [start, end] pair from the provided set of span_ids,
        mask the corresponding tokens from the input_ids,
        and return the masked section as the labels.
        Resize the timestamps to match.
        """
        label_timestamps=None
        selected_span_ids = [random.choice(span_pairs) for span_pairs in span_ids]
        labels = [
            [self.special_token_0]
            + input[span_pair[0] : span_pair[1]]
            + [self.special_token_1, self.eos_token]
            for input, span_pair in zip(inputs, selected_span_ids)
        ]
        inputs = [
            input[: span_pair[0]] + [self.special_token_0] + input[span_pair[1] :]
            for input, span_pair in zip(inputs, selected_span_ids)
        ]
        if timestamps:
            timestamps = [
                [timestamp[0]] * len(input)
                for timestamp, input in zip(timestamps, inputs)
            ]
            label_timestamps = [
                [timestamp[0]] * len(label)
                for timestamp, label in zip(timestamps, labels)
            ]

        return inputs, labels, timestamps, label_timestamps
