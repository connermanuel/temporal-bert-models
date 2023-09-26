import torch
import numpy as np
from dataclasses import dataclass
from transformers import PreTrainedTokenizerBase, BatchEncoding
from typing import Optional, List, Union, Any, Dict, Mapping, Tuple, Iterable
from tempo_models.utils import to_list, to_tensor
from tempo_models.models.bert.orthogonal_bert import TIMESTAMP_PAD


def pad_column(seq_len: int, column: Iterable, pad_value: Union[int, float]):
    return torch.tensor(
        [to_list(value) + [pad_value] * (seq_len - len(value)) for value in column],
        dtype=torch.int,
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
class CollatorSeq2Seq:
    tokenizer: PreTrainedTokenizerBase
    model: Optional[Any] = None
    label_pad_token_id: int = -100
    timestamp_pad_value: int = -TIMESTAMP_PAD

    def __call__(self, features):
        labels = (
            [feature["labels"] for feature in features]
            if "labels" in features[0].keys()
            else None
        )
        timestamps = (
            [feature["timestamps"] for feature in features]
            if "timestamps" in features[0].keys()
            else None
        )

        features = [
            {k: v for k, v in feature.items() if k != "timestamps"}
            for feature in features
        ]

        features = self.tokenizer.pad(
            features,
            return_tensors="pt",
        )
        
        if timestamps is not None:
            features["timestamps"] = pad_column(
                features["input_ids"].shape[1], timestamps, self.timestamp_pad_value
            )
        
        if labels is not None:
            features["labels"] = pad_column(
                features["input_ids"].shape[1], labels, self.label_pad_token_id
            )

        # prepare decoder_input_ids
        if (
            labels is not None
            and self.model is not None
            and hasattr(self.model, "prepare_decoder_input_ids_from_labels")
        ):
            decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(
                labels=features["labels"]
            )
            features["decoder_input_ids"] = decoder_input_ids

        return features
