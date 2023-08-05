import torch
from dataclasses import dataclass
from transformers import PreTrainedTokenizerBase, BatchEncoding
from typing import Optional, List, Union, Any, Dict, Mapping, Tuple, Iterable
from tempo_models.utils.utils import to_list, to_tensor
from tempo_models.models.bert.orthogonal_bert import TIMESTAMP_PAD

def insert_padded_column(batch: BatchEncoding, column_name: str, column: Iterable, pad_value: Union[int, float]):
    sequence_length = batch["input_ids"].shape[1]
    batch[column_name] = torch.tensor([
        to_list(value) + [pad_value] * (sequence_length - len(value)) for value in column
    ], dtype=torch.int)
    return batch

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

    def __post_init__(self):
        if self.mlm and self.tokenizer.mask_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary for masked language modeling. "
                "You should pass `mlm=False` to train on causal language modeling instead."
            )

    def __call__(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        timestamps = [example["timestamps"] for example in examples] if "timestamps" in examples[0].keys() else None
        examples_no_ts = [{k: v for k, v in example.items() if k != "timestamps"} for example in examples]
        
        if isinstance(examples[0], Mapping):
            batch = self.tokenizer.pad(examples_no_ts, return_tensors="pt", pad_to_multiple_of=self.pad_to_multiple_of)
        else:
            raise AssertionError("Dataset in the wrong format. Each entry must be a mapping.")

        # If special token mask has been preprocessed, pop it from the dict.
        special_tokens_mask = batch.pop("special_tokens_mask", None)
        if self.mlm:
            batch["input_ids"], batch["labels"] = self.mask_tokens(
                batch["input_ids"], special_tokens_mask=special_tokens_mask)
        else:
            labels = batch["input_ids"].clone()
            if self.tokenizer.pad_token_id is not None:
                labels[labels == self.tokenizer.pad_token_id] = -100
            batch["labels"] = labels

        if timestamps:
            batch = insert_padded_column(batch, "timestamps", timestamps, self.timestamp_pad_value)
        return batch

    def mask_tokens(self, inputs: Any, special_tokens_mask: Optional[Any] = None) -> Tuple[Any, Any]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """

        labels = inputs.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels

@dataclass
class CollatorCLS:
    tokenizer: PreTrainedTokenizerBase
    timestamp_pad_value: int = -TIMESTAMP_PAD

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        timestamps = [example["timestamps"] for example in examples] if "timestamps" in examples[0].keys() else None
        examples_no_ts = [{k: v for k, v in example.items() if k != "timestamps"} for example in examples]
        batch = self.tokenizer.pad(examples_no_ts, return_tensors="pt")
        if timestamps:
            batch = insert_padded_column(batch, "timestamps", timestamps, self.timestamp_pad_value)
        return batch