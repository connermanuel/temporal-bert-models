import torch
import numpy as np
from transformers import DataCollatorWithPadding, BatchEncoding
from torch.nn.utils.rnn import pad_sequence

def add_zero_timestamp(dataset):
    """For a dataset with no timestamps, add the zero timestamp."""
    def helper(examples):
        examples['timestamps'] = torch.zeros((len(examples['input_ids']), len(examples['input_ids'][0])), dtype=int)
        return examples
    return dataset.map(helper, batched=True)

def get_time_token_collator(tokenizer):
    padding_collator = DataCollatorWithPadding(tokenizer)
    wwm_probability = 0.15

    def time_token_collator(features):
        """A data collator that skips over the first few tokens in the dataset."""
        for feature in features:
            # Randomly mask words. We exclude the first 8 tokens ([CLS] + time prefix) and the last one ([SEP])
            feature.pop('word_ids')
            mask = np.random.binomial(1, wwm_probability, (len(feature["input_ids"]) - 9))
            input_ids = np.array(feature["input_ids"])
            new_labels = np.full(input_ids.shape, -100, dtype=int)

            # When selecting the indices of words to mask, only start at index 8
            masked_idxs = np.where(mask)[0] + 8
            new_labels[masked_idxs] = input_ids[masked_idxs]
            feature["labels"] = new_labels
            
            # 80% chance to replace with mask token, 10% chance to replace with random token, 10% chance to leave it alone
            probabilities = np.random.random_sample(masked_idxs.shape)
            replace_with_mask_token = masked_idxs[probabilities < 0.8]
            input_ids[replace_with_mask_token] = tokenizer.mask_token_id
            replace_with_random_token = masked_idxs[probabilities > 0.9]
            input_ids[replace_with_random_token] = torch.randint(len(tokenizer), replace_with_random_token.shape)
            feature['input_ids'] = input_ids
        
        # Construct the return value
        retval = {}
        for key in features[0].keys():
            tensors = [torch.tensor(feature[key]) for feature in features]
            pad_value = -100 if key == "labels" else 0
            retval[key] = pad_sequence(tensors, batch_first=True, padding_value=pad_value)
            
        return BatchEncoding(retval) # Pads everything to right length
    
    return time_token_collator