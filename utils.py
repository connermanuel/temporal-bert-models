import torch
from transformers import BatchEncoding, AutoTokenizer, DataCollatorForLanguageModeling, Trainer
from torch.nn.utils.rnn import pad_sequence
from transformers import BatchEncoding
from datasets import Dataset

import tqdm
import math

def add_zero_timestamp(dataset):
    """For a dataset with no timestamps, add the zero timestamp."""
    def helper(examples):
        examples['timestamps'] = torch.zeros((len(examples['input_ids']), len(examples['input_ids'][0])), dtype=int)
        return examples
    return dataset.map(helper, batched=True)

def get_collator(tokenizer, n_tokens=8, mask=True):
    """Generates a collator that masks and pads."""
    wwm_probability = 0.15

    def collator(features):
        """A data collator that skips over the first few tokens in the dataset."""
        if mask:
            for feature in features:
                # Randomly mask words. We exclude the first 8 tokens ([CLS] + time prefix) and the last one ([SEP])
                mask = torch.rand((len(feature["input_ids"]) - (n_tokens + 1))) < wwm_probability
                input_ids = torch.tensor(feature["input_ids"], requires_grad=False)
                new_labels = torch.full(input_ids.shape, -100)

                # When selecting the indices of words to mask, only start at index 8
                masked_idxs = torch.nonzero(mask, as_tuple=True)[0] + n_tokens
                new_labels[masked_idxs] = input_ids[masked_idxs]
                feature["labels"] = new_labels
                
                # 80% chance to replace with mask token, 10% chance to replace with random token, 10% chance to leave it alone
                probabilities = torch.rand(masked_idxs.shape)
                replace_with_mask_token = masked_idxs[probabilities < 0.8]
                input_ids[replace_with_mask_token] = tokenizer.mask_token_id
                replace_with_random_token = masked_idxs[probabilities > 0.9]
                input_ids[replace_with_random_token] = torch.randint(len(tokenizer), replace_with_random_token.shape)
                feature['input_ids'] = input_ids
        
        # Construct the return value
        retval = {}
        for key in features[0].keys():
            if key == "input_ids" or key == "labels":
                tensors = [feature[key] for feature in features]
            else:
                tensors = [torch.tensor(feature[key]) for feature in features]
            pad_value = -100 if key == "labels" else 0
            retval[key] = pad_sequence(tensors, batch_first=True, padding_value=pad_value)
            
        return BatchEncoding(retval) # Pads everything to right length
    
    return collator

def make_batch_iterator(dataset, batch_size=32, shuffle=False):
    num_examples = len(dataset)
    idxs = torch.arange(num_examples)
    if shuffle:
        idxs = idxs[torch.randperm(num_examples)]
    
    for start_index in range(0, num_examples, batch_size):
        idx = idxs[start_index: start_index + batch_size]
        yield [dataset[int(i)] for i in idx]

def evaluate_mlm(model, dataset, data_collator, 
             device=torch.device('cuda'), batch_size=16, pad_id=-100):
    """
    Compute token-level perplexity, accuracy, and MRR metrics.
    Note that the perplexity here is over subwords, not words.
    """
    model.eval()
    model.to(device)
    total_cross_entropy = 0.0
    total_mrr = 0.0
    total_predictions = 0
    correct_predictions = 0
    total_ranks = torch.tensor([], dtype=int)
    with torch.no_grad():
        for data in tqdm.tqdm(make_batch_iterator(dataset, batch_size), total=math.ceil(len(dataset) / batch_size)):
            ipt = BatchEncoding(data_collator(data)).to(device)
            out = model(**ipt)
            logits = out['logits']
            num_predictions = (ipt['labels'] != pad_id).sum().item()
            total_predictions += num_predictions
            total_cross_entropy += out['loss'].item() * num_predictions
            batch_correct_predictions = (
              (ipt['labels'] != pad_id) &
              (ipt['labels'] == logits.argmax(2))).sum().item()
            correct_predictions += batch_correct_predictions
            idx = torch.nonzero(ipt['labels'] != pad_id)
            labels = ipt['labels'][idx[:, 0], idx[:, 1]].cuda() ## is a list of length n
            logits_masked = logits[idx[:, 0], idx[:, 1]].cuda() ## should now be of shape n x n_tokens
            logits_values = logits[idx[:, 0], idx[:, 1], labels] ## list of length n
            ranks = (logits_masked > logits_values.reshape(-1, 1)).sum(axis=1) + 1
            total_ranks = torch.cat((total_ranks, ranks.cpu()))
            total_mrr += (1/ranks).sum().item()
    perplexity = math.exp(total_cross_entropy / total_predictions)
    accuracy = 100 * correct_predictions / total_predictions
    mrr = total_mrr / total_predictions
    return {
        'perplexity': perplexity, 
        'accuracy': accuracy, 
        'mrr': mrr,
    }

def evaluate_span_accuracy(model, dataset, data_collator, 
             device=torch.device('cuda'), batch_size=16, pad_id=-100):
    """Evaluates the span accuracy"""
    
    ids = dataset['id']
    num_ids = torch.bincount(ids)
    cum_num_ids = torch.cumsum(num_ids)

    dataset.remove_columns(["id"])

    model.eval()
    model.to(device)
    total_correct_predictions = torch.tensor([])
    with torch.no_grad():
        for data in tqdm.tqdm(make_batch_iterator(dataset, batch_size), total=math.ceil(len(dataset) / batch_size)):
            ipt = BatchEncoding(data_collator(data)).to(device)
            out = model(**ipt)
            logits = out['logits']
            total_correct_predictions = torch.cat(
              total_correct_predictions, 
              ((ipt['labels'] != pad_id) &
              (ipt['labels'] == logits.argmax(2)))
            )
    
    correct_spans = 0
    start = 0
    for end in cum_num_ids:
        if torch.all(total_correct_predictions[start:end]):
            correct_spans += 1
        start = end
    
    return correct_spans / len(cum_num_ids)

    

def add_timestamp(examples):
    # Only works on correctly batched tokens.
    timestamps = torch.tensor(examples['timestamps'])
    examples['timestamp'] = timestamps[:, 0]
    return examples
    
def sort_by_timestamp(dataset):
    dataset = dataset.map(add_timestamp, batched=True)
    dataset = dataset.sort('timestamp')
    dataset = dataset.remove_columns('timestamp')
    return dataset

def shuffle_batched(dataset, batch_size):
    """Shuffles a dataset while keeping batches intact."""
    num_batches = len(dataset) // batch_size
    idxs = torch.randperm(num_batches) * batch_size
    idxs = idxs.reshape(-1, 1) + torch.arange(batch_size)
    idxs = torch.concat([idxs.flatten(), torch.arange(num_batches * batch_size, len(dataset))])
    return dataset.select(idxs)

class NonShuffledTrainer(Trainer):
    """Shuffles the training dataset while keeping batches intact."""
    def _get_train_sampler(self):
        return None

def add_special_time_tokens(dataset, tokenizer_len):
    # Inserts special time tokens into the dataset and resizes tokenizer.
    
    def insert_special_token(examples):
        for k in examples.keys():
            examples[k] = torch.tensor(examples[k])
        for k in examples.keys():
            if k == "input_ids":
                ids = examples['input_ids']
                ts = examples['timestamps']
                examples["input_ids"] = torch.hstack((ids[:, 0:1], (tokenizer_len + ts[:, 0:1]), ids[:, 1:]))
            else:
                examples[k] = torch.hstack((examples[k][:, 0:1], examples[k]))
    dataset = dataset.map(insert_special_token, batched=True)
    return dataset

def fix_timestamps(examples):
        try:
            examples['timestamps'] = torch.tensor(examples['timestamps']) + 2
        except ValueError:
            examples['timestamps'] = [torch.tensor(a) + 2 for a in examples['timestamps']]
        return examples

def copy_weights(src: torch.nn.Module, dest: torch.nn.Module):
    """Copy the weights from the source model to the destination model."""
    sd = dest.state_dict()
    src_sd = src.state_dict()
    for k in src_sd:
        sd[k] = src_sd[k]
    dest.load_state_dict(sd)
    return dest