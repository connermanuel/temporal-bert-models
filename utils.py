import torch

def add_zero_timestamp(dataset):
    """For a dataset with no timestamps, add the zero timestamp."""
    def helper(examples):
        examples['timestamps'] = torch.zeros((len(examples['input_ids']), len(examples['input_ids'][0])), dtype=int)
        return examples
    return helper(dataset)
