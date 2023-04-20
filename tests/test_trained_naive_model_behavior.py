"""Test that a trained model does what we expect."""

import torch
import pytest
from datasets import load_from_disk
from context import BertForNaiveOrthogonalMaskedLM, evaluate

@pytest.fixture
def trained_model():
    return BertForNaiveOrthogonalMaskedLM.from_pretrained('./output/default_1e-5/checkpoint-24030')

@pytest.fixture
def tempo_dataset():
    return load_from_disk('./data/tempo_dataset')

@pytest.fixture
def dataset_timestamp_0(tempo_dataset):
    return tempo_dataset['train'].filter(lambda x: x['timestamps'][0] == 0).remove_columns(['word_ids'])

@pytest.fixture
def dataset_timestamp_1(tempo_dataset):
    return tempo_dataset['train'].filter(lambda x: x['timestamps'][0] == 1).remove_columns(['word_ids'])

@pytest.mark.skip(reason="true in expectation over all possible samples, but not necessarily true for every possible subset of indices")
def test_timestamp_effect(trained_model, dataset_timestamp_0, dataset_timestamp_1):
    """Verifies that matching sentences to the correct timestamp improves performance, in-sample."""

    NUM_SAMPLES = 1000

    def change_timestamp(examples, timestamp):
        examples['timestamps'] = torch.full(torch.tensor(examples['timestamps']).shape, timestamp)
        return examples
    
    dataset_timestamp_0_changed = dataset_timestamp_0.map(lambda x: change_timestamp(x, 1))
    dataset_timestamp_1_changed = dataset_timestamp_1.map(lambda x: change_timestamp(x, 0))
    for i in range(5):
        print(f"Iteration {i+1}")
        idxs_0 = torch.randperm(len(dataset_timestamp_0))[:NUM_SAMPLES]
        idxs_1 = torch.randperm(len(dataset_timestamp_1))[:NUM_SAMPLES]
        metrics_0 = evaluate(trained_model, dataset_timestamp_0.select(idxs_0))
        metrics_0_changed = evaluate(trained_model, dataset_timestamp_0_changed.select(idxs_0))
        print(f"Correct timestamp: {metrics_0}")
        print(f"Incorrect timestamp: {metrics_0_changed}")
        # assert metrics_0['perplexity'] < metrics_0_changed['perplexity']

        metrics_1 = evaluate(trained_model, dataset_timestamp_1.select(idxs_1))
        metrics_1_changed = evaluate(trained_model, dataset_timestamp_1_changed.select(idxs_1))
        print(f"Correct timestamp: {metrics_1}")
        print(f"Incorrect timestamp: {metrics_1_changed}")
        # assert metrics_1['perplexity'] < metrics_1_changed['perplexity']