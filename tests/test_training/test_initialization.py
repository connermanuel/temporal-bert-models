"""Tests that the initialization function does work."""
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from datasets import load_from_disk
from transformers import BertConfig, BertForMaskedLM
from transformers import AutoTokenizer

from tempo_models.train import copy_weights, initialize_mlm_model
from tempo_models.utils.collator import CollatorMLM


def test_copy_weights():
    src = BertForMaskedLM(BertConfig())
    dest = BertForMaskedLM(BertConfig())
    dest = copy_weights(src, dest)

    assert torch.equal(
        src.bert.embeddings.word_embeddings.weight,
        dest.bert.embeddings.word_embeddings.weight,
    )

def test_initialization_orthogonal_mlm(dataloader_mlm, model_bert_base):
    ### This test verifies that the behavior of the initialized orthogonal model is roughly the same as the base BERT model.
    ### "Roughly" because the product of the orthogonal matrix is not exactly the identity.
    ### But this doesn't work perfectly -- the two models have similar predictions, but there isn't really a good metric
    ### for "close-enough"-ness that will allow this to work.

    sample_input = next(iter(dataloader_mlm))
    model_orth = initialize_mlm_model("bert", "orthogonal", 12)

    output_orth = model_orth(**sample_input)
    sample_input.pop("timestamps")
    output_base = model_bert_base(**sample_input)

    mask = sample_input["labels"] != 100
    logits_orth = output_orth["logits"][mask]
    logits_base = output_base["logits"][mask]

    preds_side_by_side = torch.hstack(
        (torch.topk(logits_orth, dim=1, k=3)[1], torch.topk(logits_base, dim=1, k=3)[1])
    )

    assert torch.allclose(output_orth["logits"], output_base["logits"])  # fails
