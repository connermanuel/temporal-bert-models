"""Tests that the initialization function does work."""
import pytest
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from datasets import load_from_disk
from transformers import BertConfig, BertForMaskedLM
from transformers import AutoTokenizer

from tempo_models.train import copy_weights, initialize_mlm_model
from tempo_models.utils.collator import CollatorMLM
from tempo_models.models.bert import BertOrthogonalSelfAttention, OrthogonalConfig


def test_copy_weights():
    src = BertForMaskedLM(BertConfig())
    dest = BertForMaskedLM(BertConfig())
    dest = copy_weights(src, dest)

    assert torch.equal(
        src.bert.embeddings.word_embeddings.weight,
        dest.bert.embeddings.word_embeddings.weight,
    )


def test_initialization_orthogonal_mlm(
    dataloader_mlm, model_bert_base, model_bert_orth, device
):
    ### This test verifies that the behavior of the initialized orthogonal model is roughly the same as the base BERT model.
    ### "Roughly" because the product of the orthogonal matrix is not exactly the identity.

    model_bert_orth.eval()
    model_bert_base.eval()

    sample_input = next(iter(dataloader_mlm)).to(device)

    output_orth = model_bert_orth(**sample_input)
    sample_input.pop("timestamps")
    output_base = model_bert_base(**sample_input)

    mask = sample_input["labels"] != 100
    logits_orth = output_orth["logits"][mask]
    logits_base = output_base["logits"][mask]

    assert torch.equal(torch.topk(logits_orth, dim=1, k=3)[1], torch.topk(logits_base, dim=1, k=3)[1]) # fails

def test_initialization_orthogonal_decoupling_attention():
    ### This test verifies that the query and key layers are not "linked" -- i.e. that they update independently.
    att = BertOrthogonalSelfAttention(OrthogonalConfig())
    att.init_temporal_weights()
    opt = torch.optim.Adam(att.parameters(), lr=1)

    ipt = torch.rand(att.query_time_layers[0].weight.shape)
    ipt_2 = torch.rand(att.query_time_layers[0].weight.shape)
    loss = (att.query_time_layers[0](ipt) @ att.key_time_layers[0](ipt_2).transpose(-1, -2)).sum()
    loss.backward()
    opt.step()

    assert not torch.allclose(
        att.query_time_layers[0].weight,
        att.key_time_layers[0].weight,
    )

def test_initialization_orthogonal_decoupling(dataloader_mlm, model_bert_orth, device):
    ### This test verifies that the query and key layers are not "linked" -- i.e. that they update independently.

    optim = torch.optim.Adam(model_bert_orth.parameters())
    input = next(iter(dataloader_mlm)).to(device)

    assert all(
        [
            torch.allclose(
                model_bert_orth.bert.encoder.layer[0].attention.self.query_time_layers[i].weight,
                model_bert_orth.bert.encoder.layer[0].attention.self.key_time_layers[i].weight,
            )
            for i in range(12)
        ]
    )

    optim.zero_grad()
    model_bert_orth(**input)["loss"].backward()
    optim.step()

    assert not all(
        [
            torch.allclose(
                model_bert_orth.bert.encoder.layer[0].attention.self.query_time_layers[i].weight,
                model_bert_orth.bert.encoder.layer[0].attention.self.key_time_layers[i].weight,
            )
            for i in range(12)
        ]
    )


