"""Tests that the initialization function does work."""
import pytest
import functools

import torch
from transformers import BertConfig, BertForMaskedLM

from tempo_models.train import copy_weights


def test_copy_weights():
    src = BertForMaskedLM(BertConfig())
    dest = BertForMaskedLM(BertConfig())
    dest = copy_weights(src, dest)

    assert torch.equal(
        src.bert.embeddings.word_embeddings.weight,
        dest.bert.embeddings.word_embeddings.weight,
    )


@pytest.mark.parametrize(
    "dataloader,model_base,model_orth",
    [
        ("dataloader_mlm", "model_bert_base", "model_bert_orth"),
        ("dataloader_ssm", "model_t5_base", "model_t5_orth"),
    ],
)
def test_initialization_orthogonal(dataloader, model_base, model_orth, device, request):
    ### This test verifies that the behavior of the initialized orthogonal model is roughly the same as the base model
    dataloader = request.getfixturevalue(dataloader)
    model_base = request.getfixturevalue(model_base)
    model_orth = request.getfixturevalue(model_orth)

    model_base.eval()
    model_orth.eval()

    sample_input = next(iter(dataloader)).to(device)

    output_orth = model_orth(**sample_input)
    for k in list(sample_input.keys()):
        if "timestamps" in k:
            sample_input.pop(k)
    output_base = model_base(**sample_input)

    mask = sample_input["labels"] != -100
    logits_orth = output_orth["logits"][mask]
    logits_base = output_base["logits"][mask]

    assert torch.equal(
        torch.topk(logits_orth, dim=1, k=3)[1], torch.topk(logits_base, dim=1, k=3)[1]
    )


def test_initialization_orthogonal_decoupling_bert(
    dataloader_mlm, model_bert_orth, device
):
    ### This test verifies that the query and key layers are not "linked" -- i.e. that they update independently.

    optim = torch.optim.Adam(model_bert_orth.parameters())
    input = next(iter(dataloader_mlm)).to(device)

    assert all(
        [
            torch.allclose(
                model_bert_orth.bert.encoder.layer[0].attention.self.q_time[i].weight,
                model_bert_orth.bert.encoder.layer[0].attention.self.k_time[i].weight,
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
                model_bert_orth.bert.encoder.layer[0].attention.self.q_time[i].weight,
                model_bert_orth.bert.encoder.layer[0].attention.self.k_time[i].weight,
            )
            for i in range(12)
        ]
    )


def test_initialization_orthogonal_decoupling_t5(dataloader_ssm, model_t5_orth, device):
    ### This test verifies that the query and key layers are not "linked" -- i.e. that they update independently.

    optim = torch.optim.Adam(model_t5_orth.parameters())
    input = next(iter(dataloader_ssm)).to(device)

    assert all(
        [
            torch.allclose(
                model_t5_orth.encoder.block[0].layer[0].SelfAttention.q_time[i].weight,
                model_t5_orth.encoder.block[0].layer[0].SelfAttention.k_time[i].weight,
            )
            for i in range(11)
        ]
    )

    optim.zero_grad()
    model_t5_orth(**input)["loss"].backward()
    optim.step()

    assert not all(
        [
            torch.allclose(
                model_t5_orth.encoder.block[0].layer[0].SelfAttention.q_time[i].weight,
                model_t5_orth.encoder.block[0].layer[0].SelfAttention.k_time[i].weight,
            )
            for i in range(11)
        ]
    )
