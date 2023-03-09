"""Test basic properties about orthogonal layers."""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import orthogonal

H_SIZE = 20

@pytest.fixture 
def orth_linear():
    return orthogonal(nn.Linear(H_SIZE, H_SIZE, bias=False))

@pytest.fixture 
def orth_linear_2():
    return orthogonal(nn.Linear(H_SIZE, H_SIZE, bias=False))

@pytest.fixture 
def orth_linear_complement(orth_linear):
    layer = orthogonal(nn.Linear(H_SIZE, H_SIZE, bias=False))
    layer.weight = orth_linear.weight.T
    return layer

@pytest.fixture
def random_input():
    return torch.rand(H_SIZE)

@pytest.fixture
def eye():
    return torch.eye(H_SIZE)

def test_orthogonality(orth_linear, orth_linear_complement, eye):
    """Verifies that the layers are orthogonal."""
    orig_weights = (orth_linear.weight.clone().detach(), orth_linear_complement.weight.clone().detach())
    assert torch.dist(orig_weights[0] @ orig_weights[1], eye) < 1e-05

def test_backprop(orth_linear, random_input, eye):
    """Verifies that backprop reduces loss, and retains orthogonality."""
    orig_weight = orth_linear.weight.detach().clone()
    optim = torch.optim.Adam(orth_linear.parameters())
    loss = F.mse_loss(orth_linear(random_input), random_input)
    loss.backward()
    optim.step()
    assert torch.dist(orth_linear.weight, eye) < torch.dist(orig_weight, eye)

def test_composition_as_identity_loss(orth_linear, orth_linear_2, random_input, eye):
    """Verifies that decreasing the distance between two weights brings the output closer to the identity."""
    optim = torch.optim.Adam(list(orth_linear.parameters()) + list(orth_linear_2.parameters()))
    out_1 = orth_linear(orth_linear_2(eye).T)
    loss = torch.dist(orth_linear.weight, orth_linear_2.weight)
    loss.backward()
    optim.step()
    out_2 = orth_linear(orth_linear_2(eye).T)
    assert torch.dist(out_2, eye) < torch.dist(out_1, eye)
