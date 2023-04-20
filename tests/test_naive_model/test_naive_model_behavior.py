"""Tests that the presence of temporal weights outputs what we expect."""

import pytest
import torch
from transformers import BertForMaskedLM, BertConfig
from context import BertForNaiveOrthogonalMaskedLM

@pytest.fixture
def bert_config():
    return BertConfig()

@pytest.fixture
def base_bert_model(bert_config):
    """Returns a BERT model which will be used as a baseline."""
    return BertForMaskedLM(bert_config)

@pytest.fixture
def create_orthogonal_model(base_bert_model, bert_config):
    """Wrapper to create an orthogonal model with the same weights as bert."""
    bert_sd = base_bert_model.state_dict()

    def _create_model(init_temporal_weights):
        model = BertForNaiveOrthogonalMaskedLM(bert_config, init_temporal_weights=init_temporal_weights)
        sd = model.state_dict()
        for k in bert_sd.keys():
            sd[k] = bert_sd[k]
        model.load_state_dict(sd)
        model.eval()
        return model
    
    return _create_model

@pytest.fixture
def base_orthogonal_model(create_orthogonal_model):
    return create_orthogonal_model(init_temporal_weights=True)

@pytest.fixture
def non_init_orthogonal_model(create_orthogonal_model):
    return create_orthogonal_model(init_temporal_weights=False)

@pytest.fixture
def sample_input():
    l = [1998, 2137, 15372, 10123, 15304, 2013, 1996, 11421, 1998, 19306, 1997, 1005, 2656, 1998, 1005, 2676, 
             102, 101, 2085, 2009, 2022, 2074, 1037, 3043, 2005, 2216, 3124, 2000, 2022, 2583, 2000, 2079, 2009, 
             2733, 1999, 1998, 2733, 2041, 102, 101, 8558, 2191, 1996, 2332, 1005, 1055, 2711, 2004, 2092, 2004, 
             2010, 2436, 2298, 7746, 2135, 18708, 102, 101, 1996, 4101, 2323, 2022, 3698, 1011, 11655, 5422, 2000, 
             2507, 6882, 2491, 1997, 4101, 6375, 3012, 1998, 1996, 3014, 1997, 3684, 2224, 102, 101, 1045, 2052, 2022, 
             2053, 2393, 2006, 3011, 2000, 2017, 102, 101, 1045, 2071, 2693, 2046, 1996, 4845, 1005, 1055, 2282, 1045, 
             6814, 102, 101, 2021, 2011, 2085, 1996, 2060, 1996, 20889, 2031, 2991, 2524, 2005, 2032, 102, 101, 2002, 
             2022, 1037, 4151, 2158, 2040, 2228, 2069]
    d = {
        'input_ids': l,
        'token_type_ids': [0] * 128, 
        'attention_mask': [1] * 128,
        'labels': l
        }
    return {k: torch.tensor(d[k])[None, ...] for k in d}

@pytest.mark.skip(reason="more focused on implementation, not interface")
def test_weight_matrix_construction(base_orthogonal_model, non_init_orthogonal_model):
    """
    Verify that a model initialized with the same timestamp weights constructs the same temporal weight matrix for different timestamps,
    and that different timestamp weights constructs a different temporal weight matrix.
    """
    with torch.no_grad():
        sample_query = torch.rand([1, 12, 128, 64])
        zero_timestamps = torch.tensor([0]*128)[None, ...]
        one_timestamps = torch.tensor([1]*128)[None, ...]
        for layer in base_orthogonal_model.bert.encoder.layer:
            sa = layer.attention.self
            for i in range(12):
                assert torch.allclose(sa.query_time_layers[2][i].weight, sa.query_time_layers[3][i].weight)
            zero_timed_query_layer = sa.construct_time_matrix(sample_query, sa.query_time_layers, zero_timestamps)
            one_timed_query_layer = sa.construct_time_matrix(sample_query, sa.query_time_layers, one_timestamps)
            assert torch.allclose(zero_timed_query_layer, one_timed_query_layer)
        
        for layer in non_init_orthogonal_model.bert.encoder.layer:
            sa = layer.attention.self
            zero_timed_query_layer = sa.construct_time_matrix(sample_query, sa.query_time_layers, zero_timestamps)
            one_timed_query_layer = sa.construct_time_matrix(sample_query, sa.query_time_layers, one_timestamps)
            assert not torch.allclose(zero_timed_query_layer, one_timed_query_layer)


def test_effect_of_weight_on_output(base_orthogonal_model, non_init_orthogonal_model, sample_input):
    """
    Verify that the same sentence with different timestamps is encoded differently if temporal weights are different,
    and encoded the same way if temporal weights are the same.
    """
    with torch.no_grad():
        zero_timestamps = torch.tensor([0]*128)[None, ...]
        one_timestamps = torch.tensor([1]*128)[None, ...]
        base_out_0 = base_orthogonal_model(**sample_input, timestamps=zero_timestamps)
        base_out_0_2 = base_orthogonal_model(**sample_input, timestamps=zero_timestamps)
        base_out_1 = base_orthogonal_model(**sample_input, timestamps=one_timestamps)
        print(base_out_0.logits - base_out_0_2.logits)
        print(base_out_0.logits - base_out_1.logits)
        assert torch.allclose(base_out_0.logits, base_out_1.logits)

        non_init_out_0 = non_init_orthogonal_model(**sample_input, timestamps=zero_timestamps)
        non_init_out_1 = non_init_orthogonal_model(**sample_input, timestamps=one_timestamps)
        assert not torch.allclose(non_init_out_0.logits, non_init_out_1.logits)

def test_alpha_training_loss(base_orthogonal_model, non_init_orthogonal_model, sample_input):
    """
    Verify that the alpha term works: i.e., that for a large alpha, training loss increases if and only if the weight matrices 
    change the output from the identity.
    """
    base_orthogonal_model.train()
    non_init_orthogonal_model.train()
    with torch.no_grad():
        zero_timestamps = torch.tensor([0]*128)[None, ...]

        base_orthogonal_model.alpha = 10000
        non_init_orthogonal_model.alpha = 10000
        base_out_alpha = base_orthogonal_model(**sample_input, timestamps=zero_timestamps)
        non_init_out_alpha = non_init_orthogonal_model(**sample_input, timestamps=zero_timestamps)

        base_orthogonal_model.alpha = 0
        non_init_orthogonal_model.alpha = 0
        base_out_zero = base_orthogonal_model(**sample_input, timestamps=zero_timestamps)
        non_init_out_zero = non_init_orthogonal_model(**sample_input, timestamps=zero_timestamps)

        assert torch.isclose(base_out_alpha.loss, base_out_zero.loss, atol=1) # not the same because of dropout
        assert non_init_out_zero.loss < non_init_out_alpha.loss
        assert (base_out_alpha.loss - base_out_zero.loss) < (non_init_out_alpha.loss - non_init_out_zero.loss)
        
