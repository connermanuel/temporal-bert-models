import os
import pytest
import torch
from datasets import load_from_disk
from transformers import AutoTokenizer, BertForMaskedLM, T5ForConditionalGeneration
from torch.utils.data import DataLoader
from tempo_models.utils.collator import CollatorMLM, CollatorCLS, CollatorSSM
from tempo_models.train import initialize_model

@pytest.fixture(scope="session", autouse=True)
def set_cuda_visibility():
    if torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"

@pytest.fixture(scope="session")
def device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

##### TOKENIZERS #####
@pytest.fixture(scope="session")
def tokenizer_bert():
    return AutoTokenizer.from_pretrained("bert-base-uncased")

@pytest.fixture(scope="session")
def tokenizer_t5():
    return AutoTokenizer.from_pretrained("t5-base")

##### DATASETS #####
@pytest.fixture(scope="session")
def dataset_mlm():
    return load_from_disk("tests/sample_data/sample_data_mlm")

@pytest.fixture()
def dataloader_mlm(dataset_mlm, tokenizer_bert):
    return DataLoader(
        dataset=dataset_mlm,
        collate_fn=CollatorMLM(tokenizer_bert),
        batch_size=4
        )

@pytest.fixture(scope="session")
def dataset_cls():
    return load_from_disk("tests/sample_data/sample_data_cls")

@pytest.fixture()
def dataloader_cls(dataset_cls, tokenizer_bert):
    return DataLoader(
        dataset=dataset_cls,
        collate_fn=CollatorCLS(tokenizer_bert),
        batch_size=4
        )

@pytest.fixture(scope="session")
def dataset_ssm():
    return load_from_disk("tests/sample_data/sample_data_ssm")

@pytest.fixture()
def dataloader_ssm(dataset_ssm, tokenizer_t5):
    return DataLoader(
        dataset=dataset_ssm,
        collate_fn=CollatorSSM(tokenizer_t5, debug=True),
        batch_size=4
        )

##### MODELS #####

@pytest.fixture()
def model_bert_base(device):
    return BertForMaskedLM.from_pretrained("bert-base-uncased").to(device)

@pytest.fixture()
def model_bert_orth(device):
    return initialize_model("mlm", "bert", "orthogonal", 12).to(device)

@pytest.fixture()
def model_t5_base(device):
    return T5ForConditionalGeneration.from_pretrained("t5-base").to(device)

@pytest.fixture()
def model_t5_orth(device):
    return initialize_model("ssm", "t5", "orthogonal", 11).to(device)
