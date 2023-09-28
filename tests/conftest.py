import pytest
import torch
from datasets import load_from_disk
from transformers import AutoTokenizer, BertForMaskedLM
from torch.utils.data import DataLoader
from tempo_models.utils.collator import CollatorMLM, CollatorCLS
from tempo_models.train import initialize_mlm_model


@pytest.fixture(scope="session")
def device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

##### TOKENIZERS #####
@pytest.fixture(scope="session")
def tokenizer_bert():
    return AutoTokenizer.from_pretrained("bert-base-uncased")

##### DATASETS #####
@pytest.fixture(scope="session")
def dataset_mlm():
    return load_from_disk("tests/sample_data/sample_data_mlm")

@pytest.fixture(scope="session")
def dataloader_mlm(dataset_mlm, tokenizer_bert):
    return DataLoader(
        dataset=dataset_mlm,
        collate_fn=CollatorMLM(tokenizer_bert),
        batch_size=4
        )

@pytest.fixture(scope="session")
def dataset_cls():
    return load_from_disk("tests/sample_data/sample_data_cls")

@pytest.fixture(scope="session")
def dataloader_cls(dataset_cls, tokenizer_bert):
    return DataLoader(
        dataset=dataset_cls,
        collate_fn=CollatorCLS(tokenizer_bert),
        batch_size=4
        )

##### MODELS #####

@pytest.fixture()
def model_bert_base(device):
    return BertForMaskedLM.from_pretrained("bert-base-uncased").to(device)

@pytest.fixture()
def model_bert_orth(device):
    return initialize_mlm_model("bert", "orthogonal", 12).to(device)

