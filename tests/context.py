import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.bert.orthogonal_weight_attention_naive import BertNaiveOrthogonalTemporalModel, BertForNaiveOrthogonalMaskedLM
from models.bert.orthogonal_weight_attention import BertOrthogonalTemporalModel, BertForOrthogonalMaskedLM, MultiHeadLinear
from utils import *