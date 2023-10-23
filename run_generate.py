"""Script that we're using to generate the results from a model checkpoint."""
print("Import load_from_disk")
from datasets import load_from_disk
# print("Import T5ForConditionalGeneration")
# from transformers import T5ForConditionalGeneration
print("Import AutoTokenizer")
from transformers import AutoTokenizer
print("Doing the pkg imports")
from tempo_models.models.t5 import T5ForOrthogonalConditionalGeneration
from tempo_models.utils.collator import CollatorSSM
from tempo_models.utils import add_special_time_tokens

print("Doing the actual code stuff")
TIME_TOKENS = None


# model = T5ForConditionalGeneration.from_pretrained("t5-base")
tokenizer = AutoTokenizer.from_pretrained("t5-base")
collator = CollatorSSM(tokenizer)
dataset = load_from_disk("/local/scratch/shared/groupdir2/temporal_kb/data/wmt_templama")
dataset = dataset["templama_val"]



