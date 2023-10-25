"""Script that we're using to generate the results from a model checkpoint."""
from datasets import load_from_disk
from transformers import T5ForConditionalGeneration
from transformers import AutoTokenizer
from tempo_models.models.t5 import T5ForOrthogonalConditionalGeneration
from tempo_models.utils.collator import CollatorSSM
from tempo_models.utils import prepare_time_tokens
from torch.utils.data import DataLoader
from tqdm import tqdm

TIME_TOKENS = "special"
DEVICE = "cuda"
KEYS = ["input_ids"]

model = T5ForConditionalGeneration.from_pretrained("D:/Research/temporal_bert_models/output/temporal_kb/test_1/token")
tokenizer = AutoTokenizer.from_pretrained("t5-base")
collator = CollatorSSM(tokenizer)
dataset = load_from_disk("D:/Research/temporal_bert_models/data/templama/dataset_tokens")
dataset = dataset["val"]

dataset, model, tokenizer = prepare_time_tokens(
    TIME_TOKENS,
    dataset,
    tokenizer,
    model,
    "t5",
    9,
    2010,
    False
)

model.to(DEVICE)
# outputs = []
years = []
dl = DataLoader(dataset, batch_size=4, collate_fn=collator)

for input in tqdm(iter(dl)):
    years.extend([t[0] for t in input["timestamps"]])
    # input = input.to(DEVICE)    
    # input = {k: input[k] for k in KEYS}
    # output = model.generate(**input)
    # outputs.extend(tokenizer.batch_decode(output, skip_special_tokens=True))
        

with open("file.txt", "w") as f:
    for year in years:
        f.write(f"{year}\n")