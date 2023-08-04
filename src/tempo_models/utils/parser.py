import argparse

parser = argparse.ArgumentParser()

model_args = parser.add_argument_group("Model arguments")
model_args.add_argument(
    "-m", "--model_architecture",
    help="The model architecture to train.",
    choices=["bert", "tempo_bert", "orthogonal"], required=True)
model_args.add_argument(
    "--task",
    help="The task to train on. Defaults to mlm", 
    type=str, choices=["mlm", "cls"], default="mlm")
model_args.add_argument(
    "--n-contexts", 
    help='Number of contexts/timestamps. Defaults to 2, the number of timestamps in the SemEval dataset.', 
    type=int, default=2)
model_args.add_argument(
    "--pretrain-dir",
    help="Location of pretrained model. Required for classification task.",
    type=str, default=None)
model_args.add_argument(
    "--num-labels",
    help="Number of labels. Required for classification task.",
    type=int, default=None)
model_args.add_argument(
    "--alpha", 
    help="Regularization parameter. Defaults to 1. Only used for orthogonal model.",
    type=float, default=1)

data_args = parser.add_argument_group("Data arguments")
data_args.add_argument(
    "--data-dir", help="Path of the huggingface dataset.")
data_args.add_argument(
    "--add-time-tokens", help="Modifies the dataset to insert generic special time tokens. Use 'string' for tokenized strings, and 'special' for inserted special tokens.",
    choices=[None, "none", "string", "special"], default=None)
data_args.add_argument(
    "--sample", help="Indicates how many documents to use. If unset, uses the entire dataset.",
    type=int, default=0)
data_args.add_argument(
    "--process-dataset", help="Performs sorting and batch shuffling, and prepends time tokens if needed.",
    action='store_true')
data_args.add_argument(
    "--save-dataset", help="After processing, stores the dataset to this location.", default=None)
data_args.add_argument(
    "--no_mask", help="Do not use a masked language modeling collator. Used when the dataset already has tokens masked out.", action="store_true")