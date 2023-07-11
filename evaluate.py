import argparse
from tempo_models.evaluate import evaluate

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trains a model.")
    parser.add_argument(
        "-m", "--model_architecture",
        help="The model architecture to train",
        choices=["bert", "tempo_bert", "orthogonal", "naive"], required=True)
    parser.add_argument(
        "--n-contexts", 
        help='Number of contexts/timestamps. Defaults to 2, the number of timestamps in the SemEval dataset.', 
        type=int, default=2)
    parser.add_argument(
        "--data-dir", 
        help="Path of the huggingface dataset.", required=True)
    parser.add_argument(
        "--checkpoint-path", 
        help='If used, path of the huggingface checkpoint. Overrides checkpoint-group-dir.', default=None)
    parser.add_argument(
        "--checkpoint-group-dir", 
        help='If used, path of directory containing huggingface checkpoints.', default=None)
    parser.add_argument(
        "--results-dir", 
        help='Path to directory to store checkpoints to. Defaults to "results/{architecture}".', default=None)
    parser.add_argument(
        "--alpha", 
        help="Regularization parameter. Defaults to 1. Only used for orthogonal model directory naming.",
        type=float, default=1)
    parser.add_argument(
        "--batch-size", 
        help="Evaluation batch size. Defaults to 16.",
        type=int, default=16)
    parser.add_argument(
        "--no-cuda", help="If flag is used, block trainer from using cuda when available.",
        action='store_true')
    parser.add_argument(
        "--use-fp16", help="If flag is used, use the fp16 backend.",
        action='store_true')
    parser.add_argument(
        "--add-time-tokens", help="Modifies the dataset to insert generic special time tokens. Use 'string' for tokenized strings, and 'special' for inserted special tokens.",
        choices=[None, "none", "string", "special"], default=None)
    parser.add_argument(
        "--process-dataset", help="Performs sorting and batch shuffling, and prepends time tokens if needed.",
        action='store_true')
    parser.add_argument(
        "--save-dataset", help="After processing, stores the dataset to this location.", default=None)
    parser.add_argument(
        "--split", help="The split of the dataset to use for evaluation. Defaults to test.",
        default="test")    
    parser.add_argument(
        "--sample", help="Indicates how many documents to use. If unset, uses the entire dataset.",
        type=int, default=0)
    parser.add_argument(
        "--f1", help="Indicates that we should evaluate span F1.", action="store_true")
    parser.add_argument(
        "--no-mask", help="Do not use a masked language modeling collator. Used when the dataset already has tokens masked out.", action="store_true")
    
    args = parser.parse_args()
    evaluate(args)