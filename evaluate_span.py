import argparse
from utils import evaluate_span_accuracy

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trains a model.")
    parser.add_argument(
        "-m", "--model_architecture",
        help="The model architecture to train",
        choices=["bert", "tempo_bert", "orthogonal", "naive"], required=True)
    parser.add_argument(
        "--data-dir", 
        help="Path of the huggingface dataset.", required=True)
    parser.add_argument(
        "--checkpoint-dir", 
        help='If used, path of the huggingface checkpoint. Overrides checkpoint-group-dir.', default=None)
    parser.add_argument(
        "--checkpoint-group-dir", 
        help='If used, path of directory containing huggingface checkpoints.', default=None)
    parser.add_argument(
        "--results-dir", 
        help='Path to directory to store checkpoints to. Defaults to "results/{architecture}".', default=None)
    parser.add_argument(
        "--alpha", 
        help="Regularization parameter. Defaults to 1. Only used for orthogonal model.",
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
        "--use_time_tokens", help="Indicates that the dataset has prepeneded time tokens. Use 'string' for tokenized strings, and 'special' for inserted special tokens.",
        choices=[None, "none", "string", "special"], default=None)
    parser.add_argument(
        "--sample", help="Indicates that we should only use a small sample of the data.",
        action='store_true')
    
    args = parser.parse_args()
    main(args)