import argparse
from tempo_models.train import train

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trains a model.")
    parser.add_argument(
        "-m", "--model_architecture",
        help="The model architecture to train.",
        choices=["bert", "tempo_bert", "orthogonal"], required=True)
    parser.add_argument(
        "--data-dir", 
        help='Path of the huggingface dataset. Defaults to "./data/news_crawl_processed".', default="./data/news_crawl_processed")
    parser.add_argument(
        "--n-contexts", 
        help='Number of contexts/timestamps. Defaults to 2, the number of timestamps in the SemEval dataset.', 
        type=int, default=2)
    parser.add_argument(
        "--output-dir", 
        help='Path to save model checkpoints to. Defaults to "./output/{architecture}/{learning_rate}".', default=None)
    parser.add_argument(
        "--lr", 
        help="Maximum learning rate in a OneCycleLR trainer. Defaults to 1e-05.",
        type=float, default=1e-05)
    parser.add_argument(
        "--alpha", 
        help="Regularization parameter. Defaults to 1. Only used for orthogonal model.",
        type=float, default=1)
    parser.add_argument(
        "--batch-size", 
        help="Training batch size. Defaults to 16.",
        type=int, default=16)
    parser.add_argument(
        "--grad-steps", 
        help="Number of steps accumulated before backpropagating gradients. Defaults to 1.",
        type=int, default=1)
    parser.add_argument(
        "--num-epochs", 
        help="Number of epochs to train for. Defaults to 10.",
        type=int, default=10)
    parser.add_argument(
        "--saves-per-epoch", 
        help="How many checkpoints are saved in an epoch. Defaults to 5.",
        type=int, default=5)
    parser.add_argument(
        "--no-cuda", help="Block trainer from using cuda when available.",
        action='store_true')
    parser.add_argument(
        "--use-fp16", help="If flag is used, use the fp16 backend.",
        action='store_true')
    parser.add_argument(
        "--add-time-tokens", help="Modifies the dataset to insert generic special time tokens. Use 'string' for tokenized strings, and 'special' for inserted special tokens.",
        choices=[None, "none", "string", "special"], default=None)
    parser.add_argument(
        "--sample", help="Indicates how many documents to use. If unset, uses the entire dataset.",
        type=int, default=0)
    parser.add_argument(
        "--auto-batch", help="Indicates that we should automatically find the best batch size.",
        action='store_true')
    parser.add_argument(
        "--process-dataset", help="Performs sorting and batch shuffling, and prepends time tokens if needed.",
        action='store_true')
    parser.add_argument(
        "--save-dataset", help="After processing, stores the dataset to this location.", default=None)
    parser.add_argument(
        "--resume", help="Resume training from checkpoint.", action='store_true')
    parser.add_argument(
        "--no_mask", help="Do not use a masked language modeling collator. Used when the dataset already has tokens masked out.", action="store_true")
    
    args = parser.parse_args()
    train(args)