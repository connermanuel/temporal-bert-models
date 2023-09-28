import logging
from arg_parser import parser
from tempo_models.train import train

if __name__ == "__main__":    
    train_args = parser.add_argument_group("Training arguments")
    train_args.add_argument(
        "--output-dir", 
        help='Path to save model checkpoints to. Defaults to "./output/{architecture}/{learning_rate}".', default=None)
    train_args.add_argument(
        "--lr", 
        help="Maximum learning rate in a OneCycleLR trainer. Defaults to 1e-05.",
        type=float, default=1e-05)
    train_args.add_argument(
        "--batch-size", 
        help="Training batch size. Defaults to 16.",
        type=int, default=16)
    train_args.add_argument(
        "--grad-steps", 
        help="Number of steps accumulated before backpropagating gradients. Defaults to 1.",
        type=int, default=1)
    train_args.add_argument(
        "--num-epochs", 
        help="Number of epochs to train for. Defaults to 10.",
        type=int, default=10)
    train_args.add_argument(
        "--num-steps", 
        help="Number of steps to train for. If used, overrides num-epochs.",
        type=int, default=-1)
    train_args.add_argument(
        "--saves-per-epoch", 
        help="How many checkpoints are saved in an epoch. Defaults to 1.",
        type=int, default=1)
    train_args.add_argument(
        "--auto-batch", help="Indicates that we should automatically find the best batch size.",
        action='store_true')
    train_args.add_argument(
        "--no-cuda", help="Block trainer from using cuda when available.",
        action='store_true')
    train_args.add_argument(
        "--use-fp16", help="If flag is used, use the fp16 backend.",
        action='store_true')
    train_args.add_argument(
        "--resume", help="Resume training from checkpoint.", action='store_true')
    
    args = parser.parse_args()
    
    train(args)