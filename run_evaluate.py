from arg_parser import parser
from tempo_models.evaluate import evaluate
import logging

if __name__ == "__main__":
    eval_args = parser.add_argument_group("Evaluation arguments")

    checkpoint_args = eval_args.add_mutually_exclusive_group("Checkpoint loading arguments", required=True)
    checkpoint_args.add_argument(
        "--checkpoint-dir", 
        help='If used, path of the huggingface checkpoint. Overrides checkpoint-group-dir.')
    checkpoint_args.add_argument(
        "--checkpoint-group-dir", 
        help='If used, path of directory containing huggingface checkpoints.')
    
    eval_args.add_argument(
        "--results-dir", 
        help='Path to directory to store checkpoints to.', required=True)
    eval_args.add_argument(
        "--batch-size", 
        help="Evaluation batch size. Defaults to 16.",
        type=int, default=16)
    eval_args.add_argument(
        "--no-cuda", help="If flag is used, block trainer from using cuda when available.",
        action='store_true')
    eval_args.add_argument(
        "--use-fp16", help="If flag is used, use the fp16 backend.",
        action='store_true')
    eval_args.add_argument(
        "--split", help="The split of the dataset to use for evaluation. Defaults to test.",
        default="test")
    
    args = parser.parse_args()

    logging.basicConfig(
        filename = f"{args.output_dir}/run.log",
        format="%(asctime)s %(levelname)-8s %(message)s",
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S')
    

    evaluate(args)