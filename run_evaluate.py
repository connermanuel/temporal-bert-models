from tempo_models.utils.parser import parser
from tempo_models.evaluate import evaluate

if __name__ == "__main__":
    eval_args = parser.add_argument_group("Evaluation arguments")
    eval_args.add_argument(
        "--checkpoint-dir", 
        help='If used, path of the huggingface checkpoint. Overrides checkpoint-group-dir.', default=None)
    eval_args.add_argument(
        "--checkpoint-group-dir", 
        help='If used, path of directory containing huggingface checkpoints.', default=None)
    eval_args.add_argument(
        "--results-dir", 
        help='Path to directory to store checkpoints to. Defaults to "results/{architecture}".', default=None)
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
    evaluate(args)