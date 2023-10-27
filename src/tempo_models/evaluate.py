"""
Generalized evaluation script for various datasets and model architectures.
Given a directory containing model checkpoints, evaluate all of those checkpoints.
"""
import json
import logging
import os

import tqdm
from datasets import load_from_disk
from torch.nn import Module
from transformers import (
    BertForMaskedLM,
    BertForSequenceClassification,
    T5ForConditionalGeneration,
    Trainer,
    TrainingArguments,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments
)

from tempo_models.models.bert import (
    BertForOrthogonalMaskedLM,
    BertForOrthogonalSequenceClassification,
    BertForTemporalMaskedLM,
    BertForTemporalSequenceClassification,
)
from tempo_models.models.t5 import (
    T5ForOrthogonalConditionalGeneration
)
from tempo_models.utils import (
    evaluate_mlm,
    get_tokenizer,
    remove_unused_columns,
    prepare_time_tokens
)
from tempo_models.utils.metrics import (
    create_metric_func,
    cls_metric_accuracy,
    cls_metric_per_class_f1,
    cls_metric_weighted_f1,
    mlm_metric_accuracy,
    mlm_metric_mrr,
    ssm_metric_token_f1_from_predictions
)
from tempo_models.parser import get_parser
from tempo_models.utils.collator import get_collator

def evaluate(args):

    ### Prepare collator and tokenizer
    tokenizer = get_tokenizer(args.model_architecture)
    collator = get_collator(args.task, tokenizer)

    ### Load and process dataset
    logging.info(f"Loading dataset...")
    dataset = load_from_disk(args.data_dir)
    try:
        dataset = dataset[args.split]
    except KeyError:
        raise KeyError(
            f"The split {args.split} does not exist in the dataset. Existing splits are: {dataset.column_names}"
        )

    if args.sample:
        logging.info(f"Sampling {args.sample} entries")
        dataset = dataset.select(range(min(args.sample, len(dataset))))

    ### Fix ds, tokenizer, and model for time token
    dataset, _, tokenizer = prepare_time_tokens(
        args.time_token,
        dataset,
        tokenizer,
        None,
        args.model_architecture,
        args.n_contexts,
        args.start_year,
        resize_model = False
    )
    
    if args.attention == "base" and "timestamps" in dataset.features.keys():
        dataset = dataset.remove_columns("timestamps")

    ### Prepare evaluation setup
    if args.checkpoint_dir:
        model_dirs = [args.checkpoint_dir]
    elif args.checkpoint_group_dir:
        model_dirs = [
            f"{args.checkpoint_group_dir}/{path}"
            for path in os.listdir(args.checkpoint_group_dir)
            if "checkpoint" in path
        ]

    ModelClass = get_model_class(args.model_architecture, args.attention, args.task)

    logging.info(f"Evaluating models...")
    TrainerClass = Trainer
    ArgsClass = TrainingArguments
    if args.task == "ssm":
        TrainerClass = Seq2SeqTrainer
        ArgsClass = Seq2SeqTrainingArguments

    eval_args = ArgsClass(
        output_dir=".",
        per_device_eval_batch_size=args.batch_size,
        remove_unused_columns=False,
        fp16=args.use_fp16,
        use_cpu=args.no_cuda,
    )

    if args.task == "ssm":
        eval_args.predict_with_generate = True

    metric_func = create_metric_func({})
    if args.task == "mlm":
        metric_func = create_metric_func(
            {"mrr": mlm_metric_mrr, "accuracy": mlm_metric_accuracy}
        )
    elif args.task == "cls":
        metric_func = create_metric_func(
            {
                "accuracy": cls_metric_accuracy,
                "per_class_f1": cls_metric_per_class_f1,
                "weighted_f1": cls_metric_weighted_f1,
            }
        )
    elif args.task == "ssm":
        metric_func = lambda v: ssm_metric_token_f1_from_predictions(v, dataset["id"])

    results = {}
    for model_dir in tqdm.tqdm(model_dirs):
        model = ModelClass.from_pretrained(model_dir)

        if args.task == "cls" or args.task == "ssm":
            trainer = TrainerClass(
                model=model,
                args=eval_args,
                eval_dataset=dataset,
                compute_metrics=metric_func,
                data_collator=collator,
            )
            model_results = trainer.evaluate()
        elif args.task == "mlm":
            model_results = evaluate_mlm(
                model, dataset, collator, no_cuda=args.no_cuda
            )  # TODO: Figure this out!! It would be much better if we can just do MLM straight through the trainer.
        results[model_dir] = model_results

        with open(f"{args.results_dir}/results.json", "w") as f:
            json.dump(results, f)


def get_model_class(model_architecture: str, attention: str, task: str) -> type(Module):
    MODELS = {
        "mlm": {
            "bert": {
                "base": BertForMaskedLM,
                "tempo_bert": BertForTemporalMaskedLM,
                "orthogonal": BertForOrthogonalMaskedLM,
            },
        },
        "cls": {
            "bert": {
                "base": BertForSequenceClassification,
                "tempo_bert": BertForTemporalSequenceClassification,
                "orthogonal": BertForOrthogonalSequenceClassification,
            },
        },
        "ssm": {
            "t5": {
                "base": T5ForConditionalGeneration,
                "orthogonal": T5ForOrthogonalConditionalGeneration
            }
        }
    }

    try:
        return MODELS[task][model_architecture][attention]
    except KeyError:
        raise ValueError(
            f"Sorry, we don't support evaluation for {model_architecture} {attention} for {task}"
        )

def setup(args):
    ### Fix kwargs, create directories, and setup logging
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    model_str = f"{args.model_architecture}"
    if args.model_architecture == "orthogonal":
        model_str = f"{args.model_architecture}_{args.alpha}"
    if args.checkpoint_dir is None and args.checkpoint_group_dir is None:
        args.checkpoint_dir = f"outputs/{model_str}"
    if args.results_dir is None:
        args.results_dir = f"results/{model_str}"

    if args.checkpoint_dir and not os.path.exists(args.checkpoint_dir):
        raise ValueError("Checkpoint directory does not exist")
    elif args.checkpoint_group_dir and not os.path.exists(args.checkpoint_group_dir):
        raise ValueError("Checkpoint group directory does not exist")
    if not os.path.exists(args.data_dir):
        raise ValueError("Data directory does not exist")
    if not os.path.exists(args.results_dir):
        os.makedirs(args.results_dir)

    logging.basicConfig(
        filename=f"{args.results_dir}/run.log",
        format="%(asctime)s %(levelname)-8s %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return args

if __name__ == "__main__":
    parser = get_parser()
    eval_args = parser.add_argument_group("Evaluation arguments")

    checkpoint_args = eval_args.add_mutually_exclusive_group(required=True)
    checkpoint_args.add_argument(
        "--checkpoint-dir",
        help="If used, path of the huggingface checkpoint. Overrides checkpoint-group-dir.",
    )
    checkpoint_args.add_argument(
        "--checkpoint-group-dir",
        help="If used, path of directory containing huggingface checkpoints.",
    )

    eval_args.add_argument(
        "--results-dir",
        help="Path to directory to store evaluation results to.",
        required=True,
    )
    eval_args.add_argument(
        "--batch-size",
        help="Evaluation batch size. Defaults to 16.",
        type=int,
        default=16,
    )
    eval_args.add_argument(
        "--no-cuda",
        help="If flag is used, block trainer from using cuda when available.",
        action="store_true",
    )
    eval_args.add_argument(
        "--use-fp16", help="If flag is used, use the fp16 backend.", action="store_true"
    )
    eval_args.add_argument(
        "--split",
        help="The split of the dataset to use for evaluation. Defaults to test.",
        default="test",
    )

    args = parser.parse_args()
    args = setup(args)

    evaluate(args)
