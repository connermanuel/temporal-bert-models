"""Generalized training script for various datasets and model architectures."""
import gc
import logging
import os

import torch
from datasets import load_from_disk
from torch.cuda import empty_cache
from transformers import (
    T5ForConditionalGeneration,
    BertConfig,
    TrainingArguments,
    Seq2SeqTrainingArguments,
    Trainer,
    Seq2SeqTrainer,
)
from transformers.models.bert.modeling_bert import (
    BertForMaskedLM,
    BertForSequenceClassification,
)
from tempo_models.models.bert.orthogonal_bert import (
    BertForOrthogonalMaskedLM,
    BertForOrthogonalSequenceClassification,
    OrthogonalBertConfig,
)
from tempo_models.models.bert.tempo_bert import (
    BertForTemporalMaskedLM,
    BertForTemporalSequenceClassification,
    TempoBertConfig,
)
from tempo_models.models.t5.orthogonal_t5 import (
    T5ForOrthogonalConditionalGeneration,
    OrthogonalT5Config,
)
from tempo_models.utils import (
    get_tokenizer,
    shuffle_batched,
    prepare_time_tokens,
)
from tempo_models.utils.metrics import (
    ssm_metric_token_f1_from_predictions,
    trainer_get_predictions_from_logits,
)
from tempo_models.utils.collator import get_collator
from tempo_models.parser import get_parser


def train(args):
    ### Prepare collator and tokenizer
    logging.info(f"Initializing model")
    tokenizer = get_tokenizer(args.model_architecture)
    collator = get_collator(args.task, tokenizer)

    ### Load and process dataset
    logging.info(f"Loading dataset...")
    dataset = load_from_disk(args.data_dir)
    if args.sample:
        logging.info(f"Sampling {args.sample} entries")
        for k in dataset.keys():
            dataset[k] = dataset[k].select(range(min(args.sample, len(dataset[k]))))

    logging.info(f"Processing the dataset")
    for key in dataset.keys():
        dataset[key] = shuffle_batched(dataset[key], args.batch_size)

    model = initialize_model(
        args.task,
        args.model_architecture,
        args.attention,
        args.n_contexts,
        args.alpha,
        args.pretrain_dir,
        args.num_labels,
    )

    ### Fix ds, tokenizer, and model for time token
    dataset, model, tokenizer = prepare_time_tokens(
        args.time_token,
        dataset,
        tokenizer,
        model,
        args.model_architecture,
        args.n_contexts,
        args.start_year,
    )
    
    if args.attention == "base" and "timestamps" in dataset["train"].features.keys():
        dataset = dataset.remove_columns("timestamps")

    ### Prepare training setup
    TrainerClass = Trainer
    ArgsClass = TrainingArguments
    if args.task == "ssm":
        TrainerClass = Seq2SeqTrainer
        ArgsClass = Seq2SeqTrainingArguments

    train_args = ArgsClass(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        logging_strategy=args.save_strategy,
        evaluation_strategy=args.save_strategy,
        save_strategy=args.save_strategy,
        save_steps=args.save_steps,
        logging_steps=args.save_steps,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        auto_find_batch_size=args.auto_batch,
        gradient_accumulation_steps=args.grad_steps,
        fp16=args.use_fp16,
        use_cpu=args.no_cuda,
        num_train_epochs=args.num_epochs,
        max_steps=args.num_steps,
        remove_unused_columns=args.remove_unused_columns,
    )

    if args.task == "ssm":
        train_args.predict_with_generate = True

    trainer = TrainerClass(
        model=model,
        args=train_args,
        train_dataset=dataset["train"],
        eval_dataset={
            "templama_val": dataset["templama_val"],
            # "upstream_val": dataset["upstream_val"],
        },
        data_collator=collator,
        compute_metrics=ssm_metric_token_f1_from_predictions,
    )

    logging.info(f"Now training for {args.num_epochs} epochs.")
    trainer.save_model(f"{args.output_dir}/checkpoint-0")
    trainer.train(resume_from_checkpoint=args.resume)

    gc.collect()
    empty_cache()


def copy_weights(
    src: torch.nn.Module, dest: torch.nn.Module, prefix=None
) -> torch.nn.Module:
    """Copy the weights from the source model to the destination model."""
    sd = dest.state_dict()
    src_sd = src.state_dict()
    for k in src_sd:
        k = f"{prefix}.{k}" if prefix else k
        if k in sd:
            sd[k] = src_sd[k]
    dest.load_state_dict(sd)
    return dest


def initialize_model(
    task: str,
    model_architecture: str,
    attention: str,
    n_contexts: int,
    alpha: float,
    pretrained_loc: str = None,
    num_labels: int = None,
):
    if task == "mlm":
        if model_architecture == "bert":
            base_bert_model = BertForMaskedLM.from_pretrained("bert-base-uncased")
            model = base_bert_model
            if attention == "tempo_bert":
                model = BertForTemporalMaskedLM(
                    TempoBertConfig.from_pretrained(
                        "bert-base-uncased", n_contexts=n_contexts
                    )
                )
                model = copy_weights(base_bert_model, model)
            elif attention == "orthogonal":
                model = BertForOrthogonalMaskedLM(
                    OrthogonalBertConfig.from_pretrained(
                        "bert-base-uncased", n_contexts=n_contexts, alpha=alpha
                    )
                )
                model = copy_weights(base_bert_model, model)
            return model
        elif model_architecture == "t5":
            raise ValueError(
                "Sorry, we don't support T5 models for masked language modeling yet."
            )
            
    elif task == "cls":
        if model_architecture == "bert":
            if attention == "base":
                config = BertConfig(num_labels=num_labels)
                model = BertForSequenceClassification.from_pretrained(
                    pretrained_loc, config=config
                )
            elif attention == "tempo_bert":
                config = TempoBertConfig(num_labels=num_labels, n_contexts=n_contexts)
                model = BertForTemporalSequenceClassification.from_pretrained(
                    pretrained_loc, config=config
                )
            elif attention == "orthogonal":
                config = OrthogonalBertConfig(
                    num_labels=num_labels, n_contexts=n_contexts, alpha=alpha
                )
                model = BertForOrthogonalSequenceClassification.from_pretrained(
                    pretrained_loc, config=config
                )
        elif model_architecture == "t5":
            raise ValueError("Sorry, we don't support T5 models for classification yet.")
        return model
    
    elif task == "ssm":
        """Initializes a salient span masking model, ready for training."""
        if model_architecture == "bert":
            raise ValueError(
                "Sorry, we don't support BERT models for salient span masking yet."
            )
        elif model_architecture == "t5":
            if attention == "base":
                model = T5ForConditionalGeneration.from_pretrained("t5-base")
            elif attention == "tempo_bert":
                raise ValueError(
                    "Sorry, we don't support T5 models with the TempoBERT attention yet."
                )
            elif attention == "orthogonal":
                config = OrthogonalT5Config.from_pretrained(
                    "t5-base", n_contexts=n_contexts, alpha=alpha
                )
                model = T5ForOrthogonalConditionalGeneration.from_pretrained(
                    "t5-base", config=config
                )
        return model

def setup(args):
    if not os.path.exists(args.data_dir):
        raise ValueError("Data directory does not exist")
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    logging.basicConfig(
        filename=f"{args.output_dir}/run.log",
        format="%(asctime)s %(levelname)-8s %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return args


if __name__ == "__main__":
    parser = get_parser()
    train_args = parser.add_argument_group("Training arguments")
    train_args.add_argument(
        "--output-dir",
        help='Path to save model checkpoints to. Defaults to "./output/{architecture}/{learning_rate}".',
        default=None,
    )
    train_args.add_argument(
        "--lr",
        help="Maximum learning rate in a OneCycleLR trainer. Defaults to 1e-05.",
        type=float,
        default=1e-05,
    )
    train_args.add_argument(
        "--batch-size",
        help="Training batch size. Defaults to 16.",
        type=int,
        default=16,
    )
    train_args.add_argument(
        "--grad-steps",
        help="Number of steps accumulated before backpropagating gradients. Defaults to 1.",
        type=int,
        default=1,
    )
    train_args.add_argument(
        "--num-epochs",
        help="Number of epochs to train for. Defaults to 10.",
        type=int,
        default=10,
    )
    train_args.add_argument(
        "--num-steps",
        help="Number of steps to train for. If used, overrides num-epochs.",
        type=int,
        default=-1,
    )
    train_args.add_argument(
        "--saves-per-epoch",
        help="How many checkpoints are saved in an epoch. Defaults to 1.",
        type=int,
        default=1,
    )
    train_args.add_argument(
        "--save-strategy",
        help="Whether to save by epoch or steps.",
        type=str,
        choices=["epoch", "steps"],
        default="epoch",
    )
    train_args.add_argument(
        "--save-steps",
        help="After how many steps to save. Defaults to 500",
        type=int,
        default=500,
    )
    train_args.add_argument(
        "--auto-batch",
        help="Indicates that we should automatically find the best batch size.",
        action="store_true",
    )
    train_args.add_argument(
        "--no-cuda",
        help="Block trainer from using cuda when available.",
        action="store_true",
    )
    train_args.add_argument(
        "--use-fp16", help="If flag is used, use the fp16 backend.", action="store_true"
    )
    train_args.add_argument(
        "--resume", help="Resume training from checkpoint.", action="store_true"
    )

    args = parser.parse_args()
    args = setup(args)

    train(args)
