"""Generalized training script for various datasets and model architectures."""
import gc
import logging
import os

import torch
from datasets import load_from_disk
from torch.cuda import empty_cache
from transformers import AutoModelForMaskedLM, BertConfig, TrainingArguments
from transformers.models.bert.modeling_bert import (
    BertForMaskedLM,
    BertForSequenceClassification,
)

from tempo_models.models.bert.orthogonal_bert import (
    BertForOrthogonalMaskedLM,
    BertForOrthogonalSequenceClassification,
    OrthogonalConfig,
)
from tempo_models.models.bert.tempo_bert import (
    BertForTemporalMaskedLM,
    BertForTemporalSequenceClassification,
    TempoBertConfig,
)
from tempo_models.utils import (
    NonShuffledTrainer,
    add_special_time_tokens,
    fetch_tokenizer,
    shuffle_batched,
)
from tempo_models.utils.collator import CollatorCLS, CollatorMLM


def train(args):
    ### Fix kwargs, create directories, and setup logging
    if not os.path.exists(args.data_dir):
        raise ValueError("Data directory does not exist")
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    logging.basicConfig(
        filename = f"{args.output_dir}/run.log",
        format="%(asctime)s %(levelname)-8s %(message)s",
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S')

    ### Prepare collator and tokenizer
    logging.info(f"Initializing model")
    tokenizer = fetch_tokenizer(args.model_architecture, args.time_token, args.n_contexts)

    if args.task == "mlm":
        collator = CollatorMLM(tokenizer)
    elif args.task == "cls":
        collator = CollatorCLS(tokenizer)

    ### Load and process dataset
    logging.info(f"Loading dataset...")
    dataset = load_from_disk(args.data_dir)
    if args.sample:
        logging.info(f"Sampling {args.sample} entries")
        for k in dataset.keys():
            dataset[k] = dataset[k].select(range(min(args.sample, len(dataset[k]))))

    logging.info(f"Processing the dataset")
    if not args.skip_process:
        for key in dataset.keys():
            dataset[key] = shuffle_batched(dataset[key], args.batch_size)
        dataset = add_special_time_tokens(
            dataset, tokenizer.vocab_size, args.time_token
        )

    if args.save_dataset:
        logging.info(f"Saving the dataset to {args.save_dataset}")
        dataset.save_to_disk(args.save_dataset)

    ### Prepare model
    if args.task == "mlm":
        model = initialize_mlm_model(
            args.model_architecture,
            args.attention,
            args.n_contexts,
            args.alpha,
            args.time_token,
            tokenizer.vocab_size,
        )
    elif args.task == "cls":
        model = initialize_cls_model_from_mlm(
            args.model_architecture,
            args.attention,
            args.pretrain_dir,
            args.num_labels,
            args.n_contexts,
            args.alpha,
            args.time_token,
            tokenizer.vocab_size,
        )

    ### Prepare training setup
    save_strategy = "epoch"
    save_steps = len(dataset["train"])
    if args.saves_per_epoch > 1:
        save_strategy = "steps"
        save_steps = max(
            len(dataset["train"]) // (args.batch_size * args.saves_per_epoch), 1
        )

    train_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        logging_strategy="epoch",
        save_strategy=save_strategy,
        save_steps=save_steps,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        auto_find_batch_size=args.auto_batch,
        gradient_accumulation_steps=args.grad_steps,
        fp16=args.use_fp16,
        use_cpu=args.no_cuda,
        num_train_epochs=args.num_epochs,
        max_steps=args.num_steps,
        remove_unused_columns=True,
    )

    trainer = NonShuffledTrainer(
        model=model,
        args=train_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        data_collator=collator,
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


def initialize_mlm_model(
    model_architecture: str,
    attention: str,
    n_contexts: int,
    alpha: float = 0,
    time_token: str = None,
    vocab_size: int = 30522,
):
    """Initializes a model for the first time, ready for training."""
    if model_architecture == "bert":
        base_bert_model = AutoModelForMaskedLM.from_pretrained("bert-base-uncased")
        model = base_bert_model
        if attention == "tempo_bert":
            model = BertForTemporalMaskedLM(TempoBertConfig(n_contexts))
            model = copy_weights(base_bert_model, model)
        elif attention == "orthogonal":
            model = BertForOrthogonalMaskedLM(OrthogonalConfig(n_contexts, alpha))
            model = copy_weights(base_bert_model, model)
        if time_token == "special":
            model.resize_token_embeddings(vocab_size + n_contexts, pad_to_multiple_of=16)
        return model
    elif model_architecture == "t5":
        raise ValueError(
            "Sorry, we don't support T5 models for masked language modeling yet."
        )


def initialize_cls_model_from_mlm(
    model_architecture: str,
    attention: str,
    pretrained_loc: str,
    num_labels: int,
    n_contexts: int,
    alpha: float,
    time_token: str,
    vocab_size: int,
):
    dispatch_dict_mlm = {
        "bert": BertForMaskedLM,
        "tempo_bert": BertForTemporalMaskedLM,
        "orthogonal": BertForOrthogonalMaskedLM,
    }
    dispatch_dict_cls = {
        "bert": BertForSequenceClassification,
        "tempo_bert": BertForTemporalSequenceClassification,
        "orthogonal": BertForOrthogonalSequenceClassification,
    }
    dispatch_dict_config = {
        "bert": BertConfig,
        "tempo_bert": TempoBertConfig,
        "orthogonal": OrthogonalConfig,
    }

    pretrained_model = dispatch_dict_mlm[model_architecture].from_pretrained(
        pretrained_loc
    )
    config = dispatch_dict_config[model_architecture](
        num_labels=num_labels, n_contexts=n_contexts, alpha=alpha
    )
    model = dispatch_dict_cls[model_architecture](config)
    model = copy_weights(pretrained_model, model)
    if time_token == "special":
        model.resize_token_embeddings(vocab_size + n_contexts, pad_to_multiple_of=16)

    return model
