"""Generalized training script for various datasets and model architectures."""
import gc
import logging
import os

import torch
from datasets import load_from_disk
from torch.cuda import empty_cache
from transformers import T5ForConditionalGeneration, BertConfig, TrainingArguments, Seq2SeqTrainingArguments, Trainer, Seq2SeqTrainer
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
    OrthogonalT5Config    
)
from tempo_models.utils import (
    add_special_time_tokens,
    fetch_tokenizer,
    shuffle_batched
)
from tempo_models.utils.metrics import ssm_metric_token_f1_from_predictions, trainer_get_predictions_from_logits
from tempo_models.utils.collator import get_collator

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
    collator = get_collator(args.task, tokenizer)

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
            dataset, tokenizer, args.time_token, args.n_contexts,args.start_year
        )
        if args.attention == "base" and "timestamps" in dataset["train"].features.keys():
            dataset = dataset.remove_columns("timestamps")

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
    elif args.task == "ssm":
        model = initialize_ssm_model(
            args.model_architecture,
            args.attention,
            args.n_contexts,
            args.alpha,
            args.time_token,
            tokenizer.vocab_size,
        )
    

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
        # logging_first_step=True
    )

    

    trainer = TrainerClass(
        model=model,
        args=train_args,
        train_dataset=dataset["train"],
        eval_dataset={
            "templama_val": dataset["templama_val"],
            # "upstream_val": dataset["upstream_val"],
        },
        data_collator=collator,
        preprocess_logits_for_metrics=trainer_get_predictions_from_logits,
        compute_metrics=ssm_metric_token_f1_from_predictions
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
        base_bert_model = BertForMaskedLM.from_pretrained("bert-base-uncased")
        model = base_bert_model
        if attention == "tempo_bert":
            model = BertForTemporalMaskedLM(TempoBertConfig.from_pretrained("bert-base-uncased", n_contexts=n_contexts))
            model = copy_weights(base_bert_model, model)
        elif attention == "orthogonal":
            model = BertForOrthogonalMaskedLM(OrthogonalBertConfig.from_pretrained("bert-base-uncased", n_contexts=n_contexts, alpha=alpha))
            model = copy_weights(base_bert_model, model)
        if time_token == "special":
            model.resize_token_embeddings(vocab_size + n_contexts, pad_to_multiple_of=16)
        return model
    elif model_architecture == "t5":
        raise ValueError(
            "Sorry, we don't support T5 models for masked language modeling yet."
        )

def initialize_ssm_model(
    model_architecture: str,
    attention: str,
    n_contexts: int,
    alpha: float = 0,
    time_token: str = None,
    vocab_size: int = 30522,
):
    """Initializes a model for the first time, ready for training."""
    if model_architecture == "bert":
        raise ValueError(
            "Sorry, we don't support BERT models for salient span masking yet."
        )
    elif model_architecture == "t5":
        base_t5_model = T5ForConditionalGeneration.from_pretrained("t5-base")
        model = base_t5_model
        if attention == "tempo_bert":
            raise ValueError(
                "Sorry, we don't support T5 models with the TempoBERT attention yet."
            )
        elif attention == "orthogonal":
            model = T5ForOrthogonalConditionalGeneration(
                OrthogonalT5Config.from_pretrained("t5-base", n_contexts=n_contexts, alpha=alpha))
            model = copy_weights(base_t5_model, model)
        if time_token == "special":
            model.resize_token_embeddings(vocab_size + n_contexts, pad_to_multiple_of=16)
        return model


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
        "orthogonal": OrthogonalBertConfig,
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
