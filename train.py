"""
Generalized training script for various datasets and model architectures.
"""
import torch
from torch.cuda import empty_cache
from datasets import load_from_disk
from orthogonal_weight_attention import BertForOrthogonalMaskedLM
from temporal_self_attention import BertForTemporalMaskedLM
from transformers import AutoTokenizer, AutoModelForMaskedLM, DataCollatorForLanguageModeling, TrainingArguments, Trainer, AutoConfig

import argparse
import logging
import gc
import os
from pathlib import Path
from utils import get_time_token_collator


def copy_weights(src: torch.nn.Module, dest: torch.nn.Module):
    """Copy the weights from the source model to the destination model."""
    sd = dest.state_dict()
    src_sd = src.state_dict()
    for k in src_sd:
        sd[k] = src_sd[k]
    dest.load_state_dict(sd)
    return dest

def initialize_model(model_architecture: str, n_contexts: int):
    dispatch_dict = {
        "tempo_bert": BertForTemporalMaskedLM,
        "orthogonal": BertForOrthogonalMaskedLM
    }

    config = AutoConfig.from_pretrained('bert-base-uncased')
    bert_model = AutoModelForMaskedLM.from_pretrained('bert-base-uncased')
    if model_architecture == "bert":
        return bert_model
    else:
        ModelClass = dispatch_dict[model_architecture]
        model = ModelClass(config=config, n_contexts=n_contexts)
        model = copy_weights(bert_model, model)
        return model

def main(args):
    if args.output_dir is None:
        args.output_dir = f"./output/{args.model_architecture}/lr-{args.lr}"
        if args.model_architecture == "orthogonal":
            args.output_dir = f"{args.output_dir}_alpha-{args.alpha}"
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    logging.basicConfig(
        filename = f"{args.output_dir}/run.log",
        format="%(asctime)s %(levelname)-8s %(message)s",
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S')
    
    logging.info(f"Loading dataset...")
    dataset = load_from_disk(args.data_dir)

    save_strategy = 'epoch'
    save_steps = len(dataset['train'])
    if args.saves_per_epoch > 1:
        save_strategy = 'steps'
        save_steps = len(dataset['train']) // (args.batch_size * args.saves_per_epoch)

    bert_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    collator = DataCollatorForLanguageModeling(bert_tokenizer)
    if args.use_time_tokens:
        collator = get_time_token_collator(bert_tokenizer)

    logging.info(f"Initializing model")
    model = initialize_model(args.model_architecture, args.n_contexts)

    train_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        logging_strategy="epoch",
        save_strategy=save_strategy,
        save_steps=save_steps,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_steps,
        fp16=args.use_fp16,
        no_cuda=args.no_cuda,
        num_train_epochs=args.num_epochs,
        remove_unused_columns=True,
    )
    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['test'],
        data_collator=collator,
    )

    logging.info(f"Now training for {args.num_epochs} epochs.")
    trainer.train()

    gc.collect()
    empty_cache()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trains a model.")
    parser.add_argument(
        "-m", "--model_architecture",
        help="The model architecture to train",
        choices=["bert", "tempo_bert", "orthogonal"], required=True)
    parser.add_argument(
        "--data_dir", 
        help='Path of the huggingface dataset. Defaults to "./data/news_crawl_processed".', default="./data/news_crawl_processed")
    parser.add_argument(
        "--n_contexts", 
        help='Number of contexts/timestamps. Defaults to 2, the number of timestamps in the SemEval dataset.', 
        type=int, default=2)
    parser.add_argument(
        "--output_dir", 
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
        "--batch_size", 
        help="Training batch size. Defaults to 16.",
        type=int, default=16)
    parser.add_argument(
        "--grad_steps", 
        help="Number of steps accumulated before backpropagating gradients. Defaults to 1.",
        type=int, default=1)
    parser.add_argument(
        "--num_epochs", 
        help="Number of epochs to train for. Defaults to 10.",
        type=int, default=10)
    parser.add_argument(
        "--saves_per_epoch", 
        help="How many checkpoints are saved in an epoch. Defaults to 5.",
        type=int, default=5)
    parser.add_argument(
        "--no_cuda",
        help="Block trainer from using cuda when available. Defaults to false (uses cuda).",
        type=bool, default=False)
    parser.add_argument(
        "--use_fp16", help="If flag is used, use the fp16 backend.",
        action='store_true')
    parser.add_argument(
        "--use_time_tokens", help="Indicates that the dataset has prepeneded time tokens.",
        action='store_true')
    
    args = parser.parse_args()
    main(args)