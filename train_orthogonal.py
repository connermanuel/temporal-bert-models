"""
Wikipedia Finetuning
Finetune both bert-base-uncased and tempobert initialized to bert-base-uncased using the Wikipedia dataset.
"""
from torch.cuda import empty_cache
from datasets import load_from_disk
from models.orthogonal_weight_attention_naive import BertForOrthogonalMaskedLM
from transformers import AutoTokenizer, AutoModelForMaskedLM, DataCollatorForLanguageModeling, TrainingArguments, Trainer, AutoConfig

import argparse
import logging
import gc
import os
from pathlib import Path

logging.basicConfig(filename='run.log', level=logging.DEBUG)

def main(args):
    if args.output_dir is None:
        args.output_dir = f"./output/lr-{args.lr}_alpha-{args.alpha}"
    
    logging.info(f"Loading dataset...")
    dataset = load_from_disk(args.data_dir)

    save_strategy = 'epoch'
    save_steps = len(dataset['train'])
    if args.saves_per_epoch > 1:
        save_strategy = 'steps'
        save_steps = len(dataset['train']) // (args.batch_size * args.saves_per_epoch)
    
    bert_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    collator = DataCollatorForLanguageModeling(bert_tokenizer)

    logging.info(f"Initializing model")
    if os.path.exists(f"{args.output_dir}/base"):
        model = BertForOrthogonalMaskedLM.from_pretrained(f"{args.output_dir}/base")
    else:
        logging.info("First time initialization. Creating bert-base-uncased and copying weights.")
        config = AutoConfig.from_pretrained('bert-base-uncased')
        bert_model = AutoModelForMaskedLM.from_pretrained('bert-base-uncased')
        model = BertForOrthogonalMaskedLM(config, n_contexts=args.n_contexts, alpha=args.alpha)
        sd = model.state_dict()
        bert_sd = bert_model.state_dict()
        for k in bert_sd.keys():
            sd[k] = bert_sd[k]
        model.load_state_dict(sd)

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
        eval_dataset=dataset['val'],
        data_collator=collator,
    )

    logging.info(f"Now training for {args.num_epochs} epochs.")
    trainer.train()

    gc.collect()
    empty_cache()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trains orthogonal model on a timestamped dataset.")
    parser.add_argument(
        "--data_dir", 
        help='Path of the huggingface dataset. Defaults to "./data/news_crawl_processed".', default="./data/news_crawl_processed")
    parser.add_argument(
        "--n_contexts", 
        help='Number of contexts/timestamps. Defaults to 10, the number of years in the Temporal KB task.', 
        type=int, default=10)
    parser.add_argument(
        "--output_dir", 
        help='Path to save model checkpoints to. Defaults to "./output/{learning_rate}/{alpha}".', default=None)
    parser.add_argument(
        "--lr", 
        help="Maximum learning rate in a OneCycleLR trainer. Defaults to 1e-05.",
        type=float, default=1e-05)
    parser.add_argument(
        "--alpha", 
        help="Regularization parameter. Defaults to 1.",
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
    
    args = parser.parse_args()
    main(args)