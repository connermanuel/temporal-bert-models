"""
Wikipedia Finetuning
Finetune both bert-base-uncased and tempobert initialized to bert-base-uncased using the Wikipedia dataset.
"""
from torch.cuda import empty_cache
from datasets import load_from_disk
from temporal_self_attention import BertForTemporalMaskedLM
from transformers import AutoTokenizer, AutoModelForMaskedLM, DataCollatorForLanguageModeling, TrainingArguments, Trainer
from utils import add_zero_timestamp

import argparse
import logging
import gc
from pathlib import Path

logging.basicConfig(filename='run.log', level=logging.DEBUG)

def main(args):
    data_dir = args.data_dir
    output_dir = args.output_dir
    lr = args.lr
    batch_size = args.batch_size
    gradient_accumulation_steps = args.grad_steps
    num_epochs = args.num_epochs
    no_cuda = args.no_cuda
    fp16 = args.use_fp16
    
    logging.info(f"Loading dataset...")
    dataset = load_from_disk(data_dir)
    for k in dataset:
        if 'timestamps' not in dataset[k].features:
            logging.info(f"Adding timestamps to {k}")
            dataset[k] = add_zero_timestamp(dataset[k])

    save_strategy = 'epoch'
    save_steps = len(dataset['train'])
    if args.saves_per_epoch > 1:
        save_strategy = 'steps'
        save_steps = len(dataset['train']) // (batch_size * args.saves_per_epoch)
    
    bert_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    collator = DataCollatorForLanguageModeling(bert_tokenizer)
    output_dir = Path(output_dir)

    # ## Train Temporal Model
    if args.train_tempo:
        logging.info(f"Initializing Tempo-BERT")
        temporal_bert_mlm = BertForTemporalMaskedLM.from_pretrained('models/temporal_bert_mlm_base')
        
        tempo_training_args = TrainingArguments(
            output_dir=output_dir / 'temporal',
            overwrite_output_dir=True,
            logging_strategy="epoch",
            save_strategy=save_strategy,
            save_steps=save_steps,
            learning_rate=lr,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            fp16=fp16,
            no_cuda=no_cuda,
            num_train_epochs=num_epochs,
            remove_unused_columns=True,
        )

        tempo_trainer = Trainer(
            model=temporal_bert_mlm,
            args=tempo_training_args,
            train_dataset=dataset['train'],
            eval_dataset=dataset['test'],
            data_collator=collator,
        )

        logging.info(f"Now training Tempo-BERT for {num_epochs} epochs.")
        tempo_trainer.train()

        gc.collect()
        empty_cache()

    # ## Train Non-Temporal Model
    if args.train_base:
        logging.info(f"Initializing BERT")
        bert_mlm = AutoModelForMaskedLM.from_pretrained('bert-base-uncased')
        
        non_tempo_training_args = TrainingArguments(
            output_dir=output_dir / 'base',
            overwrite_output_dir=True,
            logging_strategy="epoch",
            save_strategy=save_strategy,
            save_steps=save_steps,
            learning_rate=lr,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            fp16=fp16,
            no_cuda=no_cuda,
            num_train_epochs=num_epochs,
            remove_unused_columns=True,
        )

        non_tempo_trainer = Trainer(
            model=bert_mlm,
            args=non_tempo_training_args,
            train_dataset=dataset['train'],
            eval_dataset=dataset['test'],
            data_collator=collator
        )

        logging.info(f"Now training BERT for {num_epochs} epochs.")
        non_tempo_trainer.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Finetune both non-temporal and temporal BERT models on a training set.")
    
    parser.add_argument('--train_base', dest='train_base', action='store_true')
    parser.add_argument('--skip_base', dest='train_base', action='store_false')
    parser.set_defaults(train_base=True)
    
    parser.add_argument('--train_tempo', dest='train_tempo', action='store_true')
    parser.add_argument('--skip_tempo', dest='train_tempo', action='store_false')
    parser.set_defaults(train_tempo=True)
    
    parser.add_argument(
        "--data_dir", 
        help='Path of the huggingface dataset. Defaults to "./data/wiki_dataset".', default='./data/wiki_dataset')
    parser.add_argument(
        "--output_dir", 
        help='Path to save model checkpoints to. Defaults to "./output".', default='./output')
    parser.add_argument(
        "--lr", 
        help="Learning rate. Defaults to 1e-08.",
        type=float, default=1e-08)
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
        help="Number of epochs to train for. Defaults to 5.",
        type=int, default=5)
    parser.add_argument(
        "--saves_per_epoch", 
        help="How many checkpoints are saved in an epoch. Defaults to 1 (save at the end of every epoch).",
        type=int, default=1)
    parser.add_argument(
        "--no_cuda",
        help="Block trainer from using cuda when available. Defaults to false (uses cuda).",
        type=bool, default=False)
    parser.add_argument(
        "--use_fp16", help="If flag is used, use the fp16 backend.",
        action='store_true')
    args = parser.parse_args()
    main(args)
