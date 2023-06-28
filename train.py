"""Generalized training script for various datasets and model architectures."""
import torch
from torch.cuda import empty_cache
from datasets import load_from_disk
from models.orthogonal_weight_attention import BertForOrthogonalMaskedLM, OrthogonalConfig
from models.temporal_self_attention import BertForTemporalMaskedLM, TempoBertConfig
from transformers import AutoTokenizer, AutoModelForMaskedLM, DataCollatorForLanguageModeling, TrainingArguments, AutoConfig

from utils import get_collator, NonShuffledTrainer, sort_by_timestamp, shuffle_batched, add_special_time_tokens, fix_timestamps, copy_weights
import argparse
import logging
import gc
import os

def initialize_model(model_architecture: str, n_contexts: int, alpha: float):
    """Initializes a model for the first time, ready for training."""   
    bert_model = AutoModelForMaskedLM.from_pretrained('bert-base-uncased')
    if model_architecture == "bert":
        return bert_model
    elif model_architecture == "tempo":
        config = TempoBertConfig(n_contexts)
        model = BertForTemporalMaskedLM(config)
    elif model_architecture == "orthogonal":
        config = OrthogonalConfig(n_contexts, alpha)
        model = BertForOrthogonalMaskedLM(config)
    model = copy_weights(bert_model, model)
    return model

def main(args):
    ### Fix kwargs, create directories, and setup logging
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    if args.output_dir is None:
        args.output_dir = f"./output/{args.model_architecture}/lr-{args.lr}"
        if args.model_architecture == "orthogonal":
            args.output_dir = f"{args.output_dir}_alpha-{args.alpha}"
    
    if not os.path.exists(args.data_dir):
        raise ValueError("Data directory does not exist")
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    logging.basicConfig(
        filename = f"{args.output_dir}/run.log",
        format="%(asctime)s %(levelname)-8s %(message)s",
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S')
       
    logging.info(f"Initializing model")
    bert_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    collator = DataCollatorForLanguageModeling(bert_tokenizer)

    ### Prepare collator and tokenizer
    bert_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    DEFAULT_TOKENIZER_LEN = len(bert_tokenizer)
    if args.add_time_tokens == "special":
        special_tokens = [f"timestamp: {t} text: " for t in range(args.n_contexts)]
        bert_tokenizer.add_tokens(special_tokens)
    
    mask = not args.no_mask
    if args.add_time_tokens == "string":
        collator = get_collator(bert_tokenizer, do_masking=mask)
    elif args.add_time_tokens == "special":
        collator = get_collator(bert_tokenizer, n_tokens=1, do_masking=mask)
    elif mask:
        collator = DataCollatorForLanguageModeling(bert_tokenizer)
    else:
        collator = get_collator(bert_tokenizer, do_masking=mask)

    ### Load and process dataset
    logging.info(f"Loading dataset...")
    dataset = load_from_disk(args.data_dir)
    if args.sample:
        logging.info(f"Sampling {args.sample} entries")
        for k in dataset.keys():
            dataset[k] = dataset[k].select(range(min(args.sample, len(dataset[k]))))
    
    logging.info(f"Processing the dataset")
    if args.process_dataset:
        dataset = sort_by_timestamp(dataset)
        for key in dataset.keys():
            dataset[key] = shuffle_batched(dataset[key], args.batch_size)
        if args.add_time_tokens == "string":
            logging.info(f"Adding string time tokens")
            ## TODO
        elif args.add_time_tokens == "special":
            logging.info(f"Adding special time tokens")
            dataset = add_special_time_tokens(dataset, DEFAULT_TOKENIZER_LEN)
    
    if args.save_dataset:
        logging.info(f"Saving the dataset to {args.save_dataset}")
        dataset.save_to_disk(args.save_dataset)
    
    if args.model_architecture != "bert":
        dataset = dataset.map(fix_timestamps, batched=True)

    ### Prepare model
    model = initialize_model(args.model_architecture, args.n_contexts, args.alpha)
    if args.add_time_tokens == "special":
        model.resize_token_embeddings(len(bert_tokenizer))

    ### Prepare training setup
    save_strategy = 'epoch'
    save_steps = len(dataset['train'])
    if args.saves_per_epoch > 1:
        save_strategy = 'steps'
        save_steps = max(len(dataset['train']) // (args.batch_size * args.saves_per_epoch), 1)
    
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
        no_cuda=args.no_cuda,
        num_train_epochs=args.num_epochs,
        remove_unused_columns=True,
    )
    trainer = NonShuffledTrainer(
        model=model,
        args=train_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['test'],
        data_collator=collator
    )

    logging.info(f"Now training for {args.num_epochs} epochs.")
    trainer.train(resume_from_checkpoint=args.resume)

    gc.collect()
    empty_cache()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trains a model.")
    parser.add_argument(
        "-m", "--model_architecture",
        help="The model architecture to train.",
        choices=["bert", "tempo_bert", "orthogonal"], required=True)
    parser.add_argument(
        "--data-dir", 
        help='Path of the huggingface dataset. Defaults to "./data/news_crawl_processed".', default="./data/news_crawl_processed")
    parser.add_argument(
        "--n-contexts", 
        help='Number of contexts/timestamps. Defaults to 2, the number of timestamps in the SemEval dataset.', 
        type=int, default=2)
    parser.add_argument(
        "--output-dir", 
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
        "--batch-size", 
        help="Training batch size. Defaults to 16.",
        type=int, default=16)
    parser.add_argument(
        "--grad-steps", 
        help="Number of steps accumulated before backpropagating gradients. Defaults to 1.",
        type=int, default=1)
    parser.add_argument(
        "--num-epochs", 
        help="Number of epochs to train for. Defaults to 10.",
        type=int, default=10)
    parser.add_argument(
        "--saves-per-epoch", 
        help="How many checkpoints are saved in an epoch. Defaults to 5.",
        type=int, default=5)
    parser.add_argument(
        "--no-cuda", help="Block trainer from using cuda when available.",
        action='store_true')
    parser.add_argument(
        "--use-fp16", help="If flag is used, use the fp16 backend.",
        action='store_true')
    parser.add_argument(
        "--add-time-tokens", help="Modifies the dataset to insert generic special time tokens. Use 'string' for tokenized strings, and 'special' for inserted special tokens.",
        choices=[None, "none", "string", "special"], default=None)
    parser.add_argument(
        "--sample", help="Indicates how many documents to use. If unset, uses the entire dataset.",
        type=int, default=0)
    parser.add_argument(
        "--auto-batch", help="Indicates that we should automatically find the best batch size.",
        action='store_true')
    parser.add_argument(
        "--process-dataset", help="Performs sorting and batch shuffling, and prepends time tokens if needed.",
        action='store_true')
    parser.add_argument(
        "--save-dataset", help="After processing, stores the dataset to this location.", default=None)
    parser.add_argument(
        "--resume", help="Resume training from checkpoint.", action='store_true')
    parser.add_argument(
        "--no_mask", help="Do not use a masked language modeling collator. Used when the dataset already has tokens masked out.", action="store_true")
    
    args = parser.parse_args()
    main(args)