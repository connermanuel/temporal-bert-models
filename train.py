"""Generalized training script for various datasets and model architectures."""
import torch
from torch.cuda import empty_cache
from datasets import load_from_disk
from models.orthogonal_weight_attention_naive import BertForNaiveOrthogonalMaskedLM
from models.orthogonal_weight_attention import BertForOrthogonalMaskedLM
from models.temporal_self_attention import BertForTemporalMaskedLM
from transformers import AutoTokenizer, AutoModelForMaskedLM, DataCollatorForLanguageModeling, TrainingArguments, AutoConfig

from utils import get_time_token_collator, NonShuffledTrainer, sort_by_timestamp, shuffle_batched, add_special_time_tokens
import argparse
import logging
import gc
import os


def copy_weights(src: torch.nn.Module, dest: torch.nn.Module):
    """Copy the weights from the source model to the destination model."""
    sd = dest.state_dict()
    src_sd = src.state_dict()
    for k in src_sd:
        sd[k] = src_sd[k]
    dest.load_state_dict(sd)
    return dest

def initialize_model(model_architecture: str, n_contexts: int, alpha: float):
    dispatch_dict = {
        "tempo_bert": BertForTemporalMaskedLM,
        "orthogonal": BertForOrthogonalMaskedLM,
        "naive": BertForNaiveOrthogonalMaskedLM
    }

    config = AutoConfig.from_pretrained('bert-base-uncased')
    bert_model = AutoModelForMaskedLM.from_pretrained('bert-base-uncased')
    if model_architecture == "bert":
        return bert_model
    elif model_architecture == "tempo_bert":
        model = BertForTemporalMaskedLM(config=config, n_contexts=n_contexts)
    else:
        ModelClass = dispatch_dict[model_architecture]
        model = ModelClass(config=config, n_contexts=n_contexts, alpha=alpha)
    model = copy_weights(bert_model, model)
    return model

def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
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
       
    logging.info(f"Initializing model")
    model = initialize_model(args.model_architecture, args.n_contexts, args.alpha)
    bert_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    collator = DataCollatorForLanguageModeling(bert_tokenizer)
    
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
        ## TODO: add the time tokens
        collator = get_time_token_collator(bert_tokenizer)
    elif args.add_time_tokens == "special":
        logging.info(f"Adding special time tokens")
        dataset = add_special_time_tokens(dataset, bert_tokenizer, model, args.n_contexts, args.process_dataset)
        collator = get_time_token_collator(bert_tokenizer, n_tokens=1)
    
    if args.save_dataset:
        logging.info(f"Saving the dataset to {args.save_dataset}")
        dataset.save_to_disk(args.save_dataset)
    
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
        choices=["bert", "tempo_bert", "orthogonal", "naive"], required=True)
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
        "--process-dataset", help="Indicates that the dataset should be processed (i.e. sorted and batch shuffled)",
        action='store_true')
    parser.add_argument(
        "--save-dataset", help="After processing, stores the dataset to this location.", default=None)
    parser.add_argument(
        "--resume", help="Resume training from checkpoint.", action='store_true')
    
    args = parser.parse_args()
    main(args)