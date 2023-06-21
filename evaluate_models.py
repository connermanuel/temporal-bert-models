"""
Generalized evaluation script for various datasets and model architectures.
Given a directory containing model checkpoints, evaluate all of those checkpoints.
"""
from torch import device as torch_device
from models.orthogonal_weight_attention import BertForOrthogonalMaskedLM
from models.temporal_self_attention import BertForTemporalMaskedLM
from transformers import BertForMaskedLM, AutoTokenizer, DataCollatorForLanguageModeling
from datasets import load_from_disk

from utils import get_collator, evaluate_mlm, evaluate_span_accuracy, add_special_time_tokens, fix_timestamps, sort_by_timestamp, shuffle_batched
import argparse
import os
import logging
import json
import tqdm

def fetch_model(model_architecture: str, checkpoint_path: str):
    dispatch_dict = {
        "tempo_bert": BertForTemporalMaskedLM,
        "orthogonal": BertForOrthogonalMaskedLM,
        "bert": BertForMaskedLM
    }    
    
    return dispatch_dict[model_architecture].from_pretrained(checkpoint_path)

def main(args):
    ### Fix kwargs, create directories, and setup logging
    model_str = f"{args.model_architecture}"
    if args.model_architecture == "orthogonal":
        model_str = f"{args.model_architecture}_{args.alpha}"
    if args.checkpoint_dir is None:
        args.checkpoint_dir = f"outputs/{model_str}"
    if args.results_dir is None:
        args.results_dir = f"results/{model_str}"
    
    if not os.path.exists(args.checkpoint_dir):
        raise ValueError("Checkpoint directory does not exist")
    if not os.path.exists(args.data_dir):
        raise ValueError("Data directory does not exist")
    if not os.path.exists(args.results_dir):
        os.makedirs(args.results_dir)

    logging.basicConfig(
        filename = f"{args.results_dir}/run.log",
        format="%(asctime)s %(levelname)-8s %(message)s",
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S')
    
    ### Prepare collator and tokenizer
    bert_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    DEFAULT_TOKENIZER_LEN = len(bert_tokenizer)
    if args.add_time_tokens == "special":
        special_tokens = [f"timestamp: {t} text: " for t in range(args.n_contexts)]
        bert_tokenizer.add_tokens(special_tokens)
    
    mask = not args.no_mask
    if args.add_time_tokens == "string":
        collator = get_collator(bert_tokenizer, mask=mask)
    elif args.add_time_tokens == "special":
        collator = get_collator(bert_tokenizer, n_tokens=1, mask=mask)
    elif mask:
        collator = DataCollatorForLanguageModeling(bert_tokenizer)
    else:
        collator = get_collator(bert_tokenizer, mask=mask)

    ### Load and process dataset
    logging.info(f"Loading dataset...")
    dataset = load_from_disk(args.data_dir)
    try:
        dataset = dataset[args.split]
    except KeyError:
        raise KeyError(f"The split {args.split} does not exist in the dataset. Existing splits are: {dataset.keys()}")
    
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
    
    dataset = dataset.map(fix_timestamps, batched=True)

    if "word_ids" in dataset.features:
        dataset = dataset.remove_columns("word_ids")
    if args.model_architecture == "bert" and "timestamps" in dataset.features:
        dataset = dataset.remove_columns("timestamps") 
    
    ### Prepare evaluation setup
    results = {
        "perplexity": [],
        "accuracy": [],
        "mrr": [],
        "paths": [],
    }

    if args.no_cuda:
        device = torch_device("cpu")
    else:
        device = torch_device("cuda") 
    
    
    ### Evaluate models
    logging.info(f"Evaluating models...")
    def evaluate_path(checkpoint_path, f1, batch_size, results_dir):
        if f1:
            result = evaluate_span_accuracy(model, dataset, collator, device, batch_size)
            print(result)
        else:
            result = evaluate_mlm(model, dataset, collator, device, batch_size)
            for k, v in result.items():
                results[k].append(v)
            results['paths'].append(checkpoint_path)

            with open(f"{results_dir}/results.json", "w") as f:
                json.dump(results, f)
    
    if args.checkpoint_dir:
        evaluate_path(args.checkpoint_dir, args.f1, args.batch_size, args.results_dir)
    elif args.checkpoint_group_dir:
        for checkpoint_path in tqdm.tqdm(sorted(os.listdir(args.checkpoint_group_dir))):
            if checkpoint_path == "run.log":
                continue
            try:
                model = fetch_model(args.model_architecture, checkpoint_path)
                full_checkpoint_path = f"{args.checkpoint_group_dir}/{checkpoint_path}"
                evaluate_path(full_checkpoint_path, args.f1, args.batch_size, args.results_dir)
            except OSError:
                print(f"Could not evaluate {checkpoint_path}")
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trains a model.")
    parser.add_argument(
        "-m", "--model_architecture",
        help="The model architecture to train",
        choices=["bert", "tempo_bert", "orthogonal", "naive"], required=True)
    parser.add_argument(
        "--data-dir", 
        help="Path of the huggingface dataset.", required=True)
    parser.add_argument(
        "--checkpoint-path", 
        help='If used, path of the huggingface checkpoint. Overrides checkpoint-group-dir.', default=None)
    parser.add_argument(
        "--checkpoint-group-dir", 
        help='If used, path of directory containing huggingface checkpoints.', default=None)
    parser.add_argument(
        "--results-dir", 
        help='Path to directory to store checkpoints to. Defaults to "results/{architecture}".', default=None)
    parser.add_argument(
        "--alpha", 
        help="Regularization parameter. Defaults to 1. Only used for orthogonal model directory naming.",
        type=float, default=1)
    parser.add_argument(
        "--batch-size", 
        help="Evaluation batch size. Defaults to 16.",
        type=int, default=16)
    parser.add_argument(
        "--no-cuda", help="If flag is used, block trainer from using cuda when available.",
        action='store_true')
    parser.add_argument(
        "--use-fp16", help="If flag is used, use the fp16 backend.",
        action='store_true')
    parser.add_argument(
        "--add-time-tokens", help="Modifies the dataset to insert generic special time tokens. Use 'string' for tokenized strings, and 'special' for inserted special tokens.",
        choices=[None, "none", "string", "special"], default=None)
    parser.add_argument(
        "--process-dataset", help="Performs sorting and batch shuffling, and prepends time tokens if needed.",
        action='store_true')
    parser.add_argument(
        "--save-dataset", help="After processing, stores the dataset to this location.", default=None)
    parser.add_argument(
        "--split", help="The split of the dataset to use for evaluation. Defaults to test.",
        default="test")    
    parser.add_argument(
        "--sample", help="Indicates how many documents to use. If unset, uses the entire dataset.",
        type=int, default=0)
    parser.add_argument(
        "--f1", help="Indicates that we should evaluate span F1.")
    parser.add_argument(
        "--no_mask", help="Do not use a masked language modeling collator. Used when the dataset already has tokens masked out.", action="store_true")
    
    args = parser.parse_args()
    main(args)