"""
Generalized evaluation script for various datasets and model architectures.
Given a directory containing model checkpoints, evaluate all of those checkpoints.
"""
from torch import device as torch_device
from models.orthogonal_weight_attention import BertForOrthogonalMaskedLM
from models.temporal_self_attention import BertForTemporalMaskedLM
from transformers import BertForMaskedLM, AutoTokenizer, DataCollatorForLanguageModeling
from datasets import load_from_disk

from utils import get_time_token_collator, evaluate_mlm, evaluate_span_accuracy, add_special_time_tokens
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
    
    logging.info(f"Loading dataset...")
    model = fetch_model(args.model_architecture, checkpoint_path)
    dataset = load_from_disk(args.data_dir)
    dataset = dataset['test']
    if args.sample:
        dataset = dataset.select(range(10))
    
    if args.add_time_tokens == "string":
        logging.info(f"Adding string time tokens")
        collator = get_time_token_collator(bert_tokenizer)
    elif args.add_time_tokens == "special":
        logging.info(f"Adding special time tokens")
        dataset = add_special_time_tokens(dataset, bert_tokenizer, model, args.n_contexts, args.process_dataset)
        collator = get_time_token_collator(bert_tokenizer, n_tokens=1)
    
    if "word_ids" in dataset.features:
        dataset = dataset.remove_columns("word_ids")
    if args.model_architecture == "bert" and "timestamps" in dataset.features:
        dataset = dataset.remove_columns("timestamps") 
    
    bert_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    collator = DataCollatorForLanguageModeling(bert_tokenizer)
    
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
    
    
    logging.info(f"Evaluating models...")
    def evaluate_path(checkpoint_path):
        try:
            if args.span_f1:
                result = evaluate_span_accuracy(model, dataset, collator, device, args.batch_size)
                print(result)
            else:
                result = evaluate_mlm(model, dataset, collator, device, args.batch_size)
                for k, v in result.items():
                    results[k].append(v)
                results['paths'].append(checkpoint_path)

                with open(f"{args.results_dir}/results.json", "w") as f:
                    json.dump(results, f) 
        except OSError:
            pass
    
    if args.checkpoint_dir:
        evaluate_path(args.checkpoint_dir)
    elif args.checkpoint_group_dir:
        for checkpoint_path in tqdm.tqdm(sorted(os.listdir(args.checkpoint_group_dir))):
            if checkpoint_path == "run.log":
                continue
            evaluate_path(f"{args.checkpoint_dir}/{checkpoint_path}")
        

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
        "--checkpoint-dir", 
        help='If used, path of the huggingface checkpoint. Overrides checkpoint-group-dir.', default=None)
    parser.add_argument(
        "--checkpoint-group-dir", 
        help='If used, path of directory containing huggingface checkpoints.', default=None)
    parser.add_argument(
        "--results-dir", 
        help='Path to directory to store checkpoints to. Defaults to "results/{architecture}".', default=None)
    parser.add_argument(
        "--alpha", 
        help="Regularization parameter. Defaults to 1. Only used for orthogonal model.",
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
        "--use_time_tokens", help="Indicates that the dataset has prepeneded time tokens. Use 'string' for tokenized strings, and 'special' for inserted special tokens.",
        choices=[None, "none", "string", "special"], default=None)
    parser.add_argument(
        "--sample", help="Indicates that we should only use a small sample of the data.",
        action='store_true')
    parser.add_argument(
        "--f1", help="Indicates that we should evaluate span F1.")
    
    args = parser.parse_args()
    main(args)