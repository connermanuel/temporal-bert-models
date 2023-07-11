"""
Generalized evaluation script for various datasets and model architectures.
Given a directory containing model checkpoints, evaluate all of those checkpoints.
"""
from tempo_models.models.bert.orthogonal_weight_attention import BertForOrthogonalMaskedLM
from tempo_models.models.bert.temporal_self_attention import BertForTemporalMaskedLM

from torch import device as torch_device
from transformers import BertForMaskedLM, AutoTokenizer, DataCollatorForLanguageModeling, AutoConfig
from datasets import load_from_disk

from utils import get_collator, evaluate_mlm, evaluate_span_accuracy, add_special_time_tokens, fix_timestamps, sort_by_timestamp, shuffle_batched
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

def evaluate(args):
    ### Fix kwargs, create directories, and setup logging
    model_str = f"{args.model_architecture}"
    if args.model_architecture == "orthogonal":
        model_str = f"{args.model_architecture}_{args.alpha}"
    if args.checkpoint_path is None and args.checkpoint_group_dir is None:
        args.checkpoint_path = f"outputs/{model_str}"
    if args.results_dir is None:
        args.results_dir = f"results/{model_str}"
    
    if args.checkpoint_path and not os.path.exists(args.checkpoint_path):
        raise ValueError("Checkpoint directory does not exist")
    elif args.checkpoint_group_dir and not os.path.exists(args.checkpoint_group_dir):
        raise ValueError("Checkpoint group directory does not exist")
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
    try:
        dataset = dataset[args.split]
    except KeyError:
        raise KeyError(f"The split {args.split} does not exist in the dataset. Existing splits are: {dataset.column_names}")
    
    if args.sample:
        logging.info(f"Sampling {args.sample} entries")
        dataset = dataset.select(range(min(args.sample, len(dataset))))
    
    logging.info(f"Processing the dataset")
    if args.process_dataset:
        dataset = sort_by_timestamp(dataset)
        # dataset = shuffle_batched(dataset, args.batch_size)
        if args.add_time_tokens == "string":
            logging.info(f"Adding string time tokens")
            ## TODO
        elif args.add_time_tokens == "special":
            logging.info(f"Adding special time tokens")
            dataset = add_special_time_tokens(dataset, DEFAULT_TOKENIZER_LEN)

    if args.save_dataset:
        logging.info(f"Saving the dataset to {args.save_dataset}")
        dataset.save_to_disk(args.save_dataset)
    

    if "word_ids" in dataset.features:
        dataset = dataset.remove_columns("word_ids")
    if args.model_architecture == "bert" and "timestamps" in dataset.features:
        dataset = dataset.remove_columns("timestamps") 
    else:
        dataset = dataset.map(fix_timestamps, batched=True)
    
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
    def evaluate_path(checkpoint_path, architecture, f1, batch_size, results_dir):
        model = fetch_model(architecture, checkpoint_path)
        if f1:
            result = evaluate_span_accuracy(model, dataset, collator, device, batch_size)
            with open(f"{results_dir}/results.json", "w") as f:
                json.dump({"accuracy": result}, f)
        else:
            result = evaluate_mlm(model, dataset, collator, device, batch_size)
            for k, v in result.items():
                results[k].append(v)
            results['paths'].append(checkpoint_path)

            with open(f"{results_dir}/results.json", "w") as f:
                json.dump(results, f)
    
    if args.checkpoint_path:
        evaluate_path(args.checkpoint_path, args.model_architecture, args.f1, args.batch_size, args.results_dir)
    elif args.checkpoint_group_dir:
        for checkpoint_path in tqdm.tqdm(sorted(os.listdir(args.checkpoint_group_dir))):
            if checkpoint_path == "run.log":
                continue
            try:
                full_checkpoint_path = f"{args.checkpoint_group_dir}/{checkpoint_path}"
                evaluate_path(full_checkpoint_path, args.model_architecture, args.f1, args.batch_size, args.results_dir)
            except OSError:
                print(f"Could not evaluate {checkpoint_path}")
        