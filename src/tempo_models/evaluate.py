"""
Generalized evaluation script for various datasets and model architectures.
Given a directory containing model checkpoints, evaluate all of those checkpoints.
"""
from tempo_models.models.bert.orthogonal_bert import BertForOrthogonalMaskedLM, BertForOrthogonalSequenceClassification
from tempo_models.models.bert.tempo_bert import BertForTemporalMaskedLM, BertForTemporalSequenceClassification

from torch import device as torch_device
from transformers import BertForMaskedLM, AutoTokenizer, BertForSequenceClassification, TrainingArguments
from datasets import load_from_disk

from tempo_models.utils.utils import get_collator, evaluate_mlm, evaluate_span_accuracy, add_special_time_tokens, sort_by_timestamp, NonShuffledTrainer
import os
import logging
import json
import tqdm


MODELS = {
    "tempo_bert_mlm": BertForTemporalMaskedLM,
    "orthogonal_mlm": BertForOrthogonalMaskedLM,
    "bert_mlm": BertForMaskedLM,
    "tempo_bert_cls": BertForTemporalSequenceClassification,
    "orthogonal_cls": BertForOrthogonalSequenceClassification,
    "bert_cls": BertForSequenceClassification
    }

def fetch_model(model_architecture: str, checkpoint_path: str, task: str):
    return MODELS[f"{model_architecture}_{task}"].from_pretrained(checkpoint_path)

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
    if "subreddit" in dataset.features:
        dataset = dataset.remove_columns("subreddit") 
    if args.model_architecture == "bert" and "timestamps" in dataset.features:
        dataset = dataset.remove_columns("timestamps") 
    
    ### Prepare evaluation setup
    if args.no_cuda:
        device = torch_device("cpu")
    else:
        device = torch_device("cuda") 
    
    if args.checkpoint_path:
        model_dirs = [args.evaluate_path]
    elif args.checkpoint_group_dir:
        model_dirs = [path for path in os.listdir(args.checkpoint_group_dir) if "checkpoint" in path]
    
    logging.info(f"Evaluating models...")

    eval_args = TrainingArguments(

    )

    for dir in model_dirs:
        model = fetch_model(args.model_architecture, dir, args.task)
        trainer = NonShuffledTrainer(
            model=model,
            eval_dataset=dataset,
            compute_metrics=metric_func,
            collator=collator
        )

    ### Evaluate models
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
        