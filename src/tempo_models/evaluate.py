"""
Generalized evaluation script for various datasets and model architectures.
Given a directory containing model checkpoints, evaluate all of those checkpoints.
"""
from tempo_models.models.bert.orthogonal_bert import BertForOrthogonalMaskedLM, BertForOrthogonalSequenceClassification
from tempo_models.models.bert.tempo_bert import BertForTemporalMaskedLM, BertForTemporalSequenceClassification
from tempo_models.utils.collator import CollatorCLS, CollatorMLM
from tempo_models.utils.metrics import create_metric_func, mlm_metric_accuracy, mlm_metric_mrr, cls_metric_accuracy, cls_metric_per_class_f1, cls_metric_weighted_f1

from transformers import BertForMaskedLM, AutoTokenizer, BertForSequenceClassification, TrainingArguments
from datasets import load_from_disk

from tempo_models.utils.utils import evaluate_mlm, add_special_time_tokens, NonShuffledTrainer, remove_unused_columns
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

def fetch_model(model_architecture: str, checkpoint_dir: str, task: str):
    return MODELS[f"{model_architecture}_{task}"].from_pretrained(checkpoint_dir)

def evaluate(args):
    ### Fix kwargs, create directories, and setup logging
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    model_str = f"{args.model_architecture}"
    if args.model_architecture == "orthogonal":
        model_str = f"{args.model_architecture}_{args.alpha}"
    if args.checkpoint_dir is None and args.checkpoint_group_dir is None:
        args.checkpoint_dir = f"outputs/{model_str}"
    if args.results_dir is None:
        args.results_dir = f"results/{model_str}"
    
    if args.checkpoint_dir and not os.path.exists(args.checkpoint_dir):
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
    if args.task == "mlm":
        collator = CollatorMLM(bert_tokenizer)
    elif args.task == "cls":
        collator = CollatorCLS(bert_tokenizer)

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
    if not args.skip_process:
        if args.add_time_tokens == "string":
            logging.info(f"Adding string time tokens")
            ## TODO
        elif args.add_time_tokens == "special":
            logging.info(f"Adding special time tokens")
            dataset = add_special_time_tokens(dataset, DEFAULT_TOKENIZER_LEN)

    if args.save_dataset:
        logging.info(f"Saving the dataset to {args.save_dataset}")
        dataset.save_to_disk(args.save_dataset)

    
    ### Prepare evaluation setup    
    if args.checkpoint_dir:
        model_dirs = [args.checkpoint_dir]
    elif args.checkpoint_group_dir:
        model_dirs = [f"{args.checkpoint_group_dir}/{path}" for path in os.listdir(args.checkpoint_group_dir) if "checkpoint" in path]
    
    logging.info(f"Evaluating models...")

    eval_args = TrainingArguments(
        output_dir=".",
        per_device_eval_batch_size=args.batch_size,
        remove_unused_columns=True,
        fp16=args.use_fp16,
        no_cuda=args.no_cuda,
        eval_accumulation_steps=4,
    )

    metric_func = create_metric_func({})
    if args.task == "mlm":
        metric_func = create_metric_func({
            "mrr": mlm_metric_mrr,
            "accuracy": mlm_metric_accuracy
        })
    elif args.task == "cls":
        metric_func = create_metric_func({
            "accuracy": cls_metric_accuracy,
            "per_class_f1": cls_metric_per_class_f1,
            "weighted_f1": cls_metric_weighted_f1
        })

    dataset = remove_unused_columns(dataset, MODELS[f"{args.model_architecture}_{args.task}"])

    dataset = dataset.remove_columns(["timestamps", "subreddit"])
    results = {}
    for dir in tqdm.tqdm(model_dirs):
        
        model = fetch_model(args.model_architecture, dir, args.task)

        if args.task == "cls":
            trainer = NonShuffledTrainer(
                model=model,
                args=eval_args,
                eval_dataset=dataset,
                compute_metrics=metric_func,
                data_collator=collator
            )
            model_results = trainer.evaluate()
        else:
            model_results = evaluate_mlm(model, dataset, collator)
        results[dir] = model_results
        
        with open(f"{args.results_dir}/results.json", "w") as f:
            json.dump(results, f)
        