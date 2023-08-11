"""Generalized training script for various datasets and model architectures."""
from torch.cuda import empty_cache
from datasets import load_from_disk
from tempo_models.models.bert.orthogonal_bert import BertForOrthogonalMaskedLM, BertForOrthogonalSequenceClassification, OrthogonalConfig
from tempo_models.models.bert.tempo_bert import BertForTemporalMaskedLM, BertForTemporalSequenceClassification, TempoBertConfig
from tempo_models.utils.collator import CollatorCLS, CollatorMLM
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModelForSequenceClassification, TrainingArguments, BertConfig
from transformers.models.bert.modeling_bert import BertForMaskedLM, BertForSequenceClassification

from tempo_models.utils.utils import NonShuffledTrainer, sort_by_timestamp, shuffle_batched, add_special_time_tokens, copy_weights
import logging
import gc
import os

def initialize_mlm_model(model_architecture: str, n_contexts: int, alpha: float):
    """Initializes a model for the first time, ready for training."""   
    bert_model = AutoModelForMaskedLM.from_pretrained('bert-base-uncased')
    if model_architecture == "bert":
        return bert_model
    elif model_architecture == "tempo_bert":
        config = TempoBertConfig(n_contexts)
        model = BertForTemporalMaskedLM(config)
    elif model_architecture == "orthogonal":
        config = OrthogonalConfig(n_contexts, alpha)
        model = BertForOrthogonalMaskedLM(config)
    model = copy_weights(bert_model, model)
    return model

def initialize_cls_model_from_mlm(model_architecture: str, pretrained_loc: str, num_labels: int, n_contexts: int, alpha: float):
    dispatch_dict_mlm = {
        "bert": BertForMaskedLM,
        "tempo": BertForTemporalMaskedLM,
        "orthogonal": BertForOrthogonalMaskedLM
    }
    dispatch_dict_cls = {
        "bert": BertForSequenceClassification,
        "temporal": BertForTemporalSequenceClassification,
        "orthogonal": BertForOrthogonalSequenceClassification
    }
    dispatch_dict_config = {
        "bert": BertConfig,
        "temporal": TempoBertConfig,
        "orthogonal": OrthogonalConfig
    }

    base_cls_model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")
    pretrained_model = dispatch_dict_mlm[model_architecture].from_pretrained(pretrained_loc)
    config = dispatch_dict_config[model_architecture](num_labels=num_labels, n_contexts=n_contexts, alpha=alpha)
    model = dispatch_dict_cls[model_architecture](config)

    model = copy_weights(base_cls_model.classifier, model, prefix="classifier")
    model = copy_weights(pretrained_model, model)
    
    return model


def train(args):
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
    if args.sample:
        logging.info(f"Sampling {args.sample} entries")
        for k in dataset.keys():
            dataset[k] = dataset[k].select(range(min(args.sample, len(dataset[k]))))
    
    logging.info(f"Processing the dataset")
    if not args.skip_process:
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

    ### Prepare model
    if args.task == "mlm":
        model = initialize_mlm_model(args.model_architecture, args.n_contexts, args.alpha)
    elif args.task == "cls":
        model = initialize_cls_model_from_mlm(args.model_architecture, args.pretrain_dir, args.num_labels, args.n_contexts, args.alpha)
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
        max_steps=args.num_steps,
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