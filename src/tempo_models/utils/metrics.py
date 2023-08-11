import evaluate
import numpy as np
from transformers import EvalPrediction
from typing import Dict, Callable

f1_metric = evaluate.load("f1")

def create_metric_func(metric_funcs: Dict[str, Callable]):
    """
    A function that creates a callable which runs the provided metric functions
    and returns a dict with the given keys.  
    Each metric function must take in an EvalPrediction.
    """
    def callable(eval_prediction: EvalPrediction):
        metrics = {}
        for key in metric_funcs:
            metrics[key] = metric_funcs[key](eval_prediction)
        return metrics
    
    return callable

def _mask_logits_and_labels_for_mlm(logits: np.ndarray, labels: np.ndarray, label_pad_id: int = -100):
    idx = np.nonzero(labels != label_pad_id)
    labels = labels[idx[0], idx[1]] ## shape (n,)
    logits = logits[idx[0], idx[1]] ## shape (n, n_tokens)
    return logits, labels

def metric_loss(eval_prediction: EvalPrediction):
    return eval_prediction.metrics["test_loss"]

def mlm_metric_mrr(eval_prediction: EvalPrediction):
    logits, labels = _mask_logits_and_labels_for_mlm(eval_prediction.predictions, eval_prediction.label_ids)
    gold_logits_values = np.take_along_axis(logits, labels[:, None], axis=1) ## shape(n, 1)
    ranks = (logits > gold_logits_values).sum(axis=1) + 1
    mrr = (1/ranks).sum()
    return mrr

def mlm_metric_accuracy(eval_prediction: EvalPrediction):
    logits, labels = _mask_logits_and_labels_for_mlm(eval_prediction.predictions, eval_prediction.label_ids)
    predictions = logits.argmax(axis=1)
    return np.mean(predictions == labels)

def cls_metric_accuracy(eval_prediction: EvalPrediction):
    logits = eval_prediction.predictions
    labels = eval_prediction.label_ids
    predictions = logits.argmax(axis=1)
    return np.mean(predictions == labels)

def cls_metric_weighted_f1(eval_prediction: EvalPrediction):
    logits = eval_prediction.predictions
    labels = eval_prediction.label_ids
    predictions = logits.argmax(axis=1)
    return f1_metric.compute(references=labels, predictions=predictions, average="weighted")["f1"]

def cls_metric_per_class_f1(eval_prediction: EvalPrediction):
    logits = eval_prediction.predictions
    labels = eval_prediction.label_ids
    predictions = logits.argmax(axis=1)
    return f1_metric.compute(references=labels, predictions=predictions, average=None)["f1"].tolist()