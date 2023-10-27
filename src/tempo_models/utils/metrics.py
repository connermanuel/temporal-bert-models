import evaluate
import numpy as np
from transformers import EvalPrediction
from typing import Dict, Callable
import torch
from collections import defaultdict

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

### SSM TRAINER STYLE METRICS
def ssm_metric_token_f1_from_predictions(eval_prediction: EvalPrediction, ids=None) -> dict:
    predictions = eval_prediction.predictions
    label_ids = eval_prediction.label_ids
    if isinstance(label_ids, tuple):
        label_ids = label_ids[0]
    num_rows = predictions.shape[0]

    extra_end_tokens = (np.ones((predictions.shape[0], 1), dtype=int) * 32098)
    predictions = np.hstack((predictions, extra_end_tokens))
    rows, positions = np.nonzero(predictions == 32098)
    _, idxs = np.unique(rows, return_index=True)
    prediction_row_end_idxs = positions[idxs]
    label_row_end_idxs = np.nonzero(label_ids == 32098)[1]

    f1_scores = []
    for row in range(num_rows):
        prediction = predictions[row][2:prediction_row_end_idxs[row]]
        label = label_ids[row][1:label_row_end_idxs[row]]
        overlap = np.intersect1d(prediction, label)

        f1 = (2 * len(overlap)) / (len(prediction) + len(label))
        f1_scores.append(f1)
    
    if ids and len(ids) == len(f1_scores):
        score_per_id = defaultdict(float)
        for id, score in zip(ids, f1_scores):
            score_per_id[id] = max(score_per_id[id], score)
        f1_scores = list(score_per_id.values())

    return {"f1": np.mean(f1_scores)}


def trainer_get_predictions_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    if isinstance(logits, tuple):
        logits = logits[0]
    return torch.argmax(logits, dim=-1)