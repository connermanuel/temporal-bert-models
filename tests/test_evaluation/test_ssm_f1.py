import numpy as np
from transformers import EvalPrediction
from tempo_models.utils.metrics import ssm_metric_token_f1_from_predictions

def test_predictions():
    predictions = np.array([
    [32099, 1, 2, 3, 4, 32098],
    [32099, 1, 2, 3, 32098, 1],
    [32099, 1, 2, 32098, 1, 0],
    [32099, 1, 32098, 1, 0, 4],
    ])

    labels = np.array([
    [32099, 1, 2, 3, 4, 32098],
    [32099, 1, 2, 3, 4, 32098],
    [32099, 1, 2, 3, 4, 32098],
    [32099, 1, 2, 3, 32098, 1],
    ])

    eval_prediction = EvalPrediction(predictions=predictions, label_ids=labels)
    metric = ssm_metric_token_f1_from_predictions(eval_prediction)
    assert "f1" in metric.keys()