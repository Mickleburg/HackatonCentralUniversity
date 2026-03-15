import pandas as pd


def compute_metrics(predictions, targets):
    """Strict span match: start, end, label должны совпадать."""
    tp = 0
    fp = 0
    fn = 0

    for pred, target in zip(predictions, targets):
        pred_set = set()
        target_set = set()
        
        for span in pred:
            if isinstance(span, (list, tuple)) and len(span) >= 3:
                pred_set.add((span[0], span[1], span[2]))
        
        for span in target:
            if isinstance(span, (list, tuple)) and len(span) >= 3:
                target_set.add((span[0], span[1], span[2]))

        tp += len(pred_set & target_set)
        fp += len(pred_set - target_set)
        fn += len(target_set - pred_set)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    micro_f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "micro_f1": micro_f1,
    }


def save_metrics(rows, output_path: str):
    """Сохраняем метрики."""
    if not rows:
        rows = [{"model": "none", "dataset": "none", "precision": 0, "recall": 0, "micro_f1": 0}]
    
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
