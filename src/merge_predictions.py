from typing import List, Tuple, Set


def _spans_overlap(span1: Tuple[int, int], span2: Tuple[int, int]) -> bool:
    """
    Проверяет, пересекаются ли два span'а.
    """
    start1, end1 = span1
    start2, end2 = span2
    return not (end1 <= start2 or end2 <= start1)


def merge_predictions(
    regex_spans: List[Tuple[int, int, str]],
    ner_spans: List[Tuple[int, int, str]]
) -> List[Tuple[int, int, str]]:
    """
    Слияние regex (приоритет) и NER предсказаний с обработкой пересечений.
    """
    if not regex_spans and not ner_spans:
        return []

    result_set: Set[Tuple[int, int, str]] = set(regex_spans or [])

    for ner_start, ner_end, ner_label in ner_spans or []:
        overlaps_with_regex = any(
            _spans_overlap((ner_start, ner_end), (regex_start, regex_end))
            for regex_start, regex_end, _ in (regex_spans or [])
        )
        if not overlaps_with_regex:
            result_set.add((ner_start, ner_end, ner_label))

    return sorted(result_set, key=lambda x: (x[0], x[1], x[2]))


def merge_multiple(
    predictions: List[List[Tuple[int, int, str]]],
    weights: List[float] = None
) -> List[Tuple[int, int, str]]:
    """
    Слияние нескольких списков предсказаний.
    """
    if not predictions:
        return []

    result = sorted(set(predictions[0]), key=lambda x: (x[0], x[1], x[2]))
    for pred in predictions[1:]:
        result = merge_predictions(result, pred)

    return result


def deduplicate_spans(
    spans: List[Tuple[int, int, str]]
) -> List[Tuple[int, int, str]]:
    """
    Удаляет дубликаты из списка spans.
    """
    return sorted(set(spans or []), key=lambda x: (x[0], x[1], x[2]))


def merge_overlapping_spans(
    spans: List[Tuple[int, int, str]],
    strategy: str = "union"
) -> List[Tuple[int, int, str]]:
    """
    Слияние перекрывающихся spans.
    strategy:
      - union: объединить интервалы
      - keep_first: оставить первый
    """
    if not spans:
        return []

    spans = sorted(spans, key=lambda x: (x[0], x[1], x[2]))

    if strategy not in {"union", "keep_first"}:
        raise ValueError("strategy must be 'union' or 'keep_first'")

    result = [spans[0]]

    for start, end, label in spans[1:]:
        last_start, last_end, last_label = result[-1]

        if _spans_overlap((last_start, last_end), (start, end)):
            if strategy == "union":
                merged_label = last_label if last_label == label else f"{last_label}/{label}"
                result[-1] = (min(last_start, start), max(last_end, end), merged_label)
        else:
            result.append((start, end, label))

    return result


def filter_by_confidence(
    spans: List[Tuple[int, int, str]],
    confidence_scores: List[float] = None,
    threshold: float = 0.5
) -> List[Tuple[int, int, str]]:
    """
    Фильтрует spans по confidence.
    """
    if confidence_scores is None or len(confidence_scores) != len(spans):
        return sorted(spans or [], key=lambda x: (x[0], x[1], x[2]))

    result = [
        span for span, score in zip(spans, confidence_scores)
        if score >= threshold
    ]
    return sorted(result, key=lambda x: (x[0], x[1], x[2]))
