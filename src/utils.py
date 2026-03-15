import os

def ensure_dirs():
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("data/answer", exist_ok=True)

def spans_to_bio(text, spans):
    """Преобразуем символьные span'ы в BIO-теги."""
    bio_tags = []
    span_dict = {}
    
    for start, end, label in spans:
        for i in range(start, end):
            span_dict[i] = label
    
    for i, char in enumerate(text):
        if i not in span_dict:
            bio_tags.append("O")
        else:
            label = span_dict[i]
            if i == 0 or span_dict.get(i - 1) != label:
                bio_tags.append(f"B-{label}")
            else:
                bio_tags.append(f"I-{label}")
    
    return bio_tags

def remove_overlaps(spans):
    """Удаляем перекрывающиеся spans, оставляя первый."""
    if not spans:
        return []
    spans = sorted(spans, key=lambda x: x[0])
    result = [spans[0]]
    for curr in spans[1:]:
        if curr[0] >= result[-1][1]:
            result.append(curr)
    return result

def spans_overlap(span1, span2):
    """Проверяем, пересекаются ли два span'а."""
    return not (span1[1] <= span2[0] or span2[1] <= span1[0])
