from typing import List, Tuple


def merge_predictions(regex_spans: List[Tuple[int, int, str]], ner_spans: List[Tuple[int, int, str]]) -> List[Tuple[int, int, str]]:
    """Merge: regex имеет приоритет, NER добавляется если не пересекается."""
    
    if not regex_spans and not ner_spans:
        return []
    
    result = list(regex_spans) if regex_spans else []
    
    if not ner_spans:
        return sorted(result, key=lambda x: (x[0], x[1]))
    
    for ner_start, ner_end, ner_label in ner_spans:
        overlaps = False
        
        for regex_start, regex_end, _ in regex_spans:
            if not (ner_end <= regex_start or ner_start >= regex_end):
                overlaps = True
                break
        
        if not overlaps:
            result.append((ner_start, ner_end, ner_label))
    
    return sorted(result, key=lambda x: (x[0], x[1]))
