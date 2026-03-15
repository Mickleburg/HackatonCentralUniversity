import re
from typing import Dict, List, Tuple


PII_PATTERNS = {
    "Номер телефона": re.compile(
        r"(?:\+?7|8)[-\s]?\(?[0-9]{3}\)?[-\s]?[0-9]{3}-?[0-9]{2}-?[0-9]{2}"
    ),
    "Email": re.compile(
        r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}"
    ),
    "Паспортные данные": re.compile(
        r"(?:[0-9]{4}[-\s]?[0-9]{6}|[0-9]{2}[-\s][0-9]{2}[-\s][0-9]{6})"
    ),
    "Сведения об ИНН": re.compile(
        r"(?:[0-9]{10}|[0-9]{12})"
    ),
    "СНИЛС клиента": re.compile(
        r"[0-9]{3}-[0-9]{3}-[0-9]{3}\s[0-9]{2}"
    ),
    "Номер карты": re.compile(
        r"(?:[0-9]{4}[-\s]?){3}[0-9]{4}"
    ),
    "CVV/CVC": re.compile(
        r"\b[0-9]{3}\b"
    ),
    "Номер банковского счета": re.compile(
        r"[0-9]{20}"
    ),
    "Водительское удостоверение": re.compile(
        r"[0-9]{2}[-\s]?[0-9]{2}[-\s]?[0-9]{6}"
    ),
    "Временное удостоверение личности": re.compile(
        r"[IVXLCDM]{1,4}[-\s]?[А-Я]{2}[-\s]?[0-9]{6}"
    ),
    "Серия и номер вида на жительство": re.compile(
        r"[0-9]{2}[-\s]?[0-9]{7}"
    ),
    "Свидетельство о рождении": re.compile(
        r"[IVXLCDM]{1,4}-[А-Я]{2}[-\s]?[0-9]{6}"
    ),
    "ПИН код": re.compile(
        r"\b[0-9]{4}\b"
    ),
    "Данные об организации/юридическом лице (ИНН, КПП, ОГРН, БИК, адреса, расчётный счёт)": re.compile(
        r"(?:[0-9]{10}|[0-9]{9}|[0-9]{13})"
    ),
    "API ключи": re.compile(
        r"[a-zA-Z0-9_]{20,}"
    ),
}

RULE_ONLY_ENTITIES = list(PII_PATTERNS.keys())

CONTEXT_KEYWORDS: Dict[str, List[str]] = {
    "CVV/CVC": ["cvv", "cvc", "cvc2"],
    "Номер карты": ["карт", "card", "номер карты"],
    "Номер телефона": ["телефон", "сбп", "смс", "звон", "номер"],
    "Email": ["email", "e-mail", "почт", "письм"],
    "ПИН код": ["пин", "pin", "код"],
    "Номер банковского счета": ["счет", "расчетн", "банковск"],
    "Паспортные данные": ["паспорт", "серия", "выдан", "мвд", "уфмс"],
    "Сведения об ИНН": ["инн", "налогов"],
    "СНИЛС клиента": ["снилс", "страхов", "пенсион"],
    "Водительское удостоверение": ["водител", "ву", "прав"],
    "API ключи": ["api", "key", "ключ", "токен"],
}


def _has_context_keyword(text: str, start: int, end: int, radius: int, label: str) -> bool:
    context_start = max(0, start - radius)
    context_end = min(len(text), end + radius)
    context = text[context_start:context_end].lower()
    keywords = CONTEXT_KEYWORDS.get(label, [])
    
    if not keywords:
        return True
    
    for keyword in keywords:
        if keyword.lower() in context:
            return True
    
    return False


def _remove_overlaps(candidates: List[Tuple[int, int, str]]) -> List[Tuple[int, int, str]]:
    """Удаляем перекрытия, оставляя первый span."""
    if not candidates:
        return []
    
    candidates = sorted(candidates, key=lambda x: (x[0], x[1]))
    result = [candidates[0]]
    
    for curr in candidates[1:]:
        if curr[0] >= result[-1][1]:
            result.append(curr)
    
    return result


def detect_pii(text: str, context_radius: int = 30) -> List[Tuple[int, int, str]]:
    """
    Детектим сущности по regex.
    Возвращаем список [(start, end, label), ...]
    """
    if not text:
        return []
    
    candidates = []
    
    for label in RULE_ONLY_ENTITIES:
        if label not in PII_PATTERNS:
            continue
        
        pattern = PII_PATTERNS[label]
        
        for match in re.finditer(pattern, text):
            start, end = match.start(), match.end()
            
            if _has_context_keyword(text, start, end, context_radius, label):
                candidates.append((start, end, label))
    
    return sorted(_remove_overlaps(candidates), key=lambda x: (x[0], x[1]))
