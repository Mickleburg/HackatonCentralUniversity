# Ячейка для тестирования модели на новых данных (CSV с колонкой 'text')
import pandas as pd
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForTokenClassification

# ------------------------------
# 1. Загружаем сохранённую модель и токенизатор
# ------------------------------
model_path = './ner_model_final'  # путь к папке с моделью
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForTokenClassification.from_pretrained(model_path)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()

# Из модели получаем словарь id -> label
id2label = model.config.id2label

# ------------------------------
# 2. Функция преобразования предсказаний в спаны (аналогично обучению)
# ------------------------------
def spans_from_labels(labels, offset_mapping, id2label):
    """
    labels: список предсказанных ID меток для токенов (только значимые токены, без специальных)
    offset_mapping: список кортежей (start, end) для тех же токенов
    id2label: словарь id -> label_str
    """
    entities = []
    i = 0
    while i < len(labels):
        label_id = labels[i]
        label_str = id2label.get(label_id, 'O')
        if label_str == 'O':
            i += 1
            continue
        if label_str.startswith('B-'):
            cat = label_str[2:]
            start_offset = offset_mapping[i][0]
            j = i + 1
            while j < len(labels):
                next_label = id2label.get(labels[j], 'O')
                if next_label == f'I-{cat}':
                    j += 1
                else:
                    break
            end_offset = offset_mapping[j-1][1]
            entities.append((int(start_offset), int(end_offset), cat))
            i = j
        else:
            # I- без B – такого не должно быть, но на всякий случай пропускаем
            i += 1
    return entities

# ------------------------------
# 3. Функция обработки одного текста
# ------------------------------
def predict_entities(text, tokenizer, model, id2label, max_length=512):
    # Токенизация с offset_mapping
    inputs = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        return_offsets_mapping=True,
        return_tensors='pt'
    )
    # Перемещаем на устройство
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)
    offset_mapping = inputs['offset_mapping'][0].cpu().numpy()  # список кортежей (start, end)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)[0].cpu().numpy()

    # Отфильтровываем специальные токены (где offset_mapping == (0,0) или (None,None))
    # и токены паддинга (attention_mask=0). Используем attention_mask.
    mask = attention_mask[0].cpu().numpy().astype(bool)
    # Также исключаем токены с нулевым смещением (специальные)
    valid_indices = [i for i, (start, end) in enumerate(offset_mapping) if mask[i] and not (start == 0 and end == 0)]
    if not valid_indices:
        return []

    pred_labels = predictions[valid_indices]
    valid_offsets = [offset_mapping[i] for i in valid_indices]

    # Преобразуем в спаны
    entities = spans_from_labels(pred_labels, valid_offsets, id2label)
    return entities

# ------------------------------
# 4. Загрузка CSV с текстами
# ------------------------------
# Предположим, что файл называется 'input.csv' и содержит колонку 'text'.
# Если есть колонка 'id', она будет сохранена.
df_input = pd.read_csv('input.csv')  # укажите ваш путь

# Применяем функцию к каждому тексту
df_input['entities'] = df_input['text'].apply(lambda x: predict_entities(x, tokenizer, model, id2label))

# Выводим результат на экран
for idx, row in df_input.iterrows():
    print(f"Текст: {row['text'][:80]}...")  # обрезаем для вывода
    print(f"Найденные сущности: {row['entities']}")
    print("-" * 80)

# Если нужно сохранить результат в новый CSV
df_output = df_input[['text', 'entities']]  # или добавьте 'id'
df_output.to_csv('output_with_entities.csv', index=False)
print("Результат сохранён в output_with_entities.csv")