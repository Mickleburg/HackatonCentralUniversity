## Пример использования

### Активизация окружения

`python -m venv .venv` 

`.venv\Scripts\activate.bat`

### Установка зависимостей

`python -m pip install --upgrade pip`

`python -m pip install -r requirements.txt`

### Подготовка данных
`python main.py prepare`

### Обучение NER
`python main.py ner_train`

### Запуск regex
`python main.py regex --input data/raw/train_dataset.tsv --output data/answer/regex_predictions.csv`

### Предсказание NER
`python main.py ner_predict --input data/raw/private_test_dataset.csv --output data/answer/ner_predictions.csv`

### Merge
`python main.py merge`

### Всё за раз
`python main.py all`
