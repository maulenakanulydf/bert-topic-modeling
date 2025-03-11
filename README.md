
# Классификация текста с использованием RuBERT и Hugging Face Transformers

Этот проект демонстрирует обучение модели `DeepPavlov/rubert-base-cased` для классификации текста на основе пользовательских данных (например, описаний и кластеров). Проект выполнен в Google Colab с использованием GPU (T4) и библиотеки `transformers` для подготовки данных, обучения и оценки модели.

## Описание

Проект включает следующие этапы:
1. **Подготовка данных**: Разделение текстов и меток на обучающую и валидационную выборки.
2. **Токенизация**: Использование токенизатора RuBERT для подготовки входных данных.
3. **Создание датасета**: Преобразование данных в формат, совместимый с PyTorch.
4. **Обучение модели**: Настройка и обучение `BertForSequenceClassification` с помощью `Trainer`.
5. **Оценка**: Проверка производительности модели на валидационной выборке.

Дата последнего обновления знаний: март 2025 года (на основе текущей даты — 11 марта 2025).

## Установка

1. **Требования**:
   - Python 3.10+
   - Google Colab с GPU (например, T4)
   - Установленные библиотеки: `transformers`, `torch`, `sklearn`

2. **Установка зависимостей**:
   В Google Colab выполните:
   ```bash
   !pip install transformers torch scikit-learn
   ```

3. **Данные**:
   - Подготовьте данные в формате pandas DataFrame с колонками `description` (текст) и `cluster` (метки).

## Использование

### Подготовка данных
Предполагается, что у вас есть DataFrame `reg` с колонками `description` и `cluster`.

```python
from sklearn.model_selection import train_test_split

# Разделение данных
train_texts, val_texts, train_labels, val_labels = train_test_split(
    reg['description'], reg['cluster'], test_size=0.2, random_state=42
)
```

### Токенизация
Используйте токенизатор RuBERT для подготовки текстов:
```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('DeepPavlov/rubert-base-cased')
train_encodings = tokenizer(list(train_texts), truncation=True, padding=True, max_length=512)
val_encodings = tokenizer(list(val_texts), truncation=True, padding=True, max_length=512)
```

### Создание датасета
Создайте PyTorch Dataset для обучения и валидации:
```python
import torch

class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx]).float()
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = Dataset(train_encodings, train_labels.tolist())
val_dataset = Dataset(val_encodings, val_labels.tolist())
```

### Настройка и обучение модели
Настройте модель и параметры обучения:
```python
from transformers import BertForSequenceClassification, Trainer, TrainingArguments

model = BertForSequenceClassification.from_pretrained(
    'DeepPavlov/rubert-base-cased', num_labels=len(set(reg['cluster']))
)

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    logging_dir='./logs',
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

trainer.train()
```

### Оценка модели
Оцените модель на валидационной выборке:
```python
trainer.evaluate()
```

## Структура проекта

- **Токенизация**: Преобразование текста в формат, подходящий для RuBERT.
- **Dataset**: Класс для работы с данными в PyTorch.
- **Обучение**: Использование `Trainer` для обучения и сохранения модели.
- **Оценка**: Вычисление потерь на валидационной выборке.

## Зависимости

- `transformers` — для работы с моделью и токенизатором.
- `torch` — для работы с тензорами и GPU.
- `sklearn` — для разделения данных.

## Известные проблемы

1. **Ошибка сохранения модели**:
   Во время обучения возникает ошибка:
   ```
   ValueError: You are trying to save a non contiguous tensor: `bert.encoder.layer.0.attention.self.query.weight` which is not allowed.
   ```
   **Решение**: Перед сохранением вызовите `.contiguous()` для тензоров модели. Обновите код сохранения, добавив:
   ```python
   state_dict = {k: v.contiguous() for k, v in model.state_dict().items()}
   trainer.save_model(output_dir='./results', state_dict=state_dict)
   ```
   Это обеспечит непрерывность тензоров перед сохранением.

2. **Предупреждения**:
   - Отсутствие `HF_TOKEN` в Colab secrets (рекомендуется для доступа к Hugging Face Hub).
   - Устаревший параметр `evaluation_strategy` (замените на `eval_strategy` в новых версиях `transformers`).

## Пример вывода

Пример результатов обучения:
```
Epoch | Training Loss | Validation Loss
1     | No log        | 0.003429
2     | No log        | 0.000190
```

## Ограничения

- Требуется GPU для эффективного обучения.
- Размер батча и максимальная длина текста (`max_length=512`) могут потребовать настройки в зависимости от объема данных и доступной памяти.
- Ошибка сохранения модели требует ручного исправления.
