# Credit Scoring

Проект по прогнозированию риска дефолта клиента на данных соревнования
[Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk).
Задача решается как бинарная классификация, основная метрика — `ROC-AUC`.

## Результаты

| Модель | ROC-AUC | PR-AUC |
|---|---:|---:|
| CatBoost | **0.7910** | **0.2931** |
| HistGradientBoosting | 0.7866 | 0.2831 |
| Random Forest | 0.7669 | 0.2529 |
| Decision Tree | 0.7161 | 0.1941 |
| Logistic Regression | 0.6451 | 0.1301 |

Kaggle Public Score финального CatBoost: **0.78910**.

## Что сделано

- обработана основная таблица заявок и дополнительные таблицы соревнования;
- построены агрегаты по кредитной истории, предыдущим заявкам и платежам;
- числовые и категориальные признаки обрабатываются раздельно;
- Logistic Regression используется как интерпретируемый baseline;
- Decision Tree, Random Forest, HistGradientBoosting и CatBoost сравниваются на
  одном validation-разбиении;
- при дисбалансе классов используются одновременно ROC-AUC и PR-AUC;
- для baseline отдельно показан выбор порога и анализ ошибок.

## Почему выбрана CatBoost

CatBoost получил лучшие ROC-AUC и PR-AUC и естественно работает со смешанными
табличными признаками. Разница между локальным ROC-AUC `0.7910` и Public Score
`0.78910` небольшая, что служит дополнительной проверкой устойчивости решения.

## Структура проекта

```text
.
├── notebooks/
│   ├── data_preprocessing.ipynb   # подготовка и агрегация таблиц
│   └── models.ipynb               # обучение, сравнение и анализ ошибок
├── models/
│   └── catboost_credit_scoring.cbm
├── processed/
│   └── submission_catboost.csv
├── home_credit_dictionary_ru.csv
├── README.md
└── requirements.txt
```

Исходные и промежуточные таблицы соревнования не публикуются в репозитории.
Файл `.gitignore` исключает `data/`, крупные parquet-файлы и служебную папку
`catboost_info/`.

## Запуск

```bash
git clone https://github.com/novaeKH/Credit-scoring.git
cd Credit-scoring
python -m venv .venv
```

```bash
# Linux / macOS
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1
```

```bash
pip install -r requirements.txt
jupyter lab
```

Загрузите Competition Data с Kaggle в локальную папку `data/`. Выполняйте
notebook’и в следующем порядке:

1. `notebooks/data_preprocessing.ipynb`;
2. `notebooks/models.ipynb`.

Первый notebook создаёт подготовленные признаки, второй обучает модели и
формирует `processed/submission_catboost.csv`.

## Ограничения

- используется одно локальное validation-разбиение, поэтому для production
  решения потребовалась бы дополнительная кросс-валидация;
- соревнование оценивает ранжирование риска, но не задаёт стоимость ошибок и
  бизнес-порог одобрения кредита;
- Kaggle score не заменяет проверку калибровки вероятностей, fairness и drift;
- сохранённая модель предназначена для воспроизводимости проекта, а не для
  непосредственного production deployment.

