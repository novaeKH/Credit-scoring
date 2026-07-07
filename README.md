# Credit Scoring

Проект по прогнозированию риска дефолта клиента на данных соревнования Home Credit Default Risk.

## Что сделано

- обработаны основные и дополнительные таблицы датасета;
- созданы агрегированные признаки по кредитной истории, платежам и прошлым заявкам;
- обучены Logistic Regression, Decision Tree, Random Forest, HistGradientBoosting и CatBoost;
- модели сравнены по ROC-AUC и PR-AUC;
- для baseline подобран порог классификации и проведён анализ ошибок.

## Результаты

| Модель | ROC-AUC | PR-AUC |
|---|---:|---:|
| CatBoost | **0.7910** | **0.2931** |
| HistGradientBoosting | 0.7866 | 0.2831 |
| Random Forest | 0.7669 | 0.2529 |
| Decision Tree | 0.7161 | 0.1941 |
| Logistic Regression | 0.6451 | 0.1301 |

Kaggle Public Score: **0.78910**

## Структура

```text
notebooks/
├── 01_data_preprocessing.ipynb
└── 02_modeling.ipynb

models/
└── catboost_credit_scoring.cbm

submissions/
└── submission_catboost.csv
```

## Используемые технологии

Python, pandas, NumPy, scikit-learn, CatBoost, matplotlib.

