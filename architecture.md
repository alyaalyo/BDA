# Архитектура конвейера

```mermaid
flowchart TD
    subgraph Sources["Слой данных · data/"]
        PARQUET["processed/features.parquet"]
        LABELS["out/reference/churn_truth.csv"]
        SNAPSHOT["out/raw/dictionaries/subscribers_snapshot_*.csv"]
        CDR["out/raw/cdr и out/late_delivery/cdr"]
    end

    CONFIG["config.json"] --> RUN["run_pipeline: выбор источников и проверка хешей"]
    PARQUET -. "при наличии" .-> RUN
    LABELS --> RUN
    SNAPSHOT -. "если нет Parquet" .-> RUN
    CDR -. "если нет Parquet" .-> RUN

    subgraph Pipeline["Слой обработки · pipeline.py"]
        BUILD["build_features: активные абоненты и агрегация CDR до среза"]
        LOAD["load_data: объединение по subscriber_key"]
        PREP["preprocess: стратификация, заполнение пропусков, StandardScaler"]
        TRAIN["train_model: обучение Random Forest"]
        EVAL["evaluate: ROC-AUC, F1, precision, recall"]
        SAVE["save_artifacts: подготовка файлов и публикация манифеста"]
        BUILD --> LOAD
        LOAD --> PREP
        PREP -->|"train"| TRAIN
        PREP -->|"validation и test"| EVAL
        TRAIN --> EVAL
        PREP -->|"скейлер и порядок признаков"| SAVE
        TRAIN -->|"модель"| SAVE
        EVAL -->|"метрики"| SAVE
    end

    RUN -->|"готовый Parquet и метки"| LOAD
    RUN -->|"снимок и разделы CDR"| BUILD
    RUN -->|"метки при восстановлении из CDR"| LOAD
    RUN -->|"совпали хеши"| REUSE["Пропуск обучения"]

    subgraph Storage["Слой хранения · artifacts/"]
        FILES["best_model.pkl · scaler.pkl · features.json · metrics.json"]
        MANIFEST["manifest.json"]
        LOG["pipeline.log"]
    end

    SAVE --> FILES
    FILES --> MANIFEST
    MANIFEST -. "сверка при следующем запуске" .-> RUN
    RUN --> LOG
    REUSE --> LOG
    SAVE --> LOG
```

При запуске в Docker каталог данных подключается в `/app/data`, а артефакты и журнал записываются в `/app/artifacts`. Оба пути соответствуют каталогам на хосте, указанным в команде запуска.
