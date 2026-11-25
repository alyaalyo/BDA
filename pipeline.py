import argparse
import fcntl
import hashlib
import json
import logging
import os
import sys
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

LOGGER = logging.getLogger(__name__)
KEY = "subscriber_key"
TARGET = "churn"
MANIFEST_NAME = "manifest.json"
LOCK_NAME = ".pipeline.lock"
ARTIFACT_KEYS = ("model_name", "scaler_name", "features_name", "metrics_name")


def cdr_sources(root, cutoff_date, last_partition_date):
    snapshot = root / "raw" / "dictionaries" / f"subscribers_snapshot_{cutoff_date}.csv"
    partitions = {}
    for layer in ("raw", "late_delivery"):
        for path in (root / layer / "cdr").glob("dt=*/cdr.csv"):
            if path.parent.name <= f"dt={last_partition_date}":
                partitions[path.parent.name] = path
    if not snapshot.is_file() or not partitions:
        raise FileNotFoundError(
            f"Для восстановления признаков нужны снимок {snapshot} и CDR в {root}."
        )
    return (snapshot, *(partitions[key] for key in sorted(partitions)))


def build_features(snapshot, partitions, cutoff_date):
    population = pd.read_csv(snapshot, usecols=[KEY, "status"], dtype="string")
    active = population["status"].str.strip().str.lower().eq("active")
    population = population.loc[active, [KEY]]
    population[KEY] = population[KEY].str.strip()
    population = population.drop_duplicates()

    columns = [KEY, "event_ts", "duration_sec", "bytes_up", "bytes_down", "charge"]
    cutoff = pd.Timestamp(cutoff_date)
    summaries = []
    for path in partitions:
        try:
            cdr = pd.read_csv(path, usecols=columns, dtype="string", on_bad_lines="skip")
        except (ValueError, UnicodeError) as error:
            LOGGER.warning("Пропущен повреждённый раздел %s: %s", path, error)
            continue

        cdr = cdr.drop_duplicates()
        timestamps = pd.to_datetime(cdr["event_ts"], format="%Y-%m-%d %H:%M:%S", errors="coerce")
        timestamps = timestamps.fillna(
            pd.to_datetime(cdr["event_ts"], format="%d.%m.%Y %H:%M:%S", errors="coerce")
        )
        cdr = cdr.loc[timestamps < cutoff].copy()
        cdr[KEY] = cdr[KEY].str.strip()
        for column in columns[2:]:
            cdr[column] = pd.to_numeric(cdr[column], errors="coerce").fillna(0).clip(lower=0)
        cdr["total_bytes"] = cdr["bytes_up"] + cdr["bytes_down"]
        summaries.append(cdr.groupby(KEY).agg(
            total_calls=("event_ts", "size"),
            total_duration=("duration_sec", "sum"),
            total_bytes=("total_bytes", "sum"),
            total_charge=("charge", "sum"),
        ))

    if not summaries or all(summary.empty for summary in summaries):
        raise ValueError("Нет читаемых событий CDR до даты среза.")
    usage = pd.concat(summaries).groupby(level=0).sum()
    return population.join(usage, on=KEY).fillna(0)


def load_data(churn_path, feature_sources, cutoff_date):
    """Загрузить признаки и присоединить метки оттока по ключу абонента."""
    if len(feature_sources) == 1:
        features = pd.read_parquet(feature_sources[0])
    else:
        snapshot, *partitions = feature_sources
        features = build_features(snapshot, partitions, cutoff_date)

    labels = pd.read_csv(churn_path, dtype={KEY: "string"})
    label_columns = labels.columns.drop(KEY)
    target_column = next((name for name in (TARGET, "churned") if name in label_columns), None)
    if target_column is None:
        if len(label_columns) != 1:
            raise ValueError("Не удалось определить целевой столбец в CSV.")
        target_column = label_columns[0]

    labels = labels[[KEY, target_column]].rename(columns={target_column: TARGET})
    labels[TARGET] = pd.to_numeric(
        labels[TARGET].astype("string").str.strip().str.lower().replace({"true": "1", "false": "0"}),
        errors="coerce",
    )
    features = features.drop(columns=[TARGET, "churned", "churn_day_index"], errors="ignore")
    for frame in (features, labels):
        frame[KEY] = frame[KEY].astype("string").str.strip()
        if frame[KEY].isna().any() or frame[KEY].eq("").any():
            raise ValueError(f"Столбец {KEY} содержит пустые ключи.")

    df_model = features.merge(labels, on=KEY, how="inner", validate="one_to_one")
    LOGGER.info("Загружено строк: признаки %d, метки %d, после объединения %d",
                len(features), len(labels), len(df_model))
    return df_model


def preprocess(df_model, test_size, val_size, random_state):
    """Разделить данные и обучить StandardScaler на тренировочной выборке."""
    if not 0 < test_size < 1 or not 0 < val_size < 1 - test_size:
        raise ValueError("Доли test_size и val_size должны быть положительны и в сумме меньше 1.")
    y = df_model[TARGET]
    if y.isna().any() or set(y.unique()) != {0, 1}:
        raise ValueError("После объединения нужны оба класса churn (0 и 1) без пропусков.")

    X = df_model.drop(columns=[TARGET, KEY]).select_dtypes(include="number").fillna(0)
    if X.shape[1] == 0:
        raise ValueError("В данных нет числовых признаков.")
    X_dev, X_test, y_dev, y_test = train_test_split(
        X, y.astype(int), test_size=test_size, random_state=random_state, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_dev, y_dev, test_size=val_size / (1 - test_size),
        random_state=random_state, stratify=y_dev,
    )
    if min(y_train.nunique(), y_val.nunique(), y_test.nunique()) < 2:
        raise ValueError("Каждая выборка должна содержать оба класса. Проверьте размеры выборок.")
    scaler = StandardScaler()
    train = (scaler.fit_transform(X_train), y_train)
    validation = (scaler.transform(X_val), y_val)
    test = (scaler.transform(X_test), y_test)
    return train, validation, test, X.columns.tolist(), scaler


def train_model(X_train, y_train, params):
    """Обучить Random Forest с параметрами из конфигурации."""
    model = RandomForestClassifier(**params)
    started = perf_counter()
    model.fit(X_train, y_train)
    LOGGER.info("Обучение заняло %.2f сек.", perf_counter() - started)
    return model


def evaluate(model, validation, test, threshold):
    """Посчитать ROC-AUC, F1, precision и recall на двух выборках."""
    metrics = {}
    positive_index = list(model.classes_).index(1)
    for name, (X, y) in (("val", validation), ("test", test)):
        probabilities = model.predict_proba(X)[:, positive_index]
        predictions = probabilities >= threshold
        metrics[f"{name}_auc"] = roc_auc_score(y, probabilities)
        metrics[f"{name}_f1"] = f1_score(y, predictions, zero_division=0)
        metrics[f"{name}_precision"] = precision_score(y, predictions, zero_division=0)
        metrics[f"{name}_recall"] = recall_score(y, predictions, zero_division=0)
    return metrics


def file_hash(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def fingerprint(config, sources):
    payload = {
        "config": config,
        "inputs": file_hashes(sources),
        "code": file_hash(Path(__file__)),
        "python": sys.version_info[:3],
        "libraries": {
            package: version(package)
            for package in ("joblib", "numpy", "pandas", "pyarrow", "scikit-learn", "scipy")
        },
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def file_hashes(paths):
    return [(path.name, file_hash(path)) for path in paths]


def artifacts_current(directory, names, run_id):
    try:
        manifest = json.loads((directory / MANIFEST_NAME).read_text(encoding="utf-8"))
        return (
            manifest["fingerprint"] == run_id
            and manifest["files"] == dict(file_hashes(directory / name for name in names))
        )
    except (FileNotFoundError, ValueError, KeyError, TypeError):
        return False


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")


def save_artifacts(directory, names, run_id, model, scaler, feature_columns, metrics):
    model_name, scaler_name, features_name, metrics_name = names
    with TemporaryDirectory(dir=directory) as temp_dir:
        staging = Path(temp_dir)
        joblib.dump(model, staging / model_name, compress=3)
        joblib.dump(scaler, staging / scaler_name, compress=3)
        write_json(staging / features_name, feature_columns)
        write_json(staging / metrics_name, metrics)
        write_json(staging / MANIFEST_NAME, {
            "fingerprint": run_id,
            "files": dict(file_hashes(staging / name for name in names)),
        })
        for name in (*names, MANIFEST_NAME):
            os.replace(staging / name, directory / name)


def run_pipeline(config_path):
    config_path = config_path.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    root = config_path.parent
    directory = (root / config["artifacts"]["model_dir"]).resolve()
    names = tuple(config["artifacts"][key] for key in ARTIFACT_KEYS)
    reserved = {".", "..", MANIFEST_NAME, LOCK_NAME}
    if len(set(names)) != len(names) or any(
        not name or Path(name).name != name or name in reserved for name in names
    ):
        raise ValueError("Имена артефактов должны быть уникальными именами файлов без каталогов.")
    log_path = (root / config["logging"]["path"]).resolve()
    if log_path in {directory / name for name in (*names, MANIFEST_NAME, LOCK_NAME)}:
        raise ValueError("Путь журнала совпадает с путём одного из артефактов.")

    threshold = config["evaluation"]["threshold"]
    if not 0 <= threshold <= 1:
        raise ValueError("Порог классификации должен лежать между 0 и 1.")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=config["logging"]["level"],
        format=config["logging"]["format"],
        handlers=[logging.StreamHandler(), logging.FileHandler(log_path, encoding="utf-8")],
        force=True,
    )

    data = config["data"]
    churn = root / data["churn_path"]
    features = root / data["features_path"]
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / LOCK_NAME).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        LOGGER.info("Запуск конвейера с конфигурацией %s", config_path)
        if not churn.is_file():
            raise FileNotFoundError(f"Файл меток отсутствует: {churn}")
        if features.is_file():
            sources = (features,)
            LOGGER.info("Готовые признаки: %s", features)
        else:
            sources = cdr_sources(root / data["raw_root"], data["cutoff_date"], data["last_partition_date"])
            LOGGER.info("Восстановление признаков из %d разделов CDR, срез %s",
                        len(sources) - 1, data["cutoff_date"])

        run_id = fingerprint(config, (churn, *sources))
        if artifacts_current(directory, names, run_id):
            LOGGER.info("Артефакты актуальны. Обучение пропущено.")
            return

        LOGGER.info("Загрузка данных")
        df_model = load_data(churn, sources, data["cutoff_date"])
        LOGGER.info("Предобработка и разделение выборок")
        train, validation, test, columns, scaler = preprocess(df_model, **config["preprocessing"])
        LOGGER.info("Строки train/val/test: %d/%d/%d. Признаков: %d",
                    len(train[1]), len(validation[1]), len(test[1]), len(columns))
        LOGGER.info("Обучение Random Forest")
        model = train_model(*train, config["model"])
        LOGGER.info("Оценка модели")
        metrics = evaluate(model, validation, test, threshold)
        metrics.update(
            n_features=len(columns), n_train=len(train[1]), n_val=len(validation[1]), n_test=len(test[1]),
            seed=config["preprocessing"]["random_state"], threshold=threshold,
        )
        save_artifacts(directory, names, run_id, model, scaler, columns, metrics)
        LOGGER.info("Артефакты сохранены в %s. Test ROC-AUC %.4f, F1 %.4f",
                    directory, metrics["test_auc"], metrics["test_f1"])


def main():
    parser = argparse.ArgumentParser(description="Обучение модели оттока абонентов")
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.json"))
    args = parser.parse_args()
    try:
        run_pipeline(args.config)
    except Exception:
        LOGGER.exception("Конвейер завершился с ошибкой")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
