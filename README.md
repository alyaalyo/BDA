# Прогнозирование оттока абонентов

Конвейер обучает Random Forest предсказывать, уйдёт ли абонент. Он объединяет признаки и метки по `subscriber_key`, стратифицированно делит данные на train, validation и test в пропорции 60/20/20 и считает ROC-AUC, F1, precision и recall на validation и test. Порог классификации по умолчанию равен 0.5.

## Входные данные

По умолчанию конвейер читает `data/out/reference/churn_truth.csv`. Целевой столбец `churn` или `churned` принимает значения `0/1` или `False/True`. Также поддерживается CSV с `subscriber_key` и одним столбцом меток с другим именем. В каждой входной таблице ключ абонента должен быть уникальным.

Если есть `data/processed/features.parquet`, конвейер берёт из него числовые признаки. Таблица должна содержать `subscriber_key` и признаки, рассчитанные на дату среза без использования будущих событий.

Если метки и готовый Parquet уже размещены по указанным путям, конвейер готов к запуску. Для запуска с архивом генератора распакуйте его из корня проекта. В архиве каталог верхнего уровня называется `out`.

```bash
mkdir -p data
unzip -n /путь/к/out.zip 'out/*' -d data
```

При отсутствии Parquet конвейер берёт активных абонентов из `data/out/raw/dictionaries/subscribers_snapshot_2025-11-15.csv` и CDR из `data/out/{raw,late_delivery}/cdr/dt=*/cdr.csv`. При совпадении дат берётся раздел из `late_delivery`. Учитываются разделы по 16 ноября 2025 года включительно и только события со временем строго раньше 15 ноября 2025 года. Из CDR рассчитываются число событий, длительность, объём трафика и начисления. Для абонентов без событий эти признаки равны нулю.

Пути, даты среза, доли выборок, seed, параметры модели, имена артефактов и логирование задаются в `config.json`. Относительные пути вычисляются от каталога конфига. Можно передать другой файл через `--config`.

## Локальный запуск

Требуются Python 3.12 и macOS или Linux. Команды выполняются из корня репозитория.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python pipeline.py --config config.json
```

## Docker

Требуется запущенный Docker Engine. Команды используют стандартный `config.json`. Каталог с данными монтируется для чтения, а каталог результатов сохраняется на хосте.

```bash
mkdir -p artifacts
docker build -t bda-churn:local .
docker run --rm \
  --mount "type=bind,source=${PWD}/data,target=/app/data,readonly" \
  --mount "type=bind,source=${PWD}/artifacts,target=/app/artifacts" \
  bda-churn:local
```

Повторите команду запуска с теми же данными и настройками. При целых артефактах журнал сообщит «Артефакты актуальны. Обучение пропущено.»

## Артефакты

| Файл | Содержимое |
| --- | --- |
| `artifacts/best_model.pkl` | Обученный Random Forest |
| `artifacts/scaler.pkl` | StandardScaler, обученный на train |
| `artifacts/features.json` | Порядок признаков для предсказаний |
| `artifacts/metrics.json` | Метрики validation и test, размеры выборок, число признаков, seed и порог |
| `artifacts/manifest.json` | Хеш запуска и контрольные суммы четырёх артефактов |
| `artifacts/pipeline.log` | Этапы, длительность обучения, сообщения о повторных запусках и ошибки |

В `artifacts/` также создаётся служебный файл блокировки `.pipeline.lock`. Схема этапов приведена в [architecture.md](architecture.md), детали рефакторинга и проверки приведены в [refactoring_report.md](refactoring_report.md).
