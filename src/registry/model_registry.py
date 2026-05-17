from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_ROOT = PROJECT_ROOT / "models"

PRICE_MODEL_NAME = "car_price_regressor"
MODEL_ARTIFACT_NAME = "model.pkl"


def get_latest_model_version(model_name: str = PRICE_MODEL_NAME) -> str:
    model_root = MODEL_ROOT / model_name
    versions = sorted(
        path.name
        for path in model_root.iterdir()
        if path.is_dir() and (path / MODEL_ARTIFACT_NAME).exists()
    )

    if not versions:
        raise RuntimeError(f"No model versions found under {model_root}.")

    return versions[-1]


def get_model_path(
    model_name: str = PRICE_MODEL_NAME,
    model_version: str | None = None,
) -> Path:
    version = model_version or get_latest_model_version(model_name)
    return MODEL_ROOT / model_name / version / MODEL_ARTIFACT_NAME


def get_next_model_version(
    model_name: str = PRICE_MODEL_NAME,
    run_date: date | None = None,
) -> str:
    date_part = (run_date or date.today()).isoformat()
    model_root = MODEL_ROOT / model_name
    model_root.mkdir(parents=True, exist_ok=True)

    sequence_numbers = []
    for path in model_root.glob(f"{date_part}_*"):
        try:
            sequence_numbers.append(int(path.name.rsplit("_", 1)[1]))
        except (IndexError, ValueError):
            continue

    return f"{date_part}_{max(sequence_numbers, default=0) + 1:03d}"
