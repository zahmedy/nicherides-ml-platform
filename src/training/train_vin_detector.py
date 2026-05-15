from ultralytics import YOLO

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
YAML_PATH = PROJECT_ROOT / "data" / "VINsion" / "data.yaml"
FINE_TUNE_MODEL_PATH = PROJECT_ROOT / "runs" / "vin_detector" / "weights" / "best.pt"

#model = YOLO("yolo12n.pt")
model = YOLO(FINE_TUNE_MODEL_PATH)

model.train(
    data=YAML_PATH,
    epochs=10,
    imgsz=640,
    batch=8,
    device="mps",
    lr0=0.0005,
    workers=4,
    project=str(PROJECT_ROOT / "runs"),
    name="vin_detector_finetune",
    patience=8
)

