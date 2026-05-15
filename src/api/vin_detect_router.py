from fastapi import APIRouter, UploadFile, File
from paddleocr import PaddleOCR
import tempfile
import os

from src.pipelines.vin_pipeline import detect_vin_crop, preprocess_vin
from src.data.validate_data import clean_vin

router = APIRouter()

ocr = PaddleOCR(lang="en")  # load once

@router.post("/v1/vin/photo")
async def vin_from_photo(file: UploadFile = File(...)):
    image_path = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            tmp.write(await file.read())
            image_path = tmp.name

        crop = detect_vin_crop(image_path)

        if crop is None:
            return {"success": False, "error": "VIN area not detected"}

        processed = preprocess_vin(crop)

        result = ocr.ocr(processed, cls=True)

        texts = []
        for line in result[0]:
            detected_text = line[1][0]
            texts.append(detected_text)

        raw_text = " ".join(texts)
        vin = clean_vin(raw_text)

        if not vin:
            return {
                "success": False,
                "error": "VIN text not readable",
                "raw_text": raw_text
            }

        return {
            "success": True,
            "vin": vin,
            "raw_text": raw_text
        }

    finally:
        if image_path and os.path.exists(image_path):
            os.remove(image_path)