from fastapi import APIRouter, UploadFile, File
import tempfile
import os
import pytesseract

from src.pipelines.vin_pipeline import detect_vin_crop, preprocess_vin
from src.data.validate_data import clean_vin

router = APIRouter()


def run_tesseract_ocr(processed_image) -> str:
    config = (
        "--psm 7 "
        "-c tessedit_char_whitelist=ABCDEFGHJKLMNPRSTUVWXYZ0123456789"
    )

    return pytesseract.image_to_string(
        processed_image,
        config=config
    )


@router.post("/v1/vin/photo")
async def vin_from_photo(file: UploadFile = File(...)):
    image_path = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            tmp.write(await file.read())
            image_path = tmp.name

        crop = detect_vin_crop(image_path)

        if crop is None:
            return {
                "success": False,
                "error": "VIN area not detected"
            }

        processed = preprocess_vin(crop)

        if processed is None:
            return {
                "success": False,
                "error": "VIN crop preprocessing failed"
            }

        raw_text = run_tesseract_ocr(processed)
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