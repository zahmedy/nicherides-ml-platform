from fastapi import APIRouter, UploadFile, File
from paddleocr import PaddleOCR
import tempfile

from src.pipelines.vin_pipeline import detect_vin_crop, preprocess_vin
from src.evaluation.evaluate_vin_detector import clean_vin

router = APIRouter()

@router.post("/v1/vin/photo")
async def vin_from_photo(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(await file.read())
        image_path = tmp.name

    crop = detect_vin_crop(image_path)

    if crop is None:
        return {"success": False, "error": "VIN area not detected"}
    
    ocr = PaddleOCR(lang='en')

    processed = preprocess_vin(crop)
    text = ocr.predict(processed)
    vin = clean_vin(text[0])

    if not vin:
        return {"success": False, "error": "VIN text not readable"}

    return { "success": True, "vin": vin }