import io
import os
from typing import List, Optional

from fastapi import FastAPI, File, UploadFile, Query, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware

from ultralytics import YOLO
from PIL import Image
import numpy as np

import pytesseract  # OCR
import cv2          # Procesamiento de imágenes


# -------- Configuración ----------
DEFAULT_MODEL = r"best.pt"  # Ajusta si usas otro run
MODEL_PATH = os.getenv("MODEL_PATH", DEFAULT_MODEL)
DEVICE = os.getenv("DEVICE", "cpu")  # "cpu" o "0" si luego tienes CUDA
IMG_SIZE = int(os.getenv("IMG_SIZE", "640"))
CONF = float(os.getenv("CONF", "0.01"))


# -------- App -------------
app = FastAPI(title="YOLO License-Plate API", version="1.0.0")

# Configuración CORS (abierto para pruebas)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cargar modelo YOLO una sola vez
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"No se encontró el modelo en: {MODEL_PATH}")

model = YOLO(MODEL_PATH)
CLASS_NAMES = model.names  # dict {id: name}


# -------- Rutas ----------
@app.get("/", response_class=PlainTextResponse)
def root():
    return (
        "YOLO Plates API\n"
        "Endpoints:\n"
        "  GET  /health\n"
        "  POST /predict  (multipart: file=@imagen)\n" # Modificado para reflejar el nuevo retorno
    )


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_path": MODEL_PATH,
        "device": DEVICE,
        "imgsz": IMG_SIZE,
        "conf": CONF,
        "classes": CLASS_NAMES,
    }


# -------- Función auxiliar ----------
def _predict_image_bytes(img_bytes: bytes):
    try:
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img_np = np.array(img)  # PIL -> NumPy
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Imagen inválida: {e}")

    # Inferencia con YOLO
    results = model.predict(
        img, imgsz=IMG_SIZE, conf=CONF, device=DEVICE, verbose=False
    )
    r = results[0]

    preds = []
    for b in r.boxes:
        # Coordenadas de la caja
        x1, y1, x2, y2 = [float(v) for v in b.xyxy[0].tolist()]

        # Recortar imagen de la placa
        x1_int, y1_int, x2_int, y2_int = int(x1), int(y1), int(x2), int(y2)
        cropped_img = img_np[y1_int:y2_int, x1_int:x2_int]

        # Preprocesar para OCR
        cropped_img_gray = cv2.cvtColor(cropped_img, cv2.COLOR_RGB2GRAY)

        # Extraer texto con OCR
        plate_text = pytesseract.image_to_string(
            cropped_img_gray, config="--psm 8"
        ).strip()
        
        preds.append({"plate_text": plate_text})
    
    # Se devuelve solo el texto reconocido, no la lista completa de predicciones
    return preds, r.plot()[..., ::-1] # Se mantiene el retorno de la imagen anotada por si la quieres usar más adelante.

# -------- Endpoint principal modificado ----------
@app.post("/predict", response_class=PlainTextResponse)
async def predict(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="Sube un archivo de imagen.")

    img_bytes = await file.read()
    preds, _ = _predict_image_bytes(img_bytes)

    # Si se encuentra alguna patente, se devuelve el texto de la primera.
    if preds:
        return preds[0]["plate_text"]

    # Si no se detecta ninguna patente, se devuelve un mensaje claro.
    return "No se ha detectado ninguna patente."