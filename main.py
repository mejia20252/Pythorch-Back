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

print(f"--- Configuración Inicial ---")
print(f"MODEL_PATH: {MODEL_PATH}")
print(f"DEVICE: {DEVICE}")
print(f"IMG_SIZE: {IMG_SIZE}")
print(f"CONF: {CONF}")

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
print(f"Verificando si el modelo existe en: {MODEL_PATH}")
if not os.path.exists(MODEL_PATH):
    print(f"ERROR: No se encontró el modelo en: {MODEL_PATH}. ¡La aplicación no puede iniciar!")
    raise FileNotFoundError(f"No se encontró el modelo en: {MODEL_PATH}")

print("Cargando modelo YOLO...")
model = YOLO(MODEL_PATH)
CLASS_NAMES = model.names  # dict {id: name}
print("Modelo YOLO cargado exitosamente.")
print(f"Clases detectadas por el modelo: {CLASS_NAMES}")


# -------- Rutas ----------
@app.get("/", response_class=PlainTextResponse)
def root():
    print("GET / - Accedida ruta raíz.")
    return (
        "YOLO Plates API\n"
        "Endpoints:\n"
        "  GET  /health\n"
        "  POST /predict  (multipart: file=@imagen)\n"
    )


@app.get("/health")
def health():
    print("GET /health - Accedida ruta de salud.")
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
    print("Iniciando _predict_image_bytes...")
    try:
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img_np = np.array(img)  # PIL -> NumPy
        print("Imagen cargada y convertida a NumPy.")
    except Exception as e:
        print(f"ERROR: Imagen inválida durante la carga: {e}")
        raise HTTPException(status_code=400, detail=f"Imagen inválida: {e}")

    # Inferencia con YOLO
    print(f"Realizando inferencia con YOLO (imgsz={IMG_SIZE}, conf={CONF}, device={DEVICE})...")
    results = model.predict(
        img, imgsz=IMG_SIZE, conf=CONF, device=DEVICE, verbose=False
    )
    r = results[0]
    print(f"Inferencia YOLO completada. Número de detecciones: {len(r.boxes)}")

    preds = []
    for i, b in enumerate(r.boxes):
        print(f"Procesando detección {i+1}...")
        # Coordenadas de la caja
        x1, y1, x2, y2 = [float(v) for v in b.xyxy[0].tolist()]
        print(f"Coordenadas de la caja (x1, y1, x2, y2): {x1, y1, x2, y2}")

        # Recortar imagen de la placa
        x1_int, y1_int, x2_int, y2_int = int(x1), int(y1), int(x2), int(y2)
        cropped_img = img_np[y1_int:y2_int, x1_int:x2_int]
        print(f"Imagen de la placa recortada. Dimensiones: {cropped_img.shape}")

        # Preprocesar para OCR
        cropped_img_gray = cv2.cvtColor(cropped_img, cv2.COLOR_RGB2GRAY)
        print("Imagen recortada convertida a escala de grises para OCR.")

        # Extraer texto con OCR
        print("Intentando extraer texto con Tesseract OCR...")
        plate_text = pytesseract.image_to_string(
            cropped_img_gray, config="--psm 8"
        ).strip()
        print(f"Texto reconocido por OCR: '{plate_text}'")
        
        preds.append({"plate_text": plate_text})
    
    print(f"Predicciones OCR completadas. Total de patentes detectadas y procesadas: {len(preds)}")
    # Se devuelve solo el texto reconocido, no la lista completa de predicciones
    return preds, r.plot()[..., ::-1] # Se mantiene el retorno de la imagen anotada por si la quieres usar más adelante.

# -------- Endpoint principal modificado ----------
@app.post("/predict", response_class=PlainTextResponse)
async def predict(file: UploadFile = File(...)):
    print(f"POST /predict - Recibida solicitud para predecir. Nombre del archivo: {file.filename}")
    if not file.filename:
        print("ERROR: No se ha subido un archivo de imagen.")
        raise HTTPException(status_code=400, detail="Sube un archivo de imagen.")

    print(f"Leyendo bytes del archivo '{file.filename}'...")
    img_bytes = await file.read()
    print(f"Bytes del archivo leídos. Tamaño: {len(img_bytes)} bytes.")
    
    try:
        preds, _ = _predict_image_bytes(img_bytes)
    except HTTPException as e:
        print(f"ERROR: Error durante la predicción de la imagen: {e.detail}")
        raise e # Re-lanzar la excepción para que FastAPI la maneje.
    except Exception as e:
        print(f"ERROR INESPERADO durante la predicción de la imagen: {e}")
        raise HTTPException(status_code=500, detail=f"Error interno del servidor durante la predicción: {e}")

    # Si se encuentra alguna patente, se devuelve el texto de la primera.
    if preds:
        first_plate_text = preds[0]["plate_text"]
        print(f"Patente detectada: '{first_plate_text}'. Devolviendo resultado.")
        return first_plate_text

    # Si no se detecta ninguna patente, se devuelve un mensaje claro.
    print("No se ha detectado ninguna patente en la imagen.")
    return "No se ha detectado ninguna patente."