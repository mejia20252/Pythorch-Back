# Dockerfile optimizado para desplegar en Render (YOLO Plates API)
# - Incluye dependencias del sistema (Tesseract, librerías OpenCV, ffmpeg)
# - Usa caching de pip copiando requirements.txt primero

FROM python:3.11-slim

# Evitar warnings de Python y forzar salida por streaming
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Instalar dependencias del sistema necesarias para OpenCV, Tesseract y compilación
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       build-essential \
       gcc \
       libgl1 \
       libglib2.0-0 \
       libsm6 \
       libxext6 \
       libxrender1 \
       tesseract-ocr \
       ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Directorio de la aplicación
WORKDIR /app

# Copiar sólo requirements primero para aprovechar caché de Docker
COPY requirements.txt /app/requirements.txt

# Instalar dependencias de Python
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copiar el resto del proyecto (incluye best.pt)
COPY . /app/

# Variables por defecto (puedes sobrescribirlas en Render)
ENV MODEL_PATH=/app/best.pt
ENV DEVICE=cpu
ENV IMG_SIZE=640
ENV CONF=0.01
ENV PORT=8000

# Exponer puerto (Render usa la variable PORT en tiempo de ejecución)
EXPOSE 8000

# Comando por defecto: uvicorn sirve la app FastAPI
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
