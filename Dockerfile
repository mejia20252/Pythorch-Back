# Utiliza una imagen base de Python con PyTorch
FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime

# Instalar las dependencias para OpenCV
# Esto es necesario para evitar los errores "libGL.so.1" y "libgthread-2.0.so.0"
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    tesseract-ocr \
    tesseract-ocr-spa \
    && rm -rf /var/lib/apt/lists/*

# Establece el directorio de trabajo dentro del contenedor
WORKDIR /app

# Copia el archivo de requisitos e instala las dependencias
# Nota: Elimina torch y torchvision de requirements.txt
COPY requirements.txt .
RUN pip install numpy==1.26.4
RUN pip install --no-cache-dir -r requirements.txt

# Instalar PyTorch y Torchvision usando el comando oficial
RUN pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu117

# Copia el código de la aplicación y el modelo
COPY main.py .
COPY best.pt .

# Expone el puerto que usará el servidor (ej. FastAPI)
EXPOSE 8000

# Comando para iniciar la aplicación (ej. con Uvicorn para FastAPI)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]