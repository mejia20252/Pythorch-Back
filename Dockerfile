# Utiliza una imagen base de Python con PyTorch
FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime

# Establece el directorio de trabajo dentro del contenedor
WORKDIR /app

# Copia el archivo de requisitos e instala las dependencias
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia el código de la aplicación y el modelo
COPY app/ .
COPY model.pt .

# Expone el puerto que usará el servidor (ej. FastAPI)
EXPOSE 8000

# Comando para iniciar la aplicación (ej. con Uvicorn para FastAPI)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]