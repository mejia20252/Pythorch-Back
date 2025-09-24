# Utiliza una imagen base de Python más reciente, basada en Debian Bookworm (estable actual).
# Esto solucionará el problema de los repositorios de apt-get.
FROM python:3.10-slim-bookworm

# Instala las dependencias del sistema necesarias para OpenCV y Tesseract OCR.
# libgl1-mesa-glx: Requerido por OpenCV en entornos sin cabeza (headless).
# libglib2.0-0: Otra dependencia común para bibliotecas gráficas.
# tesseract-ocr, tesseract-ocr-spa: El motor Tesseract y el paquete de idioma español.
# git, build-essential: Herramientas comunes que podrían ser necesarias para algunas compilaciones o dependencias.
# Nota: La lista de dependencias se mantiene igual, ya que son estándar para estas funcionalidades.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    tesseract-ocr \
    tesseract-ocr-spa \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Establece el directorio de trabajo dentro del contenedor.
WORKDIR /app

# Copia el archivo de requisitos primero para aprovechar la caché de Docker.
COPY requirements.txt .

# --- Instalación Crítica de Dependencias ---

# 1. Instala PyTorch y Torchvision primero.
#    Esto es vital. PyTorch suele venir precompilado con una versión específica de NumPy.
#    Al instalarlo primero, permitimos que PyTorch establezca su entorno correctamente.
#    Asegúrate de que 'cu117' coincida con la versión de CUDA que esperas o que esté disponible en tu entorno de Render.
#    Si no usas GPU, podrías cambiar a 'cpu'.
#    Dado que Render.com suele desplegar en CPUs para servicios web estándar,
#    es muy probable que necesites la versión 'cpu' de PyTorch.
#    Si estás seguro de que Render te proporciona una GPU con CUDA 11.7, mantén 'cu117'.
#    Si no, cambia a 'cpu' para evitar problemas de compatibilidad y tamaño de imagen.
RUN pip install --no-cache-dir \
    torch==2.0.1 \
    torchvision==0.15.2 \
    --index-url https://download.pytorch.org/whl/cpu # CAMBIADO a 'cpu' para entornos sin GPU

# 2. Instala las demás dependencias desde requirements.txt.
#    En este punto, NumPy ya debería haber sido instalado por PyTorch.
#    Asegúrate de que 'requirements.txt' NO contenga 'numpy' para evitar conflictos.
RUN pip install --no-cache-dir -r requirements.txt

# --- Fin de la Instalación Crítica ---

# Copia el código de la aplicación y el modelo entrenado.
COPY main.py .
COPY best.pt .

# Expone el puerto que usará el servidor FastAPI.
EXPOSE 8000

# Comando para iniciar la aplicación con Uvicorn.
# El host 0.0.0.0 permite que la aplicación sea accesible desde fuera del contenedor.
# El puerto 8000 es donde tu aplicación FastAPI estará escuchando.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]