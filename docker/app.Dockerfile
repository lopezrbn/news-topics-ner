# app.Dockerfile
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# System deps necesarios (git para llms_inferer, libpq para psycopg2)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    libpq-dev \
  && rm -rf /var/lib/apt/lists/*

# Instalar dependencias Python
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copiar el resto del proyecto
COPY . .

# Instalar el paquete en modo editable
RUN pip install -e .

# Comando por defecto (se sobrescribe en docker-compose para cada servicio)
CMD ["bash"]
