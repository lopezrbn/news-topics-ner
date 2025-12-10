FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Paquetes de sistema necesarios
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copiamos sólo lo necesario para instalar el paquete
COPY pyproject.toml ./
COPY requirements.txt ./
COPY src ./src
COPY data ./data

# Instalamos dependencias
RUN pip install --upgrade pip && \
    if [ -f "requirements.txt" ]; then pip install -r requirements.txt; fi && \
    pip install -e .

# Copiamos el resto del proyecto
COPY . .

# Puerto interno (la API usará 8000; se mapea desde docker-compose)
EXPOSE 8000
