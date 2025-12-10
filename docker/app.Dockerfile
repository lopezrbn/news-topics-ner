FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System packages needed
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy project files needed for installing dependencies
COPY pyproject.toml ./
COPY requirements.txt ./
COPY src ./src
COPY data ./data

# Install Python dependencies
RUN pip install --upgrade pip && \
    if [ -f "requirements.txt" ]; then pip install -r requirements.txt; fi && \
    pip install -e .

# Copy the rest of the project
COPY . .

# Internal port (the API will use 8000; mapped from docker-compose)
EXPOSE 8000
