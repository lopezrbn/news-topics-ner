FROM python:3.10-slim

# Avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install minimal system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Create working directory
WORKDIR /app

# Copy project files
COPY pyproject.toml poetry.lock* requirements.txt* .env.example ./  /app/
COPY src/ /app/src/
COPY config/ /app/config/
COPY airflow_dags/ /app/airflow_dags/

# Install Python dependencies
RUN pip install --upgrade pip && \
    if [ -f "requirements.txt" ]; then pip install -r requirements.txt; fi && \
    pip install -e .

# Install spaCy model
RUN python -m spacy download en_core_web_md

# Create directories for models/artifacts
RUN mkdir -p /app/models /app/mlruns

# Default variables (will be overwritten with .env from compose)
ENV PYTHONUNBUFFERED=1
ENV MLFLOW_ARTIFACT_ROOT=/app/mlruns

# Default command: start the API
CMD ["uvicorn", "news_nlp.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
