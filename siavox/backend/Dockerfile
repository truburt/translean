FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy dependencies list and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create directories for DB/uploads
RUN mkdir -p uploads

# Copy application source
COPY app/ ./backend/app/

# Expose FastAPI port
EXPOSE 8000

# Set environment defaults
ENV DATABASE_URL=sqlite:////app/siavox.db
ENV OLLAMA_HOST=http://host.docker.internal:11434
ENV OLLAMA_MODEL=gemma4:e4b

CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
