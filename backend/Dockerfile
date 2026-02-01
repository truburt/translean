# Copyright © 2026 Vladimir Vaulin-Belskii. All rights reserved.
FROM python:3.11-slim


WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY backend /app

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
