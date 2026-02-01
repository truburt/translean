FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --no-cache-dir --upgrade pip \
    && /opt/venv/bin/pip install --no-cache-dir -r /app/requirements.txt

ENV PATH="/opt/venv/bin:$PATH"

COPY app /app/app

EXPOSE 7860

CMD ["/bin/sh", "-c", "exec python3 -m uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-7860}"]
