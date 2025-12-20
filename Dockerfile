FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libglib2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir \
        numpy \
        opencv-python-headless \
    && pip install --no-cache-dir \
        torch --index-url https://download.pytorch.org/whl/cpu

COPY web_infer.py /app/web_infer.py

EXPOSE 8000

CMD ["python", "web_infer.py", "--model", "/models/final_model.pt", "--port", "8000"]
