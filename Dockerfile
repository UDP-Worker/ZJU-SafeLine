FROM ubuntu:22.04

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        libglib2.0-0 \
        libgl1 \
        python3 \
        python3-numpy \
        python3-opencv \
        python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --no-cache-dir --upgrade pip \
    && python3 -m pip install --no-cache-dir \
        torch --index-url https://download.pytorch.org/whl/cpu

COPY web_infer.py /app/web_infer.py

EXPOSE 8000

CMD ["python3", "web_infer.py", "--model", "/models/final_model.pt", "--port", "8000"]
