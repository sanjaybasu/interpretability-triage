FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml .
COPY Makefile .
COPY src/ src/
COPY tests/ tests/
COPY *.py ./
COPY data/ data/

RUN pip3 install --no-cache-dir -e . && apt-get update && apt-get install -y make && rm -rf /var/lib/apt/lists/*

# Default: run full pipeline
CMD ["make", "pipeline"]
