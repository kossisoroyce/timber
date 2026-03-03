FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir timber-compiler

ENV TIMBER_HOME=/data/timber
ENV PORT=11434

EXPOSE 11434

CMD timber serve "${MODEL_URL:-https://raw.githubusercontent.com/kossisoroyce/timber/main/examples/breast_cancer_model.json}" \
    --host 0.0.0.0 --port ${PORT}
