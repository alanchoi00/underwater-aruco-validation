FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 git \
    && rm -rf /var/lib/apt/lists/*

RUN git config --system --add safe.directory /work

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

WORKDIR /work
ENV PYTHONPATH=/work
