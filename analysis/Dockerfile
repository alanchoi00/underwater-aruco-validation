# Analysis environment for the offline ArUco study.
# Deliberately ROS-free: Stage 0 (extract_bags.py) runs in the ROS container and
# writes a plain PNG + CSV dataset; everything downstream reads only that.
# Pinned because the study's result depends on the detector version.
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY analysis/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

WORKDIR /work
ENV PYTHONPATH=/work
