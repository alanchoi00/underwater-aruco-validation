# Analysis environment for the offline ArUco study.
# Deliberately ROS-free: Stage 0 (extract_bags.py) runs in the ROS container and
# writes a plain PNG + CSV dataset; everything downstream reads only that.
# Pinned because the study's result depends on the detector version.
FROM python:3.12-slim

# git is here for provenance, not convenience. run_analysis.py stamps the analysis
# commit into summary.json so a figure in the report traces to the code that made it;
# without git in the image that fell back to "unknown", and the ANALYSIS_GIT_SHA env var
# was a workaround for its absence. It also lets the devcontainer's source control and
# GitLens work, since that image builds from this one. Costs ~98 MB on a 1.23 GB image
# and touches none of the pinned versions the reproducibility claim rests on.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 git \
    && rm -rf /var/lib/apt/lists/*

# The workspace is bind mounted from the host, so its files are owned by the host user
# while this container runs as root. Without this, every git call dies with "detected
# dubious ownership in repository at '/work'" and the provenance stamp silently reverts
# to "unknown". --system so it holds whatever user the container ends up running as.
RUN git config --system --add safe.directory /work

COPY analysis/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

WORKDIR /work
ENV PYTHONPATH=/work
