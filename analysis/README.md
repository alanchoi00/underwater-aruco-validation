# Offline analysis

Two environments, on purpose.

**Stage 0 (ROS container)** decodes the rosbags once into a plain dataset:

    docker run --rm -v <bags>:/bags:ro -v "$PWD":/work <ros-jazzy-image> \
        bash -lc "source /opt/ros/jazzy/setup.bash && \
                  python3 /work/analysis/extract_bags.py /bags/test1 /work/dataset/test1"

Bags are zstd-compressed; decompress first with `zstd -d <bag>.mcap.zstd`.

**Stages 1-6 (this image)** read only `dataset/` and never import ROS:

    docker build -f analysis/Dockerfile -t uwaruco-analysis .
    docker run --rm -v "$PWD":/work uwaruco-analysis python analysis/run_analysis.py dataset/

Tests:

    docker run --rm -v "$PWD":/work uwaruco-analysis python -m pytest tests/ -v

## Why the version pin matters

OpenCV rewrote the ArUco module in 4.7. The ROS 2 Jazzy container ships cv2 4.6
(legacy API: no `ArucoDetector`, no `generateImageMarker`). Detection results are not
comparable across that boundary, so this image pins 4.10 and `tests/test_env.py`
enforces the floor.
