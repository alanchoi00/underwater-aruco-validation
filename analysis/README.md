# Offline analysis

Two environments, on purpose.

**Stage 0 (ROS container)** decodes the rosbags once into a plain dataset:

    docker run --rm -v <bags>:/bags:ro -v "$PWD":/work <ros-jazzy-image> \
        bash -lc "source /opt/ros/jazzy/setup.bash && \
                  python3 /work/analysis/extract_bags.py /bags/test1 /work/dataset/test1"

Bags are zstd-compressed; decompress first with `zstd -d <bag>.mcap.zstd`.

**Stages 1-6 (this image)** read only `dataset/` and never import ROS:

    docker build -f analysis/Dockerfile -t uwaruco-analysis .
    docker run --rm -v "$PWD":/work \
      -e ANALYSIS_GIT_SHA="$(git rev-parse --short HEAD)$(git diff --quiet HEAD || echo -dirty)" \
      uwaruco-analysis python analysis/run_analysis.py dataset/

The image has no `git` binary (kept slim on purpose), so `code_version()` cannot shell
out to it; pass the commit SHA in via `ANALYSIS_GIT_SHA` as shown above so it lands in
`results/summary.json` instead of falling back to `"unknown"`.

Tests:

    docker run --rm -v "$PWD":/work uwaruco-analysis python -m pytest -q

## Editing this code

Open the repo in the devcontainer (`.devcontainer/devcontainer.json`, "Reopen in
Container"). It builds from this same `Dockerfile`, so the editor resolves against the
pinned versions the code actually runs on.

This is not cosmetic. The host here is Python 3.10 with cv2 4.5.4, where
`cv2.aruco.ArucoDetector` does not exist, so an editor pointed at the host interpreter
marks correct code as broken and accepts the 4.6 legacy API that the pin exists to avoid.

The devcontainer covers the analysis only. Stage 0 still runs in the ROS container.

## Why the version pin matters

OpenCV rewrote the ArUco module in 4.7. The ROS 2 Jazzy container ships cv2 4.6
(legacy API: no `ArucoDetector`, no `generateImageMarker`). Detection results are not
comparable across that boundary, so this image pins 4.10 and `analysis/tests/test_env.py`
enforces the floor.
