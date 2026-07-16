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
      uwaruco-analysis python analysis/run_analysis.py dataset/

`run_analysis.py` stamps the analysis commit into `results/summary.json`, so a figure in
the report traces back to the code that made it. The image carries `git` for exactly this,
and marks `/work` as a safe directory, since the workspace is bind mounted from the host
and git otherwise refuses to read a repo it thinks belongs to someone else.

Set `ANALYSIS_GIT_SHA` to override it. That matters when `.git` is not mounted, for
instance in CI, where the stamp would otherwise fall back to `"unknown"`:

    docker run --rm -v "$PWD":/work \
      -e ANALYSIS_GIT_SHA="$(git rev-parse --short HEAD)" \
      uwaruco-analysis python analysis/run_analysis.py dataset/

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
