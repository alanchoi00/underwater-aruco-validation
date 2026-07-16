# Offline analysis

Two environments, on purpose. See the root README for the container shell these commands
assume.

**Stage 0 (ROS container)** decodes the rosbags once into a plain dataset. Run from the
host, since it needs both the bags and this repo mounted:

    zstd -d <bag>.mcap.zstd                     # rosbag2 cannot read .mcap.zstd directly
    docker run --rm -v <bags>:/bags:ro -v "$PWD":/work <ros-jazzy-image> \
        bash -lc "source /opt/ros/jazzy/setup.bash && \
                  python3 /work/analysis/extract_bags.py /bags/alan1_0.mcap /work/dataset/test1"

**Stages 1-6** read only `dataset/` and never import ROS:

    python analysis/run_analysis.py dataset/
    python -m pytest -q

`run_analysis.py` stamps the analysis commit into `results/summary.json`, so a figure in
the report traces back to the code that made it. The image carries `git` for exactly this,
and marks `/work` as a safe directory, since the workspace is bind mounted from the host
and git otherwise refuses to read a repo it thinks belongs to someone else.

Set `ANALYSIS_GIT_SHA` to override it. That matters when `.git` is not mounted, for
instance in CI, where the stamp would otherwise fall back to `"unknown"`:

    ANALYSIS_GIT_SHA=$(git rev-parse --short HEAD) python analysis/run_analysis.py dataset/

## Editing this code

Open the repo in the devcontainer (`.devcontainer/devcontainer.json`, "Reopen in
Container"). It builds from the same root `Dockerfile`, so the editor resolves against the
pinned versions the code actually runs on.

This is not cosmetic. The host here is Python 3.10 with cv2 4.5.4, where
`cv2.aruco.ArucoDetector` does not exist, so an editor pointed at the host interpreter
marks correct code as broken and accepts the 4.6 legacy API that the pin exists to avoid.

The devcontainer covers the analysis only. Stage 0 still runs in the ROS container.

## Why the version pin matters

OpenCV rewrote the ArUco module in 4.7. The ROS 2 Jazzy container ships cv2 4.6
(legacy API: no `ArucoDetector`, no `generateImageMarker`). Detection results are not
comparable across that boundary, so the root `requirements.txt` pins 4.10.

`analysis/tests/test_env.py` asserts the running cv2 IS that pin, not merely that it
clears a floor. A floor cannot catch upward drift, and upward drift is what actually
happened once: a second pip install pulled an unpinned `opencv-contrib-python` 5.0.0
that shadowed the pinned headless 4.10, and the floor check passed it. One pinned
requirements file at the root now makes that unrepresentable.
