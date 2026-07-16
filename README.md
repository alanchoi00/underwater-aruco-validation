# underwater-aruco-validation

Standalone toolkit for validating underwater **ArUco** marker detection and pose
estimation on a BlueROV2, decoupled from the docking controller. It covers the
capture-to-analysis path: generate a print target, record a pool run, and characterise
detection against range and marker size offline.

This repo is **independent** of the docking pipeline. It runs its own `cv2.aruco`
detection and carries its own marker definitions; it does not import any controller or
perception package.

## Prerequisites

**Docker**, and nothing else. Every step runs in a container, so no local Python is
needed or wanted: the detection results depend on the exact OpenCV version, which is why
`requirements.txt` is pinned and `analysis/tests/test_env.py` asserts the running cv2 is
that pin rather than merely a recent one.

```bash
docker build -t uwaruco-analysis .
```

Step 3 additionally needs a **ROS 2 Jazzy image**, which is why Docker is not optional
even if you have the Python packages locally.

**VS Code with the Dev Containers extension** is optional and only affects editing. If
you "Reopen in Container", drop the `docker run --rm -v "$PWD":/work uwaruco-analysis`
prefix from every command below and just run `python ...`. Everything else is identical,
because the devcontainer builds from the same `Dockerfile`.

`capture/record_bag.sh` is the exception to all of this: it runs on the ROV against a
live ZED, not on your machine.

## 1. Make the target

```bash
docker run --rm -v "$PWD":/work uwaruco-analysis python target/aruco_collage_a4.py
```

Writes `target/aruco_collage_A4.pdf` and `target/marker_sizes_nominal.yaml`.

Print at **100% (actual size)**, then **caliper the printed markers and record what they
actually measure** in `target/marker_sizes_measured.yaml`. Do not skip this. The
2026-07-02 board printed at ~95.9%, almost certainly "fit to printable area", which
shifts every derived range by that factor. The nominal file says what the PDF asked the
printer for and is overwritten on every run; the measured file says what you got, and is
the one the analysis reads.

## 2. Record a run

On the ROV, which is where `ros2 bag` and the live ZED topics are:

```bash
capture/record_bag.sh alan-3        # -> $DATASET_DIR/alan-3
```

Before diving, confirm the camera and odometry are actually publishing:

```bash
ros2 topic hz /zed/zed_node/left/image_rect_color/compressed
ros2 topic hz /zed/zed_node/odom
```

## 3. Decode the bags

Once per run, in the **ROS 2 container**. This is the only step that needs ROS, and
`analysis/extract_bags.py` is the only file that imports it; everything downstream reads
the plain PNG and CSV dataset it writes. That split is what lets this repo pin its own
OpenCV, since the ROS image ships cv2 4.6 against the 4.10 the detection results depend
on.

```bash
zstd -d alan1_0.mcap.zstd                       # rosbag2 cannot read .mcap.zstd directly
docker run --rm -v /path/to/bags:/bags:ro -v "$PWD":/work <ros-jazzy-image> \
  bash -lc "source /opt/ros/jazzy/setup.bash && \
            python3 /work/analysis/extract_bags.py /bags/alan1_0.mcap /work/dataset/test1"
```

## 4. Analyse

```bash
docker run --rm -v "$PWD":/work uwaruco-analysis python analysis/run_analysis.py dataset/
```

Figures and CSVs land in `results/`, each stamped with the analysis commit so a figure in
the report traces back to the code that made it. Tests:

```bash
docker run --rm -v "$PWD":/work uwaruco-analysis python -m pytest -q
```

See `analysis/README.md` for what each stage does.

## On ground truth

The 2026-07-02 capture recorded **no ground truth**: range and heading were never
measured, and ZED odometry was not in the bag. `record_bag.sh` fixes this by recording
`/zed/zed_node/odom` and `/zed/zed_node/pose`, which give a metric trajectory. Record
them. Without a metric reference, detection range can only be reported against the ArUco
pose itself, which is the thing under test, and translation accuracy is not measurable at
all. See `capture/2026-07-02.md`.
