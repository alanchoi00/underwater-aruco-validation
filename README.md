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

Build the image and start a container you keep around:

```bash
docker build -t uwaruco-analysis .
docker run -d --name uwaruco -v "$PWD":/work uwaruco-analysis sleep infinity
docker exec -it uwaruco bash
```

**Every command below runs in that shell.** The workspace is bind mounted, so edits on
the host are live inside; there is no need to rebuild or restart between runs. When you
are done: `docker rm -f uwaruco`.

**VS Code with the Dev Containers extension** is an optional alternative. "Reopen in
Container" gives you the same shell, running the same commands, because it builds from
the same `Dockerfile`.

Two things fall outside that shell. Step 3 needs a **ROS 2 Jazzy** container, which is
why Docker is not optional even if you have the Python packages locally. And
`capture/record_bag.sh` runs on the ROV against a live ZED, not on your machine.

## 1. Make the target

```bash
python target/aruco_collage_a4.py
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

The odom check is the one that matters. 2026-07-02 lost it, and with it any ground truth,
so every range in those results is measured against the ArUco pose itself, which is the
thing under test. `capture/2026-07-02.md` records what that cost.

## 3. Decode the bags

Once per run, in a **ROS 2 Jazzy** container. This is the only step that needs ROS, and
`analysis/extract_bags.py` is the only file that imports it; everything downstream reads
the plain PNG and CSV dataset it writes. That split is what lets this repo pin its own
OpenCV, since the ROS image ships cv2 4.6 against the 4.10 the detection results depend
on.

From the host, since it needs both the bags and this repo mounted:

```bash
zstd -d alan1_0.mcap.zstd                       # rosbag2 cannot read .mcap.zstd directly
docker run --rm -v /path/to/bags:/bags:ro -v "$PWD":/work <ros-jazzy-image> \
  bash -lc "source /opt/ros/jazzy/setup.bash && \
            python3 /work/analysis/extract_bags.py /bags/alan1_0.mcap /work/dataset/test1"
```

If you already have a ROS container running with both mounted, `docker exec` into it and
run the inner command instead.

## 4. Analyse

```bash
python analysis/run_analysis.py dataset/
```

Figures and CSVs land in `results/`, each stamped with the analysis commit so a figure in
the report traces back to the code that made it. Tests:

```bash
python -m pytest -q
```

See `analysis/README.md` for what each stage does.
