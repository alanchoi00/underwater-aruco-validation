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

Turbidity is a second driver, answering a different question: how large must a marker be
at range d in water of attenuation beta?

    python analysis/run_turbidity.py dataset/

It measures the pool's attenuation from the board's own black/white contrast decay (no
dosing, no turbidity meter: two patches at the same range share a veiling term, so their
difference is B-free and beta is a straight-line fit), then synthesises added turbidity
onto the real frames and re-runs the real detector. The deliverable is
`results/turbidity_sizing.csv`: the marker side needed at a given range and beta.

Reading `results/turbidity_summary.json` first is worthwhile. It carries the fit quality
(beta at r2 about 0.96, veiling B at about 0.5, which is why the synthesis never inverts
the model) and the limitations of the method.

The last step, `crossval`, is the check that matters most: everything above tests the
code against itself (synthetic frames built from the same closed form the code
implements), while this tests the model against reality. It takes a near sample and a
far sample of the same marker id, so the printed contrast is identical and only the
water between differs, and predicts the far contrast from the near one using nothing but
the measured beta:

    contrast_far = contrast_near * exp(-beta * (d_far - d_near))

Both samples are also corrected for the instrument response before the ratio is formed
(apparent size falls with range, and the sampler under-reads small markers, which would
otherwise show up as the model over-predicting when the defect is measurement, not
attenuation). `results/turbidity_crossval.csv` carries every pair; `summary["crossval"]`
carries the raw and corrected relative-error distributions (n, median, p10, p90) per
channel, both reported because the gap between them is itself evidence about the
correction.

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
