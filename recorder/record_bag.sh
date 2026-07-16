#!/bin/bash
# record_bag.sh -- rosbag capture for underwater ArUco runs.
#
# Records to MCAP with file-level zstd compression. Simpler alternative to
# recorder/run_recorder.py: no run sequencing, no live ArUco cross-check -- just a
# clean bag. Use this when an operator is driving the runs by hand.
#
#   ./record_bag.sh alan-3          -> $DATASET_DIR/alan-3
#   ./record_bag.sh                 -> $DATASET_DIR/bag_<timestamp>
#
# Override the destination with DATASET_DIR=/some/path ./record_bag.sh name
#
# ---------------------------------------------------------------------------
# PROVENANCE: reconstructed from the 2026-07-02 pool capture (alan-1, alan-2).
# The topic list below was verified against those bags -- see "verified" section.
#
# TWO LESSONS FROM THAT CAPTURE, both fixed here. Read them before changing the
# topic list back.
#
#  1. /zed/zed_node/odom and /zed/zed_node/pose were NOT recorded. They are ZED's
#     visual-inertial odometry -- a metric trajectory. Without them the capture has
#     NO GROUND TRUTH: range and heading were never measured by anything, so
#     detection range can only be reported against the ArUco pose itself (which is
#     what is under test) and translation accuracy is not measurable at all.
#     These two lines are the cheapest ground truth available. Never drop them.
#
#  2. Raw image_rect_color overruns the recorder. alan-2 published at ~15 Hz and
#     the bag kept 8.8 Hz -- 42% of frames dropped (960x540 bgra8 = 2.07 MB/frame
#     = 31 MB/s). alan-1 avoided this only by running at 2.5 Hz. Record the
#     /compressed image instead; ArUco detects on it just fine.
# ---------------------------------------------------------------------------

set -euo pipefail

DATASET_DIR="${DATASET_DIR:-/home/bluerov/zimeng/dataset}"

if [ -n "${1:-}" ]; then
    OUTPUT_DIR="$DATASET_DIR/$1"
else
    OUTPUT_DIR="$DATASET_DIR/bag_$(date +%Y%m%d_%H%M%S)"
fi

if [ -d "$OUTPUT_DIR" ]; then
    echo "ERROR: a bag directory already exists at '$OUTPUT_DIR'."
    echo "Provide a unique name or remove the existing directory."
    exit 1
fi

TOPICS=(
    # --- ground truth: RECORD THESE. Their absence on 2026-07-02 is the single
    # --- biggest gap in that dataset. ZED VIO gives a metric trajectory, which is
    # --- what makes detection range and pose error measurable rather than merely
    # --- self-consistent.
    '/zed/zed_node/odom'
    '/zed/zed_node/pose'

    # --- verified present in the 2026-07-02 bags (alan-1 / alan-2) ---
    '/zed/zed_node/left/camera_info'
    '/zed/zed_node/left/image_rect_color/compressed'   # NOT raw: see lesson 2
    '/zed/zed_node/imu/data'                           # fused; carries orientation
    '/zed/zed_node/imu/data_raw'
    '/zed/zed_node/imu/mag'
    '/zed/zed_node/left_cam_imu_transform'             # ~identity, but cheap

    # --- stereo: right lens. Not recorded on 2026-07-02 (mono only). Uncomment to
    # --- enable a stereo-baseline range cross-check independent of marker size.
    # '/zed/zed_node/right/camera_info'
    # '/zed/zed_node/right/image_rect_color/compressed'
    # '/zed/zed_node/depth/camera_info'

    # --- point cloud: very large. Enable only for a short, deliberate run.
    # '/zed/zed_node/point_cloud/cloud_registered'

    # --- the sections below were in the day's script but produced NOTHING in the
    # --- 2026-07-02 bags, i.e. the drivers were not running. `ros2 bag record`
    # --- silently waits on a topic that never publishes, so a long list gives a
    # --- false sense of coverage. Uncomment only what is confirmed live with
    # --- `ros2 topic hz <topic>` BEFORE the dive.

    # waterlinked_dvl official package
    # '/waterlinked_dvl_driver/dead_reckoning_report'
    # '/waterlinked_dvl_driver/odom'
    # '/waterlinked_dvl_driver/transition_event'
    # '/waterlinked_dvl_driver/velocity_report'

    # sonar
    # '/sonar/image'
    # '/sonar/ping'
    # '/sonar/pressure'
    # '/sonar/status'

    # mavros -- '/mavros/imu/static_pressure' is the useful one (depth reference)
    # '/mavros/imu/data'
    # '/mavros/imu/data_raw'
    # '/mavros/imu/mag'
    # '/mavros/imu/static_pressure'
    # '/mavros/local_position/odom'
    # '/mavros/local_position/pose'
    # '/mavros/global_position/compass_hdg'

    # '/tf'
    # '/tf_static'
)

echo "Recording to: $OUTPUT_DIR"
echo "Topics (${#TOPICS[@]}):"
printf '  %s\n' "${TOPICS[@]}"
echo
echo "Pre-dive check -- confirm the camera is actually publishing:"
echo "  ros2 topic hz /zed/zed_node/left/image_rect_color/compressed"
echo "  ros2 topic hz /zed/zed_node/odom"
echo
echo "Press Ctrl+C to stop (allow up to 60 s to finalize)."

ros2 bag record \
    -o "$OUTPUT_DIR" \
    -s mcap \
    --compression-mode file \
    --compression-format zstd \
    "${TOPICS[@]}"
