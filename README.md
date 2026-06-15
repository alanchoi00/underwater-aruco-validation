# underwater-aruco-validation

Standalone toolkit for validating underwater **ArUco** marker detection and pose
estimation on a BlueROV2, decoupled from the docking controller. It covers the full
capture-to-analysis path: generate a print target, sequence pool capture runs, and
characterise detection vs range / size / angle / turbidity offline.

This repo is **independent** of the docking pipeline — it runs its own `cv2.aruco`
detection and carries its own marker definitions; it does not import any controller or
perception packages.

## Layout

```
recorder/   run_recorder.py    ROS 2 capture sequencer (per-run rosbag + ArUco/depth log)
collage/    aruco_collage.py   true-scale A3 ArUco print-target generator
config/     run_plan.yaml       the planned capture runs
            marker_sizes.yaml   id -> black-square size (m), used by the recorder
docs/       experiment_setup.md full experiment design, variables, acceptance criteria
            onday_runsheet.md   on-day pool checklist + enumerated runs
print/      aruco_collage_A3.pdf generated print target (A3 landscape, 2 pages)
```

## Dependencies

- Python: `numpy`, `opencv-contrib-python`, `reportlab`, `pyyaml` (`pip install -r requirements.txt`)
- ROS 2 (for the recorder only): `rclpy`, `sensor_msgs` — provided by your ROS 2 install, not pip.

## Quick start

Generate the print target (defaults match the dock: Original ArUco dict, dock IDs/sizes):

```bash
python3 collage/aruco_collage.py --out print/aruco_collage_A3.pdf
```

Print at **100% (actual size)** and verify the scale bar reads 100 mm. Marker sizes are
true; spacing on the cluster page is compressed to fit A3 (arrangement-preserving, not
metric dock geometry).

Run the capture sequencer (ENTER to start a run, ENTER to stop):

```bash
python3 recorder/run_recorder.py --ros-args \
  -p run_plan:=config/run_plan.yaml \
  -p marker_sizes:=config/marker_sizes.yaml \
  -p output_dir:=/path/to/dataset \
  -p aruco_dict:=DICT_ARUCO_ORIGINAL \
  -p image_topic:=/zed/zed_node/left/image_rect_color/compressed \
  -p camera_info_topic:=/zed/zed_node/left/camera_info \
  -p depth_topic:=/zed/zed_node/depth/depth_registered
```

It records the ZED/DVL/sonar/MAVROS topic set to MCAP (zstd) and logs an independent
ArUco range/tilt/bearing plus ZED depth at the marker per run.

## Note on measurement

Tape-measured range is the primary ground truth. The logged ArUco range and ZED depth
are **cross-checks**: underwater both are biased by port refraction and degrade with
range, and stereo error grows ~Z². Treat them as sanity checks, not metrology.
