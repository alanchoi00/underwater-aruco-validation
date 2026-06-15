#!/usr/bin/env python3
"""
run_recorder.py — operator-driven capture sequencer for underwater ArUco runs.

Independent of the blue-docking repo: it runs its OWN cv2.aruco detection (its own
dictionary + an id->size map from marker_sizes.yaml) purely to log a range/angle
cross-check while recording. It does not import or depend on the perception stack.

Per run (from run_plan.yaml): [ENTER]=start recording, [ENTER]=stop. Recording uses
Kush's full ZED/DVL/sonar/MAVROS topic set with MCAP + zstd. While recording it also
samples, from the ZED left image, an independent ArUco pose (range, board tilt,
bearing) and the ZED depth at the marker centroid, and appends them to run_log.csv.

NOTE: ArUco range and ZED depth are CROSS-CHECKS, not ground truth. Underwater both
are biased by port refraction and degrade with range; the tape measure is primary GT.

Run:
    python3 run_recorder.py --ros-args \
        -p run_plan:=run_plan.yaml \
        -p marker_sizes:=marker_sizes.yaml \
        -p output_dir:=/home/bluerov/zimeng/dataset \
        -p aruco_dict:=DICT_5X5_50 \
        -p image_topic:=/zed/zed_node/left/image_rect_color/compressed \
        -p camera_info_topic:=/zed/zed_node/left/camera_info \
        -p depth_topic:=/zed/zed_node/depth/depth_registered   # "" to disable
"""

import csv
import os
import signal
import subprocess
import threading
import time
from datetime import datetime

import cv2
import numpy as np
import rclpy
import yaml
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, CompressedImage, Image

# Kush's recording set (ZED + DVL + sonar + MAVROS + tf). Recorded verbatim so bags
# match the rest of the pipeline; edit via the `record_topics` param if needed.
DEFAULT_RECORD_TOPICS = [
    "/zed/zed_node/left_cam_imu_transform", "/zed/zed_node/imu/data",
    "/zed/zed_node/imu/data_raw", "/zed/zed_node/imu/mag",
    "/zed/zed_node/left/camera_info", "/zed/zed_node/left/image_rect_color/compressed",
    "/zed/zed_node/right/camera_info", "/zed/zed_node/right/image_rect_color/compressed",
    "/zed/zed_node/depth/camera_info", "/zed/zed_node/depth/depth_registered",
    "/zed/zed_node/point_cloud/cloud_registered", "/zed/zed_node/pose", "/zed/zed_node/odom",
    "/dvl/command/response", "/dvl/config/status", "/dvl/data", "/dvl/position",
    "/sonar/image", "/sonar/ping", "/sonar/pressure", "/sonar/status",
    "/mavros/global_position/compass_hdg", "/mavros/global_position/global",
    "/mavros/global_position/gp_lp_offset", "/mavros/global_position/gp_origin",
    "/mavros/global_position/local", "/mavros/global_position/raw/fix",
    "/mavros/global_position/raw/gps_vel", "/mavros/global_position/raw/satellites",
    "/mavros/global_position/rel_alt", "/mavros/home_position/home",
    "/mavros/imu/data", "/mavros/imu/data_raw", "/mavros/imu/diff_pressure",
    "/mavros/imu/mag", "/mavros/imu/static_pressure", "/mavros/imu/temperature_baro",
    "/mavros/imu/temperature_imu", "/mavros/local_position/accel",
    "/mavros/local_position/odom", "/mavros/local_position/pose",
    "/mavros/local_position/pose_cov", "/mavros/local_position/velocity_body",
    "/mavros/local_position/velocity_body_cov", "/mavros/local_position/velocity_local",
    "/mavros/param/event", "/tf", "/tf_static",
]


class RunRecorder(Node):
    def __init__(self):
        super().__init__("run_recorder")
        gp = lambda n, d: self.declare_parameter(n, d).value  # noqa: E731
        self.run_plan_path = gp("run_plan", "run_plan.yaml")
        self.marker_sizes_path = gp("marker_sizes", "marker_sizes.yaml")
        self.output_dir = gp("output_dir", "aruco_runs")
        self.aruco_dict_name = gp("aruco_dict", "DICT_ARUCO_ORIGINAL")
        self.image_topic = gp("image_topic", "/zed/zed_node/left/image_rect_color/compressed")
        self.info_topic = gp("camera_info_topic", "/zed/zed_node/left/camera_info")
        self.depth_topic = gp("depth_topic", "/zed/zed_node/depth/depth_registered")
        self.record_topics = list(gp("record_topics", DEFAULT_RECORD_TOPICS))

        os.makedirs(self.output_dir, exist_ok=True)
        self.csv_path = os.path.join(self.output_dir, "run_log.csv")

        # id -> marker side length (m), independent of blue-docking (from the collage)
        with open(self.marker_sizes_path) as f:
            self.marker_size_m = {int(k): float(v)
                                  for k, v in yaml.safe_load(f)["marker_size_m"].items()}

        with open(self.run_plan_path) as f:
            self.runs = yaml.safe_load(f)["runs"]

        # cv2.aruco setup
        self._dict = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, self.aruco_dict_name))
        try:
            self._detector = cv2.aruco.ArucoDetector(self._dict, cv2.aruco.DetectorParameters())
        except AttributeError:
            self._detector = None  # use legacy detectMarkers

        self._K = self._D = None
        self._gray = None
        self._depth_m = None
        self._lock = threading.Lock()

        self.create_subscription(CameraInfo, self.info_topic, self._info_cb, 10)
        self.create_subscription(CompressedImage, self.image_topic, self._img_cb, 10)
        if self.depth_topic:
            self.create_subscription(Image, self.depth_topic, self._depth_cb, 10)

        self._init_csv()
        self.get_logger().info(
            f"{len(self.runs)} runs | dict={self.aruco_dict_name} | "
            f"{len(self.marker_size_m)} marker sizes | depth={'on' if self.depth_topic else 'off'}")

    # ---------- callbacks ----------
    def _info_cb(self, msg: CameraInfo):
        with self._lock:
            self._K = np.array(msg.k, dtype=np.float64).reshape(3, 3)
            self._D = np.array(msg.d, dtype=np.float64) if msg.d else np.zeros(5)

    def _img_cb(self, msg: CompressedImage):
        arr = np.frombuffer(msg.data, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            return
        with self._lock:
            self._gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    def _depth_cb(self, msg: Image):
        try:
            if msg.encoding in ("32FC1", "32FC"):
                a = np.frombuffer(msg.data, np.float32).reshape(msg.height, msg.step // 4)[:, :msg.width]
                d = a.astype(np.float32)
            elif msg.encoding in ("16UC1", "mono16"):
                a = np.frombuffer(msg.data, np.uint16).reshape(msg.height, msg.step // 2)[:, :msg.width]
                d = a.astype(np.float32) / 1000.0
            else:
                return
        except Exception:  # noqa: BLE001
            return
        with self._lock:
            self._depth_m = d

    # ---------- aruco measurement ----------
    def measure(self):
        """Return dict with n_markers, ref_id, range_m, tilt_deg, bearing_deg, depth_m."""
        with self._lock:
            gray = None if self._gray is None else self._gray.copy()
            K, D = self._K, self._D
            depth = None if self._depth_m is None else self._depth_m.copy()
        out = {"n": 0, "ref_id": "", "range_m": "", "tilt_deg": "",
               "bearing_deg": "", "depth_m": ""}
        if gray is None or K is None:
            return out

        if self._detector is not None:
            corners, ids, _ = self._detector.detectMarkers(gray)
        else:
            corners, ids, _ = cv2.aruco.detectMarkers(gray, self._dict)
        if ids is None or len(ids) == 0:
            return out
        ids = ids.flatten()
        out["n"] = int(len(ids))

        # reference marker = largest known physical size in view (most reliable)
        best = None
        for c, mid in zip(corners, ids):
            s = self.marker_size_m.get(int(mid))
            if s is None:
                continue
            if best is None or s > best[0]:
                best = (s, int(mid), c)
        if best is None:
            return out
        s, mid, c = best
        objp = np.array([[-s / 2, s / 2, 0], [s / 2, s / 2, 0],
                         [s / 2, -s / 2, 0], [-s / 2, -s / 2, 0]], dtype=np.float64)
        img_pts = c.reshape(4, 2).astype(np.float64)
        ok, rvec, tvec = cv2.solvePnP(objp, img_pts, K, D, flags=cv2.SOLVEPNP_IPPE_SQUARE)
        if not ok:
            return out
        tvec = tvec.flatten()
        R, _ = cv2.Rodrigues(rvec)
        rng = float(np.linalg.norm(tvec))
        los = tvec / rng
        normal = R[:, 2]
        tilt = np.degrees(np.arccos(np.clip(abs(float(np.dot(normal, los))), 0, 1)))
        bearing = np.degrees(np.arctan2(float(np.hypot(tvec[0], tvec[1])), float(tvec[2])))

        out.update(ref_id=mid, range_m=round(rng, 3),
                   tilt_deg=round(float(tilt), 1), bearing_deg=round(float(bearing), 1))

        # ZED depth at the marker centroid
        if depth is not None:
            cx, cy = img_pts.mean(axis=0)
            px, py = int(round(cx)), int(round(cy))
            if 0 <= py < depth.shape[0] and 0 <= px < depth.shape[1]:
                w = depth[max(0, py - 3):py + 4, max(0, px - 3):px + 4]
                v = w[np.isfinite(w) & (w > 0.05) & (w < 30.0)]
                if v.size:
                    out["depth_m"] = round(float(np.median(v)), 3)
        return out

    # ---------- csv ----------
    def _init_csv(self):
        new = not os.path.exists(self.csv_path)
        self._csv = open(self.csv_path, "a", newline="")
        self._w = csv.writer(self._csv)
        if new:
            self._w.writerow([
                "timestamp", "run_id", "target", "planned_range_m", "tape_range_m",
                "n_markers", "ref_id", "aruco_range_m", "aruco_tilt_deg",
                "aruco_bearing_deg", "zed_depth_marker_m", "duration_s", "bag_path", "notes"])
            self._csv.flush()

    def log_row(self, row):
        self._w.writerow(row)
        self._csv.flush()

    # ---------- bag ----------
    def start_bag(self, run_id):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        bag = os.path.join(self.output_dir, f"{run_id}_{ts}")
        cmd = ["ros2", "bag", "record", "-o", bag, "-s", "mcap",
               "--compression-mode", "file", "--compression-format", "zstd"] + self.record_topics
        proc = subprocess.Popen(cmd, start_new_session=True)
        return proc, bag

    @staticmethod
    def stop_bag(proc):
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGINT)
            proc.wait(timeout=60)
        except Exception:  # noqa: BLE001
            proc.kill()


def _in(p=""):
    try:
        return input(p).strip()
    except EOFError:
        return "q"


def console(node: RunRecorder):
    print("\n=== Underwater ArUco capture sequencer (ZED) ===")
    print("Per run: position rig, [ENTER]=start, [ENTER]=stop. (s=skip r=redo q=quit)\n")
    i = 0
    while rclpy.ok() and i < len(node.runs):
        r = node.runs[i]
        rng = "operator-set" if r.get("range_m") is None else f'{r["range_m"]} m'
        print("-" * 60)
        print(f'RUN {r["id"]} [{r.get("priority","")}] {r["target"]}  range={rng} '
              f'yaw={r["yaw_deg"]} pitch={r["pitch_deg"]} min_dwell={r["dwell_s"]}s')
        if r.get("notes"):
            print(f'  notes: {r["notes"]}')
        cmd = _in("[ENTER]=record  s/r/q > ").lower()
        if cmd == "q":
            break
        if cmd == "s":
            i += 1
            continue
        if cmd == "r":
            i = max(0, i - 1)
            continue

        m0 = node.measure()
        proc, bag = node.start_bag(r["id"])
        t0 = time.time()
        print(f'  ● REC {r["id"]}  aruco@start: n={m0["n"]} range={m0["range_m"]} '
              f'tilt={m0["tilt_deg"]} depth={m0["depth_m"]}  -> {bag}')
        _in("    [ENTER] to STOP > ")
        m1 = node.measure()
        node.stop_bag(proc)
        dur = time.time() - t0
        print(f'  ■ stop {dur:.1f}s  aruco@stop: n={m1["n"]} range={m1["range_m"]} '
              f'tilt={m1["tilt_deg"]} bearing={m1["bearing_deg"]} zed_depth={m1["depth_m"]}')

        default_tape = "" if r.get("range_m") is None else str(r["range_m"])
        tape = _in(f"    tape range m [{default_tape}] > ") or default_tape
        notes = _in("    notes > ")
        node.log_row([
            datetime.now().isoformat(timespec="seconds"), r["id"], r["target"],
            r.get("range_m"), tape, m1["n"], m1["ref_id"], m1["range_m"],
            m1["tilt_deg"], m1["bearing_deg"], m1["depth_m"], f"{dur:.1f}", bag,
            notes or r.get("notes", "")])
        print(f'  ✓ logged {r["id"]}')
        i += 1
    print("\nDone ->", node.csv_path)


def main():
    rclpy.init()
    node = RunRecorder()
    threading.Thread(target=rclpy.spin, args=(node,), daemon=True).start()
    try:
        console(node)
    finally:
        node._csv.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
