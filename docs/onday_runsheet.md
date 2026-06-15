# On-Day Pool Test Run Sheet — Underwater ArUco Detection

**Date:** `TBD`  ·  **Pool:** `TBD`  ·  **Present:** Alan, Kush
**Camera:** one lens of stereo cam (monocular)  ·  **Dictionary:** `TBD` (final)
**Topics recorded:** `/camera/image_raw`, `/camera/camera_info` → MCAP bag

> Capture only — all detection/turbidity analysis is offline. Turbidity is **not** a capture variable (added later in synthesis). Each static run: hold ≥ **45 s** so there are enough frames for statistics.

---

## A. Pre-dive checklist

- [ ] Acrylic ArUco collage fabricated; **each marker side length caliper-measured** and logged
- [ ] (If running Block E) comparison board printed: 4×4 / 5×5 / 6×6 / AprilTag at one common size
- [ ] Verify a printed sample **decodes in the detector** before getting in the water
- [ ] Camera settings **locked** (no auto): exposure, gain, white balance — record below
- [ ] Tape measure / surveyed range marks laid out (0.5–5 m)
- [ ] Calibration board ready (in-water use)
- [ ] Confirm bag records both image + camera_info; disk space OK
- [ ] Ask pool operator for the day's **turbidity (NTU/FNU) log reading**

### Locked camera settings (fill before first run)

| Setting | Value |
|---|---|
| Resolution | `TBD` |
| Frame rate (fps) | `TBD` |
| Exposure | `TBD` |
| Gain / ISO | `TBD` |
| White balance | `TBD` (fixed) |

---

## B. Run plan

Priority key: **Core** (must), **Comparison** (if board printed), **Optional** (if time).
Fill `Bag file`, `Det?` (quick yes/no on the day), and `Notes` per run.

### Block S — Setup (do first) · Core

| Run | Target | Action | Bag file | Det? | Notes |
|---|---|---|---|---|---|
| S1 | Calibration board | In-water; capture ≥ 20 varied poses | | | intrinsics |
| S2 | Dock | 1.0 m, head-on, fixed exposure, ≥ 30 s | | | baseline turbidity frames (Turbidivision) |

### Block A — Dock, head-on (yaw 0°, pitch 0°) · Core

The workhorse: each frame holds all marker sizes, so this one sweep yields a detection-vs-range curve **per marker size** plus the fused pose.

| Run | Target | Range (m) | Yaw (°) | Pitch (°) | Bag file | Det? | Notes |
|---|---|---|---|---|---|---|---|
| A1 | Dock | 0.5 | 0 | 0 | | | |
| A2 | Dock | 1.0 | 0 | 0 | | | |
| A3 | Dock | 1.5 | 0 | 0 | | | |
| A4 | Dock | 2.0 | 0 | 0 | | | |
| A5 | Dock | 2.5 | 0 | 0 | | | |
| A6 | Dock | 3.0 | 0 | 0 | | | |
| A7 | Dock | 4.0 | 0 | 0 | | | |
| A8 | Dock | 5.0 | 0 | 0 | | | step further if 200 mm still detects |

### Block B — Dock, yaw 20° · Core

| Run | Target | Range (m) | Yaw (°) | Pitch (°) | Bag file | Det? | Notes |
|---|---|---|---|---|---|---|---|
| B1 | Dock | 1.0 | 20 | 0 | | | |
| B2 | Dock | 2.0 | 20 | 0 | | | |
| B3 | Dock | 3.0 | 20 | 0 | | | |

### Block C — Dock, yaw 40° · Core

| Run | Target | Range (m) | Yaw (°) | Pitch (°) | Bag file | Det? | Notes |
|---|---|---|---|---|---|---|---|
| C1 | Dock | 1.0 | 40 | 0 | | | |
| C2 | Dock | 2.0 | 40 | 0 | | | |
| C3 | Dock | 3.0 | 40 | 0 | | | watch for pose flips |

### Block D — Dock, pitch 20° (yaw 0°) · Optional

| Run | Target | Range (m) | Yaw (°) | Pitch (°) | Bag file | Det? | Notes |
|---|---|---|---|---|---|---|---|
| D1 | Dock | 1.5 | 0 | 20 | | | |
| D2 | Dock | 2.5 | 0 | 20 | | | |

### Block E — Grid / family comparison board (yaw 0°) · Comparison

Requires the comparison board (4×4 / 5×5 / 6×6 / AprilTag at one common size). Isolates dictionary/grid effect on range vs angle.

| Run | Target | Range (m) | Yaw (°) | Pitch (°) | Bag file | Det? | Notes |
|---|---|---|---|---|---|---|---|
| E1 | Compare board | 1.0 | 0 | 0 | | | |
| E2 | Compare board | 2.0 | 0 | 0 | | | |
| E3 | Compare board | 3.0 | 0 | 0 | | | |
| E4 | Compare board | 4.0 | 0 | 0 | | | |

### Block F — Dynamic / qualitative · Optional

| Run | Target | Setup | Bag file | Det? | Notes |
|---|---|---|---|---|---|
| F1 | Dock | Continuous slow approach ~5.0 → 0.3 m, head-on (one bag) | | | dynamic; controller-relevant |
| F2 | Dock | 1.5 m, 0°; partially occlude subset of markers | | | fusion recovery test |
| F3 | Dock | Repeat A4 (2.0 m, 0°) under alternate lighting | | | lighting effect |

### Block G — Repeats for variance · Core

| Run | Target | Range (m) | Yaw (°) | Pitch (°) | Bag file | Det? | Notes |
|---|---|---|---|---|---|---|---|
| G1 | Dock | 1.0 | 0 | 0 | | | repeat of A2 |
| G2 | Dock | 1.0 | 0 | 0 | | | repeat of A2 |
| G3 | Dock | 2.0 | 0 | 0 | | | repeat of A4 |
| G4 | Dock | 2.0 | 0 | 0 | | | repeat of A4 |

---

## C. Run count & triage

| Block | Runs | Priority |
|---|---|---|
| S — setup | 2 | Core |
| A — head-on sweep | 8 | Core |
| B — yaw 20° | 3 | Core |
| C — yaw 40° | 3 | Core |
| D — pitch | 2 | Optional |
| E — grid/family | 4 | Comparison |
| F — dynamic/qual | 3 | Optional |
| G — repeats | 4 | Core |
| **Total** | **29** | |

If pool time is short, drop in this order: D → F → E, keep S + A + B + C + G.

---

## D. Live sanity reference (clear water)

Rough clear-water detection expectations — use to spot anomalies on the day, not as results. Turbidity (added in post) will shrink these.

| Marker size | Expected detect to ~ | Watch |
|---|---|---|
| 47.5 mm | ~1.5–2 m | drops first |
| 60 mm | ~2–2.5 m | |
| 100 mm | ~3–4 m | |
| 200 mm | ~5 m+ | last to drop |

If a size drops far earlier than expected → check focus, exposure, or marker print quality before continuing.

---

## E. Wrap-up checklist

- [ ] All bag files named per run ID and backed up off the capture machine
- [ ] Caliper-measured marker sizes recorded
- [ ] Calibration bag (S1) captured and verified
- [ ] Baseline turbidity frames (S2) captured; operator NTU/FNU noted
- [ ] Quick spot-check: at least one A-run decodes markers in playback
- [ ] Locked camera-settings table completed
- [ ] Anomalies / failure observations noted per run

---

## F. Notes log

| Time | Run(s) | Observation / issue |
|---|---|---|
| | | |
