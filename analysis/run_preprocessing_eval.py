#!/usr/bin/env python3
"""Preprocessing evaluation: does CLAHE, or detector adaptive-threshold tuning, extend
ArUco detection into higher water turbidity?

Reuses run_turbidity.py's synthesis pipeline and trial machinery unmodified (imported,
not copied): the same posed frames, the same M.predicted_trials trial SET, and the same
T.synthesise / T.trials_at_tau. Only the OUTCOME differs per mode, never the trial set,
so the fixed 2551-trial denominator this study depends on cannot drift with mode or
multiplier. See sweep_modes for the assertion that enforces this.

Three modes, per posed frame and multiplier, applied to ONE synthesised grey image
(synthesis, not preprocessing, is the expensive step):

    none: detect.make_detector(), no preprocessing. Must reproduce run_turbidity's own
          multiplier-0 rate (the identity check).
    clahe: cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)) applied to the grey
           image, default detector.
    athresh: no image preprocessing; a detector with a tuned adaptiveThreshConstant,
             chosen by best-of-grid search (see select_athresh_constant) at the
             high-tau end of the sweep. That search has NO HOLDOUT: it is scored on
             the same frames and multiplier the athresh arm is then reported over, so
             its winning constant is an optimistic bound, not a validated optimum.

Two caveats this whole module inherits from run_turbidity and cannot remove:

1. Synthesis adds no scattering noise. It scales contrast and shifts the DC level
   (T.synthesise: I_new = B + (I_obs - B) * exp(-dbeta * d)), nothing more. The
   backscatter noise that CLAHE most amplifies in real turbid water is absent, so any
   CLAHE benefit measured here is a best case, an upper bound, not a real-water result.
2. The only real imagery is multiplier 0 (tau up to about 1.5, run_turbidity's own
   real-water span). Every multiplier above that is synthesis. A high-tau CLAHE or
   athresh benefit is therefore unproven for real water; only the multiplier-0 row of
   preprocessing_eval.csv reflects the actual pool.
"""
import json
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis import detect, geometry as g, layout as L
from analysis import metrics as M, turbidity as T, vizstyle
from analysis.run_analysis import load_run
from analysis.run_turbidity import (
    RUN,
    MULTIPLIERS,
    TAU_BINS,
    MIN_TRIALS_PER_BIN,
    board_poses,
    collect_samples,
    instrument_response_table,
    fixed_effects_beta_table,
)

MODES = ["none", "clahe", "athresh"]

CLAHE_CLIP = 2.0
CLAHE_TILE = (8, 8)
# Recorded but not the primary arm (cheap to compute alongside the primary setting).
CLAHE_CLIP_ALT = 4.0
CLAHE_TILE_ALT = (16, 16)

ATHRESH_GRID = [3.0, 5.0, 7.0]
# The high-tau end of the sweep is where a detector knob would have to earn its keep;
# testing it there, rather than at multiplier 0, is what makes the selection meaningful
# at all, at the cost of having no holdout (see the module docstring).
ATHRESH_GRID_MULTIPLIER = max(MULTIPLIERS)

# run_turbidity's own multiplier-0 overall rate (results/turbidity_summary.json,
# overall_rate_by_multiplier["0.0"]). Mode "none" here must reproduce it, within a
# tiny tolerance, or this module's wiring has diverged from the sweep it reuses.
PUBLISHED_BASELINE_RATE = 0.519
IDENTITY_TOLERANCE = 0.01

EXPECTED_N_TRIALS = 2551


def make_clahe(clip=CLAHE_CLIP, tile=CLAHE_TILE):
    return cv2.createCLAHE(clipLimit=clip, tileGridSize=tile)


def make_tuned_detector(adaptive_thresh_constant):
    """The same detector as detect.make_detector, but with adaptiveThreshConstant
    overridden. Used only by the athresh mode: 'none' and 'clahe' both detect with the
    untouched default detector, so any difference between them is attributable to the
    image, not the detector.
    """
    d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, g.DICTIONARY))
    p = cv2.aruco.DetectorParameters()
    p.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    p.adaptiveThreshConstant = adaptive_thresh_constant
    return cv2.aruco.ArucoDetector(d, p)


def preprocess(gray, mode, clahe=None):
    """Apply mode's image preprocessing to one grey frame.

    'none' and 'athresh' return the grey image unchanged: athresh's tuning lives
    entirely in the detector (make_tuned_detector), not the pixels, so this function
    must not alter the image for it.
    """
    if mode == "clahe":
        c = clahe if clahe is not None else make_clahe()
        return c.apply(gray)
    return gray


def _setup(dataset_dir):
    """Everything run_turbidity.main needs before its own sweep: poses, beta_map,
    B_map, obs. Calls run_turbidity's own functions (imported, not reimplemented) so
    this module cannot drift from how the existing sweep derives them.
    """
    det_df, _frames, _imu, ci = load_run(dataset_dir, RUN)
    Km = np.array([[ci["fx"], 0, ci["cx"]], [0, ci["fy"], ci["cy"]], [0, 0, 1]])
    lay = L.load_layout("target/board_layout.yaml")
    poses = board_poses(det_df, lay, Km)
    samples = collect_samples(dataset_dir, det_df, poses, lay)
    _k_table, responses = instrument_response_table(samples)
    fe = fixed_effects_beta_table(samples, responses)
    fe_row = {r.channel: r for r in fe.itertuples()}
    beta_map = {ch: fe_row[ch].beta_corrected for ch in T.CHANNELS}
    veil = T.measure_veiling(samples, beta_map)
    B_map = dict(zip(veil.channel, veil.B))

    obs = {}
    for r in det_df[det_df.marker_id.isin(g.SIZES)].itertuples():
        obs.setdefault((RUN, r.frame_idx), {})[int(r.marker_id)] = np.array(
            [[r.c0x, r.c0y], [r.c1x, r.c1y], [r.c2x, r.c2y], [r.c3x, r.c3y]])
    return ci, Km, lay, poses, beta_map, B_map, obs


def select_athresh_constant(dataset_dir, poses, lay, Km, ci, beta_map, B_map, frame_ids,
                             candidates=ATHRESH_GRID, multiplier=ATHRESH_GRID_MULTIPLIER):
    """Best-of-grid adaptiveThreshConstant at the high-tau end of the sweep.

    Scores each candidate by board detection rate (board hits / possible board hits)
    over every posed frame, synthesised once at `multiplier` and shared across
    candidates. NO HOLDOUT: the frames and multiplier scored here are the same ones
    the athresh arm is later reported over, so the winning constant is an OPTIMISTIC
    BOUND, not a validated optimal detector setting.
    """
    beta_vec = np.array([beta_map["b"], beta_map["g"], beta_map["r"]])
    B_vec = np.array([B_map["b"], B_map["g"], B_map["r"]])
    shape = (ci["height"], ci["width"])
    board_ids = set(g.IDS)

    detectors = {c: make_tuned_detector(c) for c in candidates}
    hits = {c: 0 for c in candidates}
    n_possible = 0
    for fi in frame_ids:
        rv, tv = poses[fi]
        depth = T.plane_depth_map(lay, rv, tv, Km, shape)
        mask = T.board_mask(lay, rv, tv, Km, shape)
        img = cv2.imread(os.path.join(dataset_dir, RUN, "frames", f"{fi:06d}.png"))
        syn = T.synthesise(img, depth, mask, B_vec, multiplier * beta_vec)
        gray = cv2.cvtColor(syn, cv2.COLOR_BGR2GRAY)
        n_possible += len(board_ids)
        for c, det in detectors.items():
            ids = {d["marker_id"] for d in detect.detect_frame(gray, det)}
            hits[c] += len(ids & board_ids)

    scores = {c: (hits[c] / n_possible if n_possible else 0.0) for c in candidates}
    best = max(candidates, key=lambda c: scores[c])
    return best, scores


def sweep_modes(dataset_dir, obs, poses, lay, Km, ci, beta_map, B_map, athresh_constant,
                 summary):
    """Stage 4 equivalent, three arms: per posed frame and multiplier, synthesise ONCE,
    convert to grey ONCE, then run all three modes against that one grey image.

    The trial SET (M.predicted_trials, filtered to the same frame_ids every time) is
    identical across modes and multipliers; only detected_by_frame, the OUTCOME, comes
    from the (possibly preprocessed) frame. This mirrors run_turbidity.sweep's own
    pattern exactly, so the invariant that makes that sweep valid carries over here.
    """
    pred = M.predicted_trials(obs, lay, Km, ci["width"], ci["height"])
    beta_vec = np.array([beta_map["b"], beta_map["g"], beta_map["r"]])
    B_vec = np.array([B_map["b"], B_map["g"], B_map["r"]])
    beta_grey = beta_map["grey"]
    shape = (ci["height"], ci["width"])

    detectors = {
        "none": detect.make_detector(),
        "clahe": detect.make_detector(),
        "athresh": make_tuned_detector(athresh_constant),
    }
    clahe = make_clahe()

    frame_ids = sorted(set(int(f) for f in pred.frame_idx.unique()) & set(poses))
    pred = pred[pred.frame_idx.isin(frame_ids)]

    cache = {}
    rng_by_frame = {}
    for fi in frame_ids:
        rv, tv = poses[fi]
        cache[fi] = (T.plane_depth_map(lay, rv, tv, Km, shape),
                     T.board_mask(lay, rv, tv, Km, shape))
        rng_by_frame[fi] = T.marker_ranges(lay, rv, tv)

    trial_rows = []
    identity_rate = None
    for m in MULTIPLIERS:
        per_mode_detected = {mode: {} for mode in MODES}
        per_mode_mis_id = {mode: {} for mode in MODES}
        for fi in frame_ids:
            depth, mask = cache[fi]
            img = cv2.imread(os.path.join(dataset_dir, RUN, "frames", f"{fi:06d}.png"))
            syn = T.synthesise(img, depth, mask, B_vec, m * beta_vec)
            gray = cv2.cvtColor(syn, cv2.COLOR_BGR2GRAY)
            for mode in MODES:
                proc = preprocess(gray, mode, clahe=clahe if mode == "clahe" else None)
                dets = detect.detect_frame(proc, detectors[mode])
                per_mode_detected[mode][fi] = {d["marker_id"] for d in dets
                                               if d["marker_id"] in g.SIZES}
                per_mode_mis_id[mode][fi] = sum(1 for d in dets
                                                if d["marker_id"] not in g.SIZES)

        for mode in MODES:
            trials = T.trials_at_tau(pred, per_mode_detected[mode])
            n_trials = int(len(trials))
            if n_trials != EXPECTED_N_TRIALS:
                raise AssertionError(
                    f"trial count drifted: mode {mode!r} multiplier {m} produced "
                    f"{n_trials} trials, expected {EXPECTED_N_TRIALS}. The trial SET "
                    "must come only from M.predicted_trials, never from detection.")

            t = trials.copy()
            t["range_m"] = [rng_by_frame[int(fi)][int(mid)]
                            for fi, mid in zip(t["frame_idx"], t["marker_id"])]
            t["tau_total"] = (1.0 + m) * beta_grey * t["range_m"]
            t["multiplier"] = m
            t["mode"] = mode
            trial_rows.append(t[["frame_idx", "marker_id", "size_mm", "pred_px",
                                 "multiplier", "mode", "range_m", "tau_total",
                                 "detected"]])

            rate = float(trials.detected.mean()) if n_trials else 0.0
            tau_median = float(t.tau_total.median()) if n_trials else float("nan")
            mis_id_total = int(sum(per_mode_mis_id[mode].values()))
            n_frames = len(frame_ids)

            summary.setdefault("rate_by_mode_multiplier", {}).setdefault(
                mode, {})[str(m)] = {
                "rate": round(rate, 4),
                "n_trials": n_trials,
                "tau_median": round(tau_median, 3),
                "mis_id_count": mis_id_total,
                "mis_id_rate_per_frame": round(mis_id_total / n_frames, 4)
                                        if n_frames else 0.0,
            }
            print(f"  mode {mode:>7} multiplier {m:>4}: rate {rate:.3f} "
                  f"over {n_trials} trials, mis_id {mis_id_total}")

            if m == 0.0 and mode == "none":
                identity_rate = rate

    summary["identity_check_none_rate_at_multiplier_0"] = round(identity_rate, 4)
    summary["identity_check_reference_rate"] = PUBLISHED_BASELINE_RATE
    matches = bool(abs(identity_rate - PUBLISHED_BASELINE_RATE) < IDENTITY_TOLERANCE)
    summary["identity_check_matches_published"] = matches
    assert matches, (
        f"mode 'none' at multiplier 0 gave rate {identity_rate:.4f}; expected to "
        f"match run_turbidity's own multiplier-0 rate {PUBLISHED_BASELINE_RATE:.3f} "
        f"within {IDENTITY_TOLERANCE}. This means the wiring here differs from the "
        "existing sweep and must be fixed, not the assertion.")

    return pd.concat(trial_rows, ignore_index=True)


def write_csv(summary):
    rows = []
    for mode, by_m in summary["rate_by_mode_multiplier"].items():
        for m_str, vals in by_m.items():
            rows.append({"mode": mode, "multiplier": float(m_str), **vals})
    df = pd.DataFrame(rows).sort_values(["mode", "multiplier"])
    df.to_csv("results/preprocessing_eval.csv", index=False)
    return df


MODE_COLOURS = {"none": "#52514e", "clahe": "#2a78d6", "athresh": "#c0392b"}
REAL_WATER_TAU_MAX = 1.5     # run_turbidity's own multiplier-0 real-water span
TAU_CLIFF = 2.0              # run_turbidity's collapse boundary


def make_figure(trials_df, csv_df):
    """rate vs tau_total, one line per mode, with the real-water boundary and the
    tau 2 cliff marked, plus a mis-ID panel underneath (a mode that lifts recall by
    adding false positives is not a win, so mis-IDs are shown alongside rate, not
    buried in the summary).
    """
    t = trials_df.copy()
    t["tau_bin"] = pd.cut(t.tau_total, TAU_BINS, right=False)
    binned = t.groupby(["mode", "tau_bin"], observed=True).agg(
        n=("detected", "size"), rate=("detected", "mean")).reset_index()
    binned = binned[binned.n >= MIN_TRIALS_PER_BIN].copy()
    binned["tau_mid"] = binned.tau_bin.apply(lambda iv: (iv.left + iv.right) / 2.0)

    fig, (ax, ax2) = plt.subplots(
        2, 1, figsize=(vizstyle.COL_W, vizstyle.COL_W * 1.25),
        gridspec_kw={"height_ratios": [3, 1]}, sharex=True)

    for mode in MODES:
        sub = binned[binned["mode"] == mode].sort_values("tau_mid")
        if len(sub) == 0:
            continue
        ax.plot(sub.tau_mid, sub.rate, "o-", color=MODE_COLOURS[mode], label=mode,
               lw=1.8, ms=4)

    ax.axvline(REAL_WATER_TAU_MAX, color=vizstyle.TEXT_SECONDARY, ls="--", lw=1.2)
    ax.axvspan(TAU_CLIFF, TAU_BINS[-1], color="#c0392b", alpha=0.06)
    ax.axvline(TAU_CLIFF, color="#c0392b", ls=":", lw=1.2)

    # Both annotations point back at their vline from the clear region past tau 3,
    # where every mode's rate has already collapsed to zero (see the collapsed flat
    # tail in the plotted curves), so neither the arrow nor the text lands on data.
    ax.annotate("real water ends here\n(tau up to about 1.5);\nright is synthesis",
               xy=(REAL_WATER_TAU_MAX, 0.55), xytext=(6.3, 0.55),
               textcoords="data", fontsize=6, color=vizstyle.TEXT_SECONDARY,
               va="center", arrowprops=dict(arrowstyle="-", lw=0.8,
                                            color=vizstyle.TEXT_SECONDARY))
    ax.annotate("tau 2 cliff", xy=(TAU_CLIFF, 0.35), xytext=(6.3, 0.35),
               textcoords="data", fontsize=6.5, color="#c0392b", va="center",
               arrowprops=dict(arrowstyle="-", lw=0.8, color="#c0392b"))

    ax.set_ylim(-0.02, 1.02)
    ax.set_ylabel("detection rate")
    ax.set_title("Detection rate vs tau_total, by preprocessing mode", fontsize=9)
    ax.legend(fontsize=7, loc="upper right")

    csv_df = csv_df.copy()
    csv_df["tau_approx"] = csv_df["tau_median"]
    for mode in MODES:
        sub = csv_df[csv_df["mode"] == mode].sort_values("tau_approx")
        ax2.plot(sub.tau_approx, sub.mis_id_rate_per_frame, "s--",
                color=MODE_COLOURS[mode], ms=3, lw=1.2)
    ax2.set_xlabel("total optical depth, tau (grey)")
    ax2.set_ylabel("mis-IDs\nper frame", fontsize=7)
    fig.tight_layout()
    vizstyle.save(fig, "preprocessing_eval")


def main(dataset_dir="dataset"):
    vizstyle.apply()
    os.makedirs("results", exist_ok=True)
    summary = {"opencv": cv2.__version__, "run": RUN,
              "clahe_clip_limit": CLAHE_CLIP, "clahe_tile_grid_size": list(CLAHE_TILE),
              "clahe_alt_recorded_only": {"clip_limit": CLAHE_CLIP_ALT,
                                          "tile_grid_size": list(CLAHE_TILE_ALT)},
              "athresh_grid": ATHRESH_GRID,
              "athresh_grid_multiplier": ATHRESH_GRID_MULTIPLIER}

    ci, Km, lay, poses, beta_map, B_map, obs = _setup(dataset_dir)

    pred_all = M.predicted_trials(obs, lay, Km, ci["width"], ci["height"])
    frame_ids = sorted(set(int(f) for f in pred_all.frame_idx.unique()) & set(poses))

    best_constant, grid_scores = select_athresh_constant(
        dataset_dir, poses, lay, Km, ci, beta_map, B_map, frame_ids)
    summary["athresh_grid_scores"] = {str(c): round(s, 4) for c, s in grid_scores.items()}
    summary["athresh_selected_constant"] = best_constant
    print(f"athresh grid search at multiplier {ATHRESH_GRID_MULTIPLIER}: "
         f"{summary['athresh_grid_scores']}, selected {best_constant}")

    trials_df = sweep_modes(dataset_dir, obs, poses, lay, Km, ci, beta_map, B_map,
                            best_constant, summary)
    trials_df.to_csv("results/preprocessing_eval_trials.csv", index=False)

    csv_df = write_csv(summary)
    make_figure(trials_df, csv_df)

    clahe_rows = summary["rate_by_mode_multiplier"]["clahe"]
    none_rows = summary["rate_by_mode_multiplier"]["none"]
    athresh_rows = summary["rate_by_mode_multiplier"]["athresh"]
    m_hi = str(max(MULTIPLIERS))
    summary["headline_high_tau_multiplier"] = m_hi
    summary["headline_rate_none_high_tau"] = none_rows[m_hi]["rate"]
    summary["headline_rate_clahe_high_tau"] = clahe_rows[m_hi]["rate"]
    summary["headline_rate_athresh_high_tau"] = athresh_rows[m_hi]["rate"]

    summary["limitations"] = [
        "Synthesis adds no scattering noise: T.synthesise only scales contrast and "
        "shifts the DC level (I_new = B + (I_obs - B) * exp(-dbeta * d)). The "
        "backscatter noise that CLAHE most amplifies in real turbid water is absent "
        "here, so a CLAHE benefit measured on synthesised frames is a BEST CASE, an "
        "upper bound, not a real-water result.",
        "The only real imagery is multiplier 0 (tau up to about 1.5, run_turbidity's "
        "own real-water span); every multiplier above that is synthesis. A high-tau "
        "CLAHE or athresh benefit is therefore UNPROVEN for real water; only the "
        "multiplier-0 row of preprocessing_eval.csv reflects the actual pool, where "
        "CLAHE was measured to be mildly negative in an earlier probe.",
        "The athresh arm's adaptiveThreshConstant is chosen by best-of-grid search on "
        "the SAME evaluation frames and multiplier it is then scored over (see "
        "select_athresh_constant); there is no holdout split. Its reported rate is "
        "therefore an OPTIMISTIC BOUND on what a tuned detector achieves, not a "
        "validated optimal setting.",
        "The synthesised high-tau frames are cosmetically wrong: veiling light B fits "
        "brighter than the white sheet is ever observed (run_turbidity's own "
        "veiling_B limitation). B sets only the DC level, which CLAHE is largely "
        "invariant to (CLAHE equalises local CONTRAST, not brightness), so this "
        "mostly does not bias the local-contrast comparison made here. Stated anyway "
        "because it is a real defect in the synthesised imagery, not because it is "
        "expected to change the direction of the result.",
        "One pool, one session, 2551 trials from a single dataset run: the tau axis "
        "generalises across turbidity levels within this data, but this water and "
        "this camera do not generalise beyond it (same caveat as run_turbidity).",
    ]

    with open("results/preprocessing_eval_summary.json", "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else "dataset")
