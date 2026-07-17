#!/usr/bin/env python3
"""Turbidity: measure the pool's attenuation, then synthesise more of it.

Separate from run_analysis.py, which answers a different question and is long enough.
Stage 1 measures beta from the board's own contrast decay. Stage 2 asks whether forward
scatter (blur) matters at our optical depths. Stages 3 to 5 synthesise added turbidity and
extract the marker sizing rule.
"""
import json
import os
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis import detect, geometry as g, layout as L
from analysis import metrics as M, turbidity as T, vizstyle
from analysis.run_analysis import code_version, load_run

RUN = "test1"


def board_poses(det, lay, Km):
    """Board pose per frame from ALL detected board markers in the original imagery."""
    poses = {}
    for fi, sub in det[det.marker_id.isin(g.SIZES)].groupby("frame_idx"):
        dets = {int(r.marker_id): np.array([[r.c0x, r.c0y], [r.c1x, r.c1y],
                                            [r.c2x, r.c2y], [r.c3x, r.c3y]])
                for r in sub.itertuples()}
        try:
            poses[int(fi)] = L.board_pnp(lay, dets, Km)
        except ValueError:
            continue
    return poses


def collect_samples(dataset_dir, det, poses, lay):
    """Stage 1 and 2: photometric rings and edge widths, per detected marker."""
    rows = []
    for fi, sub in det[det.marker_id.isin(g.SIZES)].groupby("frame_idx"):
        if int(fi) not in poses:
            continue
        rv, tv = poses[int(fi)]
        rng = T.marker_ranges(lay, rv, tv)
        path = os.path.join(dataset_dir, RUN, "frames", f"{int(fi):06d}.png")
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        for r in sub.itertuples():
            if r.apparent_px < T.MIN_PX_FOR_PHOTOMETRY:
                continue
            q = np.array([[r.c0x, r.c0y], [r.c1x, r.c1y], [r.c2x, r.c2y], [r.c3x, r.c3y]])
            black, white = T.patch_means(T.warp_marker(img, q))
            row = {"frame_idx": int(fi), "marker_id": int(r.marker_id),
                   "size_mm": round(g.SIZES[int(r.marker_id)] * 1000, 1),
                   "range_m": rng[int(r.marker_id)], "apparent_px": float(r.apparent_px),
                   "edge_px": T.edge_width(gray, q)}
            for i, ch in enumerate(T.CHANNELS):
                row[f"black_{ch}"] = float(black[i])
                row[f"white_{ch}"] = float(white[i])
            rows.append(row)
    return pd.DataFrame(rows)


def figure_beta(samples, betas):
    fig, ax = plt.subplots(figsize=(vizstyle.COL_W, vizstyle.COL_W * 0.75))
    colours = {"b": "#2a78d6", "g": "#2e8b57", "r": "#c0392b", "grey": "#52514e"}
    for ch in T.CHANNELS:
        c = samples[f"white_{ch}"] - samples[f"black_{ch}"]
        ok = c > 0
        ax.scatter(samples.range_m[ok], c[ok], s=4, alpha=0.25, color=colours[ch],
                   linewidths=0)
        row = betas[betas.channel == ch].iloc[0]
        d = np.linspace(samples.range_m.min(), samples.range_m.max(), 50)
        ax.plot(d, row.c0 * np.exp(-row.beta * d), color=colours[ch], lw=2,
                label=f"{ch}: beta={row.beta:.3f}/m, r2={row.r2:.2f}")
    ax.set_yscale("log")
    ax.set_xlabel("range (m, from board pose)")
    ax.set_ylabel("black/white contrast (DN)")
    ax.set_title("Contrast decay: the water measuring itself")
    ax.legend(fontsize=6, loc="lower left")
    vizstyle.save(fig, "beta_contrast_decay")


EDGE_PX_BINS = [40, 56, 72, 100, 140, 200]
MIN_EDGE_SAMPLES = 10


def figure_edge_width(samples, beta_grey, summary):
    """Stage 2, compared ONLY at matched apparent size.

    A global fit of edge width against tau, pooled across marker sizes, would be
    CONFOUNDED and must not be shipped. Apparent size and tau are both set by range, and
    edge width is biased by apparent size because edge_width bounds its inward reach to
    0.45 of the border cell, which truncates the rise on small markers. Measured on the
    real frames after that fix, edge width RISES with apparent size (2.10 px at 30-40 px
    to 4.04 px at 100-300 px). A pooled tau fit would report that truncation as physics.

    Within one apparent-size bin the bound is identical for every marker, so the
    truncation is identical too, while the 149.4 mm marker sits at 149.4/35.6 = 4.2x the
    optical depth of the 35.6 mm one. Any difference INSIDE a bin is therefore tau, not
    an artifact. That is the only comparison this figure makes.
    """
    s = samples.dropna(subset=["edge_px"]).copy()
    s["tau"] = beta_grey * s.range_m
    s["px_bin"] = pd.cut(s.apparent_px, EDGE_PX_BINS)

    rows = []
    for pb, sub in s.groupby("px_bin", observed=True):
        if len(sub) < MIN_EDGE_SAMPLES or sub.size_mm.nunique() < 2:
            continue                      # no lever without two sizes in the same bin
        k, _ = np.polyfit(sub.tau, sub.edge_px, 1)
        rows.append({"px_bin": str(pb), "n": int(len(sub)),
                     "n_sizes": int(sub.size_mm.nunique()),
                     "tau_lo": round(float(sub.tau.min()), 3),
                     "tau_hi": round(float(sub.tau.max()), 3),
                     "slope_px_per_tau": round(float(k), 3),
                     "median_edge_px": round(float(sub.edge_px.median()), 3)})
    within = pd.DataFrame(rows)
    within.to_csv("results/edge_width_within_px_bin.csv", index=False)
    summary["edge_width_within_bin"] = rows
    summary["edge_width_rejected_frac"] = round(
        float(samples.edge_px.isna().mean()), 3)

    fig, ax = plt.subplots(figsize=(vizstyle.WIDE_W, vizstyle.COL_W * 0.75))
    for size, sub in s.groupby("size_mm"):
        ax.scatter(sub.apparent_px, sub.edge_px, s=6, alpha=0.4, linewidths=0,
                   color=vizstyle.SIZE_COLORS.get(size, "#52514e"),
                   marker=vizstyle.SIZE_MARKERS.get(size, "o"), label=f"{size:.1f} mm")
    for lo, hi in zip(EDGE_PX_BINS[:-1], EDGE_PX_BINS[1:]):
        ax.axvline(hi, color=vizstyle.GRID, lw=0.8, zorder=0)
    ax.set_xscale("log")
    vizstyle.log_px_ticks(ax)
    ax.set_xlabel("apparent marker size (px)")
    ax.set_ylabel("10-90 edge rise (px)")
    ax.set_title("Edge sharpness, compared within apparent-size bins")
    ax.legend(fontsize=6)
    vizstyle.save(fig, "edge_width_vs_px")


def instrument_response_table(samples):
    """Stage 1b part 1: k(apparent_px) per channel, stacked into one table."""
    rows = []
    responses = {}
    for ch in T.CHANNELS:
        k_px, k_val = T.measure_instrument_response(samples, ch)
        responses[ch] = (k_px, k_val)
        for px, k in zip(k_px, k_val):
            rows.append({"channel": ch, "apparent_px": round(float(px), 2),
                         "k": round(float(k), 4)})
    return pd.DataFrame(rows), responses


def fixed_effects_beta_table(samples, responses):
    """Stage 1b part 2: one shared slope beta, one intercept per marker id, fit both
    without and with the instrument-response correction.

    "raw" here is the fixed-effects fit itself (k=None): per-marker intercepts already
    remove the inter-marker C0 differences (problem 2), so it is the fair baseline
    against which the k(px) correction's effect on beta (problem 1, the
    range-correlated instrument bias) can be judged on its own. A single pooled
    intercept across all 9 markers conflates both problems at once and is not a clean
    comparison; it stays available separately in turbidity_beta.csv for continuity
    with the earlier stage.
    """
    rows = []
    for ch in T.CHANNELS:
        sub = samples.dropna(subset=[f"white_{ch}", f"black_{ch}", "range_m",
                                     "apparent_px", "marker_id"])
        beta_raw, r2_raw, _ = T.fit_beta_fixed_effects(sub, ch, k=None)
        beta_corr, r2_corr, _ = T.fit_beta_fixed_effects(sub, ch, k=responses[ch])
        rows.append({"channel": ch, "beta_raw": beta_raw, "r2_raw": r2_raw,
                    "beta_corrected": beta_corr, "r2_corrected": r2_corr,
                    "n": int(len(sub))})
    return pd.DataFrame(rows)


def per_marker_beta_table(samples, responses):
    """Per marker id, per channel: raw beta (that marker's own contrast decay) versus
    corrected beta (contrast divided by the instrument response), each fit
    independently with fit_beta. Diagnostic detail behind the shared fixed-effects
    beta, not a replacement for it: each row's beta comes from that single marker's
    own range span, so it is noisier and its range span is narrower.
    """
    rows = []
    for ch in T.CHANNELS:
        k_px, k_val = responses[ch]
        for mid, sub in samples.groupby("marker_id"):
            sub = sub.dropna(subset=[f"white_{ch}", f"black_{ch}", "range_m",
                                     "apparent_px"])
            if len(sub) < 2:
                continue
            d = sub["range_m"].to_numpy()
            contrast_raw = (sub[f"white_{ch}"] - sub[f"black_{ch}"]).to_numpy()
            resp = T.apply_instrument_response(sub["apparent_px"].to_numpy(),
                                               k_px, k_val)
            contrast_corrected = contrast_raw / resp
            beta_raw, _, r2_raw = T.fit_beta(d, contrast_raw)
            beta_corr, _, r2_corr = T.fit_beta(d, contrast_corrected)
            rows.append({"channel": ch, "marker_id": int(mid),
                        "size_mm": float(sub["size_mm"].iloc[0]),
                        "n": int(len(sub)),
                        "range_lo_m": round(float(d.min()), 3),
                        "range_hi_m": round(float(d.max()), 3),
                        "beta_raw": beta_raw, "r2_raw": r2_raw,
                        "beta_corrected": beta_corr, "r2_corrected": r2_corr})
    return pd.DataFrame(rows)


MULTIPLIERS = [0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0]
MIN_TRIALS_PER_BIN = 10        # same convention as run_analysis.py
PX_BINS = [8, 12, 16, 21, 28, 38, 52, 72, 100, 140, 200, 300]


def sweep(dataset_dir, obs, poses, lay, Km, ci, beta_map, B_map, summary):
    """Stage 4: per frame, per added turbidity, synthesise and re-run the real detector.

    The trial SET is fixed by the original imagery (predicted_trials). Only the outcome
    comes from the degraded frame. Otherwise the denominator would shrink exactly where
    detection fails and the rate could never fall.
    """
    pred = M.predicted_trials(obs, lay, Km, ci["width"], ci["height"])

    beta_vec = np.array([beta_map["b"], beta_map["g"], beta_map["r"]])
    B_vec = np.array([B_map["b"], B_map["g"], B_map["r"]])
    detector = detect.make_detector()
    shape = (ci["height"], ci["width"])

    frame_ids = sorted(set(int(f) for f in pred.frame_idx.unique()) & set(poses))
    cache = {}
    for fi in frame_ids:
        rv, tv = poses[fi]
        cache[fi] = (T.plane_depth_map(lay, rv, tv, Km, shape),
                     T.board_mask(lay, rv, tv, Km, shape))

    out = []
    for m in MULTIPLIERS:
        detected_by_frame = {}
        reference_by_frame = {} if m == 0.0 else None
        for fi in frame_ids:
            depth, mask = cache[fi]
            img = cv2.imread(os.path.join(dataset_dir, RUN, "frames", f"{fi:06d}.png"))
            syn = T.synthesise(img, depth, mask, B_vec, m * beta_vec)
            gray = cv2.cvtColor(syn, cv2.COLOR_BGR2GRAY)
            detected_by_frame[fi] = {d["marker_id"] for d in detect.detect_frame(gray,
                                                                                detector)}
            if m == 0.0:
                # Identity check reference. detections.csv (from det/obs, used to build
                # `pred` above) was produced by detect.sweep_dataset, which reads frames
                # with cv2.IMREAD_GRAYSCALE. This sweep instead reads colour and does
                # cv2.cvtColor(img, COLOR_BGR2GRAY) so it can synthesise per channel.
                # Those two grey conversions are NOT the same image: they differ by up
                # to 1 DN, which is enough to flip detections for markers sitting on
                # ArUco's adaptive threshold. Comparing against detections.csv would
                # therefore be comparing two different grey paths, not testing the
                # synthesis. The correct reference is this frame's own
                # cvtColor(imread(colour)) grey, detected WITHOUT synthesis, which is
                # exactly what multiplier 0 (dbeta=0, exp(0)=1) must reproduce.
                gray_ref = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                ref_dets = detect.detect_frame(gray_ref, detector)
                reference_by_frame[fi] = {d["marker_id"] for d in ref_dets}
        if m == 0.0:
            exact = detected_by_frame == reference_by_frame
            summary["identity_check_exact"] = bool(exact)
            summary["identity_check_frames_compared"] = len(frame_ids)
            assert exact, (
                "synthesise at dbeta=0 must be the identity: detections at "
                "multiplier 0 diverged from detections on the same frame, same "
                "grey path, un-synthesised")
        trials = T.trials_at_tau(pred[pred.frame_idx.isin(frame_ids)], detected_by_frame)
        rates = M.rate_by_bin(trials, PX_BINS)
        rates["multiplier"] = m
        rates["tau_added_grey"] = m * beta_map["grey"]
        out.append(rates)
        overall = trials.detected.mean() if len(trials) else 0.0
        summary.setdefault("overall_rate_by_multiplier", {})[str(m)] = {
            "rate": round(float(overall), 3), "n_trials": int(len(trials))}
        print(f"  multiplier {m:>4}: overall rate {overall:.3f} "
              f"over {len(trials)} trials")
        if overall < 0.01:
            summary["sweep_stopped_at_multiplier"] = m
            break                       # detection is dead; further points say nothing
    return pred, pd.concat(out, ignore_index=True)


def figure_surface(sweep_df, beta_grey, summary):
    """rate(px, tau). The parent study's sigmoid is ONE DIAGONAL SLICE through this: in
    real data apparent size and optical depth are both set by range, so they cannot be
    separated. Synthesis holds pixels fixed and varies tau, which reality never permits.
    """
    s = sweep_df[sweep_df.n >= MIN_TRIALS_PER_BIN].copy()
    s["px"] = (s.bin_lo + s.bin_hi) / 2.0
    piv = s.pivot_table(index="tau_added_grey", columns="px", values="rate",
                        aggfunc="mean")
    fig, ax = plt.subplots(figsize=(vizstyle.WIDE_W, vizstyle.COL_W * 0.8))
    im = ax.pcolormesh(piv.columns, piv.index, piv.to_numpy(), cmap="viridis",
                       vmin=0, vmax=1, shading="nearest")
    ax.set_xscale("log")
    vizstyle.log_px_ticks(ax)
    ax.set_xlabel("apparent marker size (px)")
    ax.set_ylabel("added optical depth (grey)")
    ax.set_title("Detection rate vs apparent size and turbidity")
    fig.colorbar(im, ax=ax, label="detection rate")
    summary["n_suppressed_surface_cells"] = int(
        (sweep_df.n > 0).sum() - (sweep_df.n >= MIN_TRIALS_PER_BIN).sum())
    vizstyle.save(fig, "rate_px_tau_surface")


def main(dataset_dir="dataset"):
    vizstyle.apply()
    os.makedirs("results", exist_ok=True)
    summary = {"opencv": cv2.__version__, "code_version": code_version(), "run": RUN}

    det, _frames, _imu, ci = load_run(dataset_dir, RUN)
    Km = np.array([[ci["fx"], 0, ci["cx"]], [0, ci["fy"], ci["cy"]], [0, 0, 1]])
    lay = L.load_layout("target/board_layout.yaml")
    poses = board_poses(det, lay, Km)

    samples = collect_samples(dataset_dir, det, poses, lay)
    samples.to_csv("results/turbidity_samples.csv", index=False)

    betas = T.measure_beta(samples)

    k_table, responses = instrument_response_table(samples)
    k_table.to_csv("results/turbidity_k_px.csv", index=False)

    fe = fixed_effects_beta_table(samples, responses)
    fe_row = {r.channel: r for r in fe.itertuples()}

    # The naive pooled fit (measure_beta) is biased about 19% high (Task 7b): the
    # sampler reads small markers low and small means far, so its own defect imitates
    # attenuation. The corrected fixed-effects fit is the authoritative beta and is
    # what feeds veiling light, tau_at_5m, and the turbidity synthesis below.
    beta_map = {ch: fe_row[ch].beta_corrected for ch in T.CHANNELS}
    veil = T.measure_veiling(samples, beta_map)
    betas.merge(veil, on="channel").to_csv("results/turbidity_beta.csv", index=False)

    per_marker = per_marker_beta_table(samples, responses)
    per_marker.to_csv("results/turbidity_beta_per_marker.csv", index=False)

    summary["n_samples"] = int(len(samples))
    summary["range_m"] = [round(float(samples.range_m.min()), 2),
                          round(float(samples.range_m.max()), 2)]
    summary["beta"] = {r.channel: {"beta": round(r.beta, 3), "r2": round(r.r2, 3)}
                       for r in betas.itertuples()}
    summary["beta_fixed_effects"] = {
        ch: {"raw": {"beta": round(fe_row[ch].beta_raw, 3),
                    "r2": round(fe_row[ch].r2_raw, 3)},
             "corrected": {"beta": round(fe_row[ch].beta_corrected, 3),
                          "r2": round(fe_row[ch].r2_corrected, 3)},
             "n": int(fe_row[ch].n)}
        for ch in T.CHANNELS
    }
    summary["red_green_ratio_raw"] = round(
        float(fe_row["r"].beta_raw / fe_row["g"].beta_raw), 3)
    summary["red_green_ratio_corrected"] = round(
        float(fe_row["r"].beta_corrected / fe_row["g"].beta_corrected), 3)
    summary["veiling_B"] = {r.channel: {"B": round(r.B, 1), "r2": round(r.r2, 3)}
                            for r in veil.itertuples()}
    summary["tau_at_5m_grey"] = round(float(beta_map["grey"] * 5.0), 2)

    figure_beta(samples, betas)
    samples[["frame_idx", "marker_id", "size_mm", "range_m", "apparent_px",
             "edge_px"]].to_csv("results/turbidity_edge_width.csv", index=False)
    figure_edge_width(samples, beta_map["grey"], summary)

    # obs is built ONCE here, keyed (RUN, frame_idx) -> {marker_id: (4,2) corners},
    # exactly as run_analysis.py:171-175 does it, and shared by the sweep and the
    # identity check below so they cannot diverge.
    obs = {}
    for r in det[det.marker_id.isin(g.SIZES)].itertuples():
        obs.setdefault((RUN, r.frame_idx), {})[int(r.marker_id)] = np.array(
            [[r.c0x, r.c0y], [r.c1x, r.c1y], [r.c2x, r.c2y], [r.c3x, r.c3y]])

    B_map = dict(zip(veil.channel, veil.B))
    pred, sweep_df = sweep(dataset_dir, obs, poses, lay, Km, ci, beta_map, B_map, summary)
    pred.to_csv("results/turbidity_trials.csv", index=False)
    sweep_df.to_csv("results/turbidity_sweep.csv", index=False)

    # The identity check itself (multiplier 0 vs the same frames read the same way but
    # un-synthesised) runs inside sweep() and asserts there, next to the frame reads it
    # needs. See the comment at that assertion for why detections.csv is not the
    # reference.

    # The sweep's own multiplier-0 baseline rate (~0.519) differs from the parent
    # study's published figure from detections.csv (1374/2551 = 0.539, about 4%
    # higher). That gap is the same grey-path difference: detections.csv comes from
    # detect.sweep_dataset's cv2.IMREAD_GRAYSCALE, while this sweep reads colour and
    # converts with cv2.cvtColor, which differs by up to 1 DN and flips marginal
    # detections. It does not invalidate the sweep, since every multiplier inside it
    # uses one consistent grey path, so the tau dependence is unaffected. Arguably
    # cvtColor-on-colour is the more representative path anyway: the ZED publishes
    # colour, and a real docking pipeline would convert it the same way.
    published_rate = 1374 / 2551
    sweep_baseline_rate = summary["overall_rate_by_multiplier"]["0.0"]["rate"]
    summary["baseline_rate_vs_published"] = {
        "sweep_multiplier_0": round(sweep_baseline_rate, 3),
        "published_detections_csv": round(published_rate, 3),
        "note": "gap is the IMREAD_GRAYSCALE vs cvtColor(imread(colour)) grey-path "
                "difference (up to 1 DN); each multiplier inside the sweep uses one "
                "consistent path, so tau dependence is unaffected",
    }

    figure_surface(sweep_df, beta_map["grey"], summary)

    with open("results/turbidity_summary.json", "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "dataset")
