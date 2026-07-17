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
    beta_grey = beta_map["grey"]
    detector = detect.make_detector()
    shape = (ci["height"], ci["width"])

    frame_ids = sorted(set(int(f) for f in pred.frame_idx.unique()) & set(poses))
    cache = {}
    rng_by_frame = {}
    for fi in frame_ids:
        rv, tv = poses[fi]
        cache[fi] = (T.plane_depth_map(lay, rv, tv, Km, shape),
                     T.board_mask(lay, rv, tv, Km, shape))
        # Each marker's own range from the board pose, NOT from apparent size: the
        # parent study established that apparent-size range flips between markers and
        # drifts at oblique incidence, so the pose is the better estimate.
        rng_by_frame[fi] = T.marker_ranges(lay, rv, tv)

    out = []
    trial_rows = []
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
        # dbeta_added = m * beta_grey is an ADDED ATTENUATION COEFFICIENT (units 1/m):
        # it is the synthesis control knob passed into T.synthesise as dbeta, not an
        # optical depth. tau = beta * range needs a range, which this quantity does not
        # carry, so multiplying it by anything and calling it tau silently pools markers
        # at very different actual optical depths at the same apparent size (a marker's
        # size sets its range at fixed apparent px, and size varies 4.2x across the
        # board). Do not reinstate tau_added_grey. The real per-trial tau_total, using
        # each marker's own range from the board pose, is in turbidity_sweep_trials.csv
        # and feeds figure_surface below.
        rates["dbeta_added"] = m * beta_grey
        out.append(rates)

        t = trials.copy()
        t["range_m"] = [rng_by_frame[int(fi)][int(mid)]
                        for fi, mid in zip(t["frame_idx"], t["marker_id"])]
        # The pool's own water is already in every un-synthesised frame, so multiplier 0
        # is NOT clear water: its tau is beta_grey * range_m, not zero.
        t["tau_total"] = (1.0 + m) * beta_grey * t["range_m"]
        t["multiplier"] = m
        trial_rows.append(t[["frame_idx", "marker_id", "size_mm", "pred_px", "multiplier",
                             "range_m", "tau_total", "detected"]])

        overall = trials.detected.mean() if len(trials) else 0.0
        summary.setdefault("overall_rate_by_multiplier", {})[str(m)] = {
            "rate": round(float(overall), 3), "n_trials": int(len(trials))}
        print(f"  multiplier {m:>4}: overall rate {overall:.3f} "
              f"over {len(trials)} trials")
        if overall < 0.01:
            summary["sweep_stopped_at_multiplier"] = m
            break                       # detection is dead; further points say nothing
    return pred, pd.concat(out, ignore_index=True), pd.concat(trial_rows, ignore_index=True)


# Log-spaced edges covering the observed tau_total range (about 0.18 at multiplier 0,
# short range, up to about 9 at multiplier 8, long range). Non-uniform, same as the old
# per-multiplier axis was; pcolormesh handles irregular coordinates directly.
TAU_BINS = [0.1, 0.15, 0.22, 0.32, 0.46, 0.68, 1.0, 1.5, 2.2, 3.2, 4.6, 6.8, 10.0]


def figure_surface(trials_df, summary):
    """rate(px, tau_total), tau_total = (1 + multiplier) * beta_grey * range_m from the
    board pose. The parent study's sigmoid is ONE DIAGONAL SLICE through this: in real
    data apparent size and optical depth are both set by range, so they cannot be
    separated. Synthesis holds pixels fixed and varies tau, which reality never permits.
    """
    t = trials_df.copy()
    t["px_bin"] = pd.cut(t.pred_px, PX_BINS, right=False)
    t["tau_bin"] = pd.cut(t.tau_total, TAU_BINS, right=False)
    binned = t.dropna(subset=["px_bin", "tau_bin"]).groupby(
        ["tau_bin", "px_bin"], observed=True).agg(
        n=("detected", "size"), rate=("detected", "mean")).reset_index()
    s = binned[binned.n >= MIN_TRIALS_PER_BIN].copy()
    s["px"] = s.px_bin.apply(lambda iv: (iv.left + iv.right) / 2.0)
    s["tau"] = s.tau_bin.apply(lambda iv: (iv.left + iv.right) / 2.0)
    piv = s.pivot_table(index="tau", columns="px", values="rate", aggfunc="mean",
                        observed=True)
    fig, ax = plt.subplots(figsize=(vizstyle.WIDE_W, vizstyle.COL_W * 0.8))
    im = ax.pcolormesh(piv.columns, piv.index, piv.to_numpy(), cmap="viridis",
                       vmin=0, vmax=1, shading="nearest")
    ax.set_xscale("log")
    vizstyle.log_px_ticks(ax)
    ax.set_xlabel("apparent marker size (px)")
    ax.set_ylabel("total optical depth, tau (grey)")
    ax.set_title("Detection rate vs apparent size and total optical depth")
    fig.colorbar(im, ax=ax, label="detection rate")

    # Cells above this line are still real trials (n >= MIN_TRIALS_PER_BIN), not blank
    # space: detection is uniformly dead there (rate 0.0) and stays dead all the way to
    # tau 10, which is itself the finding (the boundary sweeps right then never comes
    # back). Clipping the axis to just past the highest NONZERO-rate cell keeps that
    # boundary visible while not spending most of the canvas repeating "still zero".
    # Every dropped cell is logged below so the cut is auditable, not silent.
    nz = s[s.rate > 0]
    tau_hi = s.tau_bin.map(lambda iv: float(iv.right)).astype(float)
    if len(nz):
        nz_tau_hi = nz.tau_bin.map(lambda iv: float(iv.right)).astype(float)
        y_top = float(nz_tau_hi.max()) * 1.1
    else:
        y_top = float(tau_hi.max())
    ax.set_ylim(top=y_top)
    fig.tight_layout()

    total_cells = (len(PX_BINS) - 1) * (len(TAU_BINS) - 1)
    n_populated = int((binned.n > 0).sum())
    n_kept = int(len(s))
    n_clipped_from_view = int((tau_hi > y_top).sum())
    summary["surface_cells_total"] = int(total_cells)
    summary["surface_cells_empty_no_trials"] = int(total_cells - n_populated)
    summary["surface_cells_suppressed_below_min_trials"] = int(n_populated - n_kept)
    summary["surface_cells_kept"] = n_kept
    summary["surface_cells_kept_clipped_from_view"] = n_clipped_from_view
    summary["surface_cells_kept_clipped_from_view_max_rate"] = (
        float(s[tau_hi > y_top].rate.max()) if n_clipped_from_view else 0.0)
    vizstyle.save(fig, "rate_px_tau_surface")


TAU_BINS = [0.0, 0.35, 0.55, 0.8, 1.1, 1.5, 2.0, 2.6, 3.4, 4.5, 6.0, 14.0]
BETA_SCENARIOS = {"this pool": None, "clearer (0.15/m)": 0.15,
                  "murkier (0.60/m)": 0.60, "harbour (1.00/m)": 1.00}


def figure_px_required(trials, fx, summary):
    """The deliverable: apparent size at the 50% crossing, as a function of REAL tau.

    Bins by tau_total (= (1 + m) * beta_grey * range_m, per trial, range from the board
    pose), NOT by the sweep's control knob. m * beta_grey has units 1/m and is an added
    attenuation coefficient, not an optical depth; binning on it pools markers at very
    different real tau (at 150 px the 149.4 mm marker is at 0.79 m and the 35.6 mm at
    0.19 m, so one cell mixes tau 0.27 to 1.15).

    Three findings, all supported by the measured curve:

    1. The budget is about 27 px, not the textbook 21 px. 3(n+2) = 21 for a 5x5
       dictionary is a pure sampling argument that assumes air, so it says nothing
       about contrast; it underestimates by roughly 30 percent here.
    2. The budget is FLAT from tau 0.18 to 1.30, a 7x range of optical depth.
       ArUco's adaptive threshold is far more robust to contrast loss than the
       sampling argument predicts.
    3. Above tau about 2, detection collapses at every apparent size: the tau 2.0 to
       2.6 bin tops out at max_rate 0.161, and every bin above tau 2.6 is 0.0. This is
       a threshold, not a gradient.

    Caution on (1) and (2): PX_BINS near the crossing are [..., 21, 28, 38, ...], so
    px_at_rate interpolates every one of the tau 0.18 to 1.75 points between bin
    centres 24.5 and 33.0, an interval only 8.5 px wide. The visible 27.3 to 32.3 px
    wobble across those points is sub-bin interpolation, not a resolvable trend: do
    not read it as the budget growing with tau. Only the flatness up to tau 1.30 and
    the collapse above tau 2 are claims this binning can support.
    """
    t = trials.copy()
    t["tau_bin"] = pd.cut(t.tau_total, TAU_BINS, right=False)
    t["px_bin"] = pd.cut(t.pred_px, PX_BINS, right=False)

    rows = []
    for tb, sub in t.groupby("tau_bin", observed=True):
        agg = sub.groupby("px_bin", observed=True).agg(n=("detected", "size"),
                                                       k=("detected", "sum"))
        agg = agg[agg.n >= MIN_TRIALS_PER_BIN]
        if len(agg) < 2:
            continue                     # cannot locate a crossing from one point
        px_c = np.array([(iv.left + iv.right) / 2.0 for iv in agg.index])
        rate = (agg.k / agg.n).to_numpy()
        rows.append({"tau_lo": float(tb.left), "tau_hi": float(tb.right),
                     "tau_mid": float((tb.left + tb.right) / 2.0),
                     "n": int(agg.n.sum()), "max_rate": float(rate.max()),
                     "px_required": T.px_at_rate(px_c, rate, 0.5)})
    curve = pd.DataFrame(rows)
    curve.to_csv("results/px_required_vs_tau.csv", index=False)
    summary["px_required_vs_tau"] = curve.to_dict("records")

    # Auditable resolution fact for the docstring's caution above: the two PX_BINS
    # edges that bracket the crossing (21, 28, 38) give bin centres 24.5 and 33.0, an
    # 8.5 px interval. px_at_rate interpolates linearly inside it, so any px_required
    # value landing in this range is not resolved finer than its width.
    lo, mid, hi = 21.0, 28.0, 38.0
    interval_lo, interval_hi = (lo + mid) / 2.0, (mid + hi) / 2.0
    summary["px_required_bin_interval"] = {
        "px_bin_edges": [lo, mid, hi],
        "interval_px": [interval_lo, interval_hi],
        "width_px": round(interval_hi - interval_lo, 1)}

    ok = curve.dropna(subset=["px_required"])
    fig, ax = plt.subplots(figsize=(vizstyle.COL_W, vizstyle.COL_W * 0.75))
    ax.plot(ok.tau_mid, ok.px_required, "o-", color="#184f95", lw=2, ms=5)
    ax.axhline(M.pixel_budget_px(5), color=vizstyle.TEXT_SECONDARY, ls="--", lw=1.5)
    ax.annotate("3(n+2) = 21 px, the clear-water budget",
                xy=(ok.tau_mid.min(), M.pixel_budget_px(5)), xytext=(4, 4),
                textcoords="offset points", fontsize=6, color=vizstyle.TEXT_SECONDARY)

    # Mark where detection collapses: the first tau bin whose px_required is nan is
    # the boundary past which no apparent size we measured reaches 50%. The flat curve
    # to its left simply stops there with no visual cue otherwise, which reads as "the
    # measurement ran out" rather than "detection died"; make the boundary explicit.
    # The dead region actually runs to tau 14 (all zero), but drawing the full range
    # would squeeze the live data into a sliver of the axes, so the view is clipped
    # just past the first dead bin; the shading still implies it continues past the
    # visible edge.
    dead = curve[curve.px_required.isna()]
    if len(dead):
        tau_collapse = float(dead.iloc[0].tau_lo)
        x_right = float(dead.iloc[0].tau_hi) * 1.15
        ax.axvspan(tau_collapse, float(curve.tau_hi.max()), color="#c0392b", alpha=0.08)
        ax.axvline(tau_collapse, color="#c0392b", ls=":", lw=1.5)
        ax.annotate(f"above tau {tau_collapse:.1f}\nno size reaches 50%",
                    xy=(tau_collapse, ok.px_required.max()),
                    xytext=(4, -4), textcoords="offset points",
                    fontsize=6, color="#c0392b", va="top")
        ax.set_xlim(right=x_right)

    # Show the measurement's own resolution, because without it this figure lies. The
    # y-axis is zoomed to about 21 to 33 px, which magnifies the 4.9 px spread into what
    # looks like a rise at the last live point. It is not: px_at_rate interpolates
    # between px bin CENTRES, and every value here falls inside the single interval
    # [24.5, 33.0]. Banding that interval shows at a glance that the whole curve sits
    # within one bin's width, so no trend is resolvable and none should be read.
    ax.axhspan(interval_lo, interval_hi, color=vizstyle.GHOST_COLOR, alpha=0.45,
               zorder=0)
    ax.annotate(f"one px bin interval [{interval_lo:.1f}, {interval_hi:.1f}]:"
                " no trend is resolvable here",
                xy=(ax.get_xlim()[0], interval_lo), xytext=(4, 3),
                textcoords="offset points", fontsize=5.5,
                color=vizstyle.TEXT_SECONDARY, va="bottom")

    ax.set_xlabel("total optical depth, tau (grey)")
    ax.set_ylabel("apparent size at 50% detection (px)")
    ax.set_title("27 px needed, not 21; fails above tau 2", fontsize=9)
    fig.tight_layout()
    vizstyle.save(fig, "px_required_vs_tau")

    # Where the curve reports nan, detection never reached 50% at ANY size we measured.
    # That is a result, not a gap: above roughly tau 2.3 turbidity limits regardless of
    # apparent size. Record it rather than interpolating through it. `dead` was already
    # computed above to place the collapse marker on the figure.
    summary["tau_bins_never_reaching_50pc"] = [
        {"tau_lo": r.tau_lo, "tau_hi": r.tau_hi, "n": r.n, "max_rate": round(r.max_rate, 3)}
        for r in dead.itertuples()]

    sizing = []
    for name, beta in BETA_SCENARIOS.items():
        b = summary["beta_grey_used"] if beta is None else beta
        for d in (3.0, 5.0):
            tau = b * d
            if tau > ok.tau_mid.max() or tau < ok.tau_mid.min():
                px_req = float("nan")     # outside what was measured; do not invent it
            else:
                px_req = float(np.interp(tau, ok.tau_mid, ok.px_required))
            sizing.append({"scenario": name, "beta_grey": round(b, 3), "range_m": d,
                           "tau_total": round(tau, 2),
                           "px_required": round(px_req, 1),
                           "side_mm": round(T.required_side_m(px_req, d, fx) * 1000, 0)})
    pd.DataFrame(sizing).to_csv("results/turbidity_sizing.csv", index=False)
    summary["sizing"] = sizing


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
    pred, sweep_df, trials_df = sweep(dataset_dir, obs, poses, lay, Km, ci, beta_map,
                                      B_map, summary)
    pred.to_csv("results/turbidity_trials.csv", index=False)
    sweep_df.to_csv("results/turbidity_sweep.csv", index=False)
    trials_df.to_csv("results/turbidity_sweep_trials.csv", index=False)

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

    figure_surface(trials_df, summary)
    summary["tau_total_by_multiplier"] = {
        str(m): {"min": round(float(sub.tau_total.min()), 3),
                "max": round(float(sub.tau_total.max()), 3)}
        for m, sub in trials_df.groupby("multiplier")
    }

    summary["beta_grey_used"] = beta_map["grey"]
    figure_px_required(trials_df, ci["fx"], summary)

    with open("results/turbidity_summary.json", "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "dataset")
