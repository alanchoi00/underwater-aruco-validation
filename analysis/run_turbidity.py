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
from matplotlib.patches import Polygon

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


def figure_beta(samples, responses):
    """Plot the authoritative fixed-effects, instrument-corrected beta fit.

    measure_beta's pooled fit is NOT plotted here: it is superseded (about 19% biased
    high, and it conflates each marker's own intercept with the shared slope), and an
    earlier version of this figure plotted it anyway, so the figure contradicted the
    beta table it sits next to. Both the scatter and the curve below are
    instrument-corrected (divided by k(apparent_px)) and fit with
    fit_beta_fixed_effects, the same call that produces beta_fixed_effects in
    turbidity_summary.json, so the numbers in the legend match that table exactly.
    The plotted intercept is the median of the per-marker intercepts from that fit,
    since fixed effects gives one per marker id and this figure draws one line.
    """
    fig, ax = plt.subplots(figsize=(vizstyle.COL_W, vizstyle.COL_W * 0.75))
    colours = {"b": "#2a78d6", "g": "#2e8b57", "r": "#c0392b", "grey": "#52514e"}
    for ch in T.CHANNELS:
        sub = samples.dropna(subset=[f"white_{ch}", f"black_{ch}", "range_m",
                                     "apparent_px", "marker_id"])
        k_px, k_val = responses[ch]
        resp = T.apply_instrument_response(sub["apparent_px"].to_numpy(), k_px, k_val)
        contrast = (sub[f"white_{ch}"] - sub[f"black_{ch}"]).to_numpy() / resp
        ok = contrast > 0
        ax.scatter(sub.range_m[ok], contrast[ok], s=4, alpha=0.25, color=colours[ch],
                   linewidths=0)
        beta, r2, intercepts = T.fit_beta_fixed_effects(sub, ch, k=responses[ch])
        c0 = float(np.median(list(intercepts.values()))) if intercepts else float("nan")
        d = np.linspace(sub.range_m.min(), sub.range_m.max(), 50)
        ax.plot(d, c0 * np.exp(-beta * d), color=colours[ch], lw=2,
                label=f"{ch}: beta={beta:.3f}/m, r2={r2:.2f}")
    ax.set_yscale("log")
    ax.set_xlabel("range (m, from board pose)")
    ax.set_ylabel("corrected black/white contrast (DN)")
    ax.set_title("Contrast decay: corrected fit", fontsize=9)
    ax.legend(fontsize=6, loc="lower left")
    fig.tight_layout()
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


# Linear edges covering the observed tau_total range (about 0.18 at multiplier 0, short
# range, up to about 9 at multiplier 8, long range), with a wide last bin to catch the
# collapsed-detection tail out to tau 14. Shared by figure_surface and
# figure_px_required, both binning the same tau_total column.
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

    # side_mm interpolates px_required across the tau 0.18 to 1.75 rise that the
    # docstring above and the shipped limitations both say is sub-bin interpolation,
    # not a resolvable trend (every point in it falls inside the single px bin
    # interval [interval_lo, interval_hi], interval_lo, interval_hi above). A bare
    # side_mm therefore reads noise as signal. side_mm_lo/side_mm_hi convert that same
    # bin interval into millimetres at the same range, so the sizing table always
    # carries its own resolution alongside the point estimate; the point estimate
    # itself is unchanged, per the project owner's crossing-rule decision.
    sizing = []
    for name, beta in BETA_SCENARIOS.items():
        b = summary["beta_grey_used"] if beta is None else beta
        for d in (3.0, 5.0):
            tau = b * d
            if tau > ok.tau_mid.max() or tau < ok.tau_mid.min():
                px_req = float("nan")     # outside what was measured; do not invent it
            else:
                px_req = float(np.interp(tau, ok.tau_mid, ok.px_required))
            # The interval is derived from range alone, so it would happily populate even
            # where px_required is nan. It must not: quoting 92 to 124 mm for harbour
            # water at 3 m would invent a bracket for a regime where we measured that
            # detection collapses at every size. No crossing means no answer, bracket
            # included.
            if np.isnan(px_req):
                side_lo = side_hi = float("nan")
            else:
                side_lo = T.required_side_m(interval_lo, d, fx) * 1000
                side_hi = T.required_side_m(interval_hi, d, fx) * 1000
            sizing.append({"scenario": name, "beta_grey": round(b, 3), "range_m": d,
                           "tau_total": round(tau, 2),
                           "px_required": round(px_req, 1),
                           "side_mm": round(T.required_side_m(px_req, d, fx) * 1000, 0),
                           "side_mm_lo": round(side_lo, 0),
                           "side_mm_hi": round(side_hi, 0)})
    pd.DataFrame(sizing).to_csv("results/turbidity_sizing.csv", index=False)
    summary["sizing"] = sizing


def crossval(samples, beta_map, responses, summary):
    """Predict a far sample's contrast from a near one, using only the measured beta.

    Same marker id, so J is identical; only the water between differs. The prediction is
        contrast_far = contrast_near * exp(-beta * (d_far - d_near))
    which is the model with nothing else in it. Pairs are drawn across every marker with a
    wide enough range span, and the distribution is what gets reported: one pair agreeing
    proves nothing.

    Every sample's contrast is also divided by its instrument response k(apparent_px)
    before forming the corrected prediction. Apparent size falls with range, so the far
    sample of any pair is systematically under-read by the sampler (blur compresses the
    black ring against the white sheet as the border cell approaches the blur width, down
    to k about 0.816 at 42 px versus 1.000 at 170 px). Left uncorrected this reads as the
    model over-predicting, when the defect is the sampler under-measuring, not beta. Both
    the raw and the corrected distributions are reported; the difference between them is
    itself evidence about the correction.
    """
    rows = []
    for mid, sub in samples.groupby("marker_id"):
        sub = sub.sort_values("range_m")
        if len(sub) < 4:
            continue
        near = sub.iloc[:max(1, len(sub) // 4)]
        far = sub.iloc[-max(1, len(sub) // 4):]
        for _, a in near.iterrows():
            for _, b in far.iterrows():
                dd = b.range_m - a.range_m
                if dd < 0.5:
                    continue
                for ch in T.CHANNELS:
                    c_near = a[f"white_{ch}"] - a[f"black_{ch}"]
                    c_far = b[f"white_{ch}"] - b[f"black_{ch}"]
                    if c_near <= 0 or c_far <= 0:
                        continue
                    pred = c_near * np.exp(-beta_map[ch] * dd)
                    rel_err = (pred - c_far) / c_far

                    k_px, k_val = responses[ch]
                    k_near = T.apply_instrument_response(a.apparent_px, k_px, k_val)
                    k_far = T.apply_instrument_response(b.apparent_px, k_px, k_val)
                    c_near_corr = c_near / k_near
                    c_far_corr = c_far / k_far
                    pred_corr = c_near_corr * np.exp(-beta_map[ch] * dd)
                    rel_err_corr = (pred_corr - c_far_corr) / c_far_corr

                    rows.append({"marker_id": int(mid), "channel": ch,
                                 "d_near": a.range_m, "d_far": b.range_m,
                                 "px_near": a.apparent_px, "px_far": b.apparent_px,
                                 "pred": pred, "actual": c_far, "rel_err": rel_err,
                                 "pred_corrected": pred_corr,
                                 "actual_corrected": c_far_corr,
                                 "rel_err_corrected": rel_err_corr})
    cv = pd.DataFrame(rows)
    cv.to_csv("results/turbidity_crossval.csv", index=False)

    def dist(col):
        return {ch: {"n": int((cv.channel == ch).sum()),
                     "median_rel_err": round(float(cv[cv.channel == ch][col].median()), 3),
                     "p10": round(float(cv[cv.channel == ch][col].quantile(0.10)), 3),
                     "p90": round(float(cv[cv.channel == ch][col].quantile(0.90)), 3)}
                for ch in T.CHANNELS if (cv.channel == ch).any()}

    summary["crossval"] = {"raw": dist("rel_err"), "corrected": dist("rel_err_corrected")}
    figure_crossval_distribution(cv, summary)
    return cv


def figure_crossval_distribution(cv, summary):
    """The actual validation: raw versus corrected relative error, pooled over every
    near/far pair the dataset offers (5629 pairs per channel, all 9 markers, every
    pair with a range gap of at least 0.5 m).

    A box per channel, raw and corrected side by side, is the chosen form: the
    finding is a location shift (about +18 percent collapsing to about -2.5 percent),
    not the shape of either distribution, and box pairs read that shift at a glance
    without the bin-width judgment calls a histogram would force at this width. A
    zero line marks truth so the collapse toward it is visible without reading tick
    labels.

    What this figure legitimately establishes, and what it does not:

    This is NOT cross-validation in the statistical sense: beta was fitted on
    exactly the 934 samples these pairs are drawn from, so there is no holdout set
    and no independent data. What the pair-prediction test legitimately shows is
    (a) that the exponential FORM holds, since curvature in range would show up as
    opposite residual bias at the near and far ends of each pair, which it does not,
    and (b) that after the instrument-response correction, the model reproduces
    real far-frame contrast to within about 3 percent (median) with a p10/p90 of
    roughly -9 to +14 percent. It is not independent validation of the
    instrument-response correction itself: given the model, the raw bias is
    k_near/k_far - 1 identically (k is applied to both sides of the same
    prediction), so its magnitude and its collapse under correction are algebra,
    not evidence that k(apparent_px) is the right correction to make.
    """
    fig, ax = plt.subplots(figsize=(vizstyle.WIDE_W, vizstyle.COL_W * 0.85))
    colours = {"b": "#2a78d6", "g": "#2e8b57", "r": "#c0392b", "grey": "#52514e"}
    channels = [ch for ch in T.CHANNELS if (cv.channel == ch).any()]
    n_per_channel = int((cv.channel == channels[0]).sum()) if channels else 0

    positions = []
    data = []
    box_colours = []
    labels = []
    for i, ch in enumerate(channels):
        sub = cv[cv.channel == ch]
        base = i * 3
        positions.extend([base, base + 1])
        data.extend([sub["rel_err"].to_numpy(), sub["rel_err_corrected"].to_numpy()])
        box_colours.extend(["#b8b6b0", colours[ch]])
        labels.append(ch)

    bp = ax.boxplot(data, positions=positions, widths=0.8, vert=False, patch_artist=True,
                    showfliers=False, medianprops={"color": "#0b0b0b", "linewidth": 1.2})
    for patch, colour in zip(bp["boxes"], box_colours):
        patch.set_facecolor(colour)
        patch.set_alpha(0.75)

    ax.axvline(0.0, color=vizstyle.TEXT_SECONDARY, ls="--", lw=1.2, zorder=0)
    ax.set_yticks([i * 3 + 0.5 for i in range(len(channels))])
    ax.set_yticklabels(labels)
    ax.set_xlabel("relative error, predicted far contrast vs measured far contrast")
    ax.set_title("The model predicts real far-frame contrast to within 3 percent",
                fontsize=9)
    # A legend BELOW the axes. Two constraints pull against each other here. Inside the
    # axes there is nowhere to put it: the raw whiskers reach +0.5, so every corner a
    # legend would want is occupied, and it lands on the bottom channel's raw box. But a
    # positional caption ("lower box is raw") is no good either, because the reader
    # separates these boxes by COLOUR, not by position, and would have to decode the
    # layout first. Anchoring the legend below the axes satisfies both: keyed to colour,
    # clear of the data.
    # A colour-keyed caption below the axes, and neither a legend box nor a positional
    # caption, because both fail here. A legend inside the axes lands on the data: the
    # raw whiskers reach +0.5, so every corner is occupied. A legend anchored outside
    # lands on the x-axis label, since tight_layout reserves room for the label but not
    # for it. And "lower box is raw" would make the reader decode the layout when the
    # eye is already separating these by colour. So: say the colours.
    fig.text(0.5, -0.02,
             "pale grey is raw, where the sampler reads far markers low; "
             "the channel's own colour is corrected by k(apparent_px). "
             f"n = {n_per_channel} pairs per channel",
             ha="center", va="top", fontsize=6.5, color=vizstyle.TEXT_SECONDARY)
    fig.tight_layout()
    vizstyle.save(fig, "crossval_distribution")

    summary["crossval_distribution_n_per_channel"] = n_per_channel


SWEEP_STRIP_FRAME = 197              # all 9 markers present, 7/9 detected at m=0,
                                      # median range 1.38 m: the strip isolates water
SWEEP_STRIP_MULTIPLIERS = [0.0, 1.0, 2.0, 4.0, 8.0]
SWEEP_STRIP_CROP_MARGIN = 0.15
SWEEP_STRIP_DETECTED_COLOR = "#1a9850"
SWEEP_STRIP_MISSED_COLOR = "#c0392b"


def _board_crop_box(mask, margin=SWEEP_STRIP_CROP_MARGIN):
    """Bounding box of the True pixels in `mask`, expanded by `margin` of its own size.

    Returns (x0, x1, y0, y1), clipped to the mask's own shape, for a fixed crop that
    can be reused across every synthesised panel of the same frame.
    """
    ys, xs = np.nonzero(mask)
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    mx = int(round((x1 - x0) * margin))
    my = int(round((y1 - y0) * margin))
    h, w = mask.shape[:2]
    return (max(0, x0 - mx), min(w, x1 + mx + 1),
           max(0, y0 - my), min(h, y1 + my + 1))


def figure_sweep_strip(dataset_dir, poses, lay, Km, ci, beta_map, B_map, summary,
                       frame_idx=SWEEP_STRIP_FRAME):
    """Figure 1, a five-panel strip: the same frame, synthesised at multipliers
    0/1/2/4/8, so a reader can see what "tau" means instead of only reading it off a
    curve. Frame 197 has all 9 board markers and 7/9 detected at multiplier 0, so what
    changes across the strip is the water, not the framing; every panel uses the SAME
    crop, from the frame's own board pose, so the comparison is like for like.

    Each marker is outlined from its PROJECTED pose corners, not its detected corners,
    so a marker the detector misses still gets an outline (red); a detected marker's
    outline is green. That is what makes the strip a diagnostic and not just a picture:
    it is the same 9 outlines every panel, only their colour and the water changing.

    This figure is also where the veiling light B's incoherence (see the limitations
    entry: B_green fits at 241.9 DN, brighter than the white sheet is ever observed,
    about 144 to 154 DN) becomes visible rather than a number in a table. At high
    multiplier every degraded pixel is pulled toward B, so if B is implausibly bright
    the panel should look implausibly washed out; if it still reads as murky water,
    that is evidence the model's insensitivity to B (contrast is B-free by
    construction) is doing the work the docstring in turbidity.py claims for it.
    """
    rv, tv = poses[frame_idx]
    shape = (ci["height"], ci["width"])
    depth = T.plane_depth_map(lay, rv, tv, Km, shape)
    mask = T.board_mask(lay, rv, tv, Km, shape)
    x0, x1, y0, y1 = _board_crop_box(mask)

    beta_vec = np.array([beta_map["b"], beta_map["g"], beta_map["r"]])
    B_vec = np.array([B_map["b"], B_map["g"], B_map["r"]])
    beta_grey = beta_map["grey"]

    rng = T.marker_ranges(lay, rv, tv)
    median_range = float(np.median(list(rng.values())))

    Xb = g.board_pts(lay, np.array([g.IDX[mid] for mid in g.IDS]))
    corners_px = g.project(Xb, np.array([rv]), np.array([tv]),
                           np.zeros(len(g.IDS), dtype=int),
                           float(Km[0, 0]), float(Km[1, 1]),
                           float(Km[0, 2]), float(Km[1, 2]))

    img_path = os.path.join(dataset_dir, RUN, "frames", f"{frame_idx:06d}.png")
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(img_path)

    detector = detect.make_detector()
    # Size the figure to the crop's own aspect ratio, not a fixed guess: the board
    # crop is much wider than tall, and a fixed height left most of the figure blank
    # above the panels.
    fig_w = vizstyle.WIDE_W * 1.8
    panel_w = fig_w / len(SWEEP_STRIP_MULTIPLIERS)
    panel_h = panel_w * (y1 - y0) / (x1 - x0)
    fig_h = panel_h + 0.7           # + per-panel title, suptitle, margins
    fig, axes = plt.subplots(1, len(SWEEP_STRIP_MULTIPLIERS), figsize=(fig_w, fig_h))
    detected_by_m = {}
    for ax, m in zip(axes, SWEEP_STRIP_MULTIPLIERS):
        syn = T.synthesise(img, depth, mask, B_vec, m * beta_vec)
        gray = cv2.cvtColor(syn, cv2.COLOR_BGR2GRAY)
        detected_ids = {d["marker_id"] for d in detect.detect_frame(gray, detector)}
        n_detected = len(detected_ids & set(g.IDS))
        detected_by_m[m] = n_detected

        crop = cv2.cvtColor(syn[y0:y1, x0:x1], cv2.COLOR_BGR2RGB)
        ax.imshow(crop)
        for i, mid in enumerate(g.IDS):
            poly = corners_px[i].copy()
            poly[:, 0] -= x0
            poly[:, 1] -= y0
            colour = (SWEEP_STRIP_DETECTED_COLOR if mid in detected_ids
                     else SWEEP_STRIP_MISSED_COLOR)
            ax.add_patch(Polygon(poly, closed=True, fill=False, edgecolor=colour,
                                 linewidth=0.9))
        tau = (1.0 + m) * beta_grey * median_range
        ax.set_title(f"m={int(m)}, tau {tau:.1f}, {n_detected}/9 detected", fontsize=7)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(f"Frame {frame_idx}, board median range {median_range:.2f} m: "
                 "synthesised turbidity", fontsize=8, y=0.995)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.87])
    vizstyle.save(fig, "synthesis_sweep_strip")

    summary["sweep_strip_frame"] = int(frame_idx)
    summary["sweep_strip_median_range_m"] = round(median_range, 3)
    summary["sweep_strip_detected_of_9_by_multiplier"] = {
        str(m): int(n) for m, n in detected_by_m.items()}


VALIDATION_MARKER_ID = 201
VALIDATION_NEAR_RANGE_M = 0.62       # marker 201's own range there, at about 189.7 px
VALIDATION_FAR_RANGE_M = 3.19        # marker 201's own range there, at about 43.0 px


def _frame_closest_to_range(poses, lay, mid, target_m):
    """Frame id whose board pose puts marker `mid` closest to `target_m` range."""
    best_fi, best_gap = None, float("inf")
    for fi, (rv, tv) in poses.items():
        rng = T.marker_ranges(lay, rv, tv).get(mid)
        if rng is None:
            continue
        gap = abs(rng - target_m)
        if gap < best_gap:
            best_fi, best_gap = fi, gap
    return best_fi


def _marker_corners(det, frame_idx, mid):
    sub = det[(det.frame_idx == frame_idx) & (det.marker_id == mid)]
    if len(sub) == 0:
        return None
    r = sub.iloc[0]
    return np.array([[r.c0x, r.c0y], [r.c1x, r.c1y], [r.c2x, r.c2y], [r.c3x, r.c3y]])


def _grey_contrast(warped):
    black, white = T.patch_means(warped)
    return float(white[3] - black[3])          # index 3 is grey, see T.CHANNELS


def figure_synthesis_validation(dataset_dir, det, poses, lay, Km, ci, beta_map, B_map,
                                summary, mid=VALIDATION_MARKER_ID):
    """Synthesis illustrated: marker 201 at the widest baseline, NOT a validation figure.

    The actual validation is figure_crossval_distribution (results/crossval_distribution),
    which pools 5629 near/far pairs across every marker and reports a corrected median
    relative error of about -2.5 percent (p10 about -9 percent, p90 about 14 percent).
    This figure is one pair from that pool, chosen because it is the TAIL, not the
    typical case: marker 201 appears at about 0.62 m (189.7 px) and, later in the run,
    at about 3.19 m (43.0 px), the widest baseline (2.57 m of real water) available for
    any single marker in the dataset. The near observation is degraded to the far
    observation's optical depth and compared against the far observation itself.

    It is chosen for illustration, not representativeness: the widest baseline makes
    the degradation mechanism (contrast collapsing with added water) most visible to
    the eye. Its raw contrast deficit is about 70 percent, far beyond the pooled p90 of
    about 13.8 percent; do not read this panel's numbers as the study's result. Read
    crossval_distribution for that.

    Deriving dbeta: synthesise() applies exp(-dbeta * d(x)) with d(x) the plane depth
    at each pixel, which at marker 201's own location in the near frame is d_near, not
    d_far. The added optical depth wanted is beta * (d_far - d_near), so
        dbeta * d_near = beta * (d_far - d_near)
        dbeta = beta * (d_far - d_near) / d_near
    computed below from the measured beta and the two ranges, not hard-coded.

    Two things this figure does NOT hide:

    1. The far panel is upsampled from about 43 px to the canonical warp size, while
       the near panel is downsampled from about 190 px to the same size. The far panel
       is therefore blockier. That is what the data is, not a synthesis artifact.
    2. The difference panel's residual is expected to be LARGELY the instrument
       response, not model error: at 43 px apparent size, blur compresses the far
       marker's rings enough that its measured contrast reads about 18 percent low,
       which is exactly what k(apparent_px) (see turbidity.measure_instrument_response)
       corrects. A roughly uniform contrast deficit in the difference panel is that
       effect, expected and already accounted for elsewhere; anything OTHER than a
       roughly uniform deficit (structure following the rings, one side brighter than
       the other) would be a finding, not this effect, and must be reported as one.

    Measured here: the raw deficit for THIS pair is about 70 percent, not the ~18
    percent that a single k(px) reading suggests, and the difference panel does show
    structure (it follows the marker's own ring pattern, brighter over the black
    cells), which is what a CONTRAST-scaled residual looks like, not an additive
    offset. Applying the k(px) correction to both ends only brings the gap to about
    30 percent (turbidity_summary.json's crossval.corrected reports a 3 percent
    MEDIAN and a 12 to 14 percent p90 over 5629 pairs), so this specific pair, the
    largest baseline for this marker, is a genuine outlier against the pooled
    statistic, not merely the pooled statistic's typical case drawn in pixels. The
    pooled median remains the right number to cite; this figure is the worst
    single case in the pool, not the typical one, and is reported as such.

    What this figure genuinely demonstrates, separate from the deficit percentage: the
    degraded panel reads bright cyan while the real far panel reads dark teal, a mean
    difference of about +48.5 DN. That is the model getting CONTRAST approximately
    right and DC badly wrong, exactly the decomposition claimed elsewhere: veiling
    light B fits at 241.9 DN, brighter than the white sheet is ever observed (about
    144 to 154 DN), so B alone is not physically plausible. Contrast is B-free by
    construction (see turbidity.py's module docstring), and it is contrast, not the
    absolute colour, that the crossval numbers above are measuring. This imagery
    disproves B as a standalone number while confirming the structure of the argument
    that separates B from beta.
    """
    fi_near = _frame_closest_to_range(poses, lay, mid, VALIDATION_NEAR_RANGE_M)
    fi_far = _frame_closest_to_range(poses, lay, mid, VALIDATION_FAR_RANGE_M)

    rv_n, tv_n = poses[fi_near]
    rv_f, tv_f = poses[fi_far]
    d_near = T.marker_ranges(lay, rv_n, tv_n)[mid]
    d_far = T.marker_ranges(lay, rv_f, tv_f)[mid]

    shape = (ci["height"], ci["width"])
    depth_near = T.plane_depth_map(lay, rv_n, tv_n, Km, shape)
    mask_near = T.board_mask(lay, rv_n, tv_n, Km, shape)

    beta_vec = np.array([beta_map["b"], beta_map["g"], beta_map["r"]])
    beta_grey = beta_map["grey"]
    dbeta = beta_vec * (d_far - d_near) / d_near
    dbeta_grey = beta_grey * (d_far - d_near) / d_near

    img_near_path = os.path.join(dataset_dir, RUN, "frames", f"{fi_near:06d}.png")
    img_far_path = os.path.join(dataset_dir, RUN, "frames", f"{fi_far:06d}.png")
    img_near = cv2.imread(img_near_path)
    img_far = cv2.imread(img_far_path)
    if img_near is None:
        raise FileNotFoundError(img_near_path)
    if img_far is None:
        raise FileNotFoundError(img_far_path)

    B_vec = np.array([B_map["b"], B_map["g"], B_map["r"]])
    degraded = T.synthesise(img_near, depth_near, mask_near, B_vec, dbeta)

    corners_near = _marker_corners(det, fi_near, mid)
    corners_far = _marker_corners(det, fi_far, mid)
    if corners_near is None or corners_far is None:
        raise ValueError(
            f"marker {mid} has no detection in frame {fi_near} or {fi_far}")
    px_near = detect.apparent_size_px(corners_near)
    px_far = detect.apparent_size_px(corners_far)

    raw_near_w = T.warp_marker(img_near, corners_near)
    deg_near_w = T.warp_marker(degraded, corners_near)
    far_w = T.warp_marker(img_far, corners_far)

    c_near = _grey_contrast(raw_near_w)
    c_deg = _grey_contrast(deg_near_w)
    c_far = _grey_contrast(far_w)
    deficit_pct = (c_deg - c_far) / c_far * 100.0

    deg_grey = cv2.cvtColor(deg_near_w, cv2.COLOR_BGR2GRAY).astype(float)
    far_grey = cv2.cvtColor(far_w, cv2.COLOR_BGR2GRAY).astype(float)
    diff = deg_grey - far_grey
    vlim = max(float(np.percentile(np.abs(diff), 99)), 1.0)

    # Titles are kept short on purpose: a long third line (e.g. spelling out
    # "downsampled"/"upsampled" per panel) overflows its own axes and bleeds into the
    # neighbouring panel's title at this width. That resampling note goes in the
    # shared caption below instead, where it has the whole figure width to sit in.
    fig, axes = plt.subplots(1, 4, figsize=(vizstyle.WIDE_W * 1.6, vizstyle.WIDE_W * 0.5))
    axes[0].imshow(cv2.cvtColor(raw_near_w, cv2.COLOR_BGR2RGB))
    axes[0].set_title(f"raw near\nrange {d_near:.2f} m, tau {beta_grey * d_near:.2f}\n"
                      f"contrast {c_near:.1f} DN, {px_near:.0f} px", fontsize=6.5)

    axes[1].imshow(cv2.cvtColor(deg_near_w, cv2.COLOR_BGR2RGB))
    axes[1].set_title("near degraded to tau_far\n"
                      f"dbeta_grey added {dbeta_grey:.3f} /m\n"
                      f"contrast {c_deg:.1f} DN, {px_near:.0f} px", fontsize=6.5)

    axes[2].imshow(cv2.cvtColor(far_w, cv2.COLOR_BGR2RGB))
    axes[2].set_title(f"real far\nrange {d_far:.2f} m, tau {beta_grey * d_far:.2f}\n"
                      f"contrast {c_far:.1f} DN, {px_far:.0f} px", fontsize=6.5)

    im = axes[3].imshow(diff, cmap="RdBu_r", vmin=-vlim, vmax=vlim)
    axes[3].set_title("difference (degraded minus real far)\n"
                      f"mean {float(diff.mean()):+.1f} DN, "
                      f"contrast deficit {deficit_pct:.0f} pct", fontsize=6.5)
    fig.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04, label="DN")

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(f"Synthesis illustrated: marker {mid} at the widest baseline "
                f"({d_far - d_near:.2f} m, frame {fi_near} to frame {fi_far})",
                fontsize=8, y=1.06)
    fig.text(0.5, -0.04,
             f"near panels resampled DOWN from {px_near:.0f} px; far panel resampled "
             f"UP from {px_far:.0f} px to the same canonical size (not a synthesis "
             "artifact)", ha="center", va="top", fontsize=6.5,
             color=vizstyle.TEXT_SECONDARY)
    fig.text(0.5, -0.10,
             "tail case, not typical: pooled corrected median is about -2.5 pct, p90 "
             "about 13.8 pct, over 5629 pairs; see crossval_distribution",
             ha="center", va="top", fontsize=6.5, color=vizstyle.TEXT_SECONDARY)
    fig.tight_layout()
    vizstyle.save(fig, "synthesis_validation")

    summary["synthesis_validation"] = {
        "marker_id": int(mid), "frame_near": int(fi_near), "frame_far": int(fi_far),
        "range_near_m": round(float(d_near), 3), "range_far_m": round(float(d_far), 3),
        "px_near_native": round(float(px_near), 1),
        "px_far_native": round(float(px_far), 1),
        "dbeta_grey_added": round(float(dbeta_grey), 4),
        "contrast_raw_near": round(c_near, 1), "contrast_degraded_near": round(c_deg, 1),
        "contrast_real_far": round(c_far, 1),
        "contrast_deficit_pct": round(float(deficit_pct), 1),
        "diff_mean_dn": round(float(diff.mean()), 2),
        "diff_std_dn": round(float(diff.std()), 2),
    }


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
    # Superseded pooled fit (measure_beta): kept for continuity with turbidity_beta.csv,
    # not the authoritative beta. That is beta_fixed_effects[ch]["corrected"] below.
    summary["beta_superseded_pooled"] = {
        r.channel: {"beta": round(r.beta, 3), "r2": round(r.r2, 3)}
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

    figure_beta(samples, responses)
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

    figure_sweep_strip(dataset_dir, poses, lay, Km, ci, beta_map, B_map, summary)
    figure_synthesis_validation(dataset_dir, det, poses, lay, Km, ci, beta_map, B_map,
                                summary)

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

    crossval(samples, beta_map, responses, summary)

    summary["limitations"] = [
        "The pool is already in every frame: multiplier 0 is tau 0.18 to 1.49 by "
        "range, not clear water. tau_total is the honest axis, not added turbidity.",
        "Veiling B's fit is bad enough to be physically incoherent, not merely "
        "uncertain: it fits with r2 negative (-0.57 to -2.72 across channels, from "
        "veiling_B in this summary; not the r2 about 0.5 once claimed for it), and "
        "B_green (241.9 DN) exceeds the white sheet's own observed brightness "
        "(about 154 DN near, 144 DN far); B_blue (225.8 DN) is the same story. The "
        "cause is that the corrected, lower beta shrinks x = 1 - exp(-beta*d) in "
        "fit_veiling, which inflates B = sum(xy)/sum(x^2) by roughly 29 percent. B "
        "is used quantitatively, as B_vec in synthesise, so this is not a dead "
        "parameter. The result survives it only because contrast is B-free by "
        "construction (see turbidity.py's module docstring) and B stays under 255, "
        "so synthesise never clips: the design's insensitivity to B was load-bearing "
        "here, not merely lucky. B sets only the DC level, which adaptive "
        "thresholding largely rejects, but at extreme tau contrast approaches "
        "quantisation and the model degrades.",
        "Synthesis is not new imagery: the turbidity levels from one frame are "
        "correlated samples, so the Wilson intervals here understate the true "
        "uncertainty across levels.",
        "Motion blur is present in the source frames (hand-held capture) and is not "
        "modelled.",
        "beta was fitted over the measured range span only; extrapolating tau beyond it "
        "assumes beta is constant with range.",
        "One pool, one session. The tau axis generalises; this water does not.",
        "The background outside the board mask is not degraded, so a synthesised frame "
        "is physically incoherent away from the board. ArUco thresholds locally against "
        "the sheet, which IS degraded, so this does not reach the decision.",
        "The pixel budget's apparent variation with tau in figure_px_required is "
        "sub-bin and not resolvable: every tau 0.18 to 1.75 point falls inside the "
        "single px bin interval [24.5, 33.0], 8.5 px wide, so no trend should be read "
        "from it beyond flatness up to tau 1.30 and collapse above tau 2.",
    ]

    with open("results/turbidity_summary.json", "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "dataset")
