
"""
Plot deadline violations, power, and latency with x-axis = job rate (λ).
Supports both classic per-GPU POWER exports and dvfs_energy_windows exports.

Outputs (under runs/<out>/):
  - latency_box_by_lambda.png
  - power_bar_by_lambda.png
  - violations_bar_by_lambda.png
"""

from __future__ import annotations
import argparse
import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import matplotlib.lines as mlines


# ------------------------------------------------------------------
# Boxplot / outlier settings
# ------------------------------------------------------------------
# Clip whiskers to these percentiles instead of 1.5 IQR
BOX_WHISKER_PERCENTILES = [1, 99]   # 1st–99th percentile
# Do not draw individual outlier points
BOX_SHOW_FLIERS = False


# ---------------- Regex patterns ----------------

SEED_RE     = re.compile(r"--seed\s+(\d+)\b")
LAM_RUN_RE  = re.compile(r"_L(?P<lam>\d+(?:\.\d+)?)__run(?P<run>\d+)", re.I)
L_SEED_RE   = re.compile(r"_L(?P<lam>\d+(?:_\d+)?)_seed(?P<seed>\d+)")
STRAT_RE = re.compile(
    r"__task_packets_summary_(?P<strat>.+?)_(?:freq-(?:adaptive|fixed))_",
    re.IGNORECASE,
)


COLOR_BY_STRAT = {
    "least-load_fixed":     "#4C78A8",
    "least-load_adaptive":  "#F58518",
    "random_fixed":         "#54A24B",
    "random_adaptive":      "#E45756",
    "opt_latency_fixed":    "#72B7B2",
    "opt_latency_adaptive": "#EECA3B",
    "opt_power_fixed":      "#B279A2",
    "opt_power_adaptive":   "#FF9DA6",
    "opt_efficiency_fixed": "#9C755F",
    "opt_efficiency_adaptive": "#BAB0AC",
}

COL_UP = "#1f77b4"   # blue
COL_GPU = "#ff7f0e"  # orange
COL_DL = "#2ca02c"   # green

# ---------------- Small helpers ----------------
def load_table(path: str) -> pd.DataFrame:
    p = Path(path)
    s = str(p).lower()
    if s.endswith(".parquet"):
        return pd.read_parquet(p)
    if s.endswith(".csv.gz") or s.endswith(".gz"):
        return pd.read_csv(p, compression="gzip")
    # plain csv
    return pd.read_csv(p)

def _col(df: pd.DataFrame, *cands: str, default=None):
    for c in cands:
        if c in df.columns:
            return c
    return default

def _num(s) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

def _read_text(p: Path) -> str:
    try:
        return p.read_text(errors="ignore")
    except Exception:
        return ""


# ---------------- λ helpers ----------------

def lambda_from_filename(p: Path) -> Tuple[float | None, int | None]:
    """
    Try to get λ and seed from filename tokens like:
      exp1_L3.5__run4__task_packets_summary_...
      ..._L3_5_seed44_...
    """
    m = L_SEED_RE.search(p.stem)
    if m:
        lam_str = m.group("lam").replace("_", ".")
        seed = int(m.group("seed"))
        try:
            return float(lam_str), seed
        except Exception:
            return None, seed

    m2 = LAM_RUN_RE.search(p.stem)
    if m2:
        try:
            return float(m2.group("lam")), None
        except Exception:
            return None, None

    return None, None

def build_seed_to_lambda(runs_dir: Path, prefix: str) -> Dict[int, float]:
    """
    Build mapping seed -> λ by scanning pregen logs:
      <prefix>_L<lam>__run<r>.pregen.log

    NOTE: If the same seed appears under multiple λ values, we WARN and skip that
    seed in the map. Callers should prefer λ from filenames and only use this
    mapping as a fallback.
    """
    pairs: List[Tuple[int, float]] = []
    for log in sorted(runs_dir.glob(f"{prefix}_L*__run*.pregen.log")):
        lam, _ = lambda_from_filename(log)
        if lam is None:
            m = LAM_RUN_RE.search(log.stem)
            if m:
                try:
                    lam = float(m.group("lam"))
                except Exception:
                    lam = None
        txt = _read_text(log)
        smo = SEED_RE.search(txt)
        if lam is not None and smo:
            seed = int(smo.group(1))
            pairs.append((seed, lam))

    if not pairs:
        print("[WARN] Could not build seed→λ mapping; loaders will rely on filenames.")
        return {}

    
    by_seed = defaultdict(set)
    for s, lam in pairs:
        by_seed[s].add(lam)

    mapping: Dict[int, float] = {}
    collisions = []
    for s, lset in sorted(by_seed.items()):
        if len(lset) == 1:
            mapping[s] = next(iter(lset))
        else:
            collisions.append((s, sorted(lset)))

    if collisions:
        print("[WARN] seed→λ collisions detected (seed used for multiple λ):", collisions)
        print("[WARN] For those seeds, filename-derived λ will be used instead of the map.")

    print("[INFO] seed→λ (unique) pairs:", sorted(mapping.items()))
    return mapping

# ---------------- File discovery ----------------

def find_latency_files(runs_dir: Path, prefix: str = "exp1") -> list[Path]:
    """
    Return ONLY the per-task summary CSVs (the new seed.csv equivalents).
    Exclude dvfs_energy_windows and *_violations.csv, etc.
    """
    # Grab all task_packets_summary_*_seed*.csv
    pat = f"{prefix}_L*__run*__task_packets_summary_*_seed*.csv"
    files = sorted(runs_dir.glob(pat))

    # Filter out non-task summaries (power windows, violations)
    clean = []
    for p in files:
        name = p.name
        if "dvfs_energy_windows" in name:
            continue
        if "per_gpu_POWER" in name:
            continue
        if "violations" in name:
            continue
        clean.append(p)

    return clean


def find_power_files(runs_dir: Path, prefix: str = "exp1") -> list[Path]:
    """
    Find the per-GPU DVFS energy window exports.
    Accept both new and legacy naming.
    """
    pats = [
        # NEW naming (no _latency_)
        f"{prefix}_L*__run*__task_packets_summary_*_dvfs_energy_windows.csv",

        # legacy fallbacks if they still exist in folder
        f"{prefix}_L*__run*__task_packets_summary_*_latency_*_dvfs_energy_windows.csv",
        f"{prefix}_L*__run*__task_packets_summary_*_latency_per_gpu_POWER.csv",
        f"{prefix}_L*__run*__task_packets_summary_*_per_gpu_POWER.csv",
    ]

    out: list[Path] = []
    for pat in pats:
        out.extend(sorted(runs_dir.glob(pat)))

    # de-dup
    seen = set()
    uniq = []
    for p in out:
        if p not in seen:
            seen.add(p)
            uniq.append(p)
    return uniq


def find_link_stats_files(runs_dir: Path, prefix: str = "exp1") -> list[Path]:
    """
    Find the per-link utilization summary files:
        <prefix>_L*__run*__task_packets_summary_*_link_stats.csv
    """
    pat = f"{prefix}_L*__run*__task_packets_summary_*_link_stats.csv"
    return sorted(runs_dir.glob(pat))


def read_gpu_utilization(
    files: Iterable[Path],
    seed_to_lambda: Dict[int, float],
) -> pd.DataFrame:
    """
    Returns DF: Lambda, Lambda_per_s, Strategy, GPU_Util (mean over file)
    """
    rows: List[Dict[str, object]] = []
    for p in files:
        strat = parse_strategy_from_filename(p.name)

        lam, _ = lambda_from_filename(p)
        if lam is None:
            m = re.search(r"seed(\d+)", p.name)
            lam = seed_to_lambda.get(int(m.group(1))) if m else None
        if lam is None:
            continue

        try:
            df = pd.read_csv(
                p,
                dtype=str,
                low_memory=False,
                usecols=lambda c: c.lower() in {
                    "utilization_u", "gpu_util", "util", "utilization"
                },
            )
        except Exception:
            continue

        util_col = None
        for c in df.columns:
            if c.lower() in {"utilization_u", "gpu_util", "util", "utilization"}:
                util_col = c
                break
        if util_col is None:
            continue

        util = pd.to_numeric(df[util_col], errors="coerce").dropna()
        if util.empty:
            continue

        rows.append({
            "Lambda": lam,
            "Lambda_per_s": lam,
            "Strategy": strat,
            "GPU_Util": util.mean(),
        })

    if not rows:
        return pd.DataFrame(columns=["Lambda", "Lambda_per_s", "Strategy", "GPU_Util"])
    return pd.DataFrame(rows)


def read_link_utilization(
    files: Iterable[Path],
    seed_to_lambda: Dict[int, float],
) -> pd.DataFrame:
    """
    Expects link_stats CSVs, returns:
        Lambda, Link_ID, util   (0–1)
    """
    rows: List[Dict[str, object]] = []

    for p in files:
        try:
            df = pd.read_csv(
                p,
                dtype=str,
                low_memory=False,
                usecols=lambda c: c.lower() in {"lambda_per_s", "link_id", "util", "utilization"},
            )
        except Exception:
            continue

        strat = parse_strategy_from_filename(p.name)  # unused, but harmless

        cols = {c.lower(): c for c in df.columns}
        lam_col  = cols.get("lambda_per_s")
        link_col = cols.get("link_id")
        util_col = cols.get("util") or cols.get("utilization")

        if lam_col is None or link_col is None or util_col is None:
            continue

        df["Lambda"] = pd.to_numeric(df[lam_col], errors="coerce")
        df["util"]   = pd.to_numeric(df[util_col], errors="coerce")

        base = df[["Lambda", link_col, "util"]].dropna()
        base = base.rename(columns={link_col: "Link_ID"})
        if not base.empty:
            rows.append(base)

    if not rows:
        return pd.DataFrame(columns=["Lambda", "Link_ID", "util"])

    return pd.concat(rows, ignore_index=True)


# ---------------- Parsing helpers ----------------

def parse_strategy_from_filename(name: str) -> str:
    m = STRAT_RE.search(name)
    if m:
        return m.group("strat")
    # Fallback: try manual split
    try:
        after = name.split("__task_packets_summary_", 1)[1]
        # e.g. "least-load_fixed_freq-fixed_adm-soft_L3_5_seed41..."
        return after.split("_freq-", 1)[0]
    except Exception:
        return "unknown"

def _ensure_lambda_col(df: pd.DataFrame) -> pd.DataFrame:
    """Guarantee a 'Lambda' column; mirror from 'Lambda_per_s' if present."""
    if "Lambda" not in df.columns and "Lambda_per_s" in df.columns:
        df = df.copy()
        df["Lambda"] = df["Lambda_per_s"]
    return df

# ---------------- Readers ----------------

def read_latency_samples(
    files: Iterable[Path],
    seed_to_lambda: Dict[int, float],
    sample_n: int = 500_000,
) -> pd.DataFrame:
    """
    Load per-task summaries and build a unified latency sample table:

        Lambda, Strategy, Freq_Mode, Admission, Latency, (delay breakdown...)

    Latency comes from `latency_s` / `Latency_s` or end-start.
    """
    rows = []

    # only columns we ever use in plotting/violations
    BASE_COLS = {
        "Lambda", "Lambda_per_s", "Strategy", "Freq_Mode", "Admission",
        "latency_s", "Latency_s", "Latency",
        "start_time", "end_time",
        "ul_delay_s", "gpu_delay_s", "dl_delay_s",
        "deadline_violation", "violation_cause",
    }

    for p in files:
        try:
            df = pd.read_csv(
                p,
                dtype=str,
                low_memory=False,
                usecols=lambda c: True if c in BASE_COLS else False,
            )
        except Exception:
            continue

        # 1. Lambda from filename or seed
        lam, _ = lambda_from_filename(p)
        if lam is not None:
            df["Lambda"] = float(lam)
            df["Lambda_per_s"] = float(lam)
        if "Lambda" not in df.columns:
            m = re.search(r"seed(\d+)", p.name)
            if m:
                seed_guess = int(m.group(1))
                lam_guess = seed_to_lambda.get(seed_guess)
                if lam_guess is not None:
                    df["Lambda"] = float(lam_guess)
                    df["Lambda_per_s"] = float(lam_guess)

        df = _ensure_lambda_col(df)

        # 2. Strategy / Freq_Mode / Admission
        if "Strategy" not in df.columns:
            df["Strategy"] = parse_strategy_from_filename(p.stem)

        if "Freq_Mode" not in df.columns:
            if "_freq-fixed_" in p.stem:
                df["Freq_Mode"] = "freq-fixed"
            elif "_freq-adaptive_" in p.stem:
                df["Freq_Mode"] = "freq-adaptive"
            else:
                df["Freq_Mode"] = "unknown"

        if "Admission" not in df.columns:
            m_adm = re.search(r"_adm-([^-_]+)", p.stem)
            df["Admission"] = f"adm-{m_adm.group(1)}" if m_adm else "adm-unknown"

        # 3. Latency → ONE numeric conversion
        if "Latency" in df.columns:
            df["Latency"] = pd.to_numeric(df["Latency"], errors="coerce")
        elif "latency_s" in df.columns:
            df["Latency"] = pd.to_numeric(df["latency_s"], errors="coerce")
        elif "Latency_s" in df.columns:
            df["Latency"] = pd.to_numeric(df["Latency_s"], errors="coerce")
        elif {"start_time", "end_time"} <= set(df.columns):
            df["Latency"] = (
                pd.to_numeric(df["end_time"], errors="coerce")
                - pd.to_numeric(df["start_time"], errors="coerce")
            )
        else:
            continue

        keep_cols = [
            "Lambda", "Strategy", "Freq_Mode", "Admission",
            "Latency",
            "ul_delay_s", "gpu_delay_s", "dl_delay_s",
            "deadline_violation", "violation_cause",
        ]
        base = df[[c for c in keep_cols if c in df.columns]].dropna(subset=["Latency"])
        if not base.empty:
            rows.append(base)

    if not rows:
        return pd.DataFrame(columns=[
            "Lambda", "Strategy", "Freq_Mode", "Admission",
            "Latency",
            "ul_delay_s", "gpu_delay_s", "dl_delay_s",
            "deadline_violation", "violation_cause",
        ])

    all_lat = pd.concat(rows, ignore_index=True)

    # downsample early so later plots are cheap
    if len(all_lat) > sample_n:
        all_lat = all_lat.sample(sample_n, random_state=17)

    return all_lat


def read_power(files: Iterable[Path], seed_to_lambda: Dict[int, float]) -> pd.DataFrame:
    rows = []
    for p in files:
        strat = parse_strategy_from_filename(p.name)

        lam, _ = lambda_from_filename(p)
        if lam is None:
            m = re.search(r"seed(\d+)", p.name)
            lam = seed_to_lambda.get(int(m.group(1))) if m else None
        if lam is None:
            continue

        try:
            df = pd.read_csv(
                p,
                dtype=str,
                low_memory=False,
                usecols=lambda c: c.lower() in {
                    "p_window_avg_w", "e_total_j", "duration",
                    "p_active_w", "avg_power_w", "average_power_w", "power_w",
                },
            )
        except Exception:
            continue

        lower = {c.lower(): c for c in df.columns}
        power_col = None
        if "p_window_avg_w" in lower:
            power_col = lower["p_window_avg_w"]
        elif "e_total_j" in lower and "duration" in lower:
            e = pd.to_numeric(df[lower["e_total_j"]], errors="coerce")
            d = pd.to_numeric(df[lower["duration"]], errors="coerce").replace(0, pd.NA)
            df["__avg_power__"] = e / d
            power_col = "__avg_power__"
        else:
            for c in df.columns:
                if c.lower() in {"p_active_w", "avg_power_w", "average_power_w", "power_w"}:
                    power_col = c
                    break

        if power_col is None:
            continue

        pw = pd.to_numeric(df[power_col], errors="coerce").dropna()
        if pw.empty:
            continue

        rows.append({
            "Lambda": lam,
            "Lambda_per_s": lam,
            "Strategy": strat,
            "Power": pw.mean(),
        })

    if not rows:
        return pd.DataFrame(columns=["Lambda", "Lambda_per_s", "Strategy", "Power"])
    return pd.DataFrame(rows)

# ---------------- Plotting ----------------

def _ensure_lambda_order(vals: Iterable[float], target=(4.0, 5.0, 6.0, 7.0)) -> List[float]:
    avals = sorted(set(float(x) for x in vals if pd.notna(x)))
    wanted = [x for x in target if x in avals]
    return wanted or avals

def plot_latency_breakdown_stacked(
    df: pd.DataFrame,
    out_path: Path,
    violated_only: bool = False,
) -> None:
    """
    Stacked bar of high-percentile UL / GPU / DL delay per (λ, strategy).

    Uses the 99th percentile of per-task delays within each (Lambda, Strategy).
    If violated_only=True, only tasks with deadline_violation > 0 are used.
    """
    df = _ensure_lambda_col(df)
    required = {"ul_delay_s", "gpu_delay_s", "dl_delay_s"}
    if df.empty or not required.issubset(df.columns):
        print("[WARN] no delay breakdown columns for stacked plot.")
        return

    # optional filter: violations only
    if violated_only:
        if "deadline_violation" in df.columns:
            v = pd.to_numeric(df["deadline_violation"], errors="coerce").fillna(0)
            df = df[v > 0]
        else:
            print("[WARN] violated_only=True but 'deadline_violation' missing.")
            return

    if df.empty:
        print("[WARN] no rows for latency breakdown plot after filtering.")
        return

    lambdas = _ensure_lambda_order(df["Lambda"].unique(), target=(4.0, 5.0, 6.0, 7.0))
    strategies = sorted(df["Strategy"].dropna().unique())
    if not lambdas or not strategies:
        print("[WARN] missing Lambda/Strategy for latency breakdown plot.")
        return

    # --- fixed colors (match legend) ---
    COL_UP  = "#1f77b4"  # blue
    COL_GPU = "#ff7f0e"  # orange
    COL_DL  = "#2ca02c"  # green

    # --- ensure delay columns exist and are numeric floats ---
    delay_cols = ["ul_delay_s", "gpu_delay_s", "dl_delay_s"]

    df = df.copy()
    
    for col in delay_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
        else:
            # create as float column of NaNs so groupby().quantile() still works
            df[col] = np.nan

    df_num = df.dropna(subset=delay_cols, how="all")
    if df_num.empty:
        print("[WARN] no numeric delay rows for latency breakdown plot.")
        return

    # 99th percentile per (Lambda, Strategy)
    agg = (
        df_num
        .groupby(["Lambda", "Strategy"])[delay_cols]
        .quantile(0.99)
        .reset_index()
    )

    comp_map = {
        (float(r["Lambda"]), r["Strategy"]): (
            float(r["ul_delay_s"]),
            float(r["gpu_delay_s"]),
            float(r["dl_delay_s"]),
        )
        for _, r in agg.iterrows()
    }

    stride, cluster_width, cluster_gap, bar_width = \
        _cluster_layout_params(len(strategies))[:4]

    fig, ax = plt.subplots(figsize=(14, 6))
    for k, lam in enumerate(lambdas):
        base_x = k * cluster_gap
        for i, s in enumerate(strategies):
            key = (lam, s)
            if key not in comp_map:
                continue
            ul_d, gpu_d, dl_d = comp_map[key]
            x = base_x + i * stride

            # UL segment
            ax.bar(
                x, ul_d,
                width=bar_width,
                color=COL_UP,
                edgecolor="black",
                linewidth=0.6,
                label="_nolegend_",
            )
            # GPU segment
            ax.bar(
                x, gpu_d,
                width=bar_width,
                bottom=ul_d,
                color=COL_GPU,
                edgecolor="black",
                linewidth=0.6,
                label="_nolegend_",
            )
            # DL segment
            ax.bar(
                x, dl_d,
                width=bar_width,
                bottom=ul_d + gpu_d,
                color=COL_DL,
                edgecolor="black",
                linewidth=0.6,
                label="_nolegend_",
            )

    centers = [k * cluster_gap + cluster_width / 2.0 for k in range(len(lambdas))]
    ax.set_xticks(centers, [f"λ={lam}" for lam in lambdas])
    ax.set_ylabel("Latency (s) — 99th percentile")

    title = "Latency breakdown (UL + GPU + DL) by Strategy and λ (99th pct)"
    if violated_only:
        title += " — violated tasks only"
    ax.set_title(title)

    # legend: colors exactly match the segments
    legend_handles = [
        mpatches.Patch(label="Uplink",   facecolor=COL_UP,  ec="black"),
        mpatches.Patch(label="GPU",      facecolor=COL_GPU, ec="black"),
        mpatches.Patch(label="Downlink", facecolor=COL_DL,  ec="black"),
    ]
    ax.legend(handles=legend_handles, frameon=False, loc="upper right")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] wrote {out_path}")

def plot_latency_boxes(df: pd.DataFrame, out_path: Path) -> None:
    df = _ensure_lambda_col(df)

    if df.empty or "Lambda" not in df.columns:
        print("[WARN] no latency rows to plot (missing 'Lambda')")
        return

    if "Latency" not in df.columns:
        print("[WARN] no latency column to plot (expected 'Latency')")
        return

    # Avoid NaN strategy names
    strategies = sorted(df["Strategy"].dropna().unique())
    if not strategies:
        print("[WARN] no strategies found for latency box plot.")
        return

    lambdas = _ensure_lambda_order(df["Lambda"].unique())

    cluster_gap = 1.5
    box_width = 0.7
    boxes_data, positions, colors = [], [], []
    xpos = 0.0

    for lam in lambdas:
        sub = df[df["Lambda"] == lam]
        n = max(1, len(strategies))
        for i, s in enumerate(strategies):
            y = _num(sub[sub["Strategy"] == s]["Latency"]).dropna().values
            if y.size == 0:
                continue
            pos = xpos + (i - (n - 1) / 2.0) * (box_width + 0.05)
            boxes_data.append(y)
            positions.append(pos)
            colors.append(COLOR_BY_STRAT.get(s, "#777"))
        xpos += cluster_gap

    if not boxes_data:
        print("[WARN] no latency samples found for requested λ/strategy sets.")
        return

    fig, ax = plt.subplots(figsize=(11, 5))
    bp = ax.boxplot(
        boxes_data,
        positions=positions,
        widths=box_width * 0.9,
        manage_ticks=False,
        showfliers=BOX_SHOW_FLIERS,          # uses global toggle
        whis=BOX_WHISKER_PERCENTILES,        # uses percentile config
        patch_artist=True,
    )

    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_edgecolor("black")
    for elem in ("whiskers", "caps", "medians"):
        for l in bp[elem]:
            l.set_color("black")

    # x tick labels at cluster centers
    centers = []
    xpos = 0.0
    for _ in lambdas:
        centers.append(xpos)
        xpos += cluster_gap
    ax.set_xticks(centers, [f"λ={lam}" for lam in lambdas])

    ax.set_ylabel("Latency (s)")
    ax.set_title("Latency distribution by Strategy and λ")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[OK] wrote {out_path}")


# --- two-panel latency plot (λ=3.5 and λ=4.0) with strategies as legend ---
def plot_latency_two_panels(df: pd.DataFrame, out_path: Path) -> None:
    df = _ensure_lambda_col(df)
    if df.empty or "Lambda" not in df.columns or "Latency" not in df.columns:
        print("[WARN] no latency data to plot for two-panel view.")
        return

    # Which lambdas to plot (will show up to 2; extend cols if you have more)
    lambdas = sorted(x for x in set(df["Lambda"]) if pd.notna(x))
    if not lambdas:
        print("[WARN] no Lambda values found.")
        return
    if len(lambdas) > 2:
        # keep first two by default; tweak if you want more subplots
        lambdas = lambdas[:4]

    strategies = sorted(df["Strategy"].dropna().unique())
    if not strategies:
        print("[WARN] no Strategy values found.")
        return

    fig, axes = plt.subplots(1, len(lambdas), figsize=(12, 5), sharey=True)

    if len(lambdas) == 1:
        axes = [axes]  # normalize to list

    # build legend handles once, using color map
    handles = [
        mpatches.Patch(label=s, color=COLOR_BY_STRAT.get(s, "#777"), ec="black")
        for s in strategies
    ]

    for ax, lam in zip(axes, lambdas):
        sub = df[df["Lambda"] == lam].copy()
        # x positions are 1..S but we hide labels; legend shows strategy names
        positions = np.arange(1, len(strategies) + 1, dtype=float)
        box_artists = []

        # create one box per strategy so we can color them
        for i, s in enumerate(strategies, start=1):
            y = _num(sub.loc[sub["Strategy"] == s, "Latency"]).dropna().values
            if y.size == 0:
                # still reserve a placeholder, or skip position entirely
                continue
            bp = ax.boxplot(
                [y], positions=[i], widths=0.7,
                manage_ticks=False, showfliers=False, patch_artist=True
            )
            # color + edges
            col = COLOR_BY_STRAT.get(s, "#777")
            for patch in bp["boxes"]:
                patch.set_facecolor(col)
                patch.set_edgecolor("black")
            for elem in ("whiskers", "caps", "medians"):
                for l in bp[elem]:
                    l.set_color("black")
            box_artists.append(bp)

        ax.set_title(f"λ={lam}")
        ax.set_xticks([])  # no per-strategy x labels; legend handles it
        ax.grid(False)

    axes[0].set_ylabel("Latency (s)")
    fig.suptitle("Latency distribution by Strategy and λ", y=0.98)

    # build legend once
    handles = [
        mpatches.Patch(label=s, color=COLOR_BY_STRAT.get(s, "#777"), ec="black")
        for s in strategies
    ]
    fig.legend(
        handles=handles, title="Strategy",
        bbox_to_anchor=(1.02, 0.5), loc="center left",
        borderaxespad=0.0, frameon=False
    )
    fig.tight_layout(rect=[0, 0, 0.87, 1])  # leave room at right
    fig.savefig(out_path, dpi=150, bbox_inches="tight") 
    plt.close(fig)
    print(f"[OK] wrote {out_path}")


def plot_latency_two_panels_new(df: pd.DataFrame, out_path: Path) -> None:
    df = _ensure_lambda_col(df)
    if df.empty or "Lambda" not in df.columns or "Latency" not in df.columns:
        print("[WARN] no latency data to plot for multi-panel view.")
        return

    # Which lambdas to plot (show up to 4)
    lambdas = sorted(x for x in set(df["Lambda"]) if pd.notna(x))
    if not lambdas:
        print("[WARN] no Lambda values found.")
        return
    if len(lambdas) > 4:
        lambdas = lambdas[:4]

    if "Strategy" not in df.columns:
        print("[WARN] no Strategy column found.")
        return
    strategies = sorted(df["Strategy"].dropna().unique())
    if not strategies:
        print("[WARN] no Strategy values found.")
        return

    fig, axes = plt.subplots(
        1, len(lambdas),
        figsize=(4 * len(lambdas), 5),
        sharey=False,   # <- independent y-scale per λ
    )

    if len(lambdas) == 1:
        axes = [axes]  # normalize to list

    # legend handles (once)
    handles = [
        mpatches.Patch(label=s, color=COLOR_BY_STRAT.get(s, "#777"), ec="black")
        for s in strategies
    ]

    for ax, lam in zip(axes, lambdas):
        sub = df[df["Lambda"] == lam].copy()
        if sub.empty:
            ax.set_visible(False)
            continue

        # one box per strategy so we can color them
        for i, s in enumerate(strategies, start=1):
            y = _num(sub.loc[sub["Strategy"] == s, "Latency"]).dropna().values
            if y.size == 0:
                continue

            bp = ax.boxplot(
                [y],
                positions=[i],
                widths=0.7,
                manage_ticks=False,
                # showfliers=BOX_SHOW_FLIERS,
                showfliers=(lam > 5.0),   # hide fliers for λ=4,5; show for 6,7
                whis=BOX_WHISKER_PERCENTILES,
                patch_artist=True,
            )

            col = COLOR_BY_STRAT.get(s, "#777")
            for patch in bp["boxes"]:
                patch.set_facecolor(col)
                patch.set_edgecolor("black")
            for elem in ("whiskers", "caps", "medians"):
                for l in bp[elem]:
                    l.set_color("black")

        # --- dynamic y-limits per λ, with zoom for λ <= 5 ---
        lat_all = _num(sub["Latency"]).dropna().values
        if lat_all.size > 0:
            if lam <= 5.0:
                # Ignore very rare huge values when choosing the axis range
                hi = np.percentile(lat_all, 99)   # or 95 if you want more zoom
                lo = np.percentile(lat_all, 1)

                # a bit of padding
                pad = 0.1 * max(hi - lo, 1e-3)
                ymin = max(0.0, float(lo) - pad)
                ymax = float(hi) + pad
            else:
                # high lambdas keep full range
                lo = float(lat_all.min())
                hi = float(lat_all.max())
                pad = 0.1 * max(hi - lo, 1e-3)
                ymin = max(0.0, lo - pad)
                ymax = hi + pad

            if ymin == ymax:
                ymax = ymin + 1e-3
            ax.set_ylim(ymin, ymax)



        ax.set_title(f"λ={lam}")
        ax.set_xticks([])    # legend explains strategies
        ax.grid(False)

    axes[0].set_ylabel("Latency (s)")
    fig.suptitle("Latency distribution by Strategy and λ", y=0.98)

    fig.legend(
        handles=handles,
        title="Strategy",
        bbox_to_anchor=(1.02, 0.5),
        loc="center left",
        borderaxespad=0.0,
        frameon=False,
    )

    fig.tight_layout(rect=[0, 0, 0.87, 1])
    plt.subplots_adjust(wspace=0.4)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] wrote {out_path}")

def plot_latency_two_panels_without_outliers(df: pd.DataFrame, out_path: Path) -> None:
    """
    Multi-panel latency boxplot by λ and Strategy.

    - One panel per Lambda (up to 4: 4,5,6,7)
    - Outliers hidden
    - Y-limits per panel based on 5–95% latency band (plus padding)
    """
    df = _ensure_lambda_col(df)
    if df.empty or "Lambda" not in df.columns or "Latency" not in df.columns:
        print("[WARN] no latency data to plot for multi-panel view (no outliers).")
        return

    lambdas = sorted(x for x in set(df["Lambda"]) if pd.notna(x))
    if not lambdas:
        print("[WARN] no Lambda values found.")
        return
    if len(lambdas) > 4:
        lambdas = lambdas[:4]

    if "Strategy" not in df.columns:
        print("[WARN] no Strategy column found.")
        return
    strategies = sorted(df["Strategy"].dropna().unique())
    if not strategies:
        print("[WARN] no Strategy values found.")
        return

    # one column per lambda
    fig, axes = plt.subplots(
        1, len(lambdas),
        figsize=(4 * len(lambdas), 5.5),
        sharey=False,
    )
    if len(lambdas) == 1:
        axes = [axes]

    # legend handles
    handles = [
        mpatches.Patch(label=s, color=COLOR_BY_STRAT.get(s, "#777"), ec="black")
        for s in strategies
    ]

    for ax, lam in zip(axes, lambdas):
        sub = df[df["Lambda"] == lam].copy()
        if sub.empty:
            ax.set_visible(False)
            continue

        for i, s in enumerate(strategies, start=1):
            y = _num(sub.loc[sub["Strategy"] == s, "Latency"]).dropna().values
            if y.size == 0:
                continue

            bp = ax.boxplot(
                [y],
                positions=[i],
                widths=0.7,
                manage_ticks=False,
                showfliers=False,          # hide outliers
                whis=[5, 95],              # whiskers at 5th / 95th percentiles
                patch_artist=True,
            )

            col = COLOR_BY_STRAT.get(s, "#777")
            for patch in bp["boxes"]:
                patch.set_facecolor(col)
                patch.set_edgecolor("black")
            for elem in ("whiskers", "caps", "medians"):
                for l in bp[elem]:
                    l.set_color("black")

        # y-limits from 5–95% band with padding
        lat_all = _num(sub["Latency"]).dropna().values
        if lat_all.size > 0:
            lo = np.percentile(lat_all, 5)
            hi = np.percentile(lat_all, 95)
            span = hi - lo
            pad = 0.1 * max(span, 1e-3)
            ymin = max(0.0, lo - pad)
            ymax = hi + pad
            if ymin == ymax:
                ymax = ymin + 1e-3
            ax.set_ylim(ymin, ymax)

        ax.set_title(f"λ={lam}")
        ax.set_xticks([])   # keep x clean; legend carries strategy labels
        ax.grid(False)

    axes[0].set_ylabel("Latency (s)")
    fig.suptitle("Latency distribution by Strategy and λ", y=0.98)

    fig.legend(
        handles=handles,
        title="Strategy",
        bbox_to_anchor=(1.02, 0.5),
        loc="center left",
        borderaxespad=0.0,
        frameon=False,
    )

    fig.tight_layout(rect=[0, 0, 0.87, 1])
    plt.subplots_adjust(wspace=0.4)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] wrote {out_path}")

# --- two-panel power plot with mean ± std error bars ---
def plot_power_mean_std_two_panels(df: pd.DataFrame, out_path: Path) -> None:
    df = _ensure_lambda_col(df)
    if df.empty or "Lambda" not in df.columns or "Power" not in df.columns:
        print("[WARN] no power data to plot for two-panel view.")
        return

    # Which lambdas to show (two panels)
    lambdas = _ensure_lambda_order(df["Lambda"].unique(), target=(4.0, 5.0, 6.0, 7.0, 8.0))
    if not lambdas:
        print("[WARN] no Lambda values found.")
        return
    if len(lambdas) > 2:
        lambdas = lambdas[:4]

    # Union of strategies across both lambdas (keep a stable order)
    strategies = sorted(df["Strategy"].dropna().unique())
    if not strategies:
        print("[WARN] no Strategy values found.")
        return

    # Aggregate mean/std per (λ, strategy)
    agg = (df.groupby(["Lambda", "Strategy"])["Power"]
             .agg(["mean", "std", "count"])
             .reset_index())
    # if a strategy appears only once at a given λ, std will be NaN → set to 0
    agg["std"] = agg["std"].fillna(0.0)

    # Build lookups for fast access
    mean_map = {(row["Lambda"], row["Strategy"]): row["mean"] for _, row in agg.iterrows()}
    std_map  = {(row["Lambda"], row["Strategy"]): row["std"]  for _, row in agg.iterrows()}

    fig, axes = plt.subplots(1, len(lambdas), figsize=(12, 5), sharey=False)

    if len(lambdas) == 1:
        axes = [axes]

    handles = [
        mpatches.Patch(label=s, color=COLOR_BY_STRAT.get(s, "#777"), ec="black")
        for s in strategies
    ]

    for ax, lam in zip(axes, lambdas):
        x = np.arange(len(strategies), dtype=float)
        means = np.array([mean_map.get((lam, s), np.nan) for s in strategies], dtype=float)
        stds  = np.array([std_map.get((lam, s), 0.0) for s in strategies], dtype=float)

        # Some (λ, strategy) may be missing → mask them out so matplotlib doesn’t plot NaNs
        mask = ~np.isnan(means)
        ax.bar(x[mask],
               means[mask],
               yerr=stds[mask],
               capsize=3,
               linewidth=0.6,
               edgecolor="black",
               color=[COLOR_BY_STRAT.get(s, "#777") for s, m in zip(strategies, mask) if m])

        ax.set_title(f"λ={lam}")
        ax.set_xticks([])  # strategies are shown in the legend
        ax.set_ylim(bottom=0)
        ax.grid(False)

    axes[0].set_ylabel("Average Power (W)")
    fig.suptitle("Average Power by Strategy and λ", y=0.98)

    # Single legend to the right
    fig.legend(handles=handles, title="Strategy",
               bbox_to_anchor=(1.02, 0.5), loc="center left",
               borderaxespad=0.0, frameon=False)
    fig.tight_layout(rect=[0, 0, 0.87, 1])  # room for legend
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] wrote {out_path}")

def read_violations(files: Iterable[Path], seed_to_lambda: Dict[int, float]) -> pd.DataFrame:
    """
    Return ONE ROW PER FILE (run) with:
        Lambda, Strategy, Violations, Total_Tasks
    """
    rows: List[Dict[str, object]] = []

    for p in files:
        try:
            df = pd.read_csv(p, dtype=str, low_memory=False)
        except Exception:
            continue

        strat = parse_strategy_from_filename(p.name)

        # Prefer λ from filename, then seed map
        lam, _ = lambda_from_filename(p)
        if lam is None:
            m = re.search(r"seed(\d+)", p.name)
            lam = seed_to_lambda.get(int(m.group(1))) if m else None
        if lam is None:
            continue

        # find a violation column (row-level boolean/0–1)
        vcol = None
        for c in df.columns:
            cl = c.lower()
            if cl in {
                "deadline_violation",          
                "violations",
                "deadline_violations",
                "task_deadline_violation",
            }:
                vcol = c
                break

        if vcol is None:
            # no recognizable per-row violation flag; skip file
            continue

        v = pd.to_numeric(df[vcol], errors="coerce").fillna(0)
        count = int((v > 0).sum())
        total = int(len(df))   # one row per task in *_task_packets_summary

        rows.append({
            "Lambda":      float(lam),
            "Strategy":    strat,
            "Violations":  count,
            "Total_Tasks": total,
        })

    if not rows:
        return pd.DataFrame(columns=["Lambda", "Strategy", "Violations", "Total_Tasks"])
    return pd.DataFrame(rows)


# --- two-panel violations plot (counts) with strategies legend ---
def plot_violations_two_panels(df: pd.DataFrame, out_path: Path) -> None:
    df = _ensure_lambda_col(df)
    # expect df to be the aggregated output from read_violations: [Lambda, Strategy, Violations]
    if df.empty or "Lambda" not in df.columns or "Violations" not in df.columns:
        print("[WARN] no violations data to plot for two-panel view.")
        return

    lambdas = _ensure_lambda_order(df["Lambda"].unique(), target=(4.0, 5.0, 6.0, 7.0))
    if not lambdas:
        print("[WARN] no Lambda values found for violations plot.")
        return
    if len(lambdas) > 2:
        lambdas = lambdas[:4]

    strategies = sorted(df["Strategy"].dropna().unique())
    if not strategies:
        print("[WARN] no Strategy values found for violations plot.")
        return

    # map (λ, strategy) -> count; fill missing with 0
    vmap = {(float(row["Lambda"]), row["Strategy"]): int(row["Violations"])
            for _, row in df.iterrows()}

    fig, axes = plt.subplots(1, len(lambdas), figsize=(12, 5), sharey=True)
    if len(lambdas) == 1:
        axes = [axes]

    handles = [
        mpatches.Patch(label=s, color=COLOR_BY_STRAT.get(s, "#777"), ec="black")
        for s in strategies
    ]

    for ax, lam in zip(axes, lambdas):
        x = np.arange(len(strategies), dtype=float)
        vals = np.array([vmap.get((lam, s), 0) for s in strategies], dtype=float)

        ax.bar(
            x, vals,
            edgecolor="black", linewidth=0.6,
            color=[COLOR_BY_STRAT.get(s, "#777") for s in strategies],
        )
        ax.set_title(f"λ={lam}")
        ax.set_xticks([])  # strategies shown in legend
        ax.set_ylim(bottom=0)
        ax.grid(False)

    axes[0].set_ylabel("Violations (count)")
    fig.suptitle("Deadline Violations by Strategy and λ", y=0.98)

    fig.legend(
        handles=handles, title="Strategy",
        bbox_to_anchor=(1.02, 0.5), loc="center left",
        borderaxespad=0.0, frameon=False
    )
    fig.tight_layout(rect=[0, 0, 0.87, 1])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] wrote {out_path}")


# Shared helpers for spacing/legend -------------------------------------------
def _cluster_layout_params(n_strategies: int):
    # spacing that scales with number of strategies
    stride = 1.0                      # gap between strategies inside a cluster
    cluster_width = (n_strategies - 1) * stride
    gutter = max(2.5, 0.25 * n_strategies)  # gap between clusters
    cluster_gap = cluster_width + gutter
    box_width = 0.75                  # for latency boxes
    bar_width = 0.8                   # for bars
    return stride, cluster_width, cluster_gap, box_width, bar_width

def _bottom_legend(fig, handles_list):
    if not handles_list:
        return
    labels_list = [h.get_label() for h in handles_list]
    fig.legend(handles=handles_list, labels=labels_list, title="Strategy",
               loc="lower center", ncol=min(5, len(handles_list)),
               frameon=False, borderaxespad=0.0, bbox_to_anchor=(0.5, -0.02))


def plot_latency_oneaxis(df: pd.DataFrame, out_path: Path) -> None:
    df = _ensure_lambda_col(df)
    if df.empty or "Lambda" not in df.columns or "Latency" not in df.columns:
        print("[WARN] no latency data to plot.")
        return

    # Convert to numeric; 
    df = df.copy()
    df["Latency"] = pd.to_numeric(df["Latency"], errors="coerce")
    df = df.dropna(subset=["Latency"])

    lambdas = _ensure_lambda_order(df["Lambda"].unique(), target=(4.0, 5.0, 6.0, 7.0))
    if not lambdas:
        print("[WARN] no Lambda values found.")
        return
    if len(lambdas) > 4:
        lambdas = lambdas[:4]

    strategies = sorted(df["Strategy"].dropna().unique())
    if not strategies:
        print("[WARN] no Strategy values found.")
        return

    # Group raw latency samples
    grouped = {
        (lam, strat): df.loc[
            (df["Lambda"] == lam) & (df["Strategy"] == strat),
            "Latency",
        ].to_numpy()
        for lam in lambdas for strat in strategies
    }

    stride, cluster_width, cluster_gap, box_width, _ = _cluster_layout_params(
        len(strategies)
    )

    fig, ax = plt.subplots(figsize=(14, 6))
    handles_map = {}

    for k, lam in enumerate(lambdas):
        base_x = k * cluster_gap
        for i, s in enumerate(strategies):
            y = grouped[(lam, s)]
            if y.size == 0:
                continue

            x = base_x + i * stride

            bp = ax.boxplot(
                [y],
                positions=[x],
                widths=box_width,
                manage_ticks=False,
                showfliers=False,
                whis=[5, 95],    # SAME settings as panels
                patch_artist=True,
            )

            col = COLOR_BY_STRAT.get(s, "#777")
            for patch in bp["boxes"]:
                patch.set_facecolor(col)
                patch.set_edgecolor("black")

            for elem in ("whiskers", "caps", "medians"):
                for l in bp[elem]:
                    l.set_color("black")

            if s not in handles_map:
                handles_map[s] = mpatches.Patch(label=s, color=col, ec="black")

    centers = [k * cluster_gap + cluster_width / 2.0 for k in range(len(lambdas))]
    ax.set_xticks(centers, [f"λ={lam}" for lam in lambdas])
    ax.set_ylabel("Latency (s)")
    ax.set_title("Latency distribution by Strategy and λ")

    # → Log scaling with raw latency data
    ax.set_yscale("log")

    _bottom_legend(fig, list(handles_map.values()))
    fig.tight_layout(rect=[0, 0.07, 1, 1])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"[OK] wrote {out_path}")


def plot_latency_oneaxis_new(df: pd.DataFrame, out_path: Path) -> None:
    df = _ensure_lambda_col(df)
    if df.empty or "Lambda" not in df.columns or "Latency" not in df.columns:
        print("[WARN] no latency data to plot.")
        return

    lambdas = _ensure_lambda_order(df["Lambda"].unique(), target=(4.0, 5.0, 6.0, 7.0))
    if not lambdas:
        print("[WARN] no Lambda values found.")
        return
    if len(lambdas) > 2:
        lambdas = lambdas[:4]

    strategies = sorted(df["Strategy"].dropna().unique())
    if not strategies:
        print("[WARN] no Strategy values found.")
        return

    stride, cluster_width, cluster_gap, box_width, _ = _cluster_layout_params(len(strategies))

    fig, ax = plt.subplots(figsize=(14, 6))  # wider
    handles_map = {}

    for k, lam in enumerate(lambdas):
        sub = df[df["Lambda"] == lam]
        base_x = k * cluster_gap
        for i, s in enumerate(strategies):
            y = _num(sub.loc[sub["Strategy"] == s, "Latency"]).dropna().values
            if y.size == 0:
                continue
            x = base_x + i * stride
            bp = ax.boxplot(
                [y],
                positions=[x],
                widths=box_width,
                manage_ticks=False,
                showfliers=BOX_SHOW_FLIERS,
                whis=BOX_WHISKER_PERCENTILES,
                patch_artist=True,
            )

            col = COLOR_BY_STRAT.get(s, "#777")
            for patch in bp["boxes"]:
                patch.set_facecolor(col); patch.set_edgecolor("black")
            for elem in ("whiskers", "caps", "medians"):
                for l in bp[elem]:
                    l.set_color("black")
            if s not in handles_map:
                handles_map[s] = mpatches.Patch(label=s, color=col, ec="black")

    # Tick at each cluster center
    centers = [k * cluster_gap + cluster_width / 2.0 for k in range(len(lambdas))]
    ax.set_xticks(centers, [f"λ={lam}" for lam in lambdas])

    ax.set_ylabel("Latency (s)")
    ax.set_title("Latency distribution by Strategy and λ")

    _bottom_legend(fig, list(handles_map.values()))
    fig.tight_layout(rect=[0, 0.07, 1, 1])  # leave space at bottom for legend
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] wrote {out_path}")


# --- ONE-AXIS power plot: mean ± std error bars, legend at bottom ------------
def plot_power_mean_std_oneaxis(df: pd.DataFrame, out_path: Path) -> None:
    df = _ensure_lambda_col(df)
    if df.empty or "Lambda" not in df.columns or "Power" not in df.columns:
        print("[WARN] no power data to plot.")
        return

    lambdas = _ensure_lambda_order(df["Lambda"].unique(), target=(3.0, 4.0, 5.0, 6.0, 7.0, 8.0))
    if not lambdas:
        print("[WARN] no Lambda values found.")
        return
    if len(lambdas) > 2:
        lambdas = lambdas[:4]

    strategies = sorted(df["Strategy"].dropna().unique())
    if not strategies:
        print("[WARN] no Strategy values found.")
        return

    agg = (df.groupby(["Lambda", "Strategy"])["Power"] .agg(["mean", "std"]).reset_index())
    agg["std"] = agg["std"].fillna(0.0)
    mean_map = {(float(r["Lambda"]), r["Strategy"]): float(r["mean"]) for _, r in agg.iterrows()}
    std_map  = {(float(r["Lambda"]), r["Strategy"]): float(r["std"])  for _, r in agg.iterrows()}

    stride, cluster_width, cluster_gap, _, bar_width = _cluster_layout_params(len(strategies))

    fig, ax = plt.subplots(figsize=(14, 6))
    handles_map = {}

    for k, lam in enumerate(lambdas):
        base_x = k * cluster_gap
        for i, s in enumerate(strategies):
            m = mean_map.get((lam, s), np.nan)
            if np.isnan(m):
                continue
            sd = std_map.get((lam, s), 0.0)
            x = base_x + i * stride
            col = COLOR_BY_STRAT.get(s, "#777")
            ax.bar(x, m, width=bar_width, yerr=sd, capsize=3,
                   edgecolor="black", linewidth=0.6, color=col)
            if s not in handles_map:
                handles_map[s] = mpatches.Patch(label=s, color=col, ec="black")

    centers = [k * cluster_gap + cluster_width / 2.0 for k in range(len(lambdas))]
    ax.set_xticks(centers, [f"λ={lam}" for lam in lambdas])
    ax.set_ylabel("Average Power (W)")
    ax.set_title("Average Power by Strategy and λ")

    _bottom_legend(fig, list(handles_map.values()))
    fig.tight_layout(rect=[0, 0.07, 1, 1])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] wrote {out_path}")


# --- ONE-AXIS violations plot: counts, legend at bottom ----------------------
def plot_violations_mean_std_oneaxis(
    df: pd.DataFrame,
    out_path: Path,
    slack_frac: float = 0.05,
) -> None:
    df = _ensure_lambda_col(df)
    if df.empty or "Lambda" not in df.columns:
        print("[WARN] no Lambda data to plot.")
        return

    # ---- 1) Recompute violations WITH slack if we have per-task data ----
    # Expect columns: Latency_s, Deadline_s, Strategy, (optional) Run
    if {"Latency_s", "Deadline_s", "Strategy"}.issubset(df.columns):
        df = df.copy()

        # numeric + effective deadline with slack
        lat = pd.to_numeric(df["Latency_s"], errors="coerce")
        dl  = pd.to_numeric(df["Deadline_s"], errors="coerce")
        eff_deadline = dl * (1.0 + slack_frac)

        df["SlackViolationFlag"] = (lat > eff_deadline).astype(int)

        group_cols = ["Lambda", "Strategy"]
        if "Run" in df.columns:
            group_cols.append("Run")

        per_run = (
            df.groupby(group_cols, dropna=True)["SlackViolationFlag"]
              .agg(["sum", "count"])
              .reset_index()
              .rename(columns={"sum": "Violations", "count": "Total_Tasks"})
        )
    else:
        # Fall back to whatever aggregated Violations / Total_Tasks we already have
        if "Violations" not in df.columns:
            print("[WARN] no Violations column and no per-task data; nothing to plot.")
            return
        per_run = df

    # ---- 2) Choose Lambdas in order ----
    lambdas = _ensure_lambda_order(per_run["Lambda"].unique(), target=(4.0, 5.0, 6.0, 7.0))
    if not lambdas:
        print("[WARN] no Lambda values found for violations plot.")
        return
    if len(lambdas) > 4:
        lambdas = lambdas[:4]
    lambdas = [float(l) for l in lambdas]

    strategies = sorted(per_run["Strategy"].dropna().unique())
    if not strategies:
        print("[WARN] no Strategy values found for violations plot.")
        return

    # ---- 3) Per-run violation rate (%) before aggregating over runs ----
    if "Total_Tasks" in per_run.columns:
        per_run = per_run.copy()
        viol = pd.to_numeric(per_run["Violations"], errors="coerce")
        total = pd.to_numeric(per_run["Total_Tasks"], errors="coerce").replace(0, np.nan)
        per_run["Rate_pct"] = (viol / total) * 100.0
        metric_col = "Rate_pct"
    else:
        # fallback: use raw counts
        metric_col = "Violations"

    agg = (
        per_run.groupby(["Lambda", "Strategy"])[metric_col]
               .agg(["mean", "std"])
               .reset_index()
    )
    agg["std"] = agg["std"].fillna(0.0)

    mean_map = {(float(r["Lambda"]), r["Strategy"]): float(r["mean"])
                for _, r in agg.iterrows()}
    std_map  = {(float(r["Lambda"]), r["Strategy"]): float(r["std"])
                for _, r in agg.iterrows()}

    # ---- 4) Plot layout (clusters per λ) ----
    def _cluster_layout_params(n_strategies: int):
        stride = 1.0
        cluster_width = (n_strategies - 1) * stride
        gutter = max(2.5, 0.25 * n_strategies)
        cluster_gap = cluster_width + gutter
        bar_width = 0.8
        return stride, cluster_width, cluster_gap, bar_width

    stride, cluster_width, cluster_gap, bar_width =_cluster_layout_params(len(strategies))

    fig, ax = plt.subplots(figsize=(14, 6))
    handles_map = {}

    for k, lam in enumerate(lambdas):
        base_x = k * cluster_gap
        for i, s in enumerate(strategies):
            m = mean_map.get((lam, s), np.nan)
            if np.isnan(m):
                continue
            sd = std_map.get((lam, s), 0.0)
            x = base_x + i * stride
            col = COLOR_BY_STRAT.get(s, "#777")
            ax.bar(
                x, m, width=bar_width, yerr=sd, capsize=3,
                edgecolor="black", linewidth=0.6, color=col
            )
            if s not in handles_map:
                handles_map[s] = mpatches.Patch(label=s, color=col, ec="black")

    # x-ticks at cluster centers
    centers = [k * cluster_gap + cluster_width / 2.0 for k in range(len(lambdas))]
    ax.set_xticks(centers)
    ax.set_xticklabels([f"λ={lam}" for lam in lambdas])

    ax.set_ylabel("Violations (% of tasks) — mean ± std")
    # ax.set_title(
    #     "Deadline Violation Rate by Strategy and λ\n"
    #     f"(violations: latency > (1+{slack_frac:.2f})×deadline)"
    # )

    # bottom legend
    handles_list = list(handles_map.values())
    if handles_list:
        labels_list = [h.get_label() for h in handles_list]
        fig.legend(
            handles=handles_list,
            labels=labels_list,
            title=f"Strategy",
            loc="lower center",
            ncol=min(5, len(handles_list)),
            frameon=False,
            borderaxespad=0.0,
            bbox_to_anchor=(0.5, -0.02),
        )

    fig.tight_layout(rect=[0, 0.08, 1, 1])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] wrote {out_path}")


def plot_latency_panels_for_lambdas(
    df: pd.DataFrame,
    out_path: Path,
    lambda_list: list,
    title_suffix: str = "",
) -> None:
    """
    Boxplots of latency by Strategy for a given list of Lambdas.
    - One panel per lambda in `lambda_list`
    - Shared y-axis across all those panels
    - Outliers hidden, whiskers at [5, 95] percentiles
    """
    df = _ensure_lambda_col(df)

    if df.empty or "Lambda" not in df.columns or "Latency" not in df.columns:
        print("[WARN] no latency data.")
        return

    # Keep only requested lambdas
    df_sub = df[df["Lambda"].isin(lambda_list)].copy()
    if df_sub.empty:
        print(f"[WARN] no data for lambdas {lambda_list}.")
        return

    lambdas = sorted(set(df_sub["Lambda"]))

    if "Strategy" not in df_sub.columns:
        print("[WARN] no Strategy column found.")
        return
    strategies = sorted(df_sub["Strategy"].dropna().unique())

    # ---------- global y range for all selected lambdas ----------
    lat_all = _num(df_sub["Latency"]).dropna().values
    if lat_all.size > 0:
        lo = np.percentile(lat_all, 5)
        hi = np.percentile(lat_all, 95)
        span = hi - lo
        pad = 0.1 * max(span, 1e-3)
        ymin = max(0.0, lo - pad)
        ymax = hi + pad
        if ymin == ymax:
            ymax = ymin + 1e-3
    else:
        ymin, ymax = 0.0, 1.0

    # ---------- one subplot per lambda, shared y ----------
    fig, axes = plt.subplots(
        1, len(lambdas),
        figsize=(4 * len(lambdas), 5),
        sharey=True,
    )
    if len(lambdas) == 1:
        axes = [axes]

    # legend handles (one per Strategy)
    handles = [
        mpatches.Patch(label=s, color=COLOR_BY_STRAT.get(s, "#777"), ec="black")
        for s in strategies
    ]

    for ax, lam in zip(axes, lambdas):
        sub = df_sub[df_sub["Lambda"] == lam].copy()
        if sub.empty:
            ax.set_visible(False)
            continue

        for i, s in enumerate(strategies, start=1):
            y = _num(sub.loc[sub["Strategy"] == s, "Latency"]).dropna().values
            if y.size == 0:
                continue

            bp = ax.boxplot(
                [y],
                positions=[i],
                widths=0.7,
                manage_ticks=False,
                showfliers=False,   # hide outliers
                whis=[5, 95],       # whiskers = 5th / 95th
                patch_artist=True,
            )

            col = COLOR_BY_STRAT.get(s, "#777")
            for patch in bp["boxes"]:
                patch.set_facecolor(col)
                patch.set_edgecolor("black")
            for elem in ("whiskers", "caps", "medians"):
                for l in bp[elem]:
                    l.set_color("black")

        ax.set_ylim(ymin, ymax)      # <- SAME y-range for all panels
        ax.set_title(f"λ={lam}")
        ax.set_xticks([])            # strategies go in legend
        ax.grid(False)

    axes[0].set_ylabel("Latency (s)")
    fig.suptitle(f"Latency distribution by Strategy and λ {title_suffix}", y=0.98)

    fig.legend(
        handles=handles,
        title="Strategy",
        bbox_to_anchor=(1.02, 0.5),
        loc="center left",
        borderaxespad=0.0,
        frameon=False,
    )

    fig.tight_layout(rect=[0, 0, 0.87, 1])
    plt.subplots_adjust(wspace=0.4)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] wrote {out_path}")


def plot_latency_percentiles_bars(
    df: pd.DataFrame,
    out_dir: Path,
    percentiles: Tuple[int, ...] = (50, 90, 99),
) -> None:
    df = _ensure_lambda_col(df)
    if df.empty or "Latency" not in df.columns or "Lambda" not in df.columns:
        print("[WARN] no latency data for percentile bars.")
        return

    lambdas = sorted(float(x) for x in df["Lambda"].unique() if pd.notna(x))

    # Only adaptive strategies (fallback to all if none found)
    all_strats = sorted(df["Strategy"].dropna().unique())
    adaptive_strats = sorted(s for s in all_strats if "adaptive" in str(s).lower())
    strategies = adaptive_strats or all_strats

    if not lambdas or not strategies:
        print("[WARN] missing Lambda/Strategy for percentile bars.")
        return


    grp = df.groupby(["Lambda", "Strategy"])["Latency"]

    stride, cluster_width, cluster_gap, _, bar_width = _cluster_layout_params(len(strategies))


    for p in percentiles:
        q = grp.quantile(p / 100.0).reset_index(name="Value")

        # map (λ, strat) -> percentile latency
        qmap = {(float(r["Lambda"]), r["Strategy"]): float(r["Value"])
                for _, r in q.iterrows()}

        fig, ax = plt.subplots(figsize=(14, 6))
        handles_map = {}

        for k, lam in enumerate(lambdas):
            base_x = k * cluster_gap
            for i, s in enumerate(strategies):
                val = qmap.get((lam, s), np.nan)
                if np.isnan(val):
                    continue
                x = base_x + i * stride
                col = COLOR_BY_STRAT.get(s, "#777")
                ax.bar(x, val, width=bar_width,
                       edgecolor="black", linewidth=0.6, color=col)
                if s not in handles_map:
                    handles_map[s] = mpatches.Patch(label=s, color=col, ec="black")

        centers = [k * cluster_gap + cluster_width / 2.0 for k in range(len(lambdas))]
        ax.set_xticks(centers, [f"λ={lam}" for lam in lambdas])
        ax.set_ylabel(f"Latency (s) — p{p}")
        ax.set_title(f"Latency p{p} by Strategy and λ")

        _bottom_legend(fig, list(handles_map.values()))
        fig.tight_layout(rect=[0, 0.07, 1, 1])
        out_path = out_dir / f"latency_p{p}_bar_by_lambda.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[OK] wrote {out_path}")


def plot_latency_cdfs(df: pd.DataFrame, out_dir: Path) -> None:
    df = _ensure_lambda_col(df)
    if df.empty or "Latency" not in df.columns or "Lambda" not in df.columns:
        print("[WARN] no latency data for CDF plots.")
        return

    lambdas = sorted(float(x) for x in df["Lambda"].unique() if pd.notna(x))

    # Only adaptive strategies (fallback to all if none found)
    all_strats = sorted(df["Strategy"].dropna().unique())
    adaptive_strats = sorted(s for s in all_strats if "adaptive" in str(s).lower())
    strategies = adaptive_strats or all_strats

    if not lambdas or not strategies:
        print("[WARN] missing Lambda/Strategy for CDF plots.")
        return


    for lam in lambdas:
        sub = df[df["Lambda"] == lam]
        if sub.empty:
            continue

        fig, ax = plt.subplots(figsize=(8, 6))
        for s in strategies:
            y = _num(sub.loc[sub["Strategy"] == s, "Latency"]).dropna().values
            if y.size == 0:
                continue
            y_sorted = np.sort(y)
            x_vals = y_sorted
            y_cdf = np.linspace(0, 1, len(y_sorted), endpoint=True)
            col = COLOR_BY_STRAT.get(s, "#777")
            ax.plot(x_vals, y_cdf, label=s, color=col)

        ax.set_xlabel("Latency (s)")
        ax.set_ylabel("CDF")
        ax.set_title(f"Latency CDF by Strategy (λ={lam})")
        ax.grid(True, linestyle=":", linewidth=0.5)
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        out_path = out_dir / f"latency_cdf_lambda_{lam}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[OK] wrote {out_path}")

def read_gpu_utilization_per_gpu(
    files: Iterable[Path],
    seed_to_lambda: Dict[int, float],
) -> pd.DataFrame:
    """
    Reads per-GPU utilization from dvfs_energy_windows files.

    Output rows:
        Lambda, Strategy, GPU_ID, Utilization
    """
    rows = []

    for p in files:
        try:
            df = pd.read_csv(p, dtype=str, low_memory=False)
        except Exception:
            continue

        strat = parse_strategy_from_filename(p.name)

        # get lambda
        lam, _ = lambda_from_filename(p)
        if lam is None:
            m = re.search(r"seed(\d+)", p.name)
            lam = seed_to_lambda.get(int(m.group(1))) if m else None
        if lam is None:
            continue

        # column normalization
        cols = {c.lower(): c for c in df.columns}

        # utilization column
        if "utilization" not in cols:
            # debug -> print actual columns
            # print(f"[DEBUG] no utilization column in {p.name}", df.columns)
            continue
        ucol = cols["utilization"]

        # GPU-ID columns (Cluster, Node, GPU)
        needed = ("cluster", "node", "gpu")
        if not all(k in cols for k in needed):
            # print(f"[DEBUG] missing GPU triple in {p.name}", df.columns)
            continue

        cluster_col = cols["cluster"]
        node_col = cols["node"]
        gpu_col = cols["gpu"]

        # Build GPU ID as "C1:N1:G1"
        df["GPU_ID"] = (
            df[cluster_col].astype(str) + ":" +
            df[node_col].astype(str) + ":" +
            df[gpu_col].astype(str)
        )

        util = pd.to_numeric(df[ucol], errors="coerce")
        gpu_ids = df["GPU_ID"]

        for gpu_id, u in zip(gpu_ids, util):
            if pd.isna(u):
                continue

            rows.append({
                "Lambda": float(lam),
                "Strategy": strat,
                "GPU_ID": gpu_id,
                "Utilization": float(u),
            })

    if not rows:
        print("[WARN] no GPU utilization entries found.")
        return pd.DataFrame(columns=["Lambda","Strategy","GPU_ID","Utilization"])

    return pd.DataFrame(rows)

def plot_gpu_utilization_per_gpu(df: pd.DataFrame, out_dir: Path) -> None:
    if df.empty:
        print("[WARN] no GPU utilization data to plot.")
        return

    strategies = sorted(df["Strategy"].unique())
    lambdas = sorted(df["Lambda"].unique())
    gpu_ids = sorted(df["GPU_ID"].unique())

    for strat in strategies:
        sub = df[df["Strategy"] == strat]

        fig, ax = plt.subplots(figsize=(12, 6))

        for lam in lambdas:
            ss = sub[sub["Lambda"] == lam]

            # average per GPU (in case multiple runs exist)
            mean = ss.groupby("GPU_ID")["Utilization"].mean().reindex(gpu_ids)

            ax.plot(
                gpu_ids,
                mean,
                marker="o",
                linewidth=2,
                label=f"λ={lam}"
            )

        ax.set_title(f"Per-GPU Utilization — {strat}")
        ax.set_ylabel("Utilization")
        ax.set_xlabel("GPU ID")
        ax.set_ylim(0, 1)
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.3)

        out = out_dir / f"gpu_utilization_per_gpu_{strat}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[OK] wrote {out}")


def plot_gpu_utilization_bar(df: pd.DataFrame, out_path: Path) -> None:
    """
    Bar plot of average GPU utilization per (Lambda, Strategy).
    Expects columns: Lambda, Strategy, GPU_Util (0–1).
    """
    df = _ensure_lambda_col(df)
    if df.empty or "Lambda" not in df.columns or "GPU_Util" not in df.columns:
        print("[WARN] no GPU utilization data to plot.")
        print("[DEBUG] GPU util df columns:", list(df.columns))
        return

    df = df.copy()
    df["GPU_Util"] = pd.to_numeric(df["GPU_Util"], errors="coerce")
    df = df.dropna(subset=["GPU_Util"])
    if df.empty:
        print("[WARN] GPU_Util is all NaN.")
        return

    df["util_pct"] = df["GPU_Util"] * 100.0

    lambdas = sorted(float(x) for x in df["Lambda"].unique() if pd.notna(x))
    strategies = sorted(df["Strategy"].dropna().unique())
    if not lambdas or not strategies:
        print("[WARN] missing Lambda/Strategy for GPU util plot.")
        return

    agg = (
        df.groupby(["Lambda", "Strategy"])["util_pct"]
          .mean()
          .reset_index()
    )
    util_map = {
        (float(r["Lambda"]), r["Strategy"]): float(r["util_pct"])
        for _, r in agg.iterrows()
    }

    stride, cluster_width, cluster_gap, bar_width = _cluster_layout_params(len(strategies))[:4]

    fig, ax = plt.subplots(figsize=(14, 6))
    handles_map: dict[str, mpatches.Patch] = {}

    for k, lam in enumerate(lambdas):
        base_x = k * cluster_gap
        for i, s in enumerate(strategies):
            u = util_map.get((lam, s), np.nan)
            if np.isnan(u):
                continue
            x = base_x + i * stride
            col = COLOR_BY_STRAT.get(s, "#777")
            ax.bar(
                x, u,
                width=bar_width,
                edgecolor="black",
                linewidth=0.6,
                color=col,
            )
            if s not in handles_map:
                handles_map[s] = mpatches.Patch(label=s, color=col, ec="black")

    centers = [k * cluster_gap + cluster_width / 2.0 for k in range(len(lambdas))]
    ax.set_xticks(centers, [f"λ={lam}" for lam in lambdas])
    ax.set_ylabel("Average GPU utilization (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Average GPU utilization by Strategy and λ")

    _bottom_legend(fig, list(handles_map.values()))
    fig.tight_layout(rect=[0, 0.07, 1, 1])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] wrote {out_path}")


def plot_link_utilization_bar(df: pd.DataFrame, out_path: Path) -> None:
    """
    Bar plot of average link utilization per (Lambda, Link_ID).
    Expects columns: Lambda, Link_ID, util (0–1).
    """
    df = _ensure_lambda_col(df)
    if df.empty or "Lambda" not in df.columns or \
       "Link_ID" not in df.columns or "util" not in df.columns:
        print("[WARN] no link utilization data to plot.")
        return

    df = df.copy()
    df["util"] = pd.to_numeric(df["util"], errors="coerce")
    df = df.dropna(subset=["util"])
    if df.empty:
        print("[WARN] link util column is all NaN.")
        return

    df["util_pct"] = df["util"] * 100.0

    lambdas = sorted(float(x) for x in df["Lambda"].unique() if pd.notna(x))
    links = sorted(df["Link_ID"].dropna().unique())
    if not lambdas or not links:
        print("[WARN] missing Lambda/Link_ID for link util plot.")
        return

    util_map = {
        (float(r["Lambda"]), r["Link_ID"]): float(r["util_pct"])
        for _, r in df.groupby(["Lambda", "Link_ID"])["util_pct"].mean().reset_index().iterrows()
    }

    stride, cluster_width, cluster_gap, bar_width = \
        _cluster_layout_params(len(links))[:4]

    
    color_cycle = iter(plt.rcParams["axes.prop_cycle"].by_key()["color"])
    color_by_link = {lk: next(color_cycle) for lk in links}

    fig, ax = plt.subplots(figsize=(14, 6))

    for k, lam in enumerate(lambdas):
        base_x = k * cluster_gap
        for i, lk in enumerate(links):
            u = util_map.get((lam, lk), np.nan)
            if np.isnan(u):
                continue
            x = base_x + i * stride
            col = color_by_link[lk]
            ax.bar(
                x, u,
                width=bar_width,
                edgecolor="black",
                linewidth=0.6,
                color=col,
            )

    handles = [
        mpatches.Patch(label=str(lk), color=color_by_link[lk], ec="black")
        for lk in links
    ]

    centers = [k * cluster_gap + cluster_width / 2.0 for k in range(len(lambdas))]
    ax.set_xticks(centers, [f"λ={lam}" for lam in lambdas])
    ax.set_ylabel("Average link utilization (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Average link utilization by Link and λ")

    _bottom_legend(fig, handles)
    fig.tight_layout(rect=[0, 0.07, 1, 1])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] wrote {out_path}")


# --- Scatter: Power vs p99 latency -------------------------------------------
def plot_scatter_power_vs_latency(
    power_df: pd.DataFrame,
    latency_df: pd.DataFrame,
    out_path: Path,
    percentile: float = 99.0,
) -> None:
    power_df = _ensure_lambda_col(power_df)
    latency_df = _ensure_lambda_col(latency_df)

    if power_df.empty or latency_df.empty:
        print("[WARN] missing power/latency for scatter plot.")
        return

    # mean power per (λ, strategy)
    p_agg = (power_df.groupby(["Lambda", "Strategy"])["Power"]
                      .mean()
                      .reset_index())
    # pXX latency per (λ, strategy)
    lat_grp = latency_df.groupby(["Lambda", "Strategy"])["Latency"]
    lat_agg = lat_grp.quantile(percentile / 100.0).reset_index(name="Latency_pXX")

    merged = pd.merge(p_agg, lat_agg, on=["Lambda", "Strategy"], how="inner")
    if merged.empty:
        print("[WARN] no overlapping (λ,strategy) for power-latency scatter.")
        return

    lambdas = sorted(merged["Lambda"].unique())
    strategies = sorted(merged["Strategy"].unique())

    markers = ["o", "s", "D", "^", "v", "P", "X", "*", "<", ">"]
    marker_map = {lam: markers[i % len(markers)] for i, lam in enumerate(lambdas)}

    fig, ax = plt.subplots(figsize=(8, 6))
    for s in strategies:
        sub = merged[merged["Strategy"] == s]
        if sub.empty:
            continue
        col = COLOR_BY_STRAT.get(s, "#777")
        for lam in lambdas:
            sub_l = sub[sub["Lambda"] == lam]
            if sub_l.empty:
                continue
            ax.scatter(
                sub_l["Power"],
                sub_l["Latency_pXX"],
                label=f"{s}, λ={lam}",
                color=col,
                marker=marker_map[lam],
                edgecolors="black",
                linewidths=0.4,
            )

    ax.set_xlabel("Average Power (W)")
    ax.set_ylabel(f"Latency p{int(percentile)} (s)")
    ax.set_title(f"Power vs Latency p{int(percentile)} by Strategy and λ")
    ax.grid(True, linestyle=":", linewidth=0.5)

    # Build legend with separate entries for strategies and lambdas

    strat_handles = [
        mpatches.Patch(label=s, color=COLOR_BY_STRAT.get(s, "#777"))
        for s in strategies
    ]

    lam_handles = [
        mlines.Line2D([], [], color="black", marker=marker_map[lam],
                      linestyle="None", label=f"λ={lam}")
        for lam in lambdas
    ]
    handles = strat_handles + lam_handles
    ax.legend(handles=handles, frameon=False, fontsize=8, loc="best")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] wrote {out_path}")

# --- Scatter: Power vs Violations -------------------------------------------
def plot_scatter_power_vs_violations(
    power_df: pd.DataFrame,
    viol_df: pd.DataFrame,
    out_path: Path,
) -> None:
    power_df = _ensure_lambda_col(power_df)
    viol_df = _ensure_lambda_col(viol_df)

    if power_df.empty or viol_df.empty:
        print("[WARN] missing power/violations for scatter plot.")
        return

    p_agg = (power_df.groupby(["Lambda", "Strategy"])["Power"]
                      .mean()
                      .reset_index())
    if "Total_Tasks" in viol_df.columns:
        vdf = viol_df.copy()
        vdf["Rate_pct"] = (
            pd.to_numeric(vdf["Violations"], errors="coerce") /
            pd.to_numeric(vdf["Total_Tasks"], errors="coerce").replace(0, np.nan)
        ) * 100.0
        v_agg = (vdf.groupby(["Lambda", "Strategy"])["Rate_pct"]
                    .mean()
                    .reset_index()
                    .rename(columns={"Rate_pct": "Violations"}))
    else:
        # fallback: keep old behaviour
        v_agg = (viol_df.groupby(["Lambda", "Strategy"])["Violations"]
                            .mean()
                            .reset_index())


    merged = pd.merge(p_agg, v_agg, on=["Lambda", "Strategy"], how="inner")
    if merged.empty:
        print("[WARN] no overlapping (λ,strategy) for power-violations scatter.")
        return

    lambdas = sorted(merged["Lambda"].unique())
    strategies = sorted(merged["Strategy"].unique())

    markers = ["o", "s", "D", "^", "v", "P", "X", "*", "<", ">"]
    marker_map = {lam: markers[i % len(markers)] for i, lam in enumerate(lambdas)}

    fig, ax = plt.subplots(figsize=(8, 6))
    for s in strategies:
        sub = merged[merged["Strategy"] == s]
        if sub.empty:
            continue
        col = COLOR_BY_STRAT.get(s, "#777")
        for lam in lambdas:
            sub_l = sub[sub["Lambda"] == lam]
            if sub_l.empty:
                continue
            ax.scatter(
                sub_l["Power"],
                sub_l["Violations"],
                label=f"{s}, λ={lam}",
                color=col,
                marker=marker_map[lam],
                edgecolors="black",
                linewidths=0.4,
            )

    ax.set_xlabel("Average Power (W)")
    ax.set_ylabel("Violations (mean % of tasks)")
    ax.set_title("Power vs Deadline Violation Rate by Strategy and λ")
 
    strat_handles = [
        mpatches.Patch(label=s, color=COLOR_BY_STRAT.get(s, "#777"))
        for s in strategies
    ]
    lam_handles = [
        mlines.Line2D([], [], color="black", marker=marker_map[lam],
                      linestyle="None", label=f"λ={lam}")
        for lam in lambdas
    ]
    handles = strat_handles + lam_handles
    ax.legend(handles=handles, frameon=False, fontsize=8, loc="best")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] wrote {out_path}")

def plot_violation_rate_oneaxis(df: pd.DataFrame, out_path: Path) -> None:
    """
    Plot deadline miss rate (% of tasks) per (Lambda, Strategy).

    Expects an aggregated dataframe from read_violations_new() with columns:
        Lambda, Strategy, Violations, Total_Tasks
    """
    df = _ensure_lambda_col(df)

    required = {"Lambda", "Strategy", "Violations", "Total_Tasks"}
    if df.empty or not required.issubset(df.columns):
        missing = required - set(df.columns)
        print(f"[WARN] no data for Violations rate plot (missing: {', '.join(sorted(missing))})")
        # small debug:
        print("[DEBUG] available columns:", list(df.columns))
        return

    # Aggregate across runs per (Lambda, Strategy) just in case
    agg = (
        df.groupby(["Lambda", "Strategy"])[["Violations", "Total_Tasks"]]
          .sum()
          .reset_index()
    )

    # Compute rate in %
    agg["Rate_pct"] = (
        pd.to_numeric(agg["Violations"], errors="coerce") /
        pd.to_numeric(agg["Total_Tasks"], errors="coerce").replace(0, np.nan)
    ) * 100.0

    if agg["Rate_pct"].dropna().empty:
        print("[WARN] all Violations rates are NaN; nothing to plot.")
        return

    lambdas = sorted(float(x) for x in agg["Lambda"].unique() if pd.notna(x))
    strategies = sorted(agg["Strategy"].dropna().unique())
    if not lambdas or not strategies:
        print("[WARN] missing Lambda/Strategy for Violations rate plot.")
        return

    rate_map = {
        (float(r["Lambda"]), r["Strategy"]): float(r["Rate_pct"])
        for _, r in agg.iterrows()
    }

    stride, cluster_width, cluster_gap, bar_width = _cluster_layout_params(len(strategies))[:4]

    fig, ax = plt.subplots(figsize=(14, 6))
    handles_map: dict[str, mpatches.Patch] = {}

    for k, lam in enumerate(lambdas):
        base_x = k * cluster_gap
        for i, s in enumerate(strategies):
            r_val = rate_map.get((lam, s), np.nan)
            if np.isnan(r_val):
                continue
            x = base_x + i * stride
            col = COLOR_BY_STRAT.get(s, "#777")
            ax.bar(
                x, r_val,
                width=bar_width,
                edgecolor="black",
                linewidth=0.6,
                color=col,
            )
            if s not in handles_map:
                handles_map[s] = mpatches.Patch(label=s, color=col, ec="black")

    centers = [k * cluster_gap + cluster_width / 2.0 for k in range(len(lambdas))]
    ax.set_xticks(centers, [f"λ={lam}" for lam in lambdas])
    ax.set_ylabel("Deadline Miss Rate (% of tasks)")
    ax.set_title("Deadline Violation Rate by Strategy and λ")

    _bottom_legend(fig, list(handles_map.values()))
    fig.tight_layout(rect=[0, 0.07, 1, 1])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] wrote {out_path}")


# --- Normalized mean power (relative to baseline strategy) -------------------
def plot_normalized_power_bar(
    df: pd.DataFrame,
    baseline_strategy: str,
    out_path: Path,
) -> None:
    df = _ensure_lambda_col(df)
    if df.empty or "Lambda" not in df.columns or "Power" not in df.columns:
        print("[WARN] no power data for normalized plot.")
        return

    lambdas = sorted(float(x) for x in df["Lambda"].unique() if pd.notna(x))

    strategies = sorted(df["Strategy"].dropna().unique())
    if not strategies:
        print("[WARN] no strategies found for normalized plot.")
        return

    if baseline_strategy not in strategies:
        print(f"[WARN] baseline strategy '{baseline_strategy}' not found; "
            f"using '{strategies[0]}' as baseline instead.")
        baseline_strategy = strategies[0]


    agg = (df.groupby(["Lambda", "Strategy"])["Power"]
             .mean()
             .reset_index())
    mean_map = {(float(r["Lambda"]), r["Strategy"]): float(r["Power"])
                for _, r in agg.iterrows()}

    stride, cluster_width, cluster_gap, _, bar_width = _cluster_layout_params(len(strategies))

    fig, ax = plt.subplots(figsize=(14, 6))
    handles_map = {}

    for k, lam in enumerate(lambdas):
        base_x = k * cluster_gap
        baseline_val = mean_map.get((lam, baseline_strategy), np.nan)
        if np.isnan(baseline_val) or baseline_val == 0:
            continue
        for i, s in enumerate(strategies):
            m_val = mean_map.get((lam, s), np.nan)
            if np.isnan(m_val):
                continue
            x = base_x + i * stride
            norm = m_val / baseline_val
            col = COLOR_BY_STRAT.get(s, "#777")
            ax.bar(x, norm, width=bar_width,
                   edgecolor="black", linewidth=0.6, color=col)
            if s not in handles_map:
                handles_map[s] = mpatches.Patch(label=s, color=col, ec="black")

    centers = [k * cluster_gap + cluster_width / 2.0 for k in range(len(lambdas))]
    ax.set_xticks(centers, [f"λ={lam}" for lam in lambdas])
    ax.axhline(1.0, color="black", linewidth=0.8, linestyle="--")
    ax.set_ylabel(f"Normalized Power (vs {baseline_strategy})")
    ax.set_title("Normalized Average Power by Strategy and λ")

    _bottom_legend(fig, list(handles_map.values()))
    fig.tight_layout(rect=[0, 0.07, 1, 1])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] wrote {out_path}")


# --- Normalized p99 latency (relative to baseline strategy) ------------------
def plot_normalized_latency_p99_bar(
    df: pd.DataFrame,
    baseline_strategy: str,
    out_path: Path,
) -> None:
    df = _ensure_lambda_col(df)
    if df.empty or "Lambda" not in df.columns or "Latency" not in df.columns:
        print("[WARN] no latency data for normalized p99 plot.")
        return

    lambdas = sorted(float(x) for x in df["Lambda"].unique() if pd.notna(x))

    strategies = sorted(df["Strategy"].dropna().unique())
    if not strategies:
        print("[WARN] no strategies found for normalized plot.")
        return

    if baseline_strategy not in strategies:
        print(f"[WARN] baseline strategy '{baseline_strategy}' not found; "
            f"using '{strategies[0]}' as baseline instead.")
        baseline_strategy = strategies[0]


    grp = df.groupby(["Lambda", "Strategy"])["Latency"]
    q = grp.quantile(0.99).reset_index(name="Latency_p99")
    qmap = {(float(r["Lambda"]), r["Strategy"]): float(r["Latency_p99"])
            for _, r in q.iterrows()}

    stride, cluster_width, cluster_gap, _, bar_width = _cluster_layout_params(len(strategies))

    fig, ax = plt.subplots(figsize=(14, 6))
    handles_map = {}

    for k, lam in enumerate(lambdas):
        base_x = k * cluster_gap
        baseline_val = qmap.get((lam, baseline_strategy), np.nan)
        if np.isnan(baseline_val) or baseline_val == 0:
            continue
        for i, s in enumerate(strategies):
            v = qmap.get((lam, s), np.nan)
            if np.isnan(v):
                continue
            x = base_x + i * stride
            norm = v / baseline_val
            col = COLOR_BY_STRAT.get(s, "#777")
            ax.bar(x, norm, width=bar_width,
                   edgecolor="black", linewidth=0.6, color=col)
            if s not in handles_map:
                handles_map[s] = mpatches.Patch(label=s, color=col, ec="black")

    centers = [k * cluster_gap + cluster_width / 2.0 for k in range(len(lambdas))]
    ax.set_xticks(centers, [f"λ={lam}" for lam in lambdas])
    ax.axhline(1.0, color="black", linewidth=0.8, linestyle="--")
    ax.set_ylabel(f"Normalized Latency p99 (vs {baseline_strategy})")
    ax.set_title("Normalized Latency p99 by Strategy and λ")

    _bottom_legend(fig, list(handles_map.values()))
    fig.tight_layout(rect=[0, 0.07, 1, 1])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] wrote {out_path}")


def plot_violation_cause_bar(df: pd.DataFrame, out_path: Path) -> None:
    """
    For each λ, plot percentage of violations by violation_cause.
    Requires columns: Lambda, deadline_violation, violation_cause
    """
    df = _ensure_lambda_col(df)
    req = {"Lambda", "deadline_violation", "violation_cause"}
    if df.empty or not req.issubset(df.columns):
        print("[WARN] no columns for violation cause plot.")
        return

    # only violated tasks
    v = pd.to_numeric(df["deadline_violation"], errors="coerce").fillna(0)
    dfv = df[v > 0].copy()
    if dfv.empty:
        print("[WARN] no violated tasks for cause plot.")
        return

    # counts per (λ, cause)
    grp = (dfv.groupby(["Lambda", "violation_cause"])["Task_ID"]
              .nunique()
              .reset_index(name="count"))

    total = (grp.groupby("Lambda")["count"]
                .sum().rename("total"))
    grp = grp.merge(total, on="Lambda")
    grp["pct"] = grp["count"] / grp["total"] * 100.0

    lambdas = sorted(grp["Lambda"].unique())
    causes = sorted(grp["violation_cause"].unique())

    # wide table: rows λ, columns cause → pct
    table = grp.pivot(index="Lambda", columns="violation_cause", values="pct").fillna(0)

    fig, ax = plt.subplots(figsize=(12, 5))
    bottom = np.zeros(len(table))
    x = np.arange(len(table))

    colors = {
        "uplink_bound":   "#1f77b4",
        "compute_bound":  "#ff7f0e",
        "downlink_bound": "#2ca02c",
    }

    handles = []
    for cause in causes:
        vals = table[cause].values
        col = colors.get(cause, "#777")
        ax.bar(x, vals, bottom=bottom, label=cause, color=col, edgecolor="black", linewidth=0.6)
        bottom += vals
        handles.append(mpatches.Patch(label=cause, color=col, ec="black"))

    ax.set_xticks(x, [f"λ={lam}" for lam in table.index])
    ax.set_ylabel("Share of violations (%)")
    ax.set_title("Violation causes by λ")
    ax.set_ylim(0, 100)
    _bottom_legend(fig, handles)

    fig.tight_layout(rect=[0, 0.07, 1, 1])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] wrote {out_path}")


# ---------------- CLI ----------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs", help="Folder containing csv logs")
    ap.add_argument("--prefix", default="exp1", help="Prefix for pregen logs (e.g., exp1)")
    ap.add_argument("--out", default="plots_lambda_compare", help="subfolder under --runs for outputs")
    ap.add_argument("--sample", type=int, default=500_000, help="max latency samples to keep")
    return ap.parse_args()

def main() -> None:
    args = parse_args()
    runs_dir = Path(args.runs)
    out_dir = runs_dir / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) seed → λ mapping (used only as fallback)
    seed_to_lambda = build_seed_to_lambda(runs_dir, args.prefix)

        # 2) discover files
    task_files  = find_latency_files(runs_dir, args.prefix)
    power_files = find_power_files(runs_dir, args.prefix)
    link_files  = find_link_stats_files(runs_dir, args.prefix)

    # 3) tidy frames
    latency   = read_latency_samples(task_files, seed_to_lambda, sample_n=args.sample)
    power     = read_power(power_files, seed_to_lambda)
    viol = pd.DataFrame(columns=["Lambda", "Strategy", "Violations", "Total_Tasks"])
    viol      = read_violations(task_files, seed_to_lambda)
    link_util = read_link_utilization(link_files, seed_to_lambda)


    print("[SUMMARY] rows found")
    print(f"  latency samples kept: {len(latency):>7}")
    print(f"  power rows:           {len(power):>7}")
    print(f"  violations rows:      {len(viol):>7}")

    # 4) plots


    # Latency
    plot_latency_oneaxis(latency, out_dir / "latency_box_by_lambda.png")

    # Latency per-λ subplots (each λ has its own y-scale)
    plot_latency_two_panels_new(
        latency,
        out_dir / "latency_box_by_lambda_panels.png",
    )

    # Latency per-λ panels (without outliers, zoomed)
    plot_latency_two_panels_without_outliers(
        latency, out_dir / "latency_box_by_lambda_without_outliers_panels.png"

    )

   # ----- First figure: λ = 4, 5 -----
    plot_latency_panels_for_lambdas(
        df=latency,
        out_path=out_dir / "latency_panels_lambda_four.png",
        lambda_list=[3.0, 4.0, 5.0],
        title_suffix="(λ = 4, 5)",
    )

    # ----- Second figure: λ = 6, 7, 8 -----
    plot_latency_panels_for_lambdas(
        df=latency,
        out_path=out_dir / "latency_panels_lambda_six.png",
        lambda_list=[6.0, 7.0, 8.0],
        title_suffix="(λ = 6, 7, 8)",
    )

    # Power (mean ± std)
    plot_power_mean_std_oneaxis(power, out_dir / "power_bar_by_lambda.png")


    # Violations (counts)
    plot_violations_mean_std_oneaxis(viol, out_dir / "violations_bar_by_lambda.png")


    # Latency
    plot_latency_oneaxis_new(latency, out_dir / "latency_box_by_lambda_without_outliers.png")

    # # Latency breakdown (all tasks)
    # plot_latency_breakdown_stacked(
    #     latency, out_dir / "latency_breakdown_all_tasks.png",
    #     violated_only=False,
    # )

    # # Latency breakdown (violated tasks only)
    # plot_latency_breakdown_stacked(
    #     latency, out_dir / "latency_breakdown_violated_only.png",
    #     violated_only=True,
    # )

    # # plot_link_utilization_bar(df_link, Path("runs/plots_lambda_compare/link_util_bar.png"))


    # # ---- NEW PLOTS ----

    # # # Latency percentiles
    # # plot_latency_percentiles_bars(latency, out_dir, percentiles=(50, 90, 99))

    # # # Latency CDFs (one PNG per λ)
    # # plot_latency_cdfs(latency, out_dir)

    # # # Power vs p99 latency scatter
    # # plot_scatter_power_vs_latency(
    # #     power, latency, out_dir / "scatter_power_vs_latency_p99.png", percentile=99.0
    # # )

    # # # Power vs violations scatter
    # # plot_scatter_power_vs_violations(
    # #     power, viol, out_dir / "scatter_power_vs_violations.png"
    # # )

    # # # Violations rate
    # plot_violation_rate_oneaxis(viol, out_dir / "violation_rate_by_lambda.png")

    # # GPU utilization (if available)
    # gpu_util  = read_gpu_utilization(power_files, seed_to_lambda)
    # if not gpu_util.empty:
    #     plot_gpu_utilization_bar(
    #         gpu_util, out_dir / "gpu_utilization_bar.png"
    #     )
    # else:
    #     print("[WARN] no GPU utilization rows found for plotting.")

    # gpu_util_c = read_gpu_utilization_per_gpu(power_files, seed_to_lambda)
    # plot_gpu_utilization_per_gpu(gpu_util_c, out_dir)




    # # Link utilization (if available)
    # if not link_util.empty:
    #     plot_link_utilization_bar(
    #         link_util, out_dir / "link_utilization_bar.png"
    #     )
    # else:
    #     print("[WARN] no link utilization rows found for plotting.")

    # # plot_violation_cause_bar(
    # #     latency_all,  # or whatever df has per-task rows
    # #     out_dir / "violation_causes_by_lambda.png",
    # )



    # # Normalized metrics (baseline can be changed if you like)
    # baseline = "least-load_adaptive"
    # plot_normalized_power_bar(power, baseline, out_dir / "power_normalized_bar.png")
    # plot_normalized_latency_p99_bar(
    #     latency, baseline, out_dir / "latency_p99_normalized_bar.png"
    # )



if __name__ == "__main__":
    main()
