#!/usr/bin/env python3
"""
Run a matrix of strategies against a single simulator binary/script.

Key features
------------
- Shared pregen:
    --pregen-in  <csv>    Reuse this CSV for ALL strategies
    --pregen-out <csv>    First strategy creates it; others reuse it

- Extras passthrough:
    Everything in --extras is appended to each simulator command
    (e.g., optimizer modules, seed, admission mode, etc).

- Aggregation & plots:
    --aggregate / --no-aggregate
    --plots / --no-plots
"""

from __future__ import annotations

import argparse,json,os,shlex,subprocess,sys
import pandas as pd  
import matplotlib.pyplot as plt  
import concurrent.futures as cf
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------- Strategy matrix ----------------
# You can edit or extend this to match simulator's strategy names/flags.
STRATEGY_MATRIX: Dict[str, Dict[str, Any]] = {
    "least-load_fixed":     {"strategy": "least-load_fixed",     "freq-mode": "fixed"},
    "opt_latency_fixed":   {"strategy": "opt_latency_fixed",   "freq-mode": "fixed"},
    "opt_power_fixed":     {"strategy": "opt_power_fixed",     "freq-mode": "fixed"},
    "opt_efficiency_fixed":{"strategy": "opt_efficiency_fixed","freq-mode": "fixed"},
    "least-load_adaptive":  {"strategy": "least-load_adaptive",  "freq-mode": "adaptive"},
    "opt_latency_adaptive":  {"strategy": "opt_latency_adaptive",  "freq-mode": "adaptive"},
    "opt_power_adaptive":    {"strategy": "opt_power_adaptive",    "freq-mode": "adaptive"},
    "opt_efficiency_adaptive":{"strategy": "opt_efficiency_adaptive","freq-mode": "adaptive"},
}


# ---------------- Subprocess runner ----------------
def run_cmd(cmd: List[str]) -> Tuple[bool, Optional[str]]:
    try:
        # Print the exact command that will be executed (with proper quoting)
        print("[CMD]", " ".join(shlex.quote(c) for c in cmd), flush=True)

        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        if proc.returncode == 0:
            return True, None
        return False, f"Exited {proc.returncode}\n{proc.stdout}"
    except Exception as e:
        return False, repr(e)



# ---------------- Simple aggregation ----------------
def aggregate_results(out_root: Path, successes: List[str], prefix: str) -> None:
    """
    Concatenate each strategy CSV (<prefix>__<strategy>.csv) into one
    '<prefix>_aggregate.csv' with a 'Strategy' column. Resilient to
    column mismatches by aligning on the union.
    """
    frames = []
    for key in successes:
        cand = out_root / f"{prefix}__{key}.csv"
        if not cand.exists():
            print(f"[WARN] missing CSV for {key}: {cand}")
            continue
        try:
            df = pd.read_csv(cand)
            df.insert(0, "Strategy", key)
            frames.append(df)
        except Exception as e:
            print(f"[WARN] failed to read {cand}: {e!r}")

    if not frames:
        print("[INFO] nothing to aggregate.")
        return

    # Align on union of columns
    all_cols: List[str] = []
    for f in frames:
        for c in f.columns:
            if c not in all_cols:
                all_cols.append(c)

    aligned = []
    for f in frames:
        for c in all_cols:
            if c not in f.columns:
                f[c] = None
        aligned.append(f[all_cols])

    out_csv = out_root / f"{prefix}_aggregate.csv"
    pd.concat(aligned, ignore_index=True).to_csv(out_csv, index=False)
    print(f"[OK] wrote aggregate CSV to {out_csv}")


# ---------------- CLI ----------------
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", required=True, help="Path to simulator script (python)")
    ap.add_argument("--out-dir", required=True, help="Output directory for all runs")
    ap.add_argument("--prefix", required=True, help="Filename prefix base for outputs")
    ap.add_argument("--config", default=None, help="Path to config.json (optional)")
    ap.add_argument(
        "--strategies",
        nargs="*",
        default=list(STRATEGY_MATRIX.keys()),
        help="Subset of strategy keys to run (default: all)",
    )
    # Anything you want to pass through to the simulator goes here:
    # e.g. "--seed 42 --admission soft --opt-module-... ..."
    ap.add_argument("--extras", default="", help="Raw string appended to each sim command")

    ap.add_argument("--plots", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--aggregate", action=argparse.BooleanOptionalAction, default=True)

    # NEW: run strategies in parallel
    ap.add_argument(
        "--parallel",
        action="store_true",
        help="Run all strategies in parallel (safe when using --pregen-in).",
    )

    # Shared pregen flags (mutually exclusive)
    ap.add_argument("--pregen-in", type=str, default=None,
                    help="Reuse this pregenerated CSV for all strategies")
    ap.add_argument("--pregen-out", type=str, default=None,
                    help="Create this pregenerated CSV once; others reuse it")

    args = ap.parse_args()

    if args.pregen_in and args.pregen_out:
        ap.error("Use only one: --pregen-in OR --pregen-out (not both).")

    # Ensure parent folder for a requested pregen-out
    if args.pregen_out:
        Path(args.pregen_out).parent.mkdir(parents=True, exist_ok=True)

    return args


# ---------------- Command builder ----------------
def build_cmd(
    sim_path: str,
    out_dir: Path,
    prefix: str,
    strat_key: str,                # key in STRATEGY_MATRIX
    strat_cfg: Dict[str, Any],     # dict from STRATEGY_MATRIX[key]
    extras: Optional[str],
    config_path: Optional[str],
    freq_map_path: Optional[str],
    *,
    pregen_in: Optional[str] = None,
    pregen_out: Optional[str] = None,
) -> List[str]:
    """
    Build a single-run simulator command. Uses --out-dir/--prefix.
    Strategy name comes from strat_cfg['name'] if present, else strat_key.
    """
    strategy_name = (strat_cfg.get("name") or str(strat_key)).strip()

    cmd: List[str] = [sys.executable, sim_path]
    if config_path:
        cmd += ["--config", config_path]

    cmd += [
        "--strategy", strategy_name,
        "--out-dir", str(out_dir),
        "--prefix",  prefix,
    ]

    # Frequency mode hints
    freq_mode = (strat_cfg.get("freq-mode") or "").strip().lower()
    if freq_mode == "fixed" and freq_map_path:
        cmd += ["--fixed-freq-map", freq_map_path]
    elif freq_mode in {"adjust", "adaptive"}:
        cmd += ["--enable-adaptive-freq"]

    # Optimizer modules if provided in matrix
    if strat_cfg.get("opt_latency"):
        cmd += ["--opt-module-latency", str(strat_cfg["opt_latency"])]
    if strat_cfg.get("opt_power"):
        cmd += ["--opt-module-power", str(strat_cfg["opt_power"])]
    if strat_cfg.get("opt_efficiency"):
        cmd += ["--opt-module-efficiency", str(strat_cfg["opt_efficiency"])]

    # Shared pregen flags
    if pregen_in:
        cmd += ["--pregen-in", pregen_in]
    if pregen_out:
        cmd += ["--pregen-out", pregen_out]

    # Any raw extras (e.g., "--seed 42 --admission soft ...")
    if extras:
        cmd += shlex.split(extras)

    return cmd

# ---------------- Main ----------------
def main() -> None:
    args = parse_args()

    # Resolve and create output root once
    out_root = Path(args.out_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    sim_path = args.sim

    successes: List[str] = []
    failures: Dict[str, str] = {}

    # Optional: if you ever construct a global fixed-freq map, write it once and reuse
    global_map: Dict[str, Any] = {}
    freq_map_path: Optional[str] = None

    # --- Normalize shared pregen paths so relative names go under out_root ---
    def _norm_under_out_root(p: Optional[str]) -> Optional[str]:
        if not p:
            return None
        pp = Path(p)
        # Keep absolute paths as-is; place relative ones inside out_root, flattening dirs
        return str(pp if pp.is_absolute() else (out_root / pp.name))

    shared_pregen_in  = _norm_under_out_root(args.pregen_in)
    shared_pregen_out = _norm_under_out_root(args.pregen_out)
    # ------------------------------------------------------------------------

    tasks = []  # list of (key, cmd)

    for idx, key in enumerate(args.strategies):
        strat = STRATEGY_MATRIX.get(key)
        if not strat:
            print(f"[WARN] unknown strategy key: {key} (skipping)")
            continue

        if strat.get("freq-mode") == "fixed" and global_map and freq_map_path is None:
            freq_map_path = str(out_root / f"{args.prefix}__fixed_freq_map.json")
            with open(freq_map_path, "w", encoding="utf-8") as f:
                json.dump(global_map, f, indent=2, sort_keys=True)

        out_prefix = args.prefix

        # shared pregen logic (unchanged)
        pregen_in: Optional[str]  = None
        pregen_out: Optional[str] = None
        if shared_pregen_in:
            pregen_in = shared_pregen_in
        elif shared_pregen_out:
            if Path(shared_pregen_out).exists():
                pregen_in = shared_pregen_out
            else:
                if idx == 0:
                    pregen_out = shared_pregen_out
                else:
                    pregen_in = shared_pregen_out

        cmd = build_cmd(
            sim_path=str(sim_path),
            out_dir=str(out_root),
            prefix=out_prefix,
            strat_key=key,
            strat_cfg=strat,
            extras=args.extras,
            config_path=args.config,
            freq_map_path=freq_map_path,
            pregen_in=pregen_in,
            pregen_out=pregen_out,
        )

        tasks.append((key, cmd))

    # -------- execute tasks (sequential or parallel) --------
    def _run_one(key_cmd):
        key, cmd = key_cmd
        print(">", " ".join(cmd), flush=True)
        try:
            res = subprocess.run(cmd, check=False, capture_output=True, text=True)
        except Exception as e:
            return key, None, f"spawn failed: {e!r}"
        return key, res, None

    if args.parallel:
        # max_workers = min(len(tasks), max(1, os.cpu_count() or 1))
        max_workers = min(3, len(tasks))
        with cf.ThreadPoolExecutor(max_workers=max_workers) as pool:
            for key, res, err in (f.result() for f in [pool.submit(_run_one, t) for t in tasks]):
                if err:
                    failures[key] = err
                    print(f"[FAIL] {key}: {err}")
                    continue

                log_path = out_root / f"{args.prefix}__{key}.log"
                if res.returncode != 0:
                    failures[key] = f"return code {res.returncode} (see {log_path.name})"
                    print(f"[FAIL] {key}: {failures[key]}")
                    continue

                written = list(out_root.glob(f"{args.prefix}*{key}*.csv"))
                if not written:
                    failures[key] = f"no outputs written with prefix '{args.prefix}' (see {log_path.name})"
                    print(f"[FAIL] {key}: {failures[key]}")
                else:
                    successes.append(key)
    else:
        # original sequential behavior
        for key, cmd in tasks:
            key, res, err = _run_one((key, cmd))
            log_path = out_root / f"{args.prefix}__{key}.log"
            if err:
                failures[key] = err
                print(f"[FAIL] {key}: {err}")
                continue
            if res.returncode != 0:
                failures[key] = f"return code {res.returncode} (see {log_path.name})"
                print(f"[FAIL] {key}: {failures[key]}")
                continue
            written = list(out_root.glob(f"{args.prefix}*{key}*.csv"))
            if not written:
                failures[key] = f"no outputs written with prefix '{args.prefix}' (see {log_path.name})"
                print(f"[FAIL] {key}: {failures[key]}")
            else:
                successes.append(key)

    print(f"[INFO] strategies requested: {len(args.strategies)}")
    if successes:
        print(f"[INFO] succeeded: {len(successes)} -> {successes}")
    if failures:
        print(f"[INFO] failed: {len(failures)} -> {failures}")

    # 1) Aggregate per-strategy CSVs into one file
    if args.aggregate and successes:
        try:
            aggregate_results(out_root, successes, args.prefix)
        except Exception as e:
            print(f"[WARN] aggregation failed: {e!r}")


    # Regenerate metrics aggregate + plots (separate scripts)
    if args.aggregate:
        try:
            subprocess.run(
                [sys.executable, "metrics_aggregate.py",
                 "--runs-dir", str(out_root), "--out-prefix", args.prefix],
                check=True
            )
            print("[OK] metrics_aggregate complete")
        except Exception as e:
            print(f"[WARN] metrics_aggregate failed: {e!r}")


if __name__ == "__main__":
    main()


# Run Command:
# python queuing_system_simulation_incremental.py \
#   --config config.json \
#   --multi "L:4.0, 5.0, 6.0, 7.0, 8.0;runs:5;seed0:41;outdir:runs;prefix:exp1" \
#   --enable-adaptive-freq \
#   --fast-decider \
#   --opt-module-latency    live_min_latency_optimizer_incremental \
#   --opt-module-power      live_min_power_optimizer_incremental \
#   --opt-module-efficiency live_max_efficiency_optimizer_incremental \
#   --admission soft