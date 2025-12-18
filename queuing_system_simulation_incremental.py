
"""
FEL-based end-to-end simulator with pre-generation.
Jobs → Tasks → UL packets are generated before the FEL run. At UL arrival
the simulator assigns (cluster, node, gpu, frequency) so link and service rates
are known deterministically. Dynamic behavior is read from `config.json`.

Highlights
- Deterministic FEL ordering via (time, priority, seq).
- Separate per-cluster stations: R0_UL:<C>, RC_UL:<C>, RC_DL:<C>, R0_DL:<C>.
- UL first-hop equality enforced: R0_UL_IN_entry == Packet_Arrival_Time.
- Clean CSV schema (stable) with queueing and service stamps for every hop.
- Downlink arrivals are scheduled after GPU completion (simultaneous or
  staggered) and clamped to ≥ GPU exit.

Assignment & DVFS
- Two assignment modes:
  1) Heuristics: least-load. DVFS can be:
     • heuristic-adaptive:least-load,
     • heuristic-fixed: pin to `--fixed-frequency` (fastest)
  2) Optimizer: objective-driven (latency / power / efficiency). DVFS can be:
     • solver-adaptive: solver may change freqs
     • solver-fixed: solver must honor fixed per-GPU freqs provided in the snapshot
- Incremental assignment cache: once a task is assigned, (c,n,g,freq) is reused
  for that task base ID. One frequency per GPU (C1) is kept in sync between the
  cache/store and `SimState.gpu_to_freq`.

OptimizerFacade (live only)
- Accepts a module by import name or `.py` path (per objective or shared).
- Preferred API (used if available):
    solve_incremental(snapshot, new_tasks, objective=..., pinned_assignments=..., pinned_frequencies=...)
- Fallbacks supported transparently:
    solve(...), solve_latency[_incremental](...), solve_power(...), solve_efficiency(...)
- The simulator passes `objective in {"latency","power","efficiency"}`.
- Snapshot includes:
    {"now","gpus": {(C,N,G):{"tail","queued","freq","rates","P_static_W","C_p","power_exp"}},
     "links","pinned_assignments", "pinned_frequencies"}

Objectives (semantic intent)
- "latency"     : minimize overall delay.
- "power"       : minimize power consumption (≈ service_time * power(freq)).
- "efficiency"  : maximize throughput / power.

Frequency handling
- If `pinned_frequencies` (non-empty) is present in the optimizer snapshot,
  frequencies are treated as **fixed**; the solver should not propose changes.
- Otherwise, the solver may return a frequency map and the simulator will update `gpu_to_freq`.
- Power modeling:
  • If `GPU_Specs[type].power` table exists, it is used directly.
  • Else parametric fallback P = P_static_W + C_p * f**power_exp (gentle defaults).

Configuration & Assumptions
- `snapshot["gpus"]` is keyed by (c,n,g) tuples (or "c-n-g" strings in exports):
    {"tail": float, "queued": int, "freq": str|None, "rates": {freq: rate}}
- `GPU_Specs` carries per-type defaults/power params.
- `Network_Settings` may set NUM_PORTS_PER_CLUSTER, per-cluster overrides, queue
  capacity, and FAST_RC_UL/DL shortcuts. DL arrival mode can be "simultaneous"
  or "staggered".

Pre-generation
- `GenConfig` derives from `config.json`. 

Outputs
- Task/packet CSV with stamps for UL, RC_UL, GPU, RC_DL, R0_DL, plus
  overall_start/end and deadline violation flags.
- (Optional) Optimizer bundle JSON: `tasks[]` (per-task assignment/duration) and
  `frequencies[]` (one freq per GPU), naturally ordered.

CLI (key flags)
- `--strategy` one of:
    least-load_adaptive, least-load_fixed,
    random_adaptive, random_fixed,
    opt_latency_adaptive, opt_latency_fixed,
    opt_power_adaptive,   opt_power_fixed,
    opt_efficiency_adaptive, opt_efficiency_fixed
  (Maps to {heuristic|optimizer} × DVFS policy.)
- `--opt-module[-latency|-power|-efficiency]` to select optimizer modules.
- `--freq-mode {adaptive,fixed}` and `--fixed-frequency` (for heuristic-fixed or
  solver-fixed runs).
- `--assign-key {base,full}` controls whether exports drop _U/_D.



Timing invariants (UL → GPU)
----------------------------
Aggregation rule (important):
• `gpu_entry_time` is stamped when the **task** (not a single packet) first
  enters the GPU. In other words, it equals the arrival time of the **last**
  UL packet of that task at the GPU ingress.

Therefore:
• If `net.fast_rc_ul == True` (RC uplink hop bypassed, i.e., no RC_UL queue):
  There is **no** RC_UL queue formation in this mode.


"""

from __future__ import annotations

import gc
import inspect
import argparse, csv, heapq, json, math
import random, re, time
import os as _os
import traceback
import numpy as np
import pandas as pd
import subprocess, sys, pathlib
import importlib.util
import hashlib as _hl
import importlib, importlib.util, os as _os, sys as _sys
from pathlib import Path as _Path
import concurrent.futures as cf


from collections import defaultdict, deque, Counter
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Any, Callable, DefaultDict, Dict, List, Optional, Set, Tuple, TypedDict, cast
from copy import deepcopy as _deepcopy
from collections import defaultdict as _dd
from dataclasses import field
from bisect import bisect_left
from fast_decider import FastPolicy  

# =============================
# CSV schema & helpers
# =============================

JOB_FIELDS = [
    "Lambda_per_s",
    "Job_ID", "Job_Start_Time", "Job_End_Time",
]

TASK_FIELDS = [
    "Direction", "Task_ID", "Task_Size_KB", "Task_Arrival_Rate",
    "Task_Arrival_Time", "Task_Deadline", "Workload_FLOPs",
    "Task_Is_First", "Task_Is_Last",
]

PACKET_FIELDS = [
    "Packet_id", "Packet_Size_KB", "Packet_Arrival_Time",
    "Is_First", "Is_Last",
]

ASSIGN_FIELDS = [
    "Assigned_Cluster", "Assigned_Node", "Assigned_GPU", "Assigned_Frequency",
    "Service_Frequency",
    "Freq_Mode", "Freq_Decision_Source", "GPU_Decision_Source",
    "UL_Port",
]

HOP_FIELDS = [
    "R0_UL_IN_entry", "R0_UL_service_start", "R0_UL_service_time", "R0_UL_EG_exit",
    "R0_UL_queue_delay", "R0_UL_prop_delay",
    "gpu_entry_time", "gpu_service_time", "gpu_exit_time", "gpu_queue_delay",
    "RC_DL_IN_entry", "RC_DL_service_start", "RC_DL_service_time", "RC_DL_IN_exit",
    "RC_DL_IN_delay",
    "R0_DL_prop_delay",
    "R0_DL_IN_entry", "R0_DL_service_start", "R0_DL_service_time", "R0_DL_IN_exit",
    "R0_DL_IN_delay",
]

OVERALL_FIELDS = [
    "overall_start_time", "overall_end_time", "Total_Completion_Time",
    "Task_Deadline_Violation", "Task_Status",
]

CSV_FIELDS = JOB_FIELDS + TASK_FIELDS + PACKET_FIELDS + ASSIGN_FIELDS + HOP_FIELDS + OVERALL_FIELDS

GLOBAL_OPT = None

# Global buffer to expose DVFS log to main for CSV export
__dvfs_log_buffer__: List[Dict[str, Any]] = []

def _dbg(state, *a):
    if bool(getattr(state, "debug_dvfs", False)):
        try:
            print("[DVFS][DBG]", *a)
        except Exception:
            pass

def _to_float(x, default=None):
    try:
        if x is None or x == "": return default
        return float(x)
    except Exception:
        return default

    
def _safe_float(x, default=None):
    try:
        return float(x)
    except Exception:
        return default


def clamp(x, lo, hi):
    """
    Clamp x to the closed interval [lo, hi], with safety for NaN/inf and swapped bounds.
    Falls back to lo on unexpected errors.
    """
    try:
        x = float(x); lo = float(lo); hi = float(hi)
        if not (math.isfinite(x) and math.isfinite(lo) and math.isfinite(hi)):
            # If x is NaN or inf, return lo (safe fallback)
            return lo if (not math.isfinite(x)) else max(lo, min(hi, x))
        if lo > hi:  # handle reversed bounds
            lo, hi = hi, lo
        return max(lo, min(hi, x))
    except Exception:
        return lo

# --- boolean normalization helpers ---
BOOL_KEYS = ("Is_First","Is_Last","Task_Is_First","Task_Is_Last")
_TRUE  = {"1","t","y","yes","on","true"}
_FALSE = {"0","f","n","no","off","false",""}

# top-level near imports
EPS = 1e-9  # prevents same-time ordering/rounding surprises

def to01(v) -> int:
    if v is None:
        return 0
    if isinstance(v, bool):
        return 1 if v else 0
    if isinstance(v, int):
        return 1 if v != 0 else 0
    if isinstance(v, float):
        if math.isnan(v):
            return 0
        return 1 if v != 0.0 else 0
    if isinstance(v, str):
        s = v.strip().lower()
        if s in _TRUE:
            return 1
        if s in _FALSE:
            return 0
        # last resort numeric parse
        try:
            x = float(s)
            if math.isnan(x):
                return 0
            return 1 if x != 0.0 else 0
        except Exception:
            return 0
    return 0

def normalize_packet_flags(p: dict) -> None:
    t = to01
    get = p.get
    for k in BOOL_KEYS:
        p[k] = t(get(k, 0))

# --- id + rounding helpers ---
def base_tid(tid: str) -> str:
    tid = str(tid)
    if tid.endswith("_U") or tid.endswith("_D"):
        return tid[:-2]
    return tid

def r6(x, nd=6):
    try: return round(float(x), nd)
    except Exception: return x

def _task_base_id(tid: str) -> str:
    """
    Collapse AR_J0_T0_U and AR_J0_T0_D into AR_J0_T0.
    If there's no _U/_D suffix, return as-is.
    """
    if not isinstance(tid, str):
        return str(tid)
    if tid.endswith("_U") or tid.endswith("_D"):
        return tid.rsplit("_", 1)[0]
    return tid



# == Log ==
def write_csv(path: str, rows: List[Dict[str, Any]]):
    _os.makedirs(_os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        for r in rows:
            out = {k: r.get(k, "") for k in CSV_FIELDS}
            w.writerow(out)

def write_pregen_csv(path: str, rows: List[Dict[str, Any]], lam: Optional[float] = None):
    fields = JOB_FIELDS + TASK_FIELDS + PACKET_FIELDS  # pregenerated has no assignment/hop stamps
    _os.makedirs(_os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            rr = dict(r)
            if lam is not None:
                rr["Lambda_per_s"] = lam
            w.writerow({k: rr.get(k, "") for k in fields})

def read_pregen_csv(path: str) -> List[Dict[str, Any]]:
    """
    Read a pregenerated UL/DL packet CSV and return rows in the internal format expected by run_sim().
    Backward compatible: Lambda_per_s optional.
    """
    rows: List[Dict[str, Any]] = []
    with open(path, "r", newline="") as f:
        rd = csv.DictReader(f)

        def _i(x, d=0):
            try: return int(float(x))
            except: return int(d)

        def _f(x, d=0.0):
            try: return float(x)
            except: return float(d)

        def _s(x, d=""):
            return str(x) if x is not None else d

        for r in rd:
            rows.append({
                "Lambda_per_s":         _f(r.get("Lambda_per_s", ""), d=0.0),

                "Direction":            _s(r.get("Direction", "Uplink")),
                "Service_ID":           _s(r.get("Service_ID", "AR")),

                "Job_ID":               _s(r.get("Job_ID", "")),
                "Job_Start_Time":       _f(r.get("Job_Start_Time", 0.0)),
                "Job_End_Time":         _f(r.get("Job_End_Time", 0.0)),

                "Task_ID":              _s(r.get("Task_ID", "")),
                "Task_Size_KB":         _i(r.get("Task_Size_KB", 0)),
                "Task_Arrival_Rate":    _f(r.get("Task_Arrival_Rate", 0.0)),
                "Task_Arrival_Time":    _f(r.get("Task_Arrival_Time", 0.0)),
                "Task_Deadline":        _f(r.get("Task_Deadline", 0.0)),
                "Workload_FLOPs":       _i(r.get("Workload_FLOPs", 0)),
                "Task_Is_First":        _i(r.get("Task_Is_First", 0)),
                "Task_Is_Last":         _i(r.get("Task_Is_Last", 0)),

                "Packet_id":            _s(r.get("Packet_id", "")),
                "Packet_Size_KB":       _i(r.get("Packet_Size_KB", 0)),
                "Packet_Arrival_Time":  _f(r.get("Packet_Arrival_Time", 0.0)),
                "Is_First":             _i(r.get("Is_First", 0)),
                "Is_Last":              _i(r.get("Is_Last", 0)),

                # Optional pre-assignments (keep empty if missing)
                "Assigned_Cluster":     _s(r.get("Assigned_Cluster", "")),
                "Assigned_Node":        _s(r.get("Assigned_Node", "")),
                "Assigned_GPU":         _s(r.get("Assigned_GPU", "")),
                "Assigned_Frequency":   _s(r.get("Assigned_Frequency", "")),
            })
    return rows

def build_task_summaries_from_state(sim_state) -> List[Dict[str, Any]]:
    """
    Build ONE row per logical task using sim_state.task_times.

    Expects sim_state.task_times[(Job_ID, Task_ID_base)] to contain:
        "start", "end", "deadline_violation", "status"
    """
    out_rows: List[Dict[str, Any]] = []
    task_times = getattr(sim_state, "task_times", {}) or {}

    for (job_id, task_base), tinfo in task_times.items():
        st = tinfo.get("start")
        en = tinfo.get("end")
        if st is None or en is None:
            continue

        out_rows.append({
            "Job_ID":             str(job_id),
            "Task_ID":            str(task_base),
            "start_time":         float(st),
            "end_time":           float(en),
            "latency_s":          float(en) - float(st),
            "deadline_violation": int(to01(tinfo.get("deadline_violation", 0))),
            "Task_Status":        str(tinfo.get("status", "")),
        })

    return out_rows

def build_task_summaries_from_packet_rows(packet_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Collapse per-packet rows into ONE row per logical task.
    A 'logical task' spans uplink+downlink (_U and _D).
    """
    by_task: Dict[Tuple[str, str], Dict[str, Any]] = {}

    for r in (packet_rows or []):
        job_id   = str(r.get("Job_ID", ""))
        raw_tid  = str(r.get("Task_ID", ""))
        base_tid = _task_base_id(raw_tid)
        key      = (job_id, base_tid)

        t_start  = _to_float(r.get("overall_start_time"), default=None)
        t_end    = _to_float(r.get("overall_end_time"), default=None)

        viol_flag = 1 if to01(r.get("Task_Deadline_Violation", 0)) == 1 else 0
        status    = str(r.get("Task_Status", "") or "")
        is_last   = to01(r.get("Is_Last", 0))

        rec = by_task.get(key)
        if rec is None:
            rec = {
                "Job_ID": job_id,
                "Task_ID_base": base_tid,
                "start_time": t_start,
                "end_time":   t_end,
                "deadline_violation": viol_flag,
                "Task_Status": status,
                "saw_final_packet": False,
            }
            by_task[key] = rec
        else:
            if t_start is not None and (rec["start_time"] is None or t_start < rec["start_time"]):
                rec["start_time"] = t_start
            if t_end is not None and (rec["end_time"] is None or t_end > rec["end_time"]):
                rec["end_time"] = t_end
            rec["deadline_violation"] = max(rec["deadline_violation"], viol_flag)

        if is_last == 1:
            rec["Task_Status"] = status or rec["Task_Status"]
            rec["saw_final_packet"] = True

    out_rows: List[Dict[str, Any]] = []
    for rec in by_task.values():
        st = rec.get("start_time")
        en = rec.get("end_time")
        latency = (en - st) if (st is not None and en is not None) else None

        out_rows.append({
            "Job_ID":             rec["Job_ID"],
            "Task_ID":            rec["Task_ID_base"],
            "start_time":         st,
            "end_time":           en,
            "latency_s":          latency,
            "deadline_violation": rec["deadline_violation"],
            "Task_Status":        rec.get("Task_Status", ""),
        })

    return out_rows

def write_task_summary_csv(path: str, rows: List[Dict[str, Any]]):
    fields = [
        "Strategy","Freq_Mode","Admission","Lambda_per_s","Seed",
        "Task_ID","Job_ID",
        "start_time","end_time","latency_s",
        "deadline_violation","Task_Status",
    ]

    _os.makedirs(_os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})

def _basename_no_ext(path: str) -> str:
    b = _os.path.basename(str(path))
    return _os.path.splitext(b)[0]

def build_run_tag(args) -> str:
    bits: List[str] = []

    if args.assigner == "optimizer":
        if args.objective == "latency":
            mod_tag = args.opt_module_latency or args.opt_module or \
                      args.opt_module_power or args.opt_module_efficiency or "nomod"
        elif args.objective == "power":
            mod_tag = args.opt_module_power or args.opt_module or \
                      args.opt_module_latency or args.opt_module_efficiency or "nomod"
        else:
            mod_tag = args.opt_module_efficiency or args.opt_module or \
                      args.opt_module_latency or args.opt_module_power or "nomod"
        bits += ["opt", _basename_no_ext(mod_tag)]
    else:
        bits += ["heur", args.heuristic]

    bits += [args.objective, f"freq-{args.freq_mode}"]
    if args.freq_mode == "fixed" and getattr(args, "fixed_frequency", ""):
        bits[-1] += f"-{args.fixed_frequency}"
    bits += [f"adm-{args.admission}", f"seed{args.seed}"]

    tag = "_".join(bits)
    return tag.replace("/", "-").replace(" ", "_")

def resolve_paths(args):
    """
    Compute filenames when user didn't override, using a consistent tag.
    Only create incremental-store paths if incremental mode is enabled.
    """
    tag = build_run_tag(args)

    out_csv   = args.out or f"task_packets_summary_{tag}.csv"
    pre_csv   = args.pregen_out or f"pre_generated_jobs_{tag}.csv"
    export_js = args.export_opt_array or f"optimizer_solution_{tag}.json"

    if getattr(args, "incremental_mode", "off") != "off":
        inc_json = args.incremental_store or f"assignments_cache_{tag}.json"
    else:
        inc_json = None

    return out_csv, pre_csv, export_js, inc_json



# =============================
# Incremental Assignment Store
# =============================

class AssignmentStore:
    def __init__(self, path: Optional[str] = None, mode: str = "readwrite",
                 assignments: Optional[Dict[str, Dict[str, str]]] = None,
                 frequencies: Optional[Dict[str, str]] = None):
        self.path = path
        self.mode = (mode or "readwrite").lower()  # off | readonly | readwrite
        self.assignments: Dict[str, Dict[str, str]] = dict(assignments or {})
        self.frequencies: Dict[str, str] = dict(frequencies or {})

    # -------- basic API --------
    def has(self, tid_base: str) -> bool:
        return tid_base in self.assignments

    def get(self, tid_base: str) -> Dict[str, str]:
        return self.assignments.get(tid_base, {})

    def put(self, tid_base: str, rec: Dict[str, str]) -> None:
        self.assignments[tid_base] = {
            "Cluster": str(rec.get("Cluster","")),
            "Node":    str(rec.get("Node","")),
            "GPU":     str(rec.get("GPU","")),
            "Frequency": str(rec.get("Frequency","")),
        }
        self._flush_if_needed()

    def set(self, tid_base: str, rec: dict) -> None:
        """Alias for put(); some callers use store.set(...)."""
        self.put(tid_base, rec)

    def set_freq(self, gid: str, freq: str) -> None:
        self.frequencies[str(gid)] = str(freq)
        self._flush_if_needed()

    # -------- persistence --------
    def _flush_if_needed(self) -> None:
        if not self.path or self.mode == "readonly":
            return
        _os.makedirs(_os.path.dirname(self.path) or ".", exist_ok=True)

        # Build payload: always write assignments; write frequencies only for FIXED runs
        payload = {"assignments": self.assignments}

        try:
            fixed_run = not _is_adaptive_run(getattr(self, "cfg", {}) or {})
        except Exception:
            # Be conservative: if we can't tell, treat as FIXED (old behavior)
            fixed_run = True

        if fixed_run:
            payload["frequencies"] = getattr(self, "frequencies", {})

        with open(self.path, "w") as f:
            json.dump(payload, f, indent=2)

    @classmethod
    def from_optimizer_json(cls, path: Optional[str]):
        """Load from either an incremental cache or an optimizer bundle file."""
        if not path or not _os.path.exists(path):
            return cls(path=path)
        try:
            with open(path, "r") as f:
                data = json.load(f)
        except Exception:
            return cls(path=path)

        # Case A: incremental cache schema
        if isinstance(data, dict) and "assignments" in data:
            return cls(path=path,
                       assignments=data.get("assignments") or {},
                       frequencies=data.get("frequencies") or {})

        # Case B: optimizer bundle schema
        assignments = {}
        freqs = {}
        if "tasks" in data:
            for t in data.get("tasks", []):
                tidb = base_tid(str(t.get("Task_ID","")))
                assignments[tidb] = {
                    "Cluster": str(t.get("Cluster","")),
                    "Node":    str(t.get("Node","")),
                    "GPU":     str(t.get("GPU","")),
                    "Frequency": str(t.get("Frequency","")),
                }
        if "frequencies" in data:
            if isinstance(data["frequencies"], list):
                # [{"GPU":"C1-N1-G1","Frequency":"910"}, ...]
                freqs = {str(x.get("GPU")): str(x.get("Frequency",""))
                         for x in data["frequencies"]}
            elif isinstance(data["frequencies"], dict):
                freqs = {str(k): str(v) for k, v in data["frequencies"].items()}
        return cls(path=path, assignments=assignments, frequencies=freqs)


def _reconcile_store_freqs(store: "AssignmentStore", cfg: dict) -> None:
    """
    Normalize store.frequencies and, for ADAPTIVE runs, remove any static pins.

    - ADAPTIVE: clear store.frequencies (no static pinning).
    - FIXED:    normalize keys (e.g., 'C1-N1-G1'), ensure values are strings,
                and if frequencies are missing, derive them by majority vote
                over per-task assignment 'Frequency' (ties -> higher MHz).
    """
    if store is None:
        return

    # ADAPTIVE → do not keep a static pin
    try:
        if _is_adaptive_run(cfg):
            if getattr(store, "mode", "readwrite") != "readonly":
                store.frequencies = {}
            return
    except Exception:
        # If cfg missing/bad we fall through and keep old behavior.
        pass

    # FIXED path: normalize existing frequencies map
    freqs_norm = {}
    for gid, f in (getattr(store, "frequencies", {}) or {}).items():
        gid_s = str(gid).replace(",", "-").strip()
        f_s   = "" if f is None else str(f).strip()
        if gid_s and f_s:
            freqs_norm[gid_s] = f_s

    # If nothing provided, infer per-GPU fixed frequencies from assignments
    if not freqs_norm:
        votes = defaultdict(Counter)
        for rec in (getattr(store, "assignments", {}) or {}).values():
            gid = f"{rec.get('Cluster')}-{rec.get('Node')}-{rec.get('GPU')}"
            f_s = str(rec.get("Frequency", "")).strip()
            if gid and f_s:
                votes[gid][f_s] += 1

        def _tie_break(item):
            # Prefer higher count; if tied, prefer higher MHz numerically.
            f_str, cnt = item
            try:
                return (cnt, float(f_str))
            except Exception:
                return (cnt, -1.0)

        for gid, ctr in votes.items():
            if ctr:
                best = max(ctr.items(), key=_tie_break)[0]
                freqs_norm[gid] = best

    # Persist (unless readonly)
    if getattr(store, "mode", "readwrite") != "readonly":
        store.frequencies = freqs_norm


# =============================
# Optimizer-bundle exporter
# =============================

def _key_task_id(task_id: str, mode: str = "base", job_id: str | None = None) -> str:
    """
    How to name tasks in the exported JSON.
      - "base":     strip _U/_D (e.g., AR_J0_T0_U -> AR_J0_T0)  [DEFAULT]
      - "job_base": Job + base (e.g., AR_J0_T0 -> AR_J0_T0, same as base here;
                    useful if Task_ID didn't already include Job)
      - "full":     keep full Task_ID (e.g., AR_J0_T0_U)
    """
    tbase = base_tid(str(task_id))
    if mode == "full":
        return str(task_id)
    if mode == "job_base":
        return f"{job_id}:{tbase}" if job_id else tbase
    return tbase  # "base"

_TID_RE = re.compile(r'(?P<svc>.+?)_J(?P<job>\d+)_T(?P<task>\d+)(?:_[UD])?$')
_GID_RE = re.compile(r'^C(?P<c>\d+)-N(?P<n>\d+)-G(?P<g>\d+)$')

def _tid_key(tid: str):
    m = _TID_RE.match(tid)
    return (m.group('svc'), int(m.group('job')), int(m.group('task'))) if m else (tid,)

def _gid_key(gid: str):
    m = _GID_RE.match(gid)
    return (int(m.group('c')), int(m.group('n')), int(m.group('g'))) if m else (gid,)

def _to_f(x):
    try: return float(x)
    except Exception: return None

def export_optimizer_array(rows: List[Dict[str, Any]], out_path: str, keying: str = "base") -> Dict[str, Any]:
    """
    Build:
      {
        "tasks":       [ {Task_ID, Cluster, Node, GPU, Frequency, Total_D}, ... ],
        "frequencies": [ {GPU, Frequency}, ... ]
      }
    from simulation rows and write JSON to out_path.
    Task array is *naturally* ordered by (service, job#, task#).
    """

    # def _key_task_id(tid: str, mode: str) -> str:
    #     if mode == "base":
    #         return tid[:-2] if (tid.endswith("_U") or tid.endswith("_D")) else tid
    #     return tid

    # 1) Aggregate per task
    by_task = defaultdict(list)
    for r in rows:
        job   = str(r.get("Job_ID", ""))
        tbase = _key_task_id(str(r.get("Task_ID","")), keying)  # <- use the function/arg defined above
        by_task[(job, tbase)].append(r)


    # 2) Build assignments, per-GPU freq votes, and total duration
    per_gpu_freq_counts: Dict[str, Counter] = defaultdict(Counter)
    tasks_out: List[Dict[str, Any]] = []
    seen_task_ids = set()

    # helper: pick numeric-max on tie
    def _best_freq(counter: Counter) -> str:
        def k(item):
            f_str, cnt = item
            try: f_num = float(f_str)
            except: f_num = float("-inf")
            return (cnt, f_num)
        return max(counter.items(), key=k)[0] if counter else ""

    # Make a first pass to collect GPU frequency votes
    for (job, tbase), rows_task in by_task.items():
        for r in rows_task:
            c = str(r.get("Assigned_Cluster","")).strip()
            n = str(r.get("Assigned_Node","")).strip()
            g = str(r.get("Assigned_GPU","")).strip()
            if c and n and g:
                gid = f"{c}-{n}-{g}"
                f = str(r.get("Assigned_Frequency","")).strip()
                if f:
                    per_gpu_freq_counts[gid][f] += 1

    # Decide a single frequency per GPU (C1)
    gpu_freq: Dict[str, str] = {gid: _best_freq(cnt) for gid, cnt in per_gpu_freq_counts.items() if cnt}

    # 3) Build the ordered task objects
    #    Sort keys naturally by their tbase ("AR_J12_T3" etc.)
    task_keys_sorted = sorted(by_task.keys(), key=lambda k: _tid_key(k[1]))
    for (job, tbase) in task_keys_sorted:
        rows_task = by_task[(job, tbase)]

        # assignment (prefer any UL row)
        c = n = g = ""
        for r in rows_task:
            if str(r.get("Direction","")).lower() == "uplink":
                c = str(r.get("Assigned_Cluster","")).strip()
                n = str(r.get("Assigned_Node","")).strip()
                g = str(r.get("Assigned_GPU","")).strip()
                if c and n and g:
                    break
        if not (c and n and g):
            # fallback: take from any row that has an assignment
            for r in rows_task:
                cc = str(r.get("Assigned_Cluster","")).strip()
                nn = str(r.get("Assigned_Node","")).strip()
                gg = str(r.get("Assigned_GPU","")).strip()
                if cc and nn and gg:
                    c, n, g = cc, nn, gg
                    break

        # duration (any row with overall_* is fine; they’re the same per task)
        D = None
        for r in rows_task:
            t0, t1 = _to_f(r.get("overall_start_time")), _to_f(r.get("overall_end_time"))
            if t0 is not None and t1 is not None:
                D = round(t1 - t0, 6)
                break

        # frequency chosen for this task’s GPU (from global per-GPU decision)
        freq = gpu_freq.get(f"{c}-{n}-{g}", "")

        # emit if we have at least an ID and assignment
        tid = tbase
        if tid in seen_task_ids:   # guard against dupes
            continue
        seen_task_ids.add(tid)

        tasks_out.append({
            "Task_ID":  tid,
            "Cluster":  c,
            "Node":     n,
            "GPU":      g,
            "Frequency": freq,
            "Total_D":  D
        })

    # 4) Build frequency array (naturally ordered by C,N,G)
    freqs_out = [
        {"GPU": gid, "Frequency": f}
        for gid, f in sorted(gpu_freq.items(), key=lambda kv: _gid_key(kv[0]))
    ]

    bundle = {"tasks": tasks_out, "frequencies": freqs_out}

    _os.makedirs(_os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(bundle, f, indent=2)   # arrays preserve order; no sort_keys

    return bundle



# =============================
# Config Readers
# =============================

def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)

# --- helper: build per-cluster link capacities from config --------------------
def _build_cluster_link_caps(cfg: dict) -> dict[str, dict[str, float]]:
    """
    Returns: { 'C1': {'ul_rate_kBps':..,'dl_rate_kBps':..,'rc_rate_kBps':..,
                      'ul_prop_s':..,'dl_prop_s':..}, ... }

    Sources:
      cfg['R0_RC_Links']       : { 'R0-RC1': {'r0_c','rc_0','uplink_prop_delay','downlink_prop_delay', ...}, ... }
      cfg['RC_To_Cluster_Map'] : { 'RC1':'C1', 'RC2':'C2', ... }
    Fallback: infer RCx->Cx from link keys if map is missing.
    """
    links_cfg = (cfg.get("R0_RC_Links") or {})
    rc_to_c   = dict(cfg.get("RC_To_Cluster_Map") or {})

    # infer mapping if not given (R0-RCk -> Ck)
    if not rc_to_c:
        for name in links_cfg.keys():
            if "-" in name:
                rc = name.split("-", 1)[-1]  # 'RC1'
                if rc.startswith("RC") and rc[2:].isdigit():
                    rc_to_c[rc] = f"C{rc[2:]}"

    out = {}
    for name, rec in links_cfg.items():
        if "-" not in name:
            continue
        rc = name.split("-", 1)[-1]     # 'RC1'
        c  = rc_to_c.get(rc)
        if not c:
            continue

        # capacities (KB/s)
        r0_c = float(rec.get("r0_c", 0))   # R0 -> RC
        rc_0 = float(rec.get("rc_0", 0))   # RC -> R0

        # allow explicit overrides; else default from r0_c / rc_0
        ul = float(rec.get("ul_rate_kBps", r0_c))
        dl = float(rec.get("dl_rate_kBps", rc_0))          # <-- downlink default from rc_0
        rc_rate = float(rec.get("rc_rate_kBps", min(r0_c, rc_0) or 0.0))

        up  = float(rec.get("uplink_prop_delay",   rec.get("ul_prop_s", 0.0)))
        dwn = float(rec.get("downlink_prop_delay", rec.get("dl_prop_s", 0.0)))

        out[str(c)] = {
            "ul_rate_kBps": ul,
            "dl_rate_kBps": dl,
            "rc_rate_kBps": rc_rate,     # <-- ensure present
            "ul_prop_s":    up,
            "dl_prop_s":    dwn,
        }
    return out

def gpu_rates_from_cluster_config(cfg: dict) -> Dict[Tuple[str,str,str], Dict[str, float]]:
    rates = {}
    cc = cfg.get("Cluster_Config", {})
    for c, nodes in cc.items():
        for n, gpus in nodes.items():
            for g, ginfo in gpus.items():
                fmap = ginfo.get("service_rates", {})
                rates[(str(c), str(n), str(g))] = {str(k): float(v) for k, v in fmap.items()}
    return rates

def per_cluster_links(cfg: dict):
    rc_to_cluster = cfg.get("RC_To_Cluster_Map", {})
    links = (cfg.get("R0_RC_Links") or cfg.get("R0_RC_LINKS") or cfg.get("links") or {})

    per_cluster = {}
    for link_id, l in links.items():
        rc = link_id.split("-")[1] if "-" in link_id else link_id
        cluster = rc_to_cluster.get(rc)
        if not cluster:
            continue
        ul  = l.get("ul_rate_kBps", l.get("r0_c"))
        dl  = l.get("dl_rate_kBps", l.get("rc_0"))    
        rc0 = l.get("rc_rate_kBps", l.get("rc_0"))

        per_cluster[str(cluster)] = {
            "ul_rate_kBps": float(ul  if ul  is not None else 0.0),
            "dl_rate_kBps": float(dl  if dl  is not None else 0.0),
            "rc_rate_kBps": float(rc0 if rc0 is not None else 0.0),  # <-- ensure present
            "ul_prop_s": float(l.get("uplink_prop_delay",   0.0)),
            "dl_prop_s": float(l.get("downlink_prop_delay", 0.0)),
        }
    return per_cluster

def ports_per_cluster_from_config(cfg: dict) -> Dict[str, int]:
    net = cfg.get("Network_Settings", {}) or {}
    global_default = int(net.get("NUM_PORTS_PER_CLUSTER", 4))
    overrides = net.get("PORTS_PER_CLUSTER", {}) or {}
    per = {}
    for link_id in (cfg.get("R0_RC_Links", {}) or {}):
        rc = link_id.split("-")[1] if "-" in link_id else link_id
        cluster = cfg.get("RC_To_Cluster_Map", {}).get(rc)
        if not cluster:
            continue
        per[str(cluster)] = int(overrides.get(cluster, global_default))
    if not per and "Cluster_Config" in cfg:
        for cluster in cfg["Cluster_Config"].keys():
            per[str(cluster)] = int(overrides.get(cluster, global_default))
    return per or {}

class UtilRec(TypedDict):
    ewma: float
    last_t: float
    busy: int

def _util_init(state):
    if not hasattr(state, "gpu_util"):
        state.gpu_util = cast(Dict[Tuple[str, str, str], UtilRec], {})

def _util_touch(state, key, now, tau_s: float, busy_flag: int = None):
    r"""
    Continuous-time EWMA of busy(t) \in {0,1}.

    ewma(now) = exp(-dt/tau)*ewma(prev) + (1 - exp(-dt/tau))*busy(prev)
    If busy_flag is not None, update busy state AFTER folding interval [last_t, now].
    """
    _util_init(state)
    rec = state.gpu_util.setdefault(key, {"ewma": 0.0, "last_t": float(now), "busy": 0})
    dt = max(0.0, float(now) - float(rec["last_t"]))
    if dt > 0.0:
        k = math.exp(-dt / max(1e-6, float(tau_s)))  # decay factor
        rec["ewma"] = k * rec["ewma"] + (1.0 - k) * float(rec["busy"])
        rec["last_t"] = float(now)
    if busy_flag is not None:
        rec["busy"] = 1 if int(busy_flag) else 0

def _util_get(state, key, now, tau_s: float) -> float:
    # bring EWMA current to 'now' without changing busy flag
    _util_touch(state, key, now, tau_s, busy_flag=None)
    return float(state.gpu_util[key]["ewma"])

# =============================
# Pre-generation (Pure FEL seed)
# =============================

@dataclass
class GenConfig:
    # --- Task timing knobs ---
    ARRIVAL_RATE_fps: int
    STRIDE: int
    TASK_COUNT_POLICY: str
    TASK_MIN_PER_JOB: int

    # --- Payload / workload ranges ---
    UPLINK_TASK_KB_RANGE: Tuple[int, int]
    DOWNLINK_TASK_KB_RANGE: Tuple[int, int]
    PACKET_SIZE_KB_RANGE: Tuple[int, int]
    WORKLOAD_FLOP_RANGE: Tuple[int, int]

    # --- Micro-gaps ---
    START_GAP_RANGE: Tuple[float, float]
    UL_INTERARR_RANGE: Tuple[float, float]
    # FIRST_JOB_T0_RANGE: Tuple[float, float]

    # --- Reference model (still used to estimate t_required) ---
    REF_GPU_PATH: Tuple[str, str, str] = ("C1", "N1", "G1")
    REF_RC_LINK: str = "R0-RC1"
    DEADLINE_REF_A_s: float = 0.020          # fixed overhead A0 (s)
    DEADLINE_REF_FLOOR_s: float = 0.030      # min floor F0 (s) - 0.016

    # --- FRAME-BUDGET knobs (NEW) ---
    # Headroom ε drawn per task so B = period - ε
    DEADLINE_HEADROOM_RANGE_s: Tuple[float, float] = (0.008, 0.010)  # seconds
    # Optional queue bound added to service estimate (conservative)
    DEADLINE_QUEUE_BOUND_s: float = 0.0
    # If task doesn't fit current frame, slip by at most this many frames
    DEADLINE_MAX_FRAMES_SLIP: int = 1
    # Cap deadline within a fraction η of the next frame (0<η≤1). >1 disables.
    DEADLINE_PERIOD_CAP_ETA: float = 1.0
    # Clamp to job window
    CLAMP_TO_JOB_WINDOW: bool = True

    # (Kept for backward compat; not used to scale budgets anymore)
    DEADLINE_OVERRIDES: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Packetization + pacing (UL/DL can be different)
    PACKETIZATION_UL: Dict[str, Any] = field(default_factory=dict)
    PACKETIZATION_DL: Dict[str, Any] = field(default_factory=dict)
    UL_PACKET_SPACING_s: float = 0.0
    DL_PACKET_SPACING_s: float = 0.0

    @staticmethod
    def from_config(cfg: dict) -> "GenConfig":
        TS = cfg.get("Task_Settings", {}) or {}

        def _int2(key: str, default: Tuple[int, int]) -> Tuple[int, int]:
            v = TS.get(key)
            return (int(v[0]), int(v[1])) if isinstance(v, (list, tuple)) else default

        def _float2(key: str, default: Tuple[float, float]) -> Tuple[float, float]:
            v = TS.get(key)
            return (float(v[0]), float(v[1])) if isinstance(v, (list, tuple)) else default

        # helpers (s/ms fallbacks)
        def _range_seconds(key_s: str, key_ms: str, default: Tuple[float, float]) -> Tuple[float, float]:
            if key_s in TS:
                a, b = TS[key_s]
                return (float(a), float(b))
            if key_ms in TS:
                a, b = TS[key_ms]
                return (float(a)/1000.0, float(b)/1000.0)
            return default

        def _scalar_seconds(key_s: str, key_ms: str, default: float) -> float:
            if key_s in TS:
                return float(TS[key_s])
            if key_ms in TS:
                return float(TS[key_ms]) / 1000.0
            return default
        
        # ▶︎ read Packetization blocks & pacing (UL/DL or shared)
        pkt_ul = dict(TS.get("Packetization_UL") or TS.get("Packetization") or {})
        pkt_dl = dict(TS.get("Packetization_DL") or TS.get("Packetization") or {})
        ul_pace = float(TS.get("UL_Packet_Spacing_s", TS.get("Packet_Spacing_s", 0.0)))
        dl_pace = float(TS.get("DL_Packet_Spacing_s", TS.get("Packet_Spacing_s", 0.0)))


        return GenConfig(
            ARRIVAL_RATE_fps = int(TS.get("SERVICE_ARRIVAL_RATE_fps",
                                     TS.get("ARRIVAL_RATE_fps", 30))),
            STRIDE            = int(TS.get("stride", 1)),
            TASK_COUNT_POLICY = str(TS.get("Task_Count_Policy", "ceil")).lower(),
            TASK_MIN_PER_JOB  = int(TS.get("Task_Min_Per_Job", 1)),

            UPLINK_TASK_KB_RANGE   = tuple(map(int, TS.get("Uplink_Task_KB_Range", [40, 50]))),
            DOWNLINK_TASK_KB_RANGE = tuple(map(int, TS.get("Downlink_Task_KB_Range", [40, 50]))),
            PACKET_SIZE_KB_RANGE   = tuple(map(int, TS.get("Packet_Size_KB_Range", [8, 16]))),
            WORKLOAD_FLOP_RANGE    = tuple(map(int, TS.get("Workload_FLOP_Range",  [20000000000, 70000000000]))),

            # FIRST_JOB_T0_RANGE = _float2("FIRST_JOB_T0_RANGE", (0.001, 0.003)),
            START_GAP_RANGE    = _float2("START_GAP_RANGE", [0.00000, 0.00050]),
            UL_INTERARR_RANGE  = _float2("UL_INTERARR_RANGE", [0.00001, 0.00005]),

            REF_GPU_PATH = tuple(TS.get("REF_GPU_PATH", ["C1","N1","G1"])),
            REF_RC_LINK  = str(TS.get("REF_RC_LINK", "R0-RC1")),
            DEADLINE_REF_A_s     = float(TS.get("DEADLINE_REF_A_s", 0.020)),
            DEADLINE_REF_FLOOR_s = float(TS.get("DEADLINE_REF_FLOOR_s", 0.030)), # 0.016
            DEADLINE_HEADROOM_RANGE_s = _range_seconds(
                "DEADLINE_HEADROOM_RANGE_s", "DEADLINE_HEADROOM_RANGE_ms", (0.008, 0.010)
            ),
            DEADLINE_QUEUE_BOUND_s = _scalar_seconds(
                "DEADLINE_QUEUE_BOUND_s", "DEADLINE_QUEUE_BOUND_ms", 0.0
            ),
            DEADLINE_MAX_FRAMES_SLIP = int(TS.get("DEADLINE_MAX_FRAMES_SLIP", 0)),       # 1
            DEADLINE_PERIOD_CAP_ETA  = float(TS.get("DEADLINE_PERIOD_CAP_ETA", 1.0)),
            CLAMP_TO_JOB_WINDOW = bool(cfg.get("DeadlinePolicy", {}).get("Clamp_To_Job_Window", True)),

            PACKETIZATION_UL = pkt_ul,
            PACKETIZATION_DL = pkt_dl,
            UL_PACKET_SPACING_s = ul_pace,
            DL_PACKET_SPACING_s = dl_pace,
        )

# ------------------ DVFS & run-type helpers ------------------

HEUR_FREQ_POLICIES = {"min_deadline", "balanced", "perf_first"}

def is_optimizer_run(cfg: dict, store=None) -> bool:
    """True when this execution should obey optimizer assignments for frequency."""
    try:
        if store is not None and getattr(store, "mode", None):
            return str(store.mode).lower() == "optimizer"
    except Exception:
        pass
    s = str(cfg.get("Scheduling_Strategy", cfg.get("strategy", ""))).lower()
    return ("opt" in s) or ("optimizer" in s)

def _heur_policy_from_cfg(cfg: dict, default: str = "min_deadline") -> str:
    """Resolve the heuristic DVFS policy. If absent/invalid → default ('min_deadline')."""
    sch = (cfg.get("Scheduler") or {})
    p = str(sch.get("FREQ_POLICY", "")).strip().lower()
    return p if p in HEUR_FREQ_POLICIES else default

def _is_adaptive_run(cfg: dict) -> bool:
    """
    True iff this run should adapt GPU frequency at runtime.
    Heuristics: strategy name contains 'adaptive' OR DVFS.mode != 'fixed'.
    """
    s = str(cfg.get("Scheduling_Strategy", cfg.get("strategy", ""))).lower()
    if "adaptive" in s:
        return True
    dvfs_mode = str((cfg.get("DVFS", {}) or {}).get("mode", "adaptive")).lower()
    return dvfs_mode != "fixed"

def _find_fixed_freq_for_gpu(cfg_json: dict, c: str, n: str, g: str) -> Optional[str]:
    """
    Look up a per-GPU fixed frequency in the top-level `frequencies` list.
    Returns a string MHz (e.g., '2300') if found, else None.
    """
    gpu_id = f"{c}-{n}-{g}"
    for entry in (cfg_json.get("frequencies") or []):
        try:
            if str(entry.get("GPU", "")) == gpu_id:
                f = entry.get("Frequency")
                s = "" if f is None else str(f).strip()
                return s or None
        except Exception:
            continue
    return None

def _gpu_service_rates(cfg: dict, c: str, n: str, g: str) -> Dict[str, float]:
    """
    Returns a normalized map of frequency->rate for GPU (c,n,g):
      { "2300": 4.055e13, ... }
    Missing/invalid entries are skipped.
    """
    cc   = (cfg.get("Cluster_Config") or {})
    node = (cc.get(c) or {}).get(n) or {}
    info = (node.get(g) or {})
    raw  = (info.get("service_rates") or {})

    rates: Dict[str, float] = {}
    for fk, rv in raw.items():
        try:
            val = float(rv)
            if val > 0.0:
                rates[str(fk)] = val
        except Exception:
            continue
    return rates

def _get_ref_gpu_rate_FLOPs_per_s(cfg_json: dict, path: Tuple[str, str, str]) -> float:
    """
    Choose a reference FLOPs/s from Cluster_Config[cluster][node][gpu]['service_rates'].

    FIXED runs   → per-GPU fixed ('frequencies') → highest available rate
    ADAPTIVE     → highest available rate
    (GPU_Specs.default intentionally ignored)
    """
    C, N, G = path
    rates = _gpu_service_rates(cfg_json, C, N, G)  # normalized

    if not _is_adaptive_run(cfg_json):
        fixed = _find_fixed_freq_for_gpu(cfg_json, C, N, G)
        if fixed and fixed in rates:
            return float(rates[fixed])

    # Highest available advertised rate
    if rates:
        try:
            return max(rates.values())
        except Exception:
            pass

    # Conservative fallback
    return 8.0e13

def gpu_type_and_defaults(cfg: dict):
    """
    Returns two dicts keyed by (C,N,G):

      typemap[(C,N,G)]       -> GPU type string
      default_freq[(C,N,G)]  -> the frequency that the runtime will *use*:

        - FIXED: per-GPU fixed from top-level 'frequencies', else CC 'default',
                 else the highest service_rates key.
        - ADAPTIVE: the highest service_rates key (initial/max candidate).

    NOTE: this function does NOT read GPU_Specs.default.
    """
    typemap: Dict[Tuple[str,str,str], str] = {}
    default_freq: Dict[Tuple[str,str,str], str] = {}
    cc = cfg.get("Cluster_Config", {}) or {}
    for c, nodes in cc.items():
        for n, gpus in (nodes or {}).items():
            for g, info in (gpus or {}).items():
                c_, n_, g_ = str(c), str(n), str(g)
                info = info or {}
                typemap[(c_, n_, g_)] = str(info.get("type", ""))
                rates = (info.get("service_rates", {}) or {})
                chosen = None
                if not _is_adaptive_run(cfg):
                    f = _find_fixed_freq_for_gpu(cfg, c_, n_, g_)
                    if f and f in rates:
                        chosen = f
                if chosen is None:
                    try:
                        keys = sorted((str(k) for k in rates.keys()), key=lambda x: float(x))
                        chosen = keys[-1] if keys else ""
                    except Exception:
                        chosen = ""
                default_freq[(c_, n_, g_)] = chosen
    return typemap, default_freq

# ---- GPU key normalizers (tuple keys everywhere for DVFS maps) --------------
def gkey_tuple(c, n, g):
    return (str(c), str(n), str(g))

def gkey_from_any(x):
    """Accept ('C1','N1','G1') or 'C1-N1-G1' and return ('C1','N1','G1')."""
    if isinstance(x, tuple) and len(x) >= 3:
        return (str(x[0]), str(x[1]), str(x[2]))
    if isinstance(x, list) and len(x) >= 3:
        return (str(x[0]), str(x[1]), str(x[2]))
    if isinstance(x, str):
        parts = x.replace("/", "-").split("-")
        if len(parts) >= 3:
            return (parts[0], parts[1], parts[2])
    raise ValueError(f"Unrecognized gkey format: {x!r}")

def get_optimizer_task_freq(store, task_id: str) -> Optional[str]:
    """Pull per-task frequency from optimizer store/assignment row."""
    if store is None:
        return None
    row = None
    try:
        row = store.lookup(task_id)   # use store's accessor
    except Exception:
        pass
    if not row:
        return None
    for key in ("Frequency", "Assigned_Frequency", "Service_Frequency"):
        if key in row and row[key] is not None:
            s = str(row[key]).strip()
            if s:
                return s
    return None

def _get_ref_link_coeffs(cfg_json: dict, link_key: str) -> tuple[float, float, float, float]:
    """Return (B_ul, B_dl, d_ul_prop, d_dl_prop). B_* in s/KB, d_* in s."""
    link = (cfg_json.get("R0_RC_Links", {}) or {}).get(link_key, {})
    r_ul = float(link.get("ul_rate_kBps", link.get("r0_c", 1000000)))
    r_dl = float(link.get("dl_rate_kBps", link.get("rc_0", 1000000)))
    d_ul = float(link.get("uplink_prop_delay", 0.001))
    d_dl = float(link.get("downlink_prop_delay", 0.001))
    return (1.0 / max(r_ul, 1e-9), 1.0 / max(r_dl, 1e-9), d_ul, d_dl)

# ------------------------------
# Deterministic packetizer
# ------------------------------
def split_payload_kb(payload_kb: int, pconf: Dict[str, Any]) -> List[int]:
    """Return a deterministic packet list (KB) for the given payload.
       Supports modes: fixed, by_count, mtu, exact_list. Preserves total size."""
    payload = int(max(0, payload_kb))
    if payload == 0 or not pconf:
        return [payload] if payload else []

    mode = str(pconf.get("mode", "fixed")).lower()
    last_policy = str(pconf.get("last_policy", "shrink")).lower()
    # (header_kb kept for future modeling; do not add to sizes now)
    # header_kb = float(pconf.get("header_kb", 0.0))

    if mode == "fixed":
        size = int(max(1, int(pconf.get("size_kb", 16))))
        n = max(1, math.ceil(payload / size))
        parts = [size] * (n - 1) + [payload - size * (n - 1)]
        if last_policy == "pad" and parts[-1] > 0:
            parts[-1] = size

    elif mode == "by_count":
        n = int(max(1, int(pconf.get("count", 1))))
        base = payload // n
        rem  = payload - base * n
        parts = [base] * n
        for i in range(rem):
            parts[i] += 1  # spread remainder deterministically over first 'rem' pkts

    elif mode == "mtu":
        cap = int(max(1, int(pconf.get("max_kb", 64))))
        n   = max(1, math.ceil(payload / cap))
        parts = [cap] * (n - 1) + [payload - cap * (n - 1)]
        if last_policy == "pad" and parts[-1] > 0:
            parts[-1] = cap

    elif mode == "exact_list":
        parts = [int(x) for x in pconf.get("sizes_kb", [])]
        if sum(parts) != payload and last_policy != "pad":
            raise ValueError(f"exact_list total {sum(parts)} != payload {payload}")

    else:
        parts = [payload]

    return parts

# ------------------------------
# Deterministic schedule helper
# ------------------------------
def schedule_packet_times(base_time: float, count: int, spacing_s: Optional[float],
                          start_gap_s: float, rng: np.random.Generator,
                          random_range: Tuple[float, float]) -> np.ndarray:
    """Return absolute times for 'count' packets.
       If spacing_s is not None, use deterministic spacing; else use random_range per gap."""
    if count <= 0:
        return np.array([], dtype=float)

    if spacing_s is not None:
        spacing = max(0.0, float(spacing_s))
        return np.array([base_time + start_gap_s + i * spacing for i in range(count)], dtype=float)

    # random fallback (kept for backward compatibility)
    lo, hi = map(float, random_range)
    if count == 1:
        return np.array([base_time + start_gap_s], dtype=float)
    gaps = rng.uniform(lo, hi, size=count - 1)
    return np.concatenate(([base_time + start_gap_s],[base_time + start_gap_s + float(x) for x in np.cumsum(gaps)]))


# --- JobConfig for arrivals & durations ---
@dataclass
class JobConfig:
    TIME_UNITS: str
    ARRIVALS: Dict[str, Any]
    DURATION: Dict[str, Any]

    @staticmethod
    def from_config(cfg: dict) -> "JobConfig":
        JS = cfg.get("Job_Settings", {}) or {}
        return JobConfig(
            TIME_UNITS  = str(JS.get("TIME_UNITS", "s")).lower(),
            ARRIVALS    = JS.get("Job_Arrivals", {"Model":"poisson","Window_s":300,"Seed":41}),
            DURATION    = JS.get("Job_Duration", {"Model":"uniform","Range_s":[10,60]})
        )

class JobTaskPacketGenerator:
    def __init__(
        self,
        gen_cfg: GenConfig,
        rng: Optional[np.random.Generator] = None,
        task_settings: Optional[Dict[str, Any]] = None,
    ):
        self.cfg = gen_cfg
        self.rng = rng or np.random.default_rng()
        self.task_log: List[Dict[str, Any]] = []
        # keep a reference to Task_Settings from config.json
        self.task_settings: Dict[str, Any] = task_settings or {}

    def _sample_headroom_seconds(self, TS) -> float:
            # Fixed headroom (wins if present)
            if TS.get("DEADLINE_HEADROOM_s") is not None:
                return float(TS["DEADLINE_HEADROOM_s"])

            # Sampled headroom (ms)
            if "DEADLINE_HEADROOM_RANGE_ms" in TS:
                lo, hi = map(float, TS["DEADLINE_HEADROOM_RANGE_ms"])
                if lo > hi: lo, hi = hi, lo
                return float(self.rng.uniform(lo, hi)) / 1000.0

            # Sampled headroom (s)
            if "DEADLINE_HEADROOM_RANGE_s" in TS:
                lo, hi = map(float, TS["DEADLINE_HEADROOM_RANGE_s"])
                if lo > hi: lo, hi = hi, lo
                return float(self.rng.uniform(lo, hi))

            # Fallback: cfg range (seconds), sampled
            lo, hi = map(float, self.cfg.DEADLINE_HEADROOM_RANGE_s)
            if lo > hi: lo, hi = hi, lo
            return float(self.rng.uniform(lo, hi))
    
    def _packetize(self, total_kB: int) -> List[int]:
        low, high = map(int, self.cfg.PACKET_SIZE_KB_RANGE)
        S = int(total_kB)

        # feasible packet count bounds
        n_min = max(1, (S + high - 1) // high)    # ceil(S / high)
        n_max = max(1, S // max(low, 1))          # floor(S / low)
        if n_min > n_max:
            # fall back: single packet if constraints are inconsistent
            return [S]

        # target n near S / avg
        avg = (low + high) / 2.0
        n0 = max(n_min, min(n_max, int(round(S / avg)) or 1))
        n = n0

        sizes = []
        remaining = S
        for i in range(1, n):  # choose first n-1 packets
            rem_slots = n - i
            # remaining sum must be between rem_slots*low and rem_slots*high
            lo = max(low, remaining - rem_slots * high)
            hi = min(high, remaining - rem_slots * low)
            if lo > hi:  # guard (shouldn't happen if bounds are correct)
                lo, hi = low, min(high, remaining - (rem_slots - 1) * low)
            s = int(self.rng.integers(lo, hi + 1))
            sizes.append(s)
            remaining -= s

        sizes.append(remaining)          # last packet
        # last will be within [low, high] by construction
        return sizes

    def pre_generate_jobs(self, num_jobs: Optional[int], raw_cfg: dict) -> List[Dict[str, Any]]:
        """
        Generate jobs with start times and durations based ONLY on Job_Settings.
        - No capping by NUM_OF_JOBS or num_jobs.
        - Poisson arrivals: rate = Lambda_per_s, window = Window_s.
        - Uniform arrivals: N ≈ round(Lambda_per_s * Window_s).
        - 'Anchor_First_Job' shifts the series so the first arrival is at 0 (doesn't add an extra job).
        - Optional First_Job_Offset_{s,ms} applied after anchoring; then re-clip to window.
        All times are seconds.
        """
        jobs: List[Dict[str, Any]] = []
        JS  = (raw_cfg.get("Job_Settings") or {})
        ARR = JS.get("Job_Arrivals") or JS.get("ARRIVALS") or {}
        DUR = JS.get("Job_Duration") or JS.get("DURATION") or {}

        # ---- arrivals config ----
        model    = str(ARR.get("Model", "poisson")).lower()
        window_s = float(ARR.get("Window_s", ARR.get("window_s", 300.0)))
        lam      = float(ARR.get("Lambda_per_s", ARR.get("lambda_per_s", 3.5)))
        seed     = ARR.get("Seed", None)

        # first-job offset (seconds)
        off_pair = ARR.get("First_Job_Offset_s", None)
        if off_pair is None:
            off_pair = [x / 1000.0 for x in ARR.get("First_Job_Offset_ms", [0, 0])]
        off_lo_s, off_hi_s = map(float, (off_pair or [0.0, 0.0]))

        anchor = bool(JS.get("Anchor_First_Job", False))

        # RNG
        rng = self.rng
        if seed is not None:
            try:
                rng = np.random.default_rng(int(seed))
            except Exception:
                pass  # fall back to self.rng

        # ---- generate arrivals (seconds) ----
        starts: list[float] = []
        if model == "poisson":
            t = 0.0
            inv = 1.0 / max(lam, 1e-12)
            while True:
                t += float(rng.exponential(inv))
                if t > window_s:
                    break
                starts.append(t)
        else:  # "uniform"
            N = max(1, int(round(lam * window_s)))
            starts = list(np.linspace(0.0, window_s, N, endpoint=False, dtype=float))

        starts = np.array(starts, dtype=float)

        # anchor (shift so first arrival is 0), no extra job inserted
        if anchor and starts.size > 0:
            starts = starts - float(starts[0])

        # optional first-job offset
        if (off_lo_s != 0.0 or off_hi_s != 0.0) and starts.size > 0:
            starts = starts + float(rng.uniform(off_lo_s, off_hi_s))

        # clip to window and ensure non-empty
        starts = starts[starts <= window_s]
        if starts.size == 0:
            starts = np.array([0.0], dtype=float)

        # --- sanity print for pregeneration ---
        print(f"[PREGEN] lambda_per_s={lam:.6g}, window_s={window_s:.3f}, N_jobs={starts.size}")

        # ---- durations (seconds) ----
        dmodel = str(DUR.get("Model", "uniform")).lower()
        durations_s: list[float] = []

        if dmodel == "uniform":
            lohi = DUR.get("Range_s", None)
            if lohi is None:
                lohi = [x / 1000.0 for x in DUR.get("Range_ms", [10_000, 60_000])]
            lo_s, hi_s = float(lohi[0]), float(lohi[1])
            for _ in range(starts.size):
                durations_s.append(float(rng.uniform(lo_s, hi_s)))

        elif dmodel == "lognormal":
            p = DUR.get("Lognormal_Params",
                        {"mean_s": 30, "sigma": 0.4, "clip_range_s": [10, 60]})
            m, s = float(p.get("mean_s", 30)), float(p.get("sigma", 0.4))
            lohi = p.get("clip_range_s", None)
            if lohi is None:
                lohi = [x / 1000.0 for x in p.get("clip_range_ms", [10_000, 60_000])]
            lo, hi = float(lohi[0]), float(lohi[1])
            for _ in range(starts.size):
                x = float(rng.lognormal(mean=np.log(max(m, 1e-9)), sigma=s))
                durations_s.append(float(min(max(x, lo), hi)))

        else:
            # fallback
            for _ in range(starts.size):
                durations_s.append(float(rng.uniform(10.0, 60.0)))

        # ---- assemble jobs ----
        for jid, (t0, dur) in enumerate(zip(starts, durations_s)):
            t0_s = float(t0)
            t1_s = float(round(t0_s + float(dur), 6))
            jobs.append({
                "Service_ID": "AR",
                "Job_ID": f"AR_J{jid}",
                "Start_Time": round(t0_s, 6),
                "End_Time":   t1_s,
            })

        return jobs


    # ------------------------------
    # Tasks: deterministic UL/DL/Work/Headroom
    # ------------------------------
    def pre_generate_tasks(self, job: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Pre-generate tasks for a streaming job.

        Conventions:
        - fps keys are interpreted as *frames per second* (float).
        - stride is *frames per task* (int >= 1).
        - Effective task arrival rate (Lambda) = fps / stride  [tasks/sec]
        - period_s = 1 / Lambda  [seconds/task]
        """
        TS = self.task_settings

        # fps/stride -> period (seconds)
        fps = int(TS.get("SERVICE_ARRIVAL_RATE_fps", 30))
        stride = int(TS.get("stride", 1))
        if stride <= 0:
            stride = 1  # safety clamp

        effective_fps = max(1e-9, fps / max(1, stride))  # tasks per second
        period_s = 1.0 / effective_fps

        # task count policy
        policy = str(TS.get("Task_Count_Policy", "ceil")).lower()
        min_tasks = int(TS.get("Task_Min_Per_Job", 1))

        # Job duration (seconds)
        start_s = float(job["Start_Time"])
        end_s   = float(job["End_Time"])
        duration_s = max(0.0, end_s - start_s)

        # Number of tasks
        raw = duration_s / period_s if period_s > 0 else 0.0
        num_tasks = math.ceil(raw) if policy == "ceil" else math.floor(raw)
        num_tasks = max(min_tasks, int(num_tasks))

        print(f"[Job {job['Job_ID']}] duration_s={duration_s:.3f} "
            f"raw={raw:.6f} → num_tasks={num_tasks} (fps={fps}, stride={stride}, policy={policy})")

        # First packet offset (SECONDS). Prefer *_s; treat *_ms as legacy.
        if "Task_First_Offset_s" in TS:
            first_off_s = float(TS["Task_First_Offset_s"])
        elif "Task_First_Offset_ms" in TS:
            first_off_s = float(TS["Task_First_Offset_ms"]) / 1000.0
            print("[warn] Task_First_Offset_ms is deprecated; use Task_First_Offset_s (seconds).")
        else:
            first_off_s = 0.0

        # base_time = start_s + first_off_s

        base_time = float(job["Start_Time"])  # EXACT match


        # Periodic arrivals (seconds)
        arrival_times = [round(base_time + i * period_s, 6) for i in range(num_tasks)]

        tasks: List[Dict[str, Any]] = []

        # --- Reference constants ---
        R_ref = _get_ref_gpu_rate_FLOPs_per_s(self.task_settings.get("_raw_cfg", {}), self.cfg.REF_GPU_PATH)
        B_ul, B_dl, d_ul_prop, d_dl_prop = _get_ref_link_coeffs(self.task_settings.get("_raw_cfg", {}), self.cfg.REF_RC_LINK)
        A0  = self.cfg.DEADLINE_REF_A_s
        F0  = self.cfg.DEADLINE_REF_FLOOR_s
        eta = float(self.cfg.DEADLINE_PERIOD_CAP_ETA)
        use_cap = (0.0 < eta <= 1.0)

        # Fleet min/max GPU rates (for conservative t_service)
        def _scan_min_max_gpu_rate(raw_cfg: dict) -> tuple[float, float]:
            cc = (raw_cfg or {}).get("Cluster_Config", {})
            rmin = float("inf"); rmax = 0.0
            for _c, nodes in cc.items():
                for _n, gpus in nodes.items():
                    for _g, ginfo in gpus.items():
                        rates = (ginfo or {}).get("service_rates", {})
                        for _f, val in (rates or {}).items():
                            try:
                                v = float(val)
                                if v > 0:
                                    if v < rmin: rmin = v
                                    if v > rmax: rmax = v
                            except Exception:
                                pass
            if not math.isfinite(rmin) or rmin <= 0 or rmax <= 0:
                return (0.0, 0.0)
            return (rmin, rmax)

        raw_cfg = self.task_settings.get("_raw_cfg", {})
        R_min, R_max = _scan_min_max_gpu_rate(raw_cfg)
        if R_min <= 0.0: R_min = max(R_ref, 1e-9)
        if R_max <= 0.0: R_max = max(R_ref, 1e-9)

        # --- Job-level base FLOPs (one draw per job) ---
        if "Workload_FLOP_Fixed" in TS and TS["Workload_FLOP_Fixed"] is not None:
            base_work = int(TS["Workload_FLOP_Fixed"])
        else:
            wr = TS.get("Workload_FLOP_Range", self.cfg.WORKLOAD_FLOP_RANGE)
            wlo, whi = int(wr[0]), int(wr[1])
            if wlo > whi:
                wlo, whi = whi, wlo
            base_work = int(self.rng.integers(wlo, whi + 1))

        # optional: store job-level workload for debugging / downstream use
        job["Job_Workload_FLOPs"] = base_work

        # jitter bounds (relative)
        jitter_lo = -0.000000005
        jitter_hi =  0.000000005

        # --- Generate each task with per-task UL/DL, but job-level FLOPs + small jitter ---
        for idx, t_arr in enumerate(arrival_times):
            tid = f"{job['Job_ID']}_T{idx}_U"

            # 1) TOTAL task size kB for THIS task (shared by UL & DL)
            #    If a fixed size is given, use it; else sample from a range.
            if "Task_Total_KB" in TS and TS["Task_Total_KB"] is not None:
                total_kb = int(TS["Task_Total_KB"])
            else:
                # Prefer an explicit Task_Total_KB_Range if present,
                # otherwise fall back to the Uplink_Task_KB_Range / config.
                tr = TS.get("Task_Total_KB_Range",
                            TS.get("Uplink_Task_KB_Range", self.cfg.UPLINK_TASK_KB_RANGE))
                tlo, thi = int(tr[0]), int(tr[1])
                if tlo > thi:
                    tlo, thi = thi, tlo
                total_kb = int(self.rng.integers(tlo, thi + 1))

            # 2) UL/DL totals both use this per-task size unless explicitly overridden.
            if "Uplink_Task_KB" in TS and TS["Uplink_Task_KB"] is not None:
                ul_total = int(TS["Uplink_Task_KB"])
            else:
                ul_total = total_kb

            if "Downlink_Task_KB" in TS and TS["Downlink_Task_KB"] is not None:
                dl_total = int(TS["Downlink_Task_KB"])
            else:
                dl_total = total_kb


            # 3) FLOPs for THIS task: small jitter around base_work
            eps = float(self.rng.uniform(jitter_lo, jitter_hi))
            work = int(base_work * (1.0 + eps))

            # 1) Sample headroom (seconds) from DEADLINE_HEADROOM_RANGE_ms
            eps_head = self._sample_headroom_seconds(TS)          # e.g., 8..10 ms -> 0.008..0.010
            # keep headroom sane (purely defensive)
            eps_head = max(0.0, min(eps_head, 0.95 * period_s))

            # 2) Absolute deadline = arrival + frame period + headroom
            abs_deadline = t_arr + period_s + eps_head

            # 3) Clamp to job window if requested
            if self.cfg.CLAMP_TO_JOB_WINDOW:
                abs_deadline = min(abs_deadline, end_s - 1e-6)

            # 4) Relative budget (should be ≈ period_s + eps_head)
            rel_budget = max(0.0, abs_deadline - t_arr)

            # (Optional) sanity check: verify budget is within [frame+lo, frame+hi]
            try:
                lo_ms, hi_ms = TS.get("DEADLINE_HEADROOM_RANGE_ms", [10, 12])
                lo = (1.0 / effective_fps) + (float(lo_ms) / 1000.0)
                hi = (1.0 / effective_fps) + (float(hi_ms) / 1000.0)
                delta = abs_deadline - t_arr
                if not (lo - 1e-4 <= delta <= hi + 1e-4):
                    print(f"[WARN] {job['Job_ID']} T{idx}: budget={delta*1000:.3f} ms "
                        f"outside [{lo*1000:.3f}, {hi*1000:.3f}] ms")
            except Exception:
                pass

            # record task with per-task totals
            tasks.append({
                "Service_ID": job["Service_ID"],
                "Job_ID": job["Job_ID"],
                "Task_ID": tid,
                "Arrival_Time": round(t_arr, 6),
                "Task_Deadline": round(abs_deadline, 6),
                "Task_Budget_s": round(rel_budget, 6),
                "Task_t_required_s": round(period_s, 6),
                "Workload_FLOPs": int(work),
                "Task_UL_Total_kB": int(ul_total),  # varies per task
                "Task_DL_Total_kB": int(dl_total),  # varies per task
                "Task_Arrival_Rate": int(round(effective_fps)),
                "Task_Is_First": 1 if idx == 0 else 0,
                "Task_Is_Last":  1 if idx == num_tasks - 1 else 0,
            })

        # back-fill job window
        for task in tasks:
            task["Job_Start_Time"] = job["Start_Time"]  # seconds
            task["Job_End_Time"]   = job["End_Time"]    # seconds

        return tasks


    def pre_generate_task_packets(self, task: Dict[str, Any]) -> List[Dict[str, Any]]:
        t0 = float(task["Arrival_Time"])  # seconds

        # 1) Deterministic sizes via Packetization blocks
        ul_total_kb = int(task["Task_UL_Total_kB"])
        dl_total_kb = int(task.get("Task_DL_Total_kB", 0))

        ul_sizes = split_payload_kb(ul_total_kb, self.cfg.PACKETIZATION_UL)
        n_ul     = len(ul_sizes)
        total_ul = max(1, sum(ul_sizes))

        dl_task_id = task["Task_ID"].replace("_U", "_D") if "_U" in task["Task_ID"] else f"{task['Task_ID']}_D"
        if dl_total_kb > 0:
            # proportional split to UL sizes (deterministic)
            dl_sizes, assigned = [], 0
            for i, ul_sz in enumerate(ul_sizes):
                dsz = int(round(dl_total_kb * (ul_sz / total_ul)))
                if i == n_ul - 1:
                    dsz = max(1, dl_total_kb - assigned)
                assigned += max(0, dsz)
                dl_sizes.append(max(1, dsz))
        else:
            dl_sizes = []

        TS = self.task_settings

        if "START_GAP_s" in TS:
            # explicit scalar (seconds)
            start_gap = float(TS["START_GAP_s"])
        elif "START_GAP_RANGE" in TS:
            # START_GAP_RANGE is a list → ignore it completely and use zero
            start_gap = 0.0
        else:
            # default no delay
            start_gap = 0.0

        ul_times = schedule_packet_times(
            base_time=t0,
            count=n_ul,
            spacing_s=self.cfg.UL_PACKET_SPACING_s,      # e.g., 20 µs
            start_gap_s=start_gap,
            rng=self.rng,
            random_range=self.cfg.UL_INTERARR_RANGE      # only used if spacing_s is None
        )

        is_first_task = int(task.get("Task_Is_First", 0))
        is_last_task  = int(task.get("Task_Is_Last", 0))

        # The first UL packet carries the compute FLOPs
        if "Workload_FLOPs" not in task:
            raise ValueError("Task missing Workload_FLOPs—must be assigned per job before packetization.")
        work_flops = int(task["Workload_FLOPs"])

        rows: List[Dict[str, Any]] = []

        # UL rows (seconds)
        for i, (ul_sz, t_pkt) in enumerate(zip(ul_sizes, ul_times)):
            # only the first UL packet carries the compute
            row_work = work_flops if i == 0 else 0
            ul_row = {
                "Direction": "Uplink",
                "Service_ID": task["Service_ID"],
                "Job_ID": task["Job_ID"],
                "Job_Start_Time": task["Job_Start_Time"],   # seconds
                "Job_End_Time": task["Job_End_Time"],       # seconds
                "Task_ID": task["Task_ID"],
                "Task_Size_KB": task["Task_UL_Total_kB"],
                "Task_Arrival_Rate": task["Task_Arrival_Rate"],
                "Task_Arrival_Time": task["Arrival_Time"],  # seconds
                "Task_Deadline": task["Task_Deadline"],     # seconds
                "Workload_FLOPs": row_work,
                "Task_Is_First": is_first_task,
                "Task_Is_Last":  is_last_task,
                "Packet_id": f"{task['Task_ID']}_P{i}",
                "Packet_Size_KB": int(ul_sz),
                "Packet_Arrival_Time": t_pkt,               # seconds
                "Is_First": 1 if i == 0 else 0,
                "Is_Last":  1 if i == n_ul - 1 else 0,
                "Task_DL_Total_kB": int(dl_total_kb),
            }
            self.task_log.append({
                **ul_row,
                "Task_UL_Total_kB": task["Task_UL_Total_kB"],
                "Task_DL_Total_kB": dl_total_kb,
            })
            rows.append(ul_row)

        # DL rows (skeleton; times filled during sim)
        if dl_total_kb > 0:
            for i, dl_sz in enumerate(dl_sizes):
                rows.append({
                    "Direction": "Downlink",
                    "Service_ID": task["Service_ID"],
                    "Job_ID": task["Job_ID"],
                    "Job_Start_Time": task["Job_Start_Time"],
                    "Job_End_Time": task["Job_End_Time"],
                    "Task_ID": dl_task_id,
                    "Task_Size_KB": dl_total_kb,
                    "Task_Arrival_Rate": task["Task_Arrival_Rate"],
                    "Task_Arrival_Time": "",
                    "Task_Deadline": task["Task_Deadline"],
                    "Workload_FLOPs": 0,                   # DL packets never carry compute
                    "Task_Is_First": is_first_task,
                    "Task_Is_Last":  is_last_task,
                    "Packet_id": f"{dl_task_id}_P{i}",
                    "Packet_Size_KB": int(dl_sz),
                    "Packet_Arrival_Time": "",
                    "Is_First": 1 if i == 0 else 0,
                    "Is_Last":  1 if i == len(dl_sizes) - 1 else 0,
                })

        return rows


def pre_generate_from_config(cfg: dict, seed: int) -> List[Dict[str, Any]]:
    gen_cfg = GenConfig.from_config(cfg)

    # make the raw config visible to the generator
    ts = dict(cfg.get("Task_Settings", {}) or {})
    ts["_raw_cfg"] = cfg
    gen = JobTaskPacketGenerator(
        gen_cfg,
        rng=np.random.default_rng(seed),
        task_settings=ts,
    )

    # Generate jobs purely from Job_Settings (Poisson/uniform config)
    jobs = gen.pre_generate_jobs(None, cfg)

    rows = []
    n_jobs = n_tasks = n_pkts = 0

    rows: List[Dict[str, Any]] = []
    for job in jobs:
        n_jobs += 1
        tasks = gen.pre_generate_tasks(job)
        for task in tasks:
            n_tasks += 1
            pkts = gen.pre_generate_task_packets(task)
            n_pkts += len(pkts)
            rows.extend(pkts)
    print(f"[PREGEN] jobs={n_jobs} tasks={n_tasks} packets={n_pkts}")
    return rows


def run_until_empty(fel, handle_event):
    """Drain the FEL completely (all times in seconds)."""
    for ev in fel.run():                 # FEL yields events until its queue is empty
        t = fel.t                        # current sim time (seconds)
        handle_event(t, ev.etype, ev.payload)


# -----------------------------------------------------------------------------

# =============================
# FEL Core
# =============================

@dataclass(order=True)
class Event:
    time: float
    priority: int
    seq: int
    etype: int = field(compare=False)
    payload: Dict[str, Any] = field(compare=False, default_factory=dict)

class Ev(IntEnum):
    ARRIVE_R0_UL_IN = 1
    DEPART_R0_UL_EG = 2
    ARRIVE_RC_UL_IN = 3
    # DEPART_RC_UL_IN = 4
    ARRIVE_GPU      = 4
    FINISH_GPU      = 5
    ARRIVE_RC_DL_IN = 6
    DEPART_RC_DL_IN = 7
    ARRIVE_R0_DL_IN = 8
    # DEPART_R0_DL_IN = 9
    DVFS_RETUNE     = 9


EV_PRIO = {
    Ev.DEPART_R0_UL_EG: -2,
    # Ev.DEPART_RC_UL_IN: -2,
    Ev.FINISH_GPU:      -2,
    Ev.DEPART_RC_DL_IN: -2,
    # Ev.DEPART_R0_DL_IN: -2,
    Ev.ARRIVE_R0_UL_IN: -1,
    Ev.ARRIVE_RC_UL_IN: -1,
    Ev.ARRIVE_GPU:      -1,
    Ev.ARRIVE_RC_DL_IN: -1,
    Ev.ARRIVE_R0_DL_IN: -1,
    Ev.DVFS_RETUNE:      0,   # or -1 if you want it at "arrival" priority
}

class FEL:
    def __init__(self):
        self.t = 0.0
        self._q: List[Event] = []
        self._seq = 0

    def schedule(self, t: float, etype: Ev, payload: Dict[str, Any], prio: Optional[int] = None):
        if prio is None:
            prio = EV_PRIO.get(etype, 0)
        self._seq += 1
        t = max(float(t), float(self.t))  # non-decreasing clock
        heapq.heappush(self._q, Event(t, int(prio), self._seq, int(etype), payload))

    def run(self):
        while self._q:
            ev = heapq.heappop(self._q)
            self.t = ev.time
            yield ev

    def size(self) -> int:
        return len(self._q)

    @property
    def now(self) -> float:
        return self.t


# =============================
# Stations & Network
# =============================

@dataclass
class Station:
    name: str
    service_time_fn: Callable[[Dict[str, Any]], float]
    capacity: Optional[int] = None
    busy: bool = False
    q: deque = field(default_factory=deque)
    q_len: int = 0
    q_last_t: float = 0.0
    q_area: float = 0.0
    max_q_len: int = 0
    tail_time: float = 0.0
    jobs_present: Counter = field(default_factory=Counter)
    in_service_job: str = ""

def _q_update(st: Station, t: float, new_len: int):
    dt = float(t) - float(st.q_last_t)
    if dt >= 0:
        st.q_area += st.q_len * dt
    st.q_last_t = float(t)
    st.q_len = int(new_len)
    st.max_q_len = max(st.max_q_len, st.q_len)

def enqueue_fcfs(st: Station, item: dict, arrival_field: str, now: float):
    arr_t = float(item.get(arrival_field, now))
    if not st.q or float(st.q[-1].get(arrival_field, -1e18)) <= arr_t:
        st.q.append(item)
    else:
        i = len(st.q) - 1
        st.q.append(item)
        while i >= 0 and float(st.q[i].get(arrival_field, -1e18)) > arr_t:
            st.q[i+1] = st.q[i]; i -= 1
        st.q[i+1] = item
    _q_update(st, now, len(st.q))

def fast_rc_ul_from_config(cfg: dict) -> bool:
    net = cfg.get("Network_Settings", {}) or {}
    return bool(net.get("FAST_RC_UL", net.get("RC_UL_IS_IDEAL", False)))

def fast_r0_dl_from_config(cfg: dict) -> bool:
    net = cfg.get("Network_Settings", {}) or {}
    # default True: R0_DL is ideal unless you explicitly study it
    return bool(net.get("FAST_R0_DL", True))

#  --- NetPerCluster: split UL vs DL RC times ---
class NetPerCluster:
    """
    Expects per_cluster mapping keyed by 'C1','C2',... with fields:
      ul_rate_kBps, dl_rate_kBps, rc_rate_kBps, ul_prop_s, dl_prop_s
    All rates are KB/s, props are seconds.
    """
    def __init__(self, per_cluster, defaults=None, fast_rc_ul=True, fast_rc_dl=False, fast_r0_dl=True):
        # Defaults are only used if a cluster is missing in config
        self.map = per_cluster
        self.defaults = defaults or {
            "ul_rate_kBps": 500000,  # 500 MB/s
            "dl_rate_kBps": 500000,
            "rc_rate_kBps": 500000,
            "ul_prop_s": 4e-5,         # 40 µs
            "dl_prop_s": 4e-5
        }
        self.fast_rc_ul = bool(fast_rc_ul)
        self.fast_rc_dl = bool(fast_rc_dl)
        self.fast_r0_dl = bool(fast_r0_dl)

    def _cfg(self, pkt):
        return self.map.get(str(pkt["Assigned_Cluster"]), self.defaults)

    # Transmission on R0->RC (uplink path)
    def ul_time(self, pkt):
        cfg = self._cfg(pkt)
        size_kB = float(pkt.get("Packet_Size_KB", 0.0))
        rate    = max(float(cfg["ul_rate_kBps"]), 1e-9)
        return size_kB / rate

    # Transmission inside RC (used for both UL and DL legs unless bypassed)
    def _rc_time_core(self, pkt):
        cfg = self._cfg(pkt)
        size_kB = float(pkt.get("Packet_Size_KB", 0.0))
        rate    = max(float(cfg["rc_rate_kBps"]), 1e-9)
        return size_kB / rate

    def rc_ul_time(self, pkt):
        return 0.0 if self.fast_rc_ul else self._rc_time_core(pkt)

    def rc_dl_time(self, pkt):
        return 0.0 if self.fast_rc_dl else self._rc_time_core(pkt)

    def r0dl_time(self, pkt):
        """
        R0_DL service time.
        - If fast_r0_dl is True → no queue, 0 service (and we will also
        bypass the queue in the event handlers below).
        - Else M/M/1 with mean = size / rate.
        """
        if self.fast_r0_dl:
            return 0.0

        cfg = self._cfg(pkt)
        size_kB = float(pkt.get("Packet_Size_KB", 0.0))
        rate    = max(float(cfg["dl_rate_kBps"]), 1e-9)   # kB/s

        mean_svc = size_kB / rate                         # seconds
        return float(np.random.exponential(mean_svc))

    # Propagation terms (seconds)
    def ul_prop(self, pkt):
        return float(self._cfg(pkt)["ul_prop_s"])

    def dl_prop(self, pkt):
        return float(self._cfg(pkt)["dl_prop_s"])


# =============================
# Scheduler
# =============================

@dataclass
class GPUCatalog:
    rates: Dict[Tuple[str,str,str], Dict[str, float]]
    types: Dict[Tuple[str,str,str], str] = field(default_factory=dict)
    defaults: Dict[Tuple[str,str,str], str] = field(default_factory=dict)

def patch_gpu_type_map(cfg: Dict[str, Any], catalog: GPUCatalog) -> None:
    """
    Ensures cfg['GPU_Type_Map'] has entries for every (C,N,G) in the catalog.
    - Uses catalog.types when present
    - Falls back to catalog.defaults
    - Writes multiple key encodings so lookups succeed
    """
    tmap = cfg.setdefault("GPU_Type_Map", {})

    def _insert_keys(c: str, n: str, g: str, gtype: str):
        # Store several encodings; keep tuple too (Python path) and strings (JSON path)
        tmap[(c, n, g)]          = gtype
        tmap[f"{c}-{n}-{g}"]     = gtype
        tmap[f"{c}/{n}/{g}"]     = gtype
        tmap[f"{c}:{n}:{g}"]     = gtype
        tmap[f"{c},{n},{g}"]     = gtype

    # 1) preferred mapping from catalog.types
    for key, gtype in (catalog.types or {}).items():
        c, n, g = map(str, key)
        _insert_keys(c, n, g, str(gtype))

    # 2) fill gaps using catalog.defaults
    for key, default_type in (catalog.defaults or {}).items():
        c, n, g = map(str, key)
        if (c, n, g) not in tmap and f"{c}-{n}-{g}" not in tmap:
            _insert_keys(c, n, g, str(default_type))

    # 3) final pass: if anything in rates lacks a type, raise (or set a global default)
    missing = []
    for key in catalog.rates.keys():
        c, n, g = map(str, key)
        if not any(k in tmap for k in [
            (c, n, g), f"{c}-{n}-{g}", f"{c}/{n}/{g}", f"{c}:{n}:{g}", f"{c},{n},{g}"
        ]):
            missing.append((c, n, g))

    if missing:
        # Choose one policy:
        #   (a) raise for strictness:
        raise RuntimeError(f"[cfg] Missing GPU_Type_Map entries for: {missing}. "
                           f"Add to catalog.types or catalog.defaults.")
        #   (b) OR: set a global default type (uncomment to use)
        # global_default = cfg.get("GPU_Global_Default_Type")
        # if not global_default:
        #     raise RuntimeError("Set GPU_Global_Default_Type or supply types/defaults.")
        # for c, n, g in missing:
        #     _insert_keys(c, n, g, global_default)


def _lambda_for_gpu(state, gkey) -> float:
    """
    Return λ_g (FLOPs/s) for a GPU, robust to different key encodings.

    Uses:
      - normalized key via _norm_gkey(gkey)
      - raw tuple key
      - several string encodings "C1-N1-G1", "C1/N1/G1", etc.
    """
    lam_map = getattr(state, "flops_inflow_per_gpu", {}) or {}
    if not lam_map:
        return 0.0

    try:
        c, n, g = map(str, gkey if isinstance(gkey, (tuple, list)) else _norm_gkey(gkey))
    except Exception:
        c = n = g = None

    candidates = []
    if c is not None:
        candidates.extend([
            _norm_gkey((c, n, g)),
            (c, n, g),
            f"{c}-{n}-{g}",
            f"{c}/{n}/{g}",
            f"{c}:{n}:{g}",
            f"{c},{n},{g}",
        ])
    else:
        try:
            candidates.append(_norm_gkey(gkey))
        except Exception:
            pass
        candidates.append(gkey)

    for k in candidates:
        if k in lam_map:
            try:
                return float(lam_map[k])
            except Exception:
                return 0.0

    return 0.0


def _freq_bins(state, gkey):
    """
    Return sorted frequency keys for this GPU.

    Uses state.freq_steps_by_gpu cache when available, otherwise
    computes and stores it using gpu_catalog.rates.
    """
    cache = getattr(state, "freq_steps_by_gpu", {}) or {}
    ks = cache.get(gkey)
    if ks:
        return ks

    catalog = getattr(state, "gpu_catalog", None)
    rates = (catalog.rates if catalog else {}) or {}
    fmap = (rates.get(gkey) or {})

    if fmap:
        def _as_float(k):
            try:
                # try to interpret the *key* as MHz, e.g. "1500"
                return float(str(k).rstrip(".0"))
            except Exception:
                return float("inf")

        ks = sorted(fmap.keys(), key=_as_float)
        # cache for next time
        if hasattr(state, "freq_steps_by_gpu"):
            state.freq_steps_by_gpu[gkey] = ks
        return ks

    # Fallbacks if we have no rates
    cur = (getattr(state, "current_freq", {}) or {}).get(gkey)
    if cur:
        return [cur]

    defaults = getattr(state, "gpu_catalog", None)
    defaults = getattr(defaults, "defaults", {}) if defaults else {}
    df = defaults.get(gkey)
    return [df] if df is not None else ["1500"]

def _freq_key_match_in_map(fmap: dict, f) -> str:
    """
    Return an existing key in fmap that best matches f.
    - Prefer exact string key match
    - Then match after normalizing numeric-like f (1500.0 -> "1500")
    - Then match numeric-equivalence against numeric-like keys in fmap
    Never returns a key not already present in fmap.
    """
    if not isinstance(fmap, dict) or not fmap or f is None:
        return ""

    # 1) exact match (fast path)
    s = str(f).strip()
    if s in fmap:
        return s

    # 2) normalize numeric-ish f -> "1500"
    norm = ""
    try:
        fv = float(f)
        norm = str(int(fv)) if abs(fv - int(fv)) < 1e-12 else str(fv)
    except Exception:
        fv = None

    if norm and norm in fmap:
        return norm

    # 3) numeric-equivalent match (ONLY if f was numeric)
    if fv is None:
        return ""

    for k in fmap.keys():
        try:
            kv = float(k)  # works if k is "1500" or 1500 or "1500.0"
        except Exception:
            continue
        if abs(kv - fv) < 1e-12:
            return str(k)

    return ""

def _gpu_hw_freq_key(state, gkey, fmap=None):
    """
    Return the canonical freq key for the *hardware* freq of GPU gkey.

    gkey: (C, N, G)
    fmap: rates[gkey] dict (optional, to avoid re-lookups)
    """
    catalog = getattr(state, "gpu_catalog", None)
    rates   = (catalog.rates if catalog else {}) or {}
    fmap    = fmap if fmap is not None else (rates.get(gkey) or {})

    gpu_to_freq = getattr(state, "gpu_to_freq", {}) or {}
    hw_raw = gpu_to_freq.get(_norm_gkey(gkey), "")
    key    = _freq_key_match_in_map(fmap, hw_raw) or (str(hw_raw) if hw_raw else "")

    # fall back to a sane default if we don't recognise it
    if not key and fmap:
        try:
            key = max(fmap.keys(), key=lambda k: float(str(k).rstrip(".0")))
        except Exception:
            key = next(iter(fmap.keys()))
    return key

def choose_heuristic_freq(
    cfg: dict,
    gpu_key: Tuple[str, str, str],
    work_flops: float,
    budget_s: float,
    queue_len: int,
    util: Optional[float] = None,
    policy: Optional[str] = None
) -> str:
    """Heuristic DVFS: 'min_deadline' (default), 'balanced', or 'perf_first'."""
    C, N, G = map(str, gpu_key)
    policy = (policy or _heur_policy_from_cfg(cfg)).lower()
    if policy not in HEUR_FREQ_POLICIES:
        policy = "min_deadline"

    # rates: { "1500": service_rate_in_flops_per_sec, ... }
    rates = _gpu_service_rates(cfg, C, N, G)
    if not rates:
        return ""

    # keep the *original* key to avoid format mismatches upstream
    steps = sorted(
        ((k, float(k), float(v)) for k, v in rates.items()),
        key=lambda x: x[1]  # sort by MHz numeric
    )
    key_list = [k for (k, _, _) in steps]
    fastest_key = key_list[-1]
    slowest_key = key_list[0]

    if policy == "perf_first":
        return fastest_key

    # If no work or infinite budget, pick the slowest that trivially fits.
    if work_flops <= 0.0:
        return slowest_key

    # Find the first step that meets the time budget.
    idx_fit = None
    for i, (_k, _mhz, R) in enumerate(steps):
        if work_flops / max(R, 1e-9) <= budget_s + 1e-12:
            idx_fit = i
            break

    if idx_fit is None:
        # none fit → go fastest
        return fastest_key

    if policy == "min_deadline":
        return key_list[idx_fit]

    if policy == "balanced":
       # util may be None (older callers) → treat as 0
        u = 0.0 if util is None else max(0.0, min(1.0, float(util)))

        # normalize queue pressure; k=2 means "line forming" starts around 2 deep
        q = max(0, int(queue_len))
        q_pressure = 1.0 - math.exp(-float(q) / 2.0)      # in [0,1)

        # blend weights (tune if needed)
        pressure = 0.5 * u + 0.5 * q_pressure             # in [0,1)

        # step around minimum-deadline-fit index
        step = 0
        if pressure > 0.85:
            step = +2
        elif pressure > 0.60:
            step = +1
        elif pressure < 0.25:
            step = -1

        target_idx = max(0, min(len(key_list) - 1, idx_fit + step))
        return key_list[target_idx]


def choose_freq_for(state: "SimState",
                    gpu_key: Tuple[str, str, str],
                    pkt: Optional[dict]) -> str:
    """
    Pure selector. Returns a canonical freq key in the GPU's fmap.
    Only side-effect: set `state._last_freq_reason` for provenance.

    FIXED:
      per-GPU fixed map > global fixed > fastest

    ADAPTIVE:
      - Least-Load → choose_heuristic_freq(..., policy=resolved_policy)
                     (default policy = 'balanced')
      - Non–least-load → current/default/slowest (no strategy fallback)
    """

    # ---- normalize key & fetch fmap ----
    C, N, G = map(str, gpu_key if isinstance(gpu_key, (tuple, list)) else tuple(gpu_key))
    gkey = (C, N, G)
    catalog = getattr(state, "gpu_catalog", None) or {}
    rates   = getattr(catalog, "rates", {}) or {}
    # no .copy() here – fmap is read-only
    fmap = (rates.get(gkey, {}) or {})

    if not fmap:
        state._last_freq_reason = "no-fmap"
        return ""

    # ---- helpers ----
    def _match(m: dict, x: str) -> str:
        # Prefer project helper when present
        try:
            return _freq_key_match_in_map(m, x)  # type: ignore
        except Exception:
            xs = str(x).strip()
            if xs in m:
                return xs
            try:
                xv = float(xs.rstrip(".0"))
                for k in m:
                    try:
                        if abs(float(str(k).rstrip(".0")) - xv) < 1e-12:
                            return k
                    except Exception:
                        pass
            except Exception:
                pass
            return ""

    def _as_float(k: str) -> float:
        try:
            return float(str(fmap[k]))
        except Exception:
            try:
                return float(str(k).rstrip(".0"))
            except Exception:
                return float("inf")

    def _fastest() -> str:
        return max(fmap.keys(), key=_as_float)

    def _slowest() -> str:
        return min(fmap.keys(), key=_as_float)

    def _current_only() -> str:
        cur = _match(fmap, (getattr(state, "gpu_to_freq", {}) or {}).get(_norm_gkey(gkey), ""))
        if cur:
            state._last_freq_reason = "keep-current"
            return cur
        state._last_freq_reason = "no-current"
        return ""  # <- do not invent a freq

    # ---- FIXED mode ----
    if str(getattr(state, "freq_mode", "adaptive")).lower() == "fixed":
        f = (getattr(state, "fixed_freq_map", {}) or {}).get(gkey)
        if f:
            f = _match(fmap, str(f))
            if f:
                state._last_freq_reason = "fixed-map"
                return f
        f = str(getattr(state, "fixed_frequency", "")).strip()
        if f:
            f = _match(fmap, f)
            if f:
                state._last_freq_reason = "fixed-global"
                return f
        state._last_freq_reason = "fixed-fastest"
        return _fastest()

    # ---- ADAPTIVE ----
    # Least-load flag is precomputed in run_sim
    is_least_load = bool(getattr(state, "is_least_load", False))

    # Resolve policy once (state override already set in run_sim)
    policy = str(getattr(state, "freq_policy", "min_deadline")).lower()

    # Force 'balanced' for least-load unless user explicitly asked otherwise
    if is_least_load and policy in ("", "default", "min_deadline"):
        policy = "balanced"

    if is_least_load:
        # Inputs for heuristic
        st   = (getattr(state, "stations", {}) or {}).get(f"GPU:{C}:{N}:{G}")
        now  = float(getattr(state, "t", 0.0))
        tail = max(0.0, float(getattr(st, "tail_time", 0.0)) - now) if st else 0.0

        is_pkt = isinstance(pkt, dict)
        work   = int(pkt.get("Workload_FLOPs", 0)) if is_pkt else 0
        ddl    = float(pkt.get("Task_Deadline", float("inf"))) if is_pkt else float("inf")
        budget = ddl if not math.isfinite(ddl) else max(0.0, ddl - tail)

        try:
            qlen = 0
            if st:
                qlen = int(getattr(st, "q_len", len(getattr(st, "q", [])) or 0))
        except Exception:
            qlen = 0

        # --- NEW: utilization only when needed ---
        util = None
        if policy == "balanced" and getattr(state, "use_balanced_util", True):
            util_tau = float(getattr(state, "util_tau_s", 0.5))
            util = _util_get(
                state,
                gkey,
                now=float(getattr(state, "now", 0.0)),
                tau_s=util_tau,
            )
            util = max(0.0, min(1.0, util))  # clamp

        cfg_plain = getattr(state, "cfg_plain", getattr(state, "cfg", {}))

        pick = choose_heuristic_freq(
            cfg_plain,
            gkey,
            work,
            budget,
            qlen,
            util=util,
            policy=policy,
        )
        pick = _match(fmap, pick) if pick else ""
        state._last_freq_reason = f"heuristic:{policy}" if pick else "heuristic:none"
        return pick or _current_only()

    # Non–least-load (random, optimizer, etc.) decide elsewhere;
    # avoid retunes here.
    return _current_only()

def _inflow_update(state, g, t_now, flops, tau=None):
    """Update per-GPU FLOPs/s EMA with this arrival."""
    tau = float(tau or getattr(state, "dvfs_window", 0.5))

    # FIX: do NOT default to t_now here, otherwise first dt becomes ~0
    # or huge depending on call order; start from 0.0 instead.
    last_t = (getattr(state, "_last_arrival_gpu_ts", {}) or {}).get(g, 0.0)

    dt = max(1e-9, float(t_now) - float(last_t))
    inst = float(flops) / dt                      # instantaneous FLOPs/s

    # EMA weight (equivalent to 1 - exp(-dt/tau))
    eta  = 1.0 - pow(2.718281828, -dt / tau)
    prev = float((getattr(state, "flops_inflow_per_gpu", {}) or {}).get(g, 0.0))
    new  = (1.0 - eta) * prev + eta * inst

    # store back
    if not hasattr(state, "flops_inflow_per_gpu") or state.flops_inflow_per_gpu is None:
        state.flops_inflow_per_gpu = {}
    state.flops_inflow_per_gpu[g] = new

    if not hasattr(state, "_last_arrival_gpu_ts") or state._last_arrival_gpu_ts is None:
        state._last_arrival_gpu_ts = {}
    state._last_arrival_gpu_ts[g] = float(t_now)

    # --- OPTIONAL DEBUG (comment out once checked) ---
    # print("[INFLOW]", g, "dt=", dt, "inst=", inst, "new_ema=", new)

def update_flops_inflow(state):
    """
    Compute λ_g (FLOPs/s) for each GPU based on:
      - global frame rate lambda_fps, and
      - the current outstanding FLOPs on that GPU.

    We use:
        flop_backlog[g] ≈ Σ_j W_j   (running + queued)
        λ_g ≈ lambda_fps * flop_backlog[g]
    """
    from collections import defaultdict

    inflow = defaultdict(float)

    fps = float(getattr(state, "lambda_fps", 0.0))
    if fps <= 0.0:
        state.flops_inflow_per_gpu = inflow
        return

    # NEW: use FLOP backlog, not service time
    flop_bg = getattr(state, "flop_backlog", {}) or {}

    for raw_key, W_sum in flop_bg.items():
        try:
            gkey = _norm_gkey(raw_key)
        except Exception:
            gkey = raw_key

        inflow[gkey] = fps * float(W_sum)

    state.flops_inflow_per_gpu = inflow

def dvfs_retune_event(state, g, now):
    """
    Window-based DVFS retune for the heuristic (least-load) strategy.

    Behaviour (heuristic controller ONLY, adaptive freq mode ONLY):

      * Recompute λ_g via update_flops_inflow(state).
      * Let f_cur be the current frequency bin for GPU g (from state.current_freq).
      * Compute ρ = λ_g / μ(g, f_cur).
      * If ρ >= dvfs_up_util_thresh (default 0.9), move one bin UP.
      * Otherwise keep the current bin.
      * Changes are STRICTLY up-only (monotone non-decreasing frequency).
      * Changes are applied:
          - immediately when GPU is idle, OR
          - via gpu_freq_pending when GPU is busy (applied at boundary / idle).
      * dvfs_active_windows maintains one row per (GPU, freq) window; the
        'end' timestamp is updated whenever a window is extended.
    """

    # Only heuristic controller in adaptive mode is allowed here.
    if str(getattr(state, "dvfs_controller", "heuristic")).lower() != "heuristic":
        return
    freq_mode = str(getattr(state, "freq_mode", "adaptive")).lower()
    if freq_mode != "adaptive":
        return

    # Normalise GPU key g as a tuple (C,N,G)
    if isinstance(g, (tuple, list)) and len(g) == 3:
        g = tuple(map(str, g))
    else:
        parts = str(g).replace("/", ",").replace("-", ",").replace(":", ",").split(",")
        if len(parts) < 3:
            return
        g = (parts[0].strip(), parts[1].strip(), parts[2].strip())

    # Refresh λ_g estimates based on current outstanding work
    update_flops_inflow(state)

    # Stop once system has fully drained *after* DVFS started
    active = getattr(state, "active_jobs", {}) or {}
    try:
        vals = active.values()
    except Exception:
        vals = getattr(active, "values", lambda: [])()
    any_jobs_left = any(bool(v) for v in vals)
    if not any_jobs_left and getattr(state, "dvfs_started", False):
        return

    # DVFS window length
    W = float(getattr(state, "dvfs_window", 0.5))

    catalog = getattr(state, "gpu_catalog", None)
    rates   = (catalog.rates if catalog else {}) or {}

    # ---- local helper: sorted freq bins for this GPU ----
    def _freq_bins_local(gkey):
        fb = getattr(state, "freq_steps_by_gpu", {}) or {}
        steps = fb.get(gkey)
        if steps:
            return steps

        fmap = (rates.get(gkey) or {})
        if fmap:
            def _as_float(k):
                try:
                    return float(str(k).rstrip(".0"))
                except Exception:
                    return float("inf")
            steps = sorted(fmap.keys(), key=_as_float)
            if hasattr(state, "freq_steps_by_gpu"):
                state.freq_steps_by_gpu[gkey] = steps
            return steps

        # no fmap → fallbacks
        cur = (getattr(state, "current_freq", {}) or {}).get(gkey)
        if cur:
            return [cur]
        defaults = getattr(state, "gpu_catalog", None)
        defaults = getattr(defaults, "defaults", {}) if defaults else {}
        df = defaults.get(gkey)
        return [df] if df is not None else ["1500"]

    fb = _freq_bins_local(g)
    if not fb:
        return

    # ---- local helper: μ(g,f) ----
    def _mu(gkey, f):
        fmap = (rates.get(gkey) or {})
        if not fmap:
            return 1e-12
        key = _freq_key_match_in_map(fmap, f)
        if not key:
            return 1e-12
        try:
            return max(1e-12, float(fmap[key]))
        except Exception:
            return 1e-12

    # Current freq bin (from state.current_freq, falling back to midpoint)
    cur = (getattr(state, "current_freq", {}) or {}).get(g)
    if not cur:
        # fall back to midpoint if unknown
        cur = fb[len(fb) // 2]

    try:
        idx_cur = fb.index(cur)
        fcur = fb[idx_cur]
    except ValueError:
        idx_cur = len(fb) // 2
        fcur = fb[idx_cur]

    # Load-based ρ = λ_g / μ(g,fcur)
    lam = float((getattr(state, "flops_inflow_per_gpu", {}) or {}).get(g, 0.0))
    mu  = _mu(g, fcur)
    rho = lam / mu if mu > 0 else 0.0

    # Up-shift threshold (spec: 0.9)
    up_thr = float(getattr(state, "dvfs_up_util_thresh", 0.9))

    # strictly upshift-only: index can only stay or go up
    try:
        idx_cur = fb.index(fcur)
    except ValueError:
        idx_cur = len(fb) // 2
        fcur = fb[idx_cur]

    if rho >= up_thr:
        idx = min(len(fb) - 1, max(0, idx_cur + 1))
        fnew = fb[idx]
    else:
        fnew = fcur

    # ===== Idle-only retune (busy → mark pending, idle → apply now) =====
    st_key = f"GPU:{g[0]}:{g[1]}:{g[2]}"
    st = (getattr(state, "stations", {}) or {}).get(st_key)
    is_busy = bool(getattr(st, "busy", False)) or (int(getattr(st, "q_len", 0)) > 0)

    if is_busy:
        # Remember desired freq; applied at boundary / idle
        pend = getattr(state, "gpu_freq_pending", {}) or {}
        # fnew is guaranteed to be >= fcur index-wise (no downshift)
        pend[g] = fnew
        state.gpu_freq_pending = pend

        fel = getattr(state, "fel", None)
        if fel is not None:
            fel.schedule(now + W, Ev.DVFS_RETUNE, {"gpu": g})
        return

    # ===== No-op guard: extend window if freq unchanged =====
    if str(fcur) == str(fnew):
        win = (getattr(state, "dvfs_active_windows", {}) or {}).get(g)
        if isinstance(win, dict):
            win["end"] = float(now)  # extend current window
        fel = getattr(state, "fel", None)
        if fel is not None:
            fel.schedule(now + W, Ev.DVFS_RETUNE, {"gpu": g})
        return

    # Apply real change: update state then open a *new* window
    state.current_freq[g] = fnew
    print(f"[DVFS-heur] t={now:.3f} GPU={g} rho={rho:.4f} mu={mu:.3e} lam={lam:.3e}")

    # close previous window if present
    win_map = getattr(state, "dvfs_active_windows", {}) or {}
    prev_win = win_map.get(g)
    if isinstance(prev_win, dict):
        prev_win["end"] = float(now)

    fmap = (rates.get(g) or {})
    if fmap:
        try:
            safe_set_gpu_freq(
                state,
                g,
                fnew,
                reason="heuristic-window",
                fmap=fmap,
                when=now,
                trigger_job_id="-",
                trigger_task_id="-",
            )
        except Exception:
            # never kill the sim due to a heuristic DVFS bug
            pass

    # open new window for (g, fnew)
    win_map[g] = {
        "Cluster": g[0],
        "Node":    g[1],
        "GPU":     g[2],
        "freq":    str(fnew),
        "start":   float(now),
        "end":     float(now),
    }
    state.dvfs_active_windows = win_map

    # Schedule next retune while work is present
    fel = getattr(state, "fel", None)
    if fel is not None:
        fel.schedule(now + W, Ev.DVFS_RETUNE, {"gpu": g})

def pick_best_by_least_load(state, pkt):
    """
    Least-Completion-Time heuristic + utilization-aware adaptive DVFS.

    score = tail_backlog_s + service_time_s
            + alpha * assign_count
            + beta  * rho
            + deadline_w * lateness_s

    Conventions in THIS function:
      - state.tail[g], state.pending_tail[g] are BACKLOG SECONDS (>= 0)
      - 'now' is the current time (absolute seconds)
      - when we need an absolute finish timestamp, we use:
            now + backlog_s + service_s

    Returns: ((C,N,G), "heuristic-lct-adaptive")
    """
    # ---- OPTIONAL: admission stats for consistency with FastPolicy ----
    # Count *distinct jobs* admitted, not tasks.
    if not hasattr(state, "admit_stats") or state.admit_stats is None:
        state.admit_stats = {
            "admitted": 0,
            "dropped":  0,
            "dropped_jobs": [],
            "dropped_job_ids": set(),
        }

    stats = state.admit_stats

    # Normalize Job_ID (similar to FastPolicy)
    jid = str(pkt.get("Job_ID") or "").strip()
    if not jid:
        # Fallback: Task_ID like "AR_J0_T0_U" -> "AR_J0"
        tid = str(pkt.get("Task_ID", "")).strip()
        parts = tid.split("_")
        jid = "_".join(parts[:2]) if len(parts) >= 2 else tid

    # Only increment once per job
    if jid:
        if "admitted_job_ids" not in stats:
            stats["admitted_job_ids"] = set()
        if jid not in stats["admitted_job_ids"]:
            stats["admitted"] += 1
            stats["admitted_job_ids"].add(jid)
    # -------------------------------------------------------------------

    now   = float(getattr(state, "t", 0.0))
    alpha = float(getattr(state, "alpha_penalty", 0.003))
    beta  = float(getattr(state, "beta_penalty",  0.01))
    deadline_w = float(getattr(state, "gamma_deadline_penalty", 1.0))
    hard_guard = bool(getattr(state, "hard_deadline_guard", False))

    # --- special handling for fixed-frequency least-load ----------------
    freq_mode    = str(getattr(state, "freq_mode", "adaptive")).lower()
    is_leastload = bool(getattr(state, "is_least_load", False))
    is_fixed_ll  = (freq_mode == "fixed" and is_leastload)

    if is_fixed_ll:
        # Fixed DVFS: don't let ρ dominate
        beta = 0.0
        max_queue_horizon = float(
            getattr(state, "max_queue_horizon_fixed_s",
                    getattr(state, "max_queue_horizon_s", 0.3))
        )
    else:
        max_queue_horizon = float(
            getattr(state, "max_queue_horizon_s", 0.3)
        )

    # FLOPs for the incoming task
    Wt  = float(pkt.get("Workload_FLOPs") or pkt.get("Workload") or 0.0)
    ddl = float(pkt.get("Task_Deadline", float("inf")))
    if (not math.isfinite(ddl)) or ddl <= 0.0:
        ddl = float("inf")

    catalog = getattr(state, "gpu_catalog", None)
    rates   = (catalog.rates if catalog else {}) or {}
    if not rates:
        raise RuntimeError("[pick_best_by_least_load] GPU catalog is empty.")

    # ---- candidate list (cached) ----------------------------------------
    candidates = getattr(state, "gpu_candidates", None)
    if not candidates:
        candidates = [tuple(map(str, k)) for k, fmap in rates.items() if fmap]
        state.gpu_candidates = candidates

    if not candidates:
        raise RuntimeError("No candidate GPUs available for pick_best_by_least_load")

    # ---- helpers --------------------------------------------------------
    def _mu(gkey, f):
        fmap = (rates.get(gkey) or {})
        if not fmap:
            return 1e-12
        key = _freq_key_match_in_map(fmap, f)
        if not key:
            return 1e-12
        try:
            return max(1e-12, float(fmap[key]))
        except Exception:
            return 1e-12

    def _tail_backlog(gkey):
        """
        Return backlog seconds for GPU gkey, combining committed and pending.
        Never negative.
        """
        tail_map = getattr(state, "tail", {}) or {}
        pend_map = getattr(state, "pending_tail", {}) or {}
        return max(
            float(tail_map.get(gkey, 0.0)),
            float(pend_map.get(gkey, 0.0)),
            0.0,
        )

    def _rho(gkey, f):
        lam = _lambda_for_gpu(state, gkey)
        mu  = _mu(gkey, f)
        return lam / mu if mu > 0 else 0.0

    def _tie(g1, g2, tail1, tail2):
        # tail1, tail2 are backlog seconds (already incl. cluster base delay)
        if g1 is None:
            return False
        if g2 is None:
            return True
        if abs(tail1 - tail2) > 1e-9:
            return tail1 < tail2
        idx = state._rr_index
        state._rr_index = idx + 1
        return (idx % 2) == 0

    # ---- cluster-level base delay lookup  --------------------------------
    # Example in config.json:
    #   "Cluster_Delay_Base_s": { "C1": 0.0, "C2": 0.03 }
    # Prefer state.cfg_plain; fall back to state.config / state.cfg
    cfg_plain = getattr(state, "cfg_plain", None)
    if not cfg_plain:
        cfg_plain = getattr(state, "config", {}) or getattr(state, "cfg", {}) or {}

    base_delays = (cfg_plain.get("Cluster_Delay_Base_s", {}) or {})

    # DEBUG: print this only once
    if getattr(state, "debug_cluster_delay", False) and \
       not getattr(state, "_printed_cluster_delay", False):
        print("[LL-DBG] Cluster_Delay_Base_s map:", base_delays)
        print("[LL-DBG] freq_mode =", getattr(state, "freq_mode", None),
              "strategy =", getattr(state, "strategy", None))
        state._printed_cluster_delay = True

    # ---- main loop -------------------------------------------------------
    best_g     = None
    best_score = float("inf")
    best_tail  = float("inf")   # backlog+cluster_s for tie-breaking

    gpu_to_freq  = getattr(state, "gpu_to_freq", {}) or {}
    current_freq = getattr(state, "current_freq", {}) or {}

    if not hasattr(state, "_rr_index"):
        state._rr_index = 0

    for gkey in candidates:
        gkey = tuple(map(str, gkey))

        tail_backlog = _tail_backlog(gkey)  # seconds of queued work

        # skip if queue backlog already too large
        if tail_backlog > max_queue_horizon:
            continue

        # probe freq
        fb = _freq_bins(state, gkey)
        if fb:
            probe = current_freq.get(gkey) or _gpu_hw_freq_key(state, gkey)
            if probe not in fb:
                probe = fb[len(fb) // 2]
        else:
            probe = current_freq.get(gkey, gpu_to_freq.get(_norm_gkey(gkey), ""))

        if not probe:
            fb = fb or [str(k) for k in sorted(
                (rates.get(gkey) or {}).keys(),
                key=lambda x: float(str(x).rstrip(".0"))
            )]
            probe = fb[len(fb) // 2]

        mu      = _mu(gkey, probe)
        service = Wt / mu if mu > 0 else float("inf")   # service time seconds

        # predicted absolute finish time for this task
        pred_finish_abs = now + tail_backlog + service

        lateness = 0.0
        if math.isfinite(ddl):
            lateness = max(0.0, pred_finish_abs - ddl)

        # ----- apply cluster base delay as extra backlog -------------------
        cluster_id = str(gkey[0])
        base_delay = float(base_delays.get(cluster_id, 0.0))
        cluster_penalty_w = 0.2   # e.g. 20% of its raw value
        tail_with_cluster = tail_backlog + cluster_penalty_w * base_delay
        # tail_with_cluster = tail_backlog + base_delay  # still backlog seconds

        if hard_guard and math.isfinite(ddl) and pred_finish_abs > ddl:
            continue

        acount = (getattr(state, "assigned_counts", {}) or {}).get(gkey, 0)
        rho    = _rho(gkey, probe)

        score = (
            tail_with_cluster
            + service
            + alpha * acount
            + beta  * rho
            + deadline_w * lateness
        )

        if (score < best_score - 1e-9) or (
            abs(score - best_score) <= 1e-9 and _tie(gkey, best_g, tail_with_cluster, best_tail)
        ):
            best_g, best_score, best_tail = gkey, score, tail_with_cluster

    if best_g is None:
        # fall back to smallest backlog if everything exceeded horizon
        best_g = min(candidates, key=lambda gk: _tail_backlog(gk))

    # --- choose initial DVFS bin for this GPU based on ρ thresholds ---------
    fb = _freq_bins(state, best_g)
    fb = fb or [str(k) for k in sorted(
        (rates.get(best_g) or {}).keys(),
        key=lambda x: float(str(x).rstrip(".0"))
    )]

    # probe at current hardware freq if possible
    probe = current_freq.get(best_g) or _gpu_hw_freq_key(state, best_g)
    if probe not in fb:
        probe = fb[len(fb) // 2]

    if is_fixed_ll:
        # FIXED least-load: always run at fastest DVFS step
        f_init = fb[-1] if fb else probe
    else:
        rho_now = _rho(best_g, probe)
        lam     = float(getattr(state, "lambda_fps", 0.0))

        if lam >= 5.0:
            if   rho_now > 0.30: f_init = fb[-1]         # go to max earlier
            elif rho_now > 0.15: f_init = fb[len(fb)//2]
            else:                f_init = fb[0]


        # if lam >= 5.0:
        #     # Heavier load: avoid very low freq for new tasks
        #     if rho_now > 0.40:
        #         f_init = fb[-1]          # go straight to f_max
        #     else:
        #         f_init = fb[len(fb)//2]  # at least mid bin
        # else:
        #     # Original conservative mapping for lighter load
        #     if   rho_now > 0.60: f_init = fb[-1]      # f_max
        #     elif rho_now > 0.30: f_init = fb[len(fb)//2]
        #     else:                f_init = fb[0]


    # ---- safety floor: never go below min_ll_freq_index ----
    def _freq_index(bins, f):
        try:
            return bins.index(f)
        except ValueError:
            # fall back to mid-bin if f is not in bins
            return max(0, min(len(bins) - 1, len(bins) // 2))

    min_idx = int(getattr(state, "min_ll_freq_index", -1))
    if min_idx >= 0 and fb:
        idx = _freq_index(fb, f_init)
        if idx < min_idx:
            f_init = fb[min_idx]

    # --- reserve time for this job at f_init (UPDATE BACKLOG) --------------
    mu_init = _mu(best_g, f_init)
    dt = Wt / mu_init if mu_init > 0 else float("inf")   # extra backlog seconds

    pending = getattr(state, "pending_tail", {}) or {}
    prev_backlog = float(pending.get(best_g, 0.0))
    pending[best_g] = max(prev_backlog, 0.0) + dt
    state.pending_tail = pending

    tail_map = getattr(state, "tail", {}) or {}
    tail_map[best_g] = max(float(tail_map.get(best_g, 0.0)), pending[best_g])
    state.tail = tail_map

    # bump counts and record desired freq choice
    ac = getattr(state, "assigned_counts", {}) or {}
    ac[best_g] = ac.get(best_g, 0) + 1
    state.assigned_counts = ac

    cur = getattr(state, "current_freq", {}) or {}
    cur[best_g] = f_init
    state.current_freq = cur
    state._last_freq_reason = "least-load-init"

    # --- immediate idle-arrival upshift (spec behaviour) --------------------
    st_key = f"GPU:{best_g[0]}:{best_g[1]}:{best_g[2]}"
    stations = getattr(state, "stations", {}) or {}
    st = stations.get(st_key)
    is_busy = bool(getattr(st, "busy", False)) or (int(getattr(st, "q_len", 0)) > 0)

    fmap_g = (rates.get(best_g) or {})
    cur_hw = _gpu_hw_freq_key(state, best_g, fmap=fmap_g)

    # Only upshift on idle GPU, and only if f_init is a higher bin than cur_hw
    # Only adaptive LL is allowed to change frequencies here
    if (not is_fixed_ll) and (not is_busy) and fmap_g and (f_init != cur_hw):
        fb = _freq_bins(state, best_g)
        if fb and fb.index(f_init) > fb.index(cur_hw):
            try:
                safe_set_gpu_freq(
                    state,
                    best_g,
                    f_init,
                    reason="least-load-arrival",
                    fmap=fmap_g,
                    when=now,
                    trigger_job_id=str(pkt.get("Job_ID", "-")),
                    trigger_task_id=str(pkt.get("Task_ID", "-")),
                )
            except Exception:
                pass


    return (best_g, "heuristic-lct-adaptive")

def _opened_by_for(reason: str, fm: str = "") -> str:
    """
    Map DVFS change reason → stable window 'opened_by' label.

    IMPORTANT:
    - This is only meaningful when the frequency ACTUALLY changes.
    - Boundary / heuristic checks that keep the same freq must NOT create
      new semantic windows.
    """

    fm = (fm or "").lower()
    r  = (reason or "").lower()

    # Hard lock
    if fm == "fixed":
        return "fixed"

    # Startup / seeding
    if "seed" in r or "startup" in r or "init_seed" in r:
        return "seed/startup"

    # Optimizer chose a NEW frequency
    if "optimizer" in r:
        return "optimizer"

    # First real service on GPU
    if "service-start" in r:
        return "service-start"

    # Explicit assignment-based DVFS
    if "assign" in r:
        return "assign"

    # Normalization / cleanup
    if "normalize" in r:
        return "normalize"

    # Boundary / heuristic retune
    #  ONLY if it actually caused a freq change
    if "boundary" in r or "heuristic" in r:
        return "retuner"

    # Default
    return "dvfs"


@dataclass
class Dvfsevent:
    t: float
    cluster: str
    node: str
    gpu: str
    old_f: str
    new_f: str
    reason: str
    opened_by: str
    trigger_job_id: str = ""
    trigger_task_id: str = ""


def dvfslogger_events_to_dicts(logger):
    out = []
    for e in logger._events:
        out.append({
            "time": float(e.t),
            "Cluster": str(e.cluster), "Node": str(e.node), "GPU": str(e.gpu),
            "old_f": str(e.old_f), "new_f": str(e.new_f),
            "reason": str(e.reason), "opened_by": str(e.opened_by),
            "trigger_job_id":  str(e.trigger_job_id),
            "trigger_task_id": str(e.trigger_task_id),
        })
    return out

class DVFSLogger:
    def __init__(self):
        self._events: list[Dvfsevent] = []

    def seed(self, t0, c, n, g, init_f, opened_by="init_seed", trigger_job_id="", trigger_task_id=""):
        self._events.append(
            Dvfsevent(
                float(t0), str(c), str(n), str(g),
                old_f="-", new_f=str(init_f), reason="init_seed",
                opened_by=str(opened_by),
                trigger_job_id=str(trigger_job_id),
                trigger_task_id=str(trigger_task_id),
            )
    )

    def change(self, t, c, n, g, old_f, new_f, reason, opened_by="dvfs", trigger_job_id="", trigger_task_id=""):
        self._events.append(
            Dvfsevent(
                float(t), str(c), str(n), str(g),
                old_f=str(old_f), new_f=str(new_f), reason=str(reason),
                opened_by=str(opened_by),
                trigger_job_id=str(trigger_job_id),
                trigger_task_id=str(trigger_task_id),
            )
        )


    def dump_csv(self, out_csv: str | Path):
        out_csv = Path(out_csv); out_csv.parent.mkdir(parents=True, exist_ok=True)
        self._events.sort(key=lambda e: (e.cluster, e.node, e.gpu, e.t))
        with out_csv.open("w", newline="") as f:
            w = csv.DictWriter(
                f,
                fieldnames=["t","Cluster","Node","GPU","old_f","new_f", "reason","opened_by","trigger_job_id","trigger_task_id"]
            )
            w.writeheader()
            for ev in self._events:
                w.writerow({
                    "t": ev.t, "Cluster": ev.cluster, "Node": ev.node, "GPU": ev.gpu,
                    "old_f": ev.old_f, "new_f": ev.new_f, "reason": ev.reason,
                    "opened_by": ev.opened_by, "trigger_job_id": ev.trigger_job_id,
                    "trigger_task_id": ev.trigger_task_id,
                })

def _norm_gkey(gkey):
    # tuple/list
    if isinstance(gkey, (tuple, list)) and len(gkey) == 3:
        return (str(gkey[0]), str(gkey[1]), str(gkey[2]))

    s = str(gkey).strip()

    # handle stringified tuple: "('C1', 'N1', 'G1')" or '("C1","N1","G1")'
    m = re.findall(r"[A-Za-z0-9_]+", s)
    if len(m) >= 3:
        return (m[0], m[1], m[2])

    # fallback: common delimiters
    s = s.replace(":", "-").replace(",", "-").replace(" ", "")
    parts = [p for p in s.split("-") if p]
    return (parts[0], parts[1], parts[2]) if len(parts) >= 3 else ("", "", "")


# --- DVFS epoch gate: allow at most one retune between new job arrivals ---
def _dvfs_epoch_bump(state, gkey):
    if not hasattr(state, "_dvfs_epoch") or not isinstance(state._dvfs_epoch, dict):
        state._dvfs_epoch = {}
    if not hasattr(state, "_dvfs_epoch_applied") or not isinstance(state._dvfs_epoch_applied, dict):
        state._dvfs_epoch_applied = {}

    gkey = _norm_gkey(gkey)

    # ensure applied map has a value for this GPU
    state._dvfs_epoch_applied.setdefault(gkey, -1)

    state._dvfs_epoch[gkey] = int(state._dvfs_epoch.get(gkey, 0)) + 1
    _dbg(state, f"EPOCH(bump) GPU={gkey} -> {state._dvfs_epoch[gkey]}")


def _dvfs_epoch_mark_applied(state, gkey):
    if not hasattr(state, "_dvfs_epoch") or not isinstance(state._dvfs_epoch, dict):
        state._dvfs_epoch = {}
    if not hasattr(state, "_dvfs_epoch_applied") or not isinstance(state._dvfs_epoch_applied, dict):
        state._dvfs_epoch_applied = {}
    gkey = _norm_gkey(gkey)
    # latch the current epoch index as "applied"
    cur_epoch = int(state._dvfs_epoch.get(gkey, 0))
    state._dvfs_epoch_applied[gkey] = cur_epoch
    _dbg(state, f"EPOCH(applied) GPU={gkey} -> {cur_epoch}")

def _dvfs_epoch_pending(state, gkey) -> bool:
    cur = int((getattr(state, "_dvfs_epoch", {}) or {}).get(gkey, 0))
    ap  = int((getattr(state, "_dvfs_epoch_applied", {}) or {}).get(gkey, -1))
    return cur != ap

# ========== DVFS logging & windows helpers ==========
def _adapt_dvfs_event(e) -> dict:
    if isinstance(e, dict):
        return {
            "time":      _to_float(e.get("time", e.get("t", 0.0)), 0.0),
            "Cluster":   str(e.get("Cluster", e.get("cluster",""))),
            "Node":      str(e.get("Node",    e.get("node",""))),
            "GPU":       str(e.get("GPU",     e.get("gpu",""))),
            "old_f":     str(e.get("old_f",   e.get("prev",""))),
            "new_f":     str(e.get("new_f",   e.get("Frequency", e.get("freq","")))),
            "reason":    str(e.get("reason",  e.get("opened_by",""))),
            "opened_by": str(e.get("opened_by", e.get("reason",""))),
            "trigger_job_id":  str(e.get("trigger_job_id",  e.get("job_id",""))),
            "trigger_task_id": str(e.get("trigger_task_id", e.get("task_id",""))),
        }
    return {
        "time":      _to_float(getattr(e, "t", 0.0), 0.0),
        "Cluster":   str(getattr(e, "cluster","")),
        "Node":      str(getattr(e, "node","")),
        "GPU":       str(getattr(e, "gpu","")),
        "old_f":     str(getattr(e, "old_f","")),
        "new_f":     str(getattr(e, "new_f", getattr(e, "freq",""))),
        "reason":    str(getattr(e, "reason","")),
        "opened_by": str(getattr(e, "opened_by","")),
        "trigger_job_id":  str(getattr(e, "trigger_job_id", getattr(e, "job_id",""))),
        "trigger_task_id": str(getattr(e, "trigger_task_id", getattr(e, "task_id",""))),
    }


def _collect_dvfs_events(state) -> list[dict]:
    """
    Collect DVFS events from the single canonical sink (state.dvfs_log),
    normalized via _adapt_dvfs_event and sorted by (Cluster,Node,GPU,time).
    """
    raw = getattr(state, "dvfs_log", []) or []
    evs = [_adapt_dvfs_event(e) for e in raw]

    # Filter out malformed rows
    evs = [e for e in evs if e["Cluster"] or e["Node"] or e["GPU"]]

    evs.sort(key=lambda r: (r["Cluster"], r["Node"], r["GPU"], r["time"]))
    return evs


def gpu_label(df):
    return (df["Assigned_Cluster"].astype(str) + "-" +
            df["Assigned_Node"].astype(str) + "-" +
            df["Assigned_GPU"].astype(str))


def _now(state) -> float:
    """
    Single canonical time source for the sim.

    Preference order (based on SimState definition):
      1) state.fel.t
      2) state.t (only if some code injects it)
      3) 0.0
    """
    try:
        fel = getattr(state, "fel", None)
        t = getattr(fel, "t", None) if fel is not None else None
        if t is not None:
            return float(t)
    except Exception:
        pass

    try:
        t = getattr(state, "t", None)
        if t is not None:
            return float(t)
    except Exception:
        pass

    return 0.0


def log_dvfs_change(
    state,
    gpu_key,
    old_f,
    new_f,
    reason,
    *,
    trigger_job_id=None,
    trigger_task_id=None,
    when=None,
    fmap=None,
    origin=None,
    force_log=False,
    **extras,
):
    """
    Canonical DVFS event logger.

    - Single sink: state.dvfs_log as list[dict].
    - Cheap per call: no stack introspection unless state.debug_dvfs is True.
    """
    # Drop exact no-ops unless forced
    if (str(old_f) == str(new_f)) and not force_log:
        return

    try:
        # Optional: expensive call-site tracing only when debugging
        if getattr(state, "debug_dvfs", False):
            try:
                frame = inspect.currentframe().f_back
                src = f"{frame.f_code.co_filename}:{frame.f_lineno} in {frame.f_code.co_name}"
                _dbg(state, f"[DVFS-LOG] from={src} reason={reason}")
            except Exception:
                _dbg(state, f"[DVFS-LOG] reason={reason} (no callsite)")

        # # Normalize GPU key
        # try:
        #     c, n_, g = map(str, _norm_gkey(gpu_key))
        # except Exception:
        #     c, n_, g = map(str, gpu_key)
        
        key3 = _norm_gkey(gpu_key)
        c, n_, g = map(str, key3)

        # Resolve fmap if not provided
        if fmap is None:
            try:
                fmap = (getattr(state.gpu_catalog, "rates", {}) or {}).get(
                    (c, n_, g), {}
                ) or {}
            except Exception:
                fmap = {}

        # Normalize freq keys against fmap
        old_k = _freq_key_match_in_map(fmap, old_f)
        new_k = _freq_key_match_in_map(fmap, new_f)

        # If we still don’t have keys, fall back to raw strings
        if not old_k:
            old_k = str(old_f)
        if not new_k:
            new_k = str(new_f)

        # Re-check no-op after normalization
        if not force_log and old_k == new_k:
            return

        if not force_log and str(old_k).strip() == str(new_k).strip():
            return

        t = float(_now(state) if when is None else when)
        t = round(t, 6)

        opened = _opened_by_for(reason, str(getattr(state, "freq_mode", "")))

        rec = {
            "time": t,
            "Cluster": c,
            "Node": n_,
            "GPU": g,
            "old_f": str(old_k),
            "new_f": str(new_k),
            "reason": str(reason or ""),
            "origin": str(origin or ""),
            "opened_by": opened,
            "Freq_Mode": str(getattr(state, "freq_mode", "adaptive")),
            "trigger_job_id": "" if trigger_job_id is None else str(trigger_job_id),
            "trigger_task_id": "" if trigger_task_id is None else str(trigger_task_id),
            **(extras or {}),
        }

        # Single canonical sink
        if not hasattr(state, "dvfs_log") or state.dvfs_log is None:
            state.dvfs_log = []
        state.dvfs_log.append(rec)

        if getattr(state, "debug_dvfs", False):
            _dbg(
                state,
                f"[DVFS-LOG] GPU={c}-{n_}-{g} old={old_k} new={new_k} "
                f"reason={reason} opened_by={opened}",
            )

        # Optional: keep DVFSLogger only when explicitly enabled
        if getattr(state, "debug_dvfs_logger", False) and hasattr(state, "dvfs_logger"):
            state.dvfs_logger.change(
                t,
                c,
                n_,
                g,
                old_f=str(old_k),
                new_f=str(new_k),
                reason=str(reason or ""),
                opened_by=opened,
                trigger_job_id=rec["trigger_job_id"],
                trigger_task_id=rec["trigger_task_id"],
            )

    except Exception:
        # Never crash because of logging
        pass


def _stamp_assignment_for_drop(state: "SimState", pkt: dict) -> None:
    """Stamp assignment/decision fields for a dropped task using ONLY the canonical job pin & plan."""
    job_id  = _norm_job_id(pkt)
    task_id = str(pkt.get("Task_ID", ""))

    # Always stamp mode
    pkt.setdefault("Freq_Mode", str(getattr(state, "freq_mode", "adaptive")))

    # ---- Resolve GPU strictly from the canonical job pin ----
    gpu_key = None
    ja = getattr(state, "job_assignment", {}).get(job_id)
    if isinstance(ja, (tuple, list)) and len(ja) == 2 and isinstance(ja[0], (tuple, list)) and len(ja[0]) == 3:
        gpu_key = (str(ja[0][0]), str(ja[0][1]), str(ja[0][2]))

    if gpu_key is not None:
        c, n, g = gpu_key
        pkt.setdefault("Assigned_Cluster", c)
        pkt.setdefault("Assigned_Node", n)
        pkt.setdefault("Assigned_GPU", g)

    # ---- Decision source: only from per-job provenance (no guessing) ----
    if hasattr(state, "job_gpu_src"):
        pkt.setdefault("GPU_Decision_Source", state.job_gpu_src.get(job_id, ""))

    # ---- Frequency: only from job plan or locked FIXED value ----
    f_req, f_src = "", ""

    # Planned per-job freq (must already be recorded)
    plan = getattr(state, "job_freq_plan", {}).get(job_id)
    if isinstance(plan, (tuple, list)) and len(plan) >= 2:
        # expect (gpu_tuple, freq, src)
        f_req = str(plan[1] or "")
        f_src = str(plan[2] or "") if len(plan) >= 3 else ""
    elif isinstance(plan, dict):
        f_req = str(plan.get("freq", "") or plan.get("Frequency", ""))
        f_src = str(plan.get("src", "") or plan.get("source", ""))

    # In FIXED mode, prefer the pinned freq from the canonical job pin
    if str(getattr(state, "freq_mode", "adaptive")).lower() == "fixed":
        if isinstance(ja, (tuple, list)) and len(ja) == 2:
            locked = str(ja[1] or "")
            if locked:
                f_req = locked
                if not f_src:
                    f_src = "locked"

    # Stamp only if we have a request; no defaults, no HW/current fallbacks
    if f_req:
        pkt.setdefault("Assigned_Frequency", f_req)
        pkt.setdefault("Service_Frequency", f_req)
    if f_src:
        pkt.setdefault("Freq_Decision_Source", f_src)

def derive_dvfs_windows(dvfs_log: list, run_end: float) -> list:
    run_end = _safe_float(run_end, 0.0)
    evs = list(dvfs_log or [])

    by = defaultdict(list)
    for e in evs:
        key = (str(e.get("Cluster","")), str(e.get("Node","")), str(e.get("GPU","")))
        by[key].append(e)

    out = []
    for (C, N, G), seq in by.items():
        seq.sort(key=lambda r: _safe_float(r.get("time"), 0.0))
        for i, e in enumerate(seq):
            t0 = _safe_float(e.get("time"), 0.0)
            t1 = _safe_float(seq[i+1].get("time"), run_end) if i+1 < len(seq) else run_end
            if not (math.isfinite(t0) and math.isfinite(t1)) or t1 <= t0:
                continue
            out.append({
                "Cluster": C, "Node": N, "GPU": G,
                "start": t0, "end": t1, "duration": round(t1 - t0, 6),
                # freq now kept in MHz (consistent with config)
                "freq": str(e.get("new_f", e.get("freq",""))),
                "opened_by": str(e.get("opened_by", "dvfs")),
                "trigger_job_id":  e.get("trigger_job_id",""),
                "trigger_task_id": e.get("trigger_task_id",""),
                "Freq_Mode":       e.get("Freq_Mode",""),
            })
    return out

def _sum_overlap(spans: list[tuple[float, float]], t0: float, t1: float) -> float:
    """
    Fast overlap between [t0,t1) and a GPU's busy spans.
    Assumes 'spans' is sorted, non-overlapping [(a,b), ...].
    """
    if not spans or t1 <= t0:
        return 0.0

    # find first span whose END is >= t0, then step forward
    ends = [b for _, b in spans]
    i = bisect_left(ends, t0) - 1
    if i < 0:
        i = 0

    total = 0.0
    n = len(spans)
    while i < n:
        a, b = spans[i]
        if a >= t1:
            break
        if b > t0:
            # overlap = [max(a,t0), min(b,t1))
            lo = a if a > t0 else t0
            hi = b if b < t1 else t1
            if hi > lo:
                total += (hi - lo)
        i += 1
    return total

def _intersect_len(a0, a1, b0, b1):
    lo = max(a0, b0); hi = min(a1, b1)
    return max(0.0, hi - lo)

def _busy_spans_from_rows(rows: list):
    spans = defaultdict(list)

    start_keys = ("gpu_entry_time","R0_UL_service_start","RC_DL_service_start","overall_start_time")
    end_keys   = ("gpu_exit_time","R0_UL_EG_exit","RC_DL_IN_exit","overall_end_time")

    for r in rows or []:
        try:
            C = str(r.get("Assigned_Cluster","")); N = str(r.get("Assigned_Node","")); G = str(r.get("Assigned_GPU",""))
            rs = re = None
            for k in start_keys:
                if r.get(k) not in (None, ""):
                    rs = _safe_float(r.get(k), None); break
            for k in end_keys:
                if r.get(k) not in (None, ""):
                    re = _safe_float(r.get(k), None); break
            if rs is None or re is None or not (math.isfinite(rs) and math.isfinite(re)) or re <= rs:
                continue
            spans[(C,N,G)].append((rs, re))
        except Exception:
            pass

    # merge per GPU
    for k, ivals in list(spans.items()):
        ivals.sort(key=lambda p: (p[0], p[1]))
        merged = []
        for a, b in ivals:
            if not merged or a > merged[-1][1]:
                merged.append([a, b])
            else:
                merged[-1][1] = max(merged[-1][1], b)
        spans[k] = [(a, b) for a, b in merged]
    return spans

def write_dvfs_windows_csv(path: str, windows: list) -> None:
    hdr = [
        "Cluster","Node","GPU","freq","start","end","duration",
        "active_time","idle_time","utilization",
        "opened_by","job_id","task_id","Freq_Mode",
    ]
    write_rows_csv(path, windows or [], hdr)


def write_window_tasks_csv(path, flat_rows):
    hdr = ["Cluster","Node","GPU","freq",
           "window_start","window_end","window_duration",
           "Job_ID","Task_ID","task_start","task_end","overlap_s"]
    _os.makedirs(_os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=hdr)
        w.writeheader()
        for r in flat_rows or []:
            w.writerow({k: r.get(k, "") for k in hdr})

# ========= DVFS energy: power per window & per-GPU totals =========

def _freq_norm_key(d: dict, f: str) -> str:
    f = str(f).strip()
    if f in d: return f
    if f.endswith(".0") and f[:-2] in d: return f[:-2]
    if (f + ".0") in d: return f + ".0"
    try:
        s = str(int(float(f)))
        if s in d: return s
        if s + ".0" in d: return s + ".0"
    except Exception:
        pass
    return f if f in d else ""

def _build_typemap_from_cfg(cfg: dict):
    cc = cfg.get("Cluster_Config", {})
    tmap = {}
    for c, nodes in cc.items():
        for n, gpus in nodes.items():
            for g, info in gpus.items():
                tmap[(str(c),str(n),str(g))] = str(info.get("type",""))
    return tmap

# ---- tiny helpers ------------------------------------------------------------
def _safe_float(x, default=0.0):
    try:
        s = "" if x is None else str(x).strip()
        return default if s == "" else float(s)
    except Exception:
        return default

def _as_ghz(x) -> float:
    # handles '', None, '  ', '1500', 1500, '2.505'
    f = _safe_float(x, default=0.0)
    return (f / 1000.0) if f > 100.0 else f

def _bounded01(x) -> float:
    return max(0.0, min(1.0, _safe_float(x, default=0.0)))

def _collect_freqs_for_type(cfg: dict, gtype: str):
    """
    Return DVFS points (in MHz) for this GPU type, using:
      1) GPU_Specs[gtype]["freqs"]
      2) Any 'service_rates' dicts under Cluster_Config keyed by freq strings
    """
    specs = cfg.get("GPU_Specs", {}) or {}

    lst = (specs.get(gtype, {}) or {}).get("freqs")
    if lst:
        vals = set()
        for f in lst:
            v = _safe_float(f)
            if v > 0:
                vals.add(v)
        return sorted(vals)


    found = set()
    def _scan(obj):
        if isinstance(obj, dict):
            if "service_rates" in obj and isinstance(obj["service_rates"], dict):
                for k in obj["service_rates"].keys():
                    fk = _safe_float(k)
                    if fk > 0:
                        found.add(fk)
            for v in obj.values():
                _scan(v)
        elif isinstance(obj, list):
            for v in obj:
                _scan(v)

    _scan(cfg.get("Cluster_Config", {}))
    return sorted(found)


def _rows_by_gpu_sorted(rows):
    """
    Pre-index packet rows by (Cluster,Node,GPU) and sort by start time.

    Store each entry as (start, end, row_ref) to avoid copying dicts.
    """
    by_gpu = defaultdict(list)
    if not rows:
        return by_gpu

    for r in rows:
        C = str(r.get("Assigned_Cluster", ""))
        N = str(r.get("Assigned_Node", ""))
        G = str(r.get("Assigned_GPU", ""))

        rs = (r.get("gpu_entry_time") or
              r.get("R0_UL_service_start") or
              r.get("overall_start_time"))
        re = (r.get("gpu_exit_time") or
              r.get("RC_DL_IN_exit") or
              r.get("overall_end_time"))

        rs = _safe_float(rs, None)
        re = _safe_float(re, None)
        if rs is None or re is None or not (math.isfinite(rs) and math.isfinite(re)) or re <= rs:
            continue

        # compact tuple: (start, end, original_row)
        by_gpu[(C, N, G)].append((rs, re, r))

    for seq in by_gpu.values():
        seq.sort(key=lambda x: x[0])  # sort by start
    return by_gpu


def _windows_by_gpu_sorted(windows):
    """
    Group DVFS windows by (Cluster, Node, GPU) and sort each group by start time.
    """
    by_gpu = {}
    for w in (windows or []):
        C = str(w.get("Cluster", ""))
        N = str(w.get("Node", ""))
        G = str(w.get("GPU", ""))
        key = (C, N, G)
        by_gpu.setdefault(key, []).append(w)

    for key, lst in by_gpu.items():
        lst.sort(key=lambda r: _safe_float(r.get("start"), 0.0))

    return by_gpu


def compute_power_windows(cfg, windows, rows=None):
    """
    Compute per-window and per-GPU power/energy using DVFS windows
    and packet-level logs.

    This version is O(N + M) per GPU:
      - pre-index rows per (C,N,G) and sort by start time
      - group windows per (C,N,G) and sort by start time
      - stream through rows and windows with a moving index
    """
    p_active_util, p_idle_raw = _power_lookup_from_cfg(cfg)

    if not windows:
        return [], []

    # Pre-index rows by GPU, sorted by _start (we already defined this helper)
    rows_by_gpu = _rows_by_gpu_sorted(rows or [])
    wins_by_gpu = _windows_by_gpu_sorted(windows)

    out = []
    per_gpu = {}
    per_gpu_freq = {}

    for key_gpu, wlist in wins_by_gpu.items():
        C, N, G = key_gpu

        rows_gpu = rows_by_gpu.get(key_gpu, [])
        nrows = len(rows_gpu)

        # moving pointer into rows for THIS GPU
        i = 0

        for w in wlist:
            fkey  = w.get("freq", "")
            f_mhz = _safe_float(fkey, 0.0)

            start = _safe_float(w.get("start"), 0.0)
            end   = _safe_float(w.get("end"),   start)
            if not (math.isfinite(start) and math.isfinite(end)) or end <= start:
                continue

            dur = max(0.0, end - start)

            # advance i until rows_gpu[i] ends after window start
            while i < nrows and rows_gpu[i][1] <= start:
                i += 1

            j = i
            util_num = 0.0
            util_den = 0.0
            Bk_rows  = 0.0
            jid = ""
            tid = ""

            util_keys = ("gpu_active_frac", "U_active", "U_g", "SM_occupancy")

            while j < nrows:
                rs, re, r = rows_gpu[j]   # <<< UNPACK TUPLE HERE

                if rs >= end:
                    break

                # overlap with window
                s = max(start, rs)
                e = min(end, re)
                ol = e - s
                if ol > 0:
                    if not jid:
                        jid = str(r.get("Job_ID", "") or r.get("job_id", ""))
                    if not tid:
                        tid = str(r.get("Task_ID", "") or r.get("task_id", ""))

                    u_this = None
                    for k in util_keys:
                        if k in r and r[k] not in (None, ""):
                            u_this = _bounded01(r[k])
                            break
                    if u_this is None:
                        u_this = 1.0

                    util_num += u_this * ol
                    util_den += ol
                    Bk_rows  += ol

                j += 1

            # If the window already has active_time/idle_time from
            # compute_active_idle_from_logs, respect that. Otherwise
            # use the active time we just derived from rows.
            if "active_time" in w:
                Bk = _safe_float(w.get("active_time"), 0.0)
                Ik = _safe_float(
                    w.get("idle_time"),
                    max(0.0, dur - Bk),
                )
            else:
                Bk = Bk_rows
                Ik = max(0.0, dur - Bk)

            # Utilisation: prefer window's gpu_util_active if present,
            # else use the overlapping-row weighted average.
            Ua_rows = _bounded01(util_num / util_den) if util_den > 0 else 1.0
            U_active = _bounded01(w.get("gpu_util_active", Ua_rows))

            Pi = float(p_idle_raw(key_gpu, f_mhz))
            Pa = float(p_active_util(key_gpu, f_mhz, U_active))
            if not (math.isfinite(Pa) and math.isfinite(Pi)):
                raise ValueError(f"[power] non-finite Pa/Pi for {key_gpu}@{f_mhz}MHz: {Pa}, {Pi}")
            if Pa <= 0 or Pi <= 0:
                raise ValueError(f"[power] non-positive Pa/Pi for {key_gpu}@{f_mhz}MHz: {Pa}, {Pi}")
            if Pa < Pi:  # clamp
                Pa = Pi

            E_static  = Pi * dur
            E_dynamic = (Pa - Pi) * max(0.0, Bk)
            E_total   = E_static + E_dynamic

            row = dict(w)
            row.update({
                "Cluster":     C,
                "Node":        N,
                "GPU":         G,
                "duration":    round(dur, 6),
                "active_time": round(Bk, 6),
                "idle_time":   round(Ik, 6),
                "utilization": 0.0 if dur <= 0 else round(min(1.0, max(0.0, Bk/dur)), 6),
                "P_idle_W":    round(Pi, 6),
                "P_active_W":  round(Pa, 6),
                "E_idle_J":    round(Pi * Ik, 6),
                "E_active_J":  round((Pa - Pi) * Bk, 6),
                "E_static_J":  round(E_static, 6),
                "E_total_J":   round(E_total, 6),
                "P_window_avg_W": 0.0 if dur <= 0 else round(E_total / dur, 6),
                "U_active":    round(U_active, 6),
                "opened_by":   w.get("opened_by", "dvfs"),
                "trigger_job_id":  w.get("trigger_job_id",  jid) or "-",
                "trigger_task_id": w.get("trigger_task_id", tid) or "-",
                "Freq_Mode":   w.get("Freq_Mode", cfg.get("Frequency_Mode", "")),
            })
            out.append(row)

            # ---- per-GPU totals ----
            agg = per_gpu.setdefault(key_gpu, {
                "Cluster": C, "Node": N, "GPU": G,
                "active_time": 0.0, "idle_time": 0.0,
                "E_static_J": 0.0, "E_dynamic_J": 0.0, "E_total_J": 0.0,
                "total_time_s": 0.0, "_U_active_time_weighted": 0.0,
            })
            agg["active_time"]  += Bk
            agg["idle_time"]    += Ik
            agg["E_static_J"]   += E_static
            agg["E_dynamic_J"]  += E_dynamic
            agg["E_total_J"]    += E_total
            agg["total_time_s"] += dur
            agg["_U_active_time_weighted"] += U_active * max(0.0, Bk)

            # ---- optional per-(GPU,freq) ----
            key_gf = (C, N, G, str(fkey))
            agg2 = per_gpu_freq.setdefault(key_gf, {
                "Cluster": C, "Node": N, "GPU": G, "freq": str(fkey),
                "active_time": 0.0, "idle_time": 0.0,
                "E_static_J": 0.0, "E_dynamic_J": 0.0, "E_total_J": 0.0,
                "window_s": 0.0, "_U_active_time_weighted": 0.0,
            })
            agg2["active_time"] += Bk
            agg2["idle_time"]   += Ik
            agg2["E_static_J"]  += E_static
            agg2["E_dynamic_J"] += E_dynamic
            agg2["E_total_J"]   += E_total
            agg2["window_s"]    += dur
            agg2["_U_active_time_weighted"] += U_active * max(0.0, Bk)

    # finalize per-GPU totals (unchanged behaviour)
    totals = []
    for rec in per_gpu.values():
        T  = float(rec.get("total_time_s", rec["active_time"] + rec["idle_time"]))
        Ba = float(rec["active_time"])
        rec["avg_power_w"] = (rec["E_total_J"] / T) if T > 0 else 0.0
        rec["duty_cycle"]  = (Ba / T) if T > 0 else 0.0
        rec["P_static_W"]  = (rec["E_static_J"] / T) if T > 0 else 0.0
        rec["P_dyn_when_active_W"] = (rec["E_dynamic_J"] / Ba) if Ba > 0 else 0.0
        Uw = float(rec.pop("_U_active_time_weighted", 0.0))
        rec["U_active_avg"] = (Uw / Ba) if Ba > 0 else 0.0
        totals.append(rec)

    # backfill job/task for leading windows – same logic as before
    try:
        by_gpu = {}
        for r in out:
            key = (str(r.get("Cluster","")),
                   str(r.get("Node","")),
                   str(r.get("GPU","")))
            by_gpu.setdefault(key, []).append(r)
        for key, rows_gpu in by_gpu.items():
            rows_gpu.sort(key=lambda r: _safe_float(r.get("start"), 0.0))
            for i in range(len(rows_gpu)-1):
                curr = rows_gpu[i]; nxt = rows_gpu[i+1]
                curr_blank = not (curr.get("trigger_job_id") and curr["trigger_job_id"] != "-")
                nxt_has    = nxt.get("trigger_job_id") and nxt["trigger_job_id"] != "-"
                if curr_blank and nxt_has:
                    s0 = _safe_float(curr.get("end"),   0.0)
                    s1 = _safe_float(nxt.get("start"), 0.0)
                    if s1 >= s0 and (s1 - s0) <= 1e-6:
                        curr["trigger_job_id"]  = str(nxt.get("trigger_job_id"))
                        curr["trigger_task_id"] = str(nxt.get("trigger_task_id"))
    except Exception:
        pass

    return out, totals


def _busy_spans_from_rows_gpu(rows: list):
    spans = defaultdict(list)
    for r in rows or []:
        if r.get("gpu_entry_time") in (None, "") or r.get("gpu_exit_time") in (None, ""):
            continue
        rs = _safe_float(r["gpu_entry_time"], None)
        re = _safe_float(r["gpu_exit_time"],   None)
        if rs is None or re is None or not (math.isfinite(rs) and math.isfinite(re)) or re <= rs:
            continue
        C = str(r.get("Assigned_Cluster",""))
        N = str(r.get("Assigned_Node",""))
        G = str(r.get("Assigned_GPU",""))
        spans[(C,N,G)].append((rs, re))

    for k, ivals in list(spans.items()):
        ivals.sort(key=lambda p: (p[0], p[1]))
        merged = []
        for a, b in ivals:
            if not merged or a > merged[-1][1]:
                merged.append([a, b])
            else:
                merged[-1][1] = max(merged[-1][1], b)
        spans[k] = [(a, b) for a, b in merged]
    return spans

def compute_active_idle_from_logs(rows: list, windows: list) -> list:
    spans = _busy_spans_from_rows_gpu(rows)
    out = []

    for w in (windows or []):
        C = str(w.get("Cluster","")); N = str(w.get("Node","")); G = str(w.get("GPU",""))
        t0 = _safe_float(w.get("start"), 0.0)
        t1 = _safe_float(w.get("end"),   t0)
        if not (math.isfinite(t0) and math.isfinite(t1)) or t1 <= t0:
            continue

        # **FAST PATH**: two-pointer overlap
        active = _sum_overlap(spans.get((C, N, G), []), t0, t1)
        dur  = max(0.0, t1 - t0)
        idle = max(0.0, dur - active)
        util = 0.0 if dur <= 0 else max(0.0, min(1.0, active / dur))

        w2 = dict(w)
        w2["duration"]    = round(dur, 6)
        w2["active_time"] = round(active, 6)
        w2["idle_time"]   = round(idle, 6)
        w2["utilization"] = round(util, 6)
        out.append(w2)

    out.sort(key=lambda r: (r.get("Cluster",""), r.get("Node",""), r.get("GPU",""), _safe_float(r.get("start"), 0.0)))
    return out



# ---- main factory ------------------------------------------------------------
def _power_lookup_from_cfg(cfg: dict):
    """
    Returns:
      p_active_util((C,N,G), f_mhz, U_active) -> W
      p_idle       ((C,N,G), f_mhz)           -> W
    """
    specs_by_type = dict(cfg.get("GPU_Specs") or {})
    type_map      = dict(cfg.get("GPU_Type_Map") or {})
    meta          = dict(cfg.get("GPU_Specs_Meta") or {})
    meta_e_default = float(meta.get("power_exp", 1.0))

    # ---- back-compat aliases for static power
    for _, s in specs_by_type.items():
        if "P_static" not in s and "P_st" in s:
            s["P_static"] = s["P_st"]
        if "P_static_W" not in s and "P_static" in s:
            s["P_static_W"] = s["P_static"]

    # auto-calibrate phi_power if absent
    for gtype, s in specs_by_type.items():
        e_type = float(s.get("power_exp", meta_e_default))
        s["_phi_exp"] = e_type

        if "phi_power" in s:
            continue

        Pst   = float(s.get("P_static_W", float("nan")))
        Pmax  = s.get("P_max_W", None)
        freqs = _collect_freqs_for_type(cfg, gtype)

        if Pst is None or Pmax is None or not freqs:
            raise ValueError(
                f"[power] Missing calibration for GPU type '{gtype}': "
                f"P_static={Pst}, P_max={Pmax}, freqs_found={freqs}. "
                f"Provide phi_power or (P_static, P_max, and DVFS freqs)."
            )

        # if Pmax is None or not freqs or not math.isfinite(Pst):
        #     raise ValueError(
        #         f"[power] Missing calibration for GPU type '{gtype}': "
        #         f"P_static_W={s.get('P_static_W')}, P_max_W={Pmax}, freqs_found={freqs}."
        #     )

        # Be robust if helper returns GHz: convert to MHz if clearly in GHz range
        fmax = max(float(x) for x in freqs)
        if fmax < 50.0:          # heuristic: <50 means likely GHz
            fmax *= 1000.0       # -> MHz
        denom = max(fmax, 1e-9) ** e_type
        s["phi_power"] = max(0.0, (float(Pmax) - Pst) / denom)  # W / MHz^e

    def _spec_for(gkey):
        gtype = type_map.get(gkey)
        if not gtype:
            c, n, g = gkey
            for k in (f"{c}-{n}-{g}", f"{c}/{n}/{g}", f"{c}:{n}:{g}",
                      f"{c},{n},{g}", (str(c), str(n), str(g))):
                gtype = type_map.get(k)
                if gtype:
                    break
        spec = specs_by_type.get(gtype or "", {})
        if not spec:
            raise ValueError(f"[power] Unknown GPU spec for {gkey}. Check GPU_Type_Map keys.")
        return spec

    def p_idle(gkey, f_mhz: float) -> float:
        spec = _spec_for(gkey)
        Pst = float(spec.get("P_static_W", float("nan")))
        if not math.isfinite(Pst) or Pst <= 0:
            raise ValueError(f"[power] Invalid P_static_W for {gkey}: {Pst}")
        return Pst

    def p_active_util(gkey, f_mhz: float, U_active: float) -> float:
        spec = _spec_for(gkey)
        Pst  = p_idle(gkey, f_mhz)
        phi  = float(spec.get("phi_power", 0.0))           # W / MHz^e
        e    = float(spec.get("_phi_exp", meta_e_default)) # per-type exponent
        Pd   = max(0.0, U_active) * phi * (max(0.0, float(f_mhz)) ** e)
        return max(Pst + Pd, Pst)

    return p_active_util, p_idle

def _bounded01(x: float) -> float:
    try:
        return max(0.0, min(1.0, float(x)))
    except Exception:
        return 0.0

def write_dvfs_energy_windows_csv(path, rows):
    hdr = [
        "Cluster","Node","GPU","freq","start","end","duration",
        "active_time","idle_time","utilization","U_active",
        "P_active_W","P_idle_W","E_active_J","E_idle_J",
        "E_static_J","E_total_J",
        "P_window_avg_W","opened_by",
        "trigger_job_id","trigger_task_id",
        "Freq_Mode",
    ]
    write_rows_csv(path, rows or [], hdr)


def write_per_gpu_power_csv(path, rows):
    hdr = [
        "Cluster","Node","GPU",
        "active_time","idle_time","total_time_s",
        "E_active_J","E_idle_J","E_total_J","avg_power_w",
        # # new metadata (order doesn’t matter for the plotter,
        # # but consistent is nice)
        # "Strategy","Freq_Mode","Admission","Lambda_per_s","Seed",
    ]
    _os.makedirs(_os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=hdr)
        w.writeheader()
        for r in rows or []:
            w.writerow({k: r.get(k, "") for k in hdr})


# ---- Per-GPU, Per-Frequency stats + transition matrix ----
def build_per_freq_stats(windows):
    """
    Aggregate per (Cluster,Node,GPU,freq):
      time_s, active_s, idle_s, utilization, energy (if present)
    Also build per-GPU transition counts between consecutive freqs.
    """
    from collections import defaultdict

    per_freq = defaultdict(lambda: {
        "Cluster":"", "Node":"", "GPU":"", "freq":"",
        "time_s":0.0, "active_s":0.0, "idle_s":0.0,
        "utilization":0.0,
        "E_active_J":0.0, "E_idle_J":0.0, "E_total_J":0.0
    })
    # for transition matrix
    trans = defaultdict(lambda: defaultdict(int))   # key=(C,N,G) -> { "f0->f1": count }

    # group windows by GPU and sort by time
    by_gpu = defaultdict(list)
    for w in windows or []:
        C = str(w.get("Cluster","")); N = str(w.get("Node","")); G = str(w.get("GPU",""))
        by_gpu[(C,N,G)].append(w)

    for gkey, seq in by_gpu.items():
        seq = sorted(seq, key=lambda r: float(r.get("start", 0.0)))
        prev_f = None
        for w in seq:
            C,N,G = gkey
            f = str(w.get("freq",""))
            dur  = float(w.get("duration", 0.0))
            act  = float(w.get("active_time", 0.0))
            idle = float(w.get("idle_time", max(0.0, dur - act)))
            util = (act/dur) if dur > 0 else 0.0
            util = max(0.0, min(1.0, util))

            row = per_freq[(C,N,G,f)]
            row.update({"Cluster":C,"Node":N,"GPU":G,"freq":f})
            row["time_s"]    += dur
            row["active_s"]  += act
            row["idle_s"]    += idle
            # energy columns might be absent; add if present
            row["E_active_J"] += float(w.get("E_active_J", 0.0))
            row["E_idle_J"]   += float(w.get("E_idle_J", 0.0))
            row["E_total_J"]  += float(w.get("E_total_J", 0.0))

            if prev_f is not None and prev_f != f:
                trans[gkey][f"{prev_f}->{f}"] += 1
            prev_f = f

    # finalize utilization per (GPU,freq)
    out_rows = []
    for (C,N,G,f), r in per_freq.items():
        t = r["time_s"]
        util = (r["active_s"]/t) if t > 0 else 0.0
        r["utilization"] = max(0.0, min(1.0, util))
        out_rows.append(r)

    # flatten transitions
    trans_rows = []
    for (C,N,G), d in trans.items():
        if not d:
            # still emit zeros so every GPU appears
            trans_rows.append({"Cluster":C,"Node":N,"GPU":G,"transition":"(none)","count":0})
        else:
            for k,v in sorted(d.items()):
                trans_rows.append({"Cluster":C,"Node":N,"GPU":G,"transition":k,"count":int(v)})

    return out_rows, trans_rows

def write_per_freq_csv(path, rows):
    hdr = ["Cluster","Node","GPU","freq","time_s","active_s","idle_s","utilization",
           "E_active_J","E_idle_J","E_total_J"]
    _os.makedirs(_os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=hdr)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k,"") for k in hdr})

def write_violations_csv(path, dropped):
    hdr = [
        "Job_ID","Task_ID","Reason","Predicted_Finish","Deadline",
        "Assigned_Cluster","Assigned_Node","Assigned_GPU",
        "Assigned_Frequency","Service_Frequency",
        "Freq_Mode","Freq_Decision_Source","GPU_Decision_Source",
    ]
    _os.makedirs(_os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=hdr)
        w.writeheader()
        for (job_id, task_id), info in (dropped or {}).items():
            base = {
                "Job_ID": job_id,
                "Task_ID": task_id,
                "Reason": (info or {}).get("reason", ""),
                "Predicted_Finish": (info or {}).get("predicted_finish", ""),
                "Deadline": (info or {}).get("deadline", ""),
            }
            pkt = (info or {}).get("pkt") or {}
            base.update({
                "Assigned_Cluster": pkt.get("Assigned_Cluster",""),
                "Assigned_Node": pkt.get("Assigned_Node",""),
                "Assigned_GPU": pkt.get("Assigned_GPU",""),
                "Assigned_Frequency": pkt.get("Assigned_Frequency",""),
                "Service_Frequency": pkt.get("Service_Frequency",""),
                "Freq_Mode": pkt.get("Freq_Mode",""),
                "Freq_Decision_Source": pkt.get("Freq_Decision_Source",""),
                "GPU_Decision_Source": pkt.get("GPU_Decision_Source",""),
            })
            w.writerow(base)

def write_transitions_csv(path, rows):
    hdr = ["Cluster","Node","GPU","transition","count"]
    _os.makedirs(_os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=hdr)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k,"") for k in hdr})

def attach_tasks_to_windows(rows, windows):
    """
    For diagnostics: for each DVFS window, list tasks that overlap it
    with their per-window overlap seconds. Efficient two-pointer sweep
    per GPU; returns (windows_enriched, flat_overlap_rows).
    """
    # collect task intervals per GPU (not merged; we want per-task rows)
    tasks_by = defaultdict(list)
    for r in rows or []:
        try:
            C = str(r.get("Assigned_Cluster",""))
            N = str(r.get("Assigned_Node",""))
            G = str(r.get("Assigned_GPU",""))
            s = float(r.get("gpu_entry_time"))
            e = float(r.get("gpu_exit_time"))
            if not (math.isfinite(s) and math.isfinite(e)) or e <= s:
                continue
            tasks_by[(C,N,G)].append({
                "Job_ID": r.get("Job_ID",""),
                "Task_ID": r.get("Task_ID",""),
                "start": s, "end": e,
            })
        except Exception:
            pass

    # windows per GPU
    wins_by = defaultdict(list)
    for w in windows or []:
        key = (str(w.get("Cluster","")), str(w.get("Node","")), str(w.get("GPU","")))
        wins_by[key].append(w)

    flat = []
    enriched = []
    for key, wins in wins_by.items():
        wins.sort(key=lambda w: float(w.get("start", 0.0)))
        tasks = tasks_by.get(key, [])
        tasks.sort(key=lambda t: t["start"])

        # two-pointer sweep
        ti = 0
        for w in wins:
            t0 = float(w.get("start", 0.0)); t1 = float(w.get("end", 0.0))
            win_tasks = []

            # advance ti while current task ends before window
            while ti < len(tasks) and tasks[ti]["end"] <= t0:
                ti += 1

            tj = ti
            while tj < len(tasks):
                a0 = tasks[tj]["start"]; a1 = tasks[tj]["end"]
                if a0 >= t1:
                    break  # tasks beyond this window
                ov = _intersect_len(a0, a1, t0, t1)
                if ov > 0:
                    rec = {
                        "Cluster": key[0], "Node": key[1], "GPU": key[2],
                        "freq": str(w.get("freq","")),
                        "window_start": t0, "window_end": t1,
                        "window_duration": t1 - t0,
                        "Job_ID": tasks[tj]["Job_ID"], "Task_ID": tasks[tj]["Task_ID"],
                        "task_start": a0, "task_end": a1,
                        "overlap_s": ov,
                    }
                    win_tasks.append(rec)
                    flat.append(rec)
                tj += 1

            w2 = dict(w)
            w2["tasks_in_window"] = win_tasks
            enriched.append(w2)

    return enriched, flat

def _dbg_dump_gpu_catalog(state: "SimState", gkey: Tuple[str,str,str], *, tag: str="DBG") -> None:
    try:
        cat = getattr(state, "gpu_catalog", None)
        if not cat:
            print(f"[{tag}][CAT] gpu_catalog missing")
            return
        rates    = (cat.rates or {}).get(_norm_gkey(gkey), {}) or {}
        default  = (cat.defaults or {}).get(_norm_gkey(gkey), "")
        gtype    = (getattr(cat, "types", {}) or {}).get(_norm_gkey(gkey), "")
        strat    = str(getattr(state, "strategy", "")).lower()
        fmode    = str(getattr(state, "freq_mode", "adaptive")).lower()
        ng = _norm_gkey(gkey)
        hw = (getattr(state, "gpu_to_freq", {}) or {}).get(ng, "")
        print(f"[{tag}][CAT] gkey={_norm_gkey(gkey)} type={gtype} strategy={strat} freq_mode={fmode}")
        print(f"[{tag}][CAT] default={default} hw={hw}")
        if not rates:
            print(f"[{tag}][CAT] rates=EMPTY")
        else:
            ordered = sorted(((k, rates[k]) for k in rates), key=lambda kv: float(kv[1]))
            print(f"[{tag}][CAT] rates(sorted slow→fast)={ordered}")
    except Exception as e:
        print(f"[{tag}][CAT][ERR] {e}")


# ====================================================
def get_gpu_rate(state: "SimState", task_or_pkt: Dict[str, Any]) -> float:
    gkey = _norm_gkey((
        str(task_or_pkt.get("Assigned_Cluster")),
        str(task_or_pkt.get("Assigned_Node")),
        str(task_or_pkt.get("Assigned_GPU")),
    ))

    cat = getattr(state, "gpu_catalog", None)
    rates = getattr(cat, "rates", {}) or {}
    fmap = rates.get(gkey, {}) or {}
    if not fmap:
        return 0.0

    strat = str(getattr(state, "strategy", "")).lower()
    fm    = str(getattr(state, "freq_mode", "adaptive")).lower()
    prefer_slowest = (("power" in strat) or ("eff" in strat) or ("efficiency" in strat)) and (fm == "adaptive")

    # Prefer actual service freq, then planned, then hw
    cand = (
        str(task_or_pkt.get("Service_Frequency", "") or "").strip()
        or str(task_or_pkt.get("Assigned_Frequency", "") or "").strip()
    )

    if not cand:
        cur_map = getattr(state, "gpu_to_freq", {}) or {}
        cand = str(cur_map.get(gkey, "") or "").strip()

    if not cand:
        defaults = getattr(cat, "defaults", {}) or {}
        cand = str(defaults.get(gkey, "") or "").strip()

    f = _freq_key_match_in_map(fmap, cand) if cand else ""

    if not f:
        # Choose deterministically by *service rate*
        try:
            if prefer_slowest:
                f = min(fmap.keys(), key=lambda k: float(fmap[k]))
            else:
                f = max(fmap.keys(), key=lambda k: float(fmap[k]))
        except Exception:
            return 0.0

    try:
        return float(fmap[f])
    except Exception:
        return 0.0


# ---------- Admission + predictor helpers ----------

def _task_key(pkt):
    if not isinstance(pkt, dict):
        return ("", "")
    jid = pkt.get("Job_ID") or pkt.get("job_id") or ""
    tid = pkt.get("Task_ID") or pkt.get("task_id") or ""
    return (str(jid), str(tid))

def _station_tail(state: SimState, prefix: str, c: str, svc_fn) -> Tuple[int, float]:
    n_ports = int(state.ports_per_cluster.get(str(c), 1))
    best = (1, float("inf"))
    for p in range(1, n_ports+1):
        st = get_station(state, prefix, str(c), p, svc_fn)
        tail = float(st.tail_time)
        if tail < best[1]:
            best = (p, tail)
    return best

def _gpu_tail(state: SimState, c: str, n: str, g: str, svc_fn) -> float:
    name = f"GPU:{c}:{n}:{g}"
    st = state.stations.get(name)
    if st is None:
        st = Station(name=name, service_time_fn=svc_fn, capacity=None)
        state.stations[name] = st
    return float(st.tail_time)


# ------------------------------------------------------------
# Frequency setter + logging
# ------------------------------------------------------------
def _dwell_elapsed(state, gkey, min_ms=5.0):
    if not hasattr(state, "_dvfs_last_change"): state._dvfs_last_change = {}
    last_t = float(state._dvfs_last_change.get(gkey, -1.0))
    now    = float(getattr(state, "t", 0.0))
    return last_t < 0 or (now - last_t) * 1000.0 >= float(min_ms)

def _gkey_tuple_to_str(key_tuple):
    """
    ('C1','N1','G1') -> 'C1-N1-G1'
    """
    return f"{key_tuple[0]}-{key_tuple[1]}-{key_tuple[2]}"

def merge_adjacent_same_freq_windows(windows, eps=1e-6):
    """
    Lightweight safety net: assume windows are already sorted by
    (Cluster, Node, GPU, start), and just do a single linear pass.

    All fields are assumed to be already canonical:
      - Cluster/Node/GPU: strings
      - freq: string
      - start/end/duration: floats
    """
    if not windows:
        return []

    out = []
    prev = dict(windows[0])  # copy first
    prev_gpu  = (prev["Cluster"], prev["Node"], prev["GPU"])
    prev_freq = prev["freq"]
    prev_start = prev["start"]
    prev_end   = prev["end"]

    for w in windows[1:]:
        gpu  = (w["Cluster"], w["Node"], w["GPU"])
        freq = w["freq"]
        start = w["start"]
        end   = w["end"]

        if gpu == prev_gpu and freq == prev_freq and abs(prev_end - start) <= eps:
            # merge into prev
            prev_end = end
            prev["end"] = prev_end
            prev["duration"] = max(0.0, prev_end - prev_start)
        else:
            out.append(prev)
            prev = dict(w)
            prev_gpu  = gpu
            prev_freq = freq
            prev_start = start
            prev_end   = end

    out.append(prev)
    return out

# def _close_dvfs_window(state, key_tuple, end_time):
#     """
#     Finalize the currently-open DVFS window for this GPU and append it to
#     state.dvfs_window_log. Removes it from state.dvfs_active_windows.

#     We *always* merge with the previous window when:
#       - same (Cluster,Node,GPU),
#       - same freq,
#       - contiguous in time.
#     This keeps the log compact and avoids heavy merging passes later.
#     """
#     gkey = _gkey_tuple_to_str(key_tuple)
#     win = (getattr(state, "dvfs_active_windows", {}) or {}).get(gkey)
#     if not win:
#         return

#     start_t = float(win["start"])
#     end_t   = float(end_time)
#     if end_t < start_t:
#         end_t = start_t  # guard

#     if not hasattr(state, "dvfs_window_log"):
#         state.dvfs_window_log = []

#     row = {
#         "Cluster": key_tuple[0],
#         "Node":    key_tuple[1],
#         "GPU":     key_tuple[2],
#         # store freq/start/end as *canonical* types: string + floats
#         "freq":    str(win["freq"]),
#         "start":   start_t,
#         "end":     end_t,
#         "duration": max(0.0, end_t - start_t),
#         "opened_by":        str(win["reason"]),
#         "trigger_job_id":   str(win["trigger_job"]),
#         "trigger_task_id":  str(win["trigger_task"]),
#         "Freq_Mode":        getattr(state, "freq_mode", getattr(state, "Frequency_Mode", "")),
#     }

#     log  = state.dvfs_window_log
#     minw = float(getattr(state, "dvfs_min_window_s", 0.0))

#     # --- Try to merge with previous window whenever possible ---
#     if log:
#         prev = log[-1]
#         same_gpu = (
#             prev["Cluster"], prev["Node"], prev["GPU"]
#         ) == (row["Cluster"], row["Node"], row["GPU"])
#         same_freq = prev.get("freq") == row.get("freq")  # both are strings
#         contiguous = abs(float(prev.get("end", prev["start"])) - row["start"]) <= 1e-9

#         if same_gpu and same_freq and contiguous:
#             # Just extend prev; do not add a new row
#             prev["end"] = row["end"]
#             prev["duration"] = max(0.0, float(prev["end"]) - float(prev["start"]))
#             del state.dvfs_active_windows[gkey]
#             return

#     # --- If we didn't merge, apply min-window policy (optional) ---
#     if minw > 0.0 and row["duration"] < minw and log:
#         prev = log[-1]
#         same_gpu = (
#             prev["Cluster"], prev["Node"], prev["GPU"]
#         ) == (row["Cluster"], row["Node"], row["GPU"])
#         same_freq = prev.get("freq") == row.get("freq")
#         contiguous = abs(float(prev.get("end", prev["start"])) - row["start"]) <= 1e-9
#         if same_gpu and same_freq and contiguous:
#             # merge tiny row into prev
#             prev["end"] = row["end"]
#             prev["duration"] = max(0.0, float(prev["end"]) - float(prev["start"]))
#         # else: drop micro window entirely
#     else:
#         log.append(row)

#     # remove active window
#     del state.dvfs_active_windows[gkey]

def apply_boundary_upshift_only(
    state, gkey, desired_key, *,
    t, fmap,
    trigger_job_id="", trigger_task_id="",
    reason="boundary",
):
    # Must have a desired freq and it must be a real bin
    if not desired_key:
        return False
    if not fmap or str(desired_key) not in fmap:
        return False

    ngkey  = _norm_gkey(gkey)
    hw_raw = (getattr(state, "gpu_to_freq", {}) or {}).get(ngkey, "")
    cur_key = _freq_key_match_in_map(fmap, hw_raw) or (hw_raw or "")

    # No-op
    if str(desired_key) == str(cur_key):
        return False

    # STRICT boundary gate (must be "in_service_job is None")
    if not _gpu_can_change_now(state, gkey, job_id=trigger_job_id):
        return False

    # Service-rate helpers
    def _rate(k: str) -> float:
        try:
            return float(fmap.get(str(k), -1.0))
        except Exception:
            return -1.0

    rd = _rate(desired_key)
    if rd <= 0.0:
        return False  # desired must be usable

    # If current is unknown/unseeded, allow (this is the first real set)
    rc = _rate(cur_key) if cur_key else -1.0
    if cur_key and (rd <= rc):
        return False  # UPSHIFT ONLY when current is known

    # Optional debug hook
    _dbg(state, f"[DVFS-BOUNDARY] t={float(t):.6f} GPU={gkey} cur={cur_key or '-'}->{desired_key} "
                f"in_service={getattr((getattr(state,'stations',{}) or {}).get(f'GPU:{gkey[0]}:{gkey[1]}:{gkey[2]}'), 'in_service_job', None)} "
                f"reason={reason} trig_job={trigger_job_id} trig_task={trigger_task_id}")

    # Apply change (no override lock)
    return safe_set_gpu_freq(
        state, gkey, desired_key,
        reason=reason,
        fmap=fmap,
        when=float(t),
        force_log=True,
        override_lock=False,
        trigger_job_id=str(trigger_job_id),
        trigger_task_id=str(trigger_task_id),
    )


def _assert_dvfs_freq_change(
    state,
    gpu_key,
    old_f,
    new_f,
    *,
    reason="",
):
    """
    Runtime invariant:
    A DVFS window MUST NOT open unless frequency actually changes.

    This should be called right before opening a DVFS window.
    """
    if str(old_f) == str(new_f):
        msg = (
            "[DVFS-INVARIANT VIOLATION]\n"
            f"GPU={gpu_key}\n"
            f"old_f={old_f} new_f={new_f}\n"
            f"reason={reason}\n"
            "A DVFS window was about to open without a frequency change."
        )

        # Hard fail in debug mode
        if getattr(state, "debug_dvfs", False):
            raise AssertionError(msg)

        # Soft fail otherwise (log once)
        _dbg(state, msg)


def _close_dvfs_window(state, key_tuple, end_time):
    """
    Close the currently-open DVFS window for this GPU key_tuple=(C,N,G).
    Uses tuple keys consistently (NO string conversion).
    """
    key3 = _norm_gkey(key_tuple)

    aw = getattr(state, "dvfs_active_windows", None) or {}
    win = aw.get(key3)
    if not win:
        return

    start_t = float(win.get("start", 0.0) or 0.0)
    end_t   = float(end_time or start_t)
    if end_t < start_t:
        end_t = start_t

    if not hasattr(state, "dvfs_window_log") or state.dvfs_window_log is None:
        state.dvfs_window_log = []
    log = state.dvfs_window_log

    fm = str(getattr(state, "freq_mode", getattr(state, "Frequency_Mode", "")) or "")

    row = {
        "Cluster": key3[0],
        "Node":    key3[1],
        "GPU":     key3[2],
        "freq":    str(win.get("freq", "")),
        "start":   start_t,
        "end":     end_t,
        "duration": max(0.0, end_t - start_t),
        "opened_by":       _opened_by_for(str(win.get("reason", "")), fm),
        "trigger_job_id":   str(win.get("trigger_job", "")),
        "trigger_task_id":  str(win.get("trigger_task", "")),
        "Freq_Mode":        fm,
    }

    eps  = float(getattr(state, "dvfs_merge_eps", 1e-6) or 1e-6)
    minw = float(getattr(state, "dvfs_min_window_s", 0.0) or 0.0)

    def same_gpu(a):
        return (a.get("Cluster"), a.get("Node"), a.get("GPU")) == (row["Cluster"], row["Node"], row["GPU"])

    def same_freq(a):
        return str(a.get("freq", "")) == str(row.get("freq", ""))

    def contiguous(a):
        return abs(float(a.get("end", a.get("start", 0.0))) - float(row["start"])) <= eps

    # merge if same gpu+freq and contiguous
    if log and same_gpu(log[-1]) and same_freq(log[-1]) and contiguous(log[-1]):
        log[-1]["end"] = row["end"]
        log[-1]["duration"] = max(0.0, float(log[-1]["end"]) - float(log[-1]["start"]))
    else:
        # min-window: merge into previous only if same gpu+freq and contiguous; otherwise keep
        if minw > 0.0 and row["duration"] < minw and log and same_gpu(log[-1]) and same_freq(log[-1]) and contiguous(log[-1]):
            log[-1]["end"] = row["end"]
            log[-1]["duration"] = max(0.0, float(log[-1]["end"]) - float(log[-1]["start"]))
        else:
            log.append(row)

    # remove active window
    try:
        del aw[key3]
    except KeyError:
        pass
    state.dvfs_active_windows = aw

def safe_set_gpu_freq(
    state,
    key_tuple,
    new_freq,
    reason="dvfs",
    fmap=None,
    when=None,
    force_log=False,
    override_lock=False,
    trigger_job_id="",
    trigger_task_id="",
):
    # --- ensure state dicts exist ---
    if not hasattr(state, "dvfs_active_windows") or state.dvfs_active_windows is None:
        state.dvfs_active_windows = {}          # (C,N,G) -> window dict
    if not hasattr(state, "gpu_to_freq") or state.gpu_to_freq is None:
        state.gpu_to_freq = {}                  # (C,N,G) -> freq key
    if not hasattr(state, "last_freq_change") or state.last_freq_change is None:
        state.last_freq_change = {}             # (C,N,G) -> time

    key3 = _norm_gkey(key_tuple)

    # canonical sim clock
    when = float(_now(state) if when is None else when)

    # --- normalize desired frequency key ---
    newf_raw = str(new_freq or "").strip()
    if fmap and newf_raw:
        try:
            newf = _freq_key_match_in_map(fmap, newf_raw) or newf_raw
        except Exception:
            newf = newf_raw
    else:
        newf = newf_raw

    # If caller passed empty/None, do nothing
    if not newf:
        return False

    # --- source-of-truth current HW key ---
    prev_raw = str((state.gpu_to_freq.get(key3) or "")).strip()
    if fmap and prev_raw:
        try:
            prev_freq = _freq_key_match_in_map(fmap, prev_raw) or prev_raw
        except Exception:
            prev_freq = prev_raw
    else:
        prev_freq = prev_raw

    # --- no-op guard (also ensures gpu_to_freq is at least canonicalized) ---
    if str(prev_freq) == str(newf):
        # keep gpu_to_freq consistent/canonical
        state.gpu_to_freq[key3] = prev_freq or newf
        return False

    # ===========================
    # STRICT boundary-only lock
    # ===========================
    if not override_lock:
        if not _gpu_can_change_now(state, key3, job_id=trigger_job_id):
            return False

    # cooldown guard (override_lock bypasses cooldown)
    last = float(state.last_freq_change.get(key3, -1e30))
    cooldown = float(getattr(state, "dvfs_cooldown_s", 0.0) or 0.0)
    if (when - last) < cooldown and not override_lock:
        # allow upshift during cooldown if fmap says it's faster
        try:
            if fmap and float(fmap.get(newf, -1.0)) > float(fmap.get(prev_freq, -1.0)):
                pass
            else:
                _dbg(
                    state,
                    f"[DVFS] cooldown block GPU={key3} cur={prev_freq} want={newf} "
                    f"dt={when-last:.6f} cd={cooldown}"
                )
                return False
        except Exception:
            _dbg(
                state,
                f"[DVFS] cooldown block GPU={key3} cur={prev_freq} want={newf} "
                f"dt={when-last:.6f} cd={cooldown}"
            )
            return False

    # --- close any open window first (if exists) ---
    prev_win = state.dvfs_active_windows.get(key3)
    if prev_win:
        _close_dvfs_window(state, key3, when)

    # invariant check (skip strict check if prev is unknown/blank)
    if prev_freq:
        _assert_dvfs_freq_change(
            state,
            gpu_key=key3,
            old_f=prev_freq,
            new_f=newf,
            reason=reason,
        )

    # --- open new window and WRITE BACK HW KEY (this fixes hw= blank) ---
    state.dvfs_active_windows[key3] = {
        "freq": newf,
        "start": when,
        "reason": str(reason),
        "trigger_job": str(trigger_job_id),
        "trigger_task": str(trigger_task_id),
        "prev": str(prev_freq),
    }
    state.gpu_to_freq[key3] = newf          # <<< critical for hw=
    state.last_freq_change[key3] = when

    if force_log:
        _dbg(state, f"[DVFS] GPU={key3} {prev_freq or '-'} -> {newf} @t={when:.6f} reason={reason}")

    return True

def set_gpu_freq(
    state: "SimState",
    gpu_key: tuple,
    new_f: str,
    *,
    reason: str,
    origin: str = "",
    src_obj: Optional[Dict[str, Any]] = None,
    fmap: Optional[Dict[str, float]] = None,
    when: Optional[float] = None,
    force_log: bool = False,
    **extras: Any,
):
    # Respect FIXED mode "hard lock"
    if str(getattr(state, "freq_mode", "adaptive")).lower() == "fixed":
        # allow seeding logs, but don't retune live
        if reason in {"locked", "init_seed"}:
            return (False, (getattr(state, "gpu_to_freq", {}) or {}).get(_norm_gkey(gpu_key), "") or "")
    
    # normalize key forms
    (c, n, g) = _norm_gpu_key(gpu_key)
    gpu_key = (c, n, g)
    ng = _norm_gkey(gpu_key)

    ts = float(when if when is not None else getattr(state, "t", 0.0))
    ts = round(ts, 6)

    if fmap is None:
        fmap = (getattr(state.gpu_catalog, "rates", {}) or {}).get(gpu_key, {}) or {}

    cur_f = (getattr(state, "gpu_to_freq", {}) or {}).get(ng, "")
    cur_k = _freq_key_match_in_map(fmap, cur_f) or (cur_f or "")
    new_k = _freq_key_match_in_map(fmap, new_f) or ""

    # --- safety: block optimizer-owned retunes when we're not in optimizer mode ---
    fm = str(getattr(state, "freq_mode", "adaptive")).lower()
    assigner = str(getattr(state, "assigner", "")).lower()
    if fm == "adaptive" and assigner != "optimizer" and str(reason).lower().startswith("optimizer"):
        stack = "".join(traceback.format_stack(limit=20))
        _dbg(state,
             "[TRIP] unexpected set_gpu_freq(reason='optimizer') in ADAPTIVE.\n"
             f"GPU={ng} cur={cur_k} -> new={new_k}\nCALL STACK:\n{stack}")
        return (False, cur_k)

    _dbg(state, f"REQ set_gpu_freq GPU={ng} cur={cur_k} -> new={new_k} reason={reason} at={ts}")

    # -------- FIXED mode: never change HW except initial seed --------
    if fm != "adaptive":
        if reason == "init_seed":
            log_dvfs_change(
                state, ng,
                old_f=cur_k, new_f=(new_k or new_f),
                reason="init_seed",
                trigger_job_id=(src_obj or {}).get("Job_ID", ""),
                trigger_task_id=(src_obj or {}).get("Task_ID", ""),
                when=ts, fmap=fmap, origin=origin, force_log=True,
            )

        # no actual HW update, no repeated spam
        return (False, cur_k or (new_k or new_f))

    # # -------- ADAPTIVE path --------
    # if new_k == "":
    #     if force_log:
    #         print(f"[DVFS][WARN] {ng} frequency '{new_f}' not in service_rates; "
    #               f"available={sorted(fmap.keys())}")
    #     return (False, cur_k)

    # # Skip exact no-ops unless forced
    # if not force_log and new_k == cur_k:
    #     _dbg(state, f"NO-OP set_gpu_freq GPU={ng} new==cur=={new_k} reason={reason}")
    #     return (False, cur_k)

    # # Apply update
    # state.gpu_to_freq[ng] = new_k

    # # Stamp "last change" on both state-global and station-local so hysteresis works
    # now_t = float(getattr(state, "t", ts))
    # if not hasattr(state, "_dvfs_last_change"):
    #     state._dvfs_last_change = {}
    # state._dvfs_last_change[ng] = now_t

    # st = (getattr(state, "stations", {}) or {}).get(f"GPU:{c}:{n}:{g}")
    # if st is not None:
    #     st.last_dvfs_change = now_t

    # # Log event
    # log_dvfs_change(
    #     state, ng, cur_k, new_k, reason,
    #     trigger_job_id=(src_obj or {}).get("Job_ID", ""),
    #     trigger_task_id=(src_obj or {}).get("Task_ID", ""),
    #     when=ts, fmap=fmap, origin=origin, force_log=force_log, **extras
    # )

    # -------- ADAPTIVE path --------
    if new_k == "":
        if force_log:
            print(f"[DVFS][WARN] {ng} frequency '{new_f}' not in service_rates; "
                  f"available={sorted(fmap.keys())}")
        return (False, cur_k)

    # Skip exact no-ops unless forced
    if not force_log and new_k == cur_k:
        _dbg(state, f"NO-OP set_gpu_freq GPU={ng} new==cur=={new_k} reason={reason}")
        return (False, cur_k)

    # >>>  use safe_set_gpu_freq so windows are correct
    changed = safe_set_gpu_freq(
        state,
        ng,                          # tuple key (C,N,G)
        new_k,
        reason=reason,
        fmap=fmap,
        when=ts,
        force_log=force_log,
        override_lock=extras.get("override_lock", False),
        trigger_job_id=(src_obj or {}).get("Job_ID", ""),
        trigger_task_id=(src_obj or {}).get("Task_ID", ""),
    )
    if not changed:
        return (False, cur_k)

    # Stamp last-change for hysteresis
    now_t = float(getattr(state, "t", ts))
    if not hasattr(state, "_dvfs_last_change"):
        state._dvfs_last_change = {}
    state._dvfs_last_change[ng] = now_t

    st = (getattr(state, "stations", {}) or {}).get(f"GPU:{c}:{n}:{g}")
    if st is not None:
        st.last_dvfs_change = now_t

    # Log event (one per real change)
    log_dvfs_change(
        state, ng, cur_k, new_k, reason,
        trigger_job_id=(src_obj or {}).get("Job_ID", ""),
        trigger_task_id=(src_obj or {}).get("Task_ID", ""),
        when=ts, fmap=fmap, origin=origin, force_log=force_log, **extras
    )

    return (True, new_k)


# ---------------------------------------------
# Does this job still have outstanding work?
# ---------------------------------------------
# Clear pins only when the job has no more work
def _job_has_outstanding_work(state, job_id: str) -> bool:
    jid = str(job_id)

    # Any packets still running on any GPU?
    active_jobs = getattr(state, "active_jobs", None) or {}
    for jobs in active_jobs.values():
        if jid in jobs:
            return True

    # Any packets queued at any GPU?
    stations = getattr(state, "stations", None) or {}
    for st in stations.values():
        name = str(getattr(st, "name", ""))
        if not name.startswith("GPU:"):
            continue
        q = getattr(st, "q", None) or []
        for t in q:
            try:
                if str(t.get("Job_ID", "")) == jid:
                    return True
            except AttributeError:
                # if q holds something non-dict, just ignore that entry
                continue

    return False

# ---------------------------------------------
# Clear per-job records if job is truly done
# ---------------------------------------------
def _clear_job_records_if_done(state, job_id: str) -> None:
    """
    Free the per-job GPU pin and lambda_bg load *only* when the job has
    no running/queued work.

    NOTE: least-load strategies don't use per-job pins or lambda_bg, so we
    skip all this work for them to avoid O(N) scans on every completion.
    """
    jid = str(job_id)

    # ---- fast exit for least-load / non-optimizer schedulers -------------
    strategy = str(getattr(state, "strategy", "") or "").lower()
    if "least-load" in strategy or "leastload" in strategy:
        # No job pins or lambda_bg used in least-load: nothing to clear.
        return

    # ---- check if this job still has any work in the system --------------
    if _job_has_outstanding_work(state, jid):
        return  # still running somewhere; don't clear pins yet

    # ---- safe to free pins & freq plans for optimizer-based schedulers ---
    for attr in ("job_assignment", "job_freq_plan", "job_gpu_src", "x_job"):
        try:
            d = getattr(state, attr, None)
            if isinstance(d, dict):
                d.pop(jid, None)
        except Exception:
            pass

    # # ---- decrement this job's contribution to lambda_bg, if tracked ------
    # _lambda_bg_job_finish(state, jid)

def _finalize_provenance(state: "SimState", pkt: dict, job_id: str):
    """
    Ensure Freq_Mode, Freq_Decision_Source, GPU_Decision_Source, Service_Frequency
    are always populated for this packet.
    """
    # pull plan if present
    plan = state.job_freq_plan.get(job_id)
    plan_src, plan_freq = "", ""
    if plan:
        if isinstance(plan, tuple):
            plan_freq = str(plan[1]) if len(plan) > 1 else ""
            plan_src  = str(plan[2]) if len(plan) > 2 else ""
        elif isinstance(plan, dict):
            plan_freq = str(plan.get("freq","") or plan.get("Frequency",""))
            plan_src  = str(plan.get("src","")  or plan.get("source",""))

    # choose final freq source
    raw_src     = (pkt.get("Freq_Decision_Source") or "").strip()
    mode_fixed  = str(getattr(state, "freq_mode", "adaptive")).lower() == "fixed"
    prev_reason = str(getattr(state, "_last_freq_reason", "")).strip()

    final_src = (
        raw_src or plan_src or ("locked" if mode_fixed else "") or
        (prev_reason if prev_reason else "dvfs-helper")
    )

    pkt["Freq_Decision_Source"] = final_src
    pkt["Freq_Mode"] = "fixed" if final_src == "locked" or mode_fixed else "adaptive"

    # keep Service_Frequency in sync
    svc = pkt.get("Service_Frequency") or pkt.get("Assigned_Frequency") or plan_freq
    pkt["Service_Frequency"] = svc or ""

    # GPU decision source precedence: existing > remembered > infer from final_src
    gpu_src = (pkt.get("GPU_Decision_Source") or
               state.job_gpu_src.get(job_id, "") or "")
    if not gpu_src:
        if final_src in ("optimizer", "random", "store_restore",
                         "predicted-finish", "balanced", "fastest"):
            gpu_src = final_src
        elif final_src.startswith("min_deadline"):
            gpu_src = "least-load"
    if gpu_src:
        pkt["GPU_Decision_Source"] = gpu_src

def _norm_job_id(pkt) -> str:
        # Make sure Job_ID is consistent across pregen/log paths
        j = str(pkt.get("Job_ID"))
        return j  # or return j.split("AR_")[-1] if that matches data

# =============================
# Optimizer Facade (LIVE-ONLY)
# =============================

def _import_opt_module(path_or_name: Optional[str]):
    if not path_or_name:
        return None
    if _os.path.exists(path_or_name) and path_or_name.endswith(".py"):
        spec = importlib.util.spec_from_file_location("live_opt_dyn", path_or_name)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore
        return mod
    return __import__(path_or_name)

def _median(nums):
    s = sorted([float(v) for v in nums if v is not None])
    if not s: return 0.0
    n = len(s)
    mid = n // 2
    return (s[mid-1] + s[mid]) / 2.0 if n % 2 == 0 else s[mid]

def _gpu_can_change_now(state, key3, job_id=None) -> bool:
    try:
        key3 = _norm_gkey(key3)  # ALWAYS force ('C','N','G') tuple
    except Exception:
        _dbg(state, f"LOCK bad key: GPU={key3} allow=NO")
        return False

    gname = f"GPU:{key3[0]}:{key3[1]}:{key3[2]}"
    st = (getattr(state, "stations", {}) or {}).get(gname)
    if not st:
        _dbg(state, f"LOCK miss (no station): GPU={key3} allow=YES")
        return True

    cur = getattr(st, "in_service_job", None)
    ok = (cur is None)

    if (not ok) or getattr(state, "debug_dvfs", False):
        _dbg(
            state,
            f"LOCK boundary-only: GPU={key3} in_service={cur} ask={job_id} "
            f"allow={'YES' if ok else 'NO'}"
        )
    return ok


def _unit_from_cp_units(s: str) -> str:
        s = (s or "").lower()
        return "ghz" if "per_ghz" in s else "mhz"

def _phi_freq_unit(s: str) -> str:
    s = (s or "").lower()
    if "per_" in s: s = s.split("per_", 1)[1]
    s = s.rstrip("0123456789")  # drop exponent digits
    return "ghz" if "ghz" in s else "mhz"

class OptimizerFacade:
    def __init__(
        self,
        catalog: GPUCatalog,
        objective: str = "latency",
        # Old single-module param still supported:
        opt_module: Optional[str] = None,
        # New explicit per-objective params (optional):
        latency_module: Optional[str] = None,
        power_module: Optional[str] = None,
        efficiency_module: Optional[str] = None,
        # Or pass a dict mapping objective -> module
        modules: Optional[Dict[str, Optional[str]]] = None,
        store_path: Optional[str] = None,
        store: Optional["AssignmentStore"] = None,
    ):
        self.catalog = catalog
        self.objective = (objective or "latency").lower()  

        # Load incremental store (unchanged)
        self.store = store or (AssignmentStore.from_optimizer_json(store_path) if store_path else AssignmentStore())
        self.last_solution = None
        self.config = getattr(self.store, "cfg", None) or {}

        # ---- Load modules per objective ----
        self._modules: Dict[str, Any] = {}

        def _maybe_load(name: Optional[str]):
            return _import_opt_module(name) if name else None

        if modules:
            for obj, modname in modules.items():
                m = _maybe_load(modname)
                if m:
                    self._modules[obj.lower()] = m
        else:
            # Backward compatible:
            # - if only opt_module is given, use it for all objectives
            # - otherwise use the explicit per-objective params when provided
            lat_mod = _maybe_load(latency_module) or _maybe_load(opt_module)
            pow_mod = _maybe_load(power_module) or _maybe_load(opt_module)
            eff_mod = _maybe_load(efficiency_module) or _maybe_load(opt_module)

            if lat_mod: self._modules["latency"] = lat_mod
            if pow_mod: self._modules["power"] = pow_mod
            if eff_mod: self._modules["efficiency"] = eff_mod
        
        self._opt_warmed_local = False


    # Helper: pick module by objective (with safe fallbacks)
    def _pick_module(self, objective: Optional[str]) -> Optional[Any]:
        obj = (objective or self.objective or "latency").lower()
        mod = self._modules.get(obj)
        if mod:
            return mod
        # Fallbacks if a dedicated module wasn't provided
        return self._modules.get("latency") or self._modules.get("power") or self._modules.get("efficiency")
    
    # warmup
    def warmup(self, state=None) -> None:
        if getattr(self, "_opt_warmed_local", False):
            return
        if state is not None and getattr(state, "_opt_warmed", False):
            return
        try:
            # Use the same knobs as decide() to keep the static signature identical
            snap = {
                "now": 0.0,
                "use_arrival_rates": True,
                "gpus": {
                    "Cw-Nw-Gw": {
                        "gid": "Cw-Nw-Gw",
                        "key3": ["Cw", "Nw", "Gw"],
                        "rates": {"1500": 1e6},
                        "tail_s": 0.0,
                        "queued": 0,
                        "freq": "1500",
                    }
                },
                "R0_RC_Links": {"Cw": {"ul_rate_kBps": 1e6, "dl_rate_kBps": 1e6}},
                "links": {"Cw": {"ul_rate_kBps": 1e6, "dl_rate_kBps": 1e6}},  # if solver expects "links"
                "pinned_assignments": {},
                "pinned_frequencies": {},
                "freq_mode": "adaptive",
                "solver": {"mip_gap": 0.02, "time_limit_s": 0.3},
            }

            dummy = [{
                "Task_ID": "WARMUP_T0",
                "Job_ID": "WARMUP",
                "Workload_FLOPs": 0.0,         
                "Task_Deadline": 1e9,          
                "UL_Total_kB": 0,
                "DL_Total_kB": 0,
                "Lambda": 1.0,
            }]
            
            mod = self._pick_module(getattr(state, "objective", None) if state is not None else self.objective)
            if not mod:
                return
            if hasattr(mod, "solve_incremental"):
                mod.solve_incremental(snap, dummy)
            elif hasattr(mod, "solve"):
                mod.solve(snap, dummy, objective=str(getattr(state, "objective", self.objective) or "latency").lower())

        except Exception:
            pass
        finally:
            self._opt_warmed_local = True
            if state is not None:
                state._opt_warmed = True


    def _snapshot(self, state: "SimState") -> Dict[str, Any]:
        # ---- config / cluster delay map (NEW) ------------------------------
        cfg = getattr(self, "config", {}) or {}

        raw_cd = (cfg.get("Cluster_Delay_Base_s") or {})
        cluster_delay_map: Dict[str, float] = {}
        for cid, val in raw_cd.items():
            try:
                cluster_delay_map[str(cid)] = float(val)
            except Exception:
                continue

        # ---- canonical "now" ----
        now = float(_now(state))

        # ---- per-GPU info (JSON-safe keys) ----
        # gpus keyed by gid_str to avoid tuple-key issues in json/deepcopy
        gpus: Dict[str, Dict[str, Any]] = {}

        gpu_freq_now = getattr(state, "gpu_to_freq", {}) or {}
        pending_service_time = getattr(state, "pending_service_time", {}) or {}
        pending_tasks = getattr(state, "pending_tasks", {}) or {}

        catalog = getattr(state, "gpu_catalog", None)
        rates_all = getattr(catalog, "rates", {}) or {}
        types_all = getattr(catalog, "types", {}) or {}
        # specs_all = getattr(catalog, "power_specs", {}) or {}
        gpu_specs_all = (cfg.get("GPU_Specs") or {})
        stations = getattr(state, "stations", {}) or {}

        for key3, fmap in (rates_all.items() if isinstance(rates_all, dict) else []):
            try:
                c, n, g = map(str, key3)
            except Exception:
                continue

            gid_str = f"{c}-{n}-{g}"
            st = stations.get(f"GPU:{c}:{n}:{g}")

            gtype = types_all.get((c, n, g))
            spec = (gpu_specs_all.get(gtype, {}) if gtype else {}) or {}

            # --- derive phi_power if missing (preserve old behavior) ---
            if "phi_power" not in spec:
                try:
                    e = float(spec.get("phi_power_exp", 1.0))
                    Pst = float(spec.get("P_st", spec.get("P_static_W", 0.0)))
                    Pmax = float(spec.get("P_max_W", spec.get("P_max", 0.0)))
                    # fmax from spec freqs else from fmap keys
                    fmax = max([float(x) for x in (spec.get("freqs", []) or [])] or
                            [float(k) for k in (fmap or {}).keys()])
                    if Pmax > 0 and fmax > 0:
                        spec["phi_power"] = max(0.0, (Pmax - Pst) / (fmax ** e))
                        spec["_phi_exp"] = e
                except Exception:
                    pass

            def _f(d, k, default=0.0):
                try:
                    return float(d.get(k, default))
                except Exception:
                    return float(default)

            P_idle_W  = float(spec.get("P_st", spec.get("P_static_W", 60.0)))
            P_max_W   = _f(spec, "P_max_W", 300.0)
            C_p       = _f(spec, "C_p", 0.0)
            power_exp = _f(spec, "power_exp", 1.0)

            # normalize valid service rates
            rates_norm: Dict[str, float] = {}
            for fk, rv in ((fmap or {}).items() if isinstance(fmap, dict) else []):
                try:
                    r = float(rv)
                except Exception:
                    continue
                if r > 0.0:
                    rates_norm[str(fk)] = r

            if not rates_norm:
                # skip GPUs with no valid rate table
                continue

            # --- real queue stats ---
            tail_real = 0.0
            queued_real = 0
            if st is not None:
                tail_real = float(getattr(st, "tail_time", getattr(st, "queue_time", 0.0)) or 0.0)
                if getattr(st, "qlen", None) is not None:
                    try:
                        queued_real = int(st.qlen)
                    except Exception:
                        queued_real = 0
                else:
                    q_obj = getattr(st, "q", [])
                    try:
                        queued_real = int(len(q_obj))
                    except Exception:
                        queued_real = 0

            # -------- include not-yet-enqueued reservations --------
            # pending_t  = extra seconds we already promised to this GPU
            # pending_ct = how many tasks we've promised but not queued yet
            
            # --- pending reservations: support tuple-key and gid_str ---
            key_tuple = (c, n, g)
            key_norm  = _norm_gkey(key_tuple)

            pending_t = 0.0
            try:
                v = pending_service_time.get(key_tuple, None)
                if v is None: v = pending_service_time.get(key_norm, None)
                if v is None: v = pending_service_time.get(gid_str, None)
                pending_t = float(v or 0.0)
            except Exception:
                pending_t = 0.0

            pending_ct = 0
            try:
                v = pending_tasks.get(key_tuple, None)
                if v is None: v = pending_tasks.get(key_norm, None)
                if v is None: v = pending_tasks.get(gid_str, None)
                pending_ct = int(v or 0)
            except Exception:
                pending_ct = 0

            # try:
            #     pending_t = float(pending_service_time.get(key_tuple, 0.0) or 0.0)
            # except Exception:
            #     pending_t = 0.0
            # try:
            #     pending_ct = int(pending_tasks.get(key_tuple, 0) or 0)
            # except Exception:
            #     pending_ct = 0

            tail_total = max(0.0, tail_real + pending_t)
            queued_total = queued_real + pending_ct

            gid_str = f"{c}-{n}-{g}"

            # flop backlog
            flop_bg = 0.0
            try:
                fb = getattr(state, "flop_backlog", {}) or {}
                flop_bg = float(fb.get(key_norm, fb.get(key_tuple, 0.0)) or 0.0)
            except Exception:
                flop_bg = 0.0

            base_delay = float(cluster_delay_map.get(str(c), 0.0))

            # current freq (read from gpu_to_freq; canonical keys are tuples)
            cur_freq = ""
            try:
                cur_freq = str(
                    gpu_freq_now.get(key_norm, None) or
                    gpu_freq_now.get(key_tuple, None) or
                    gpu_freq_now.get(gid_str, "") or
                    ""
                )
            except Exception:
                cur_freq = ""

            gpus[gid_str] = {
                "gid": gid_str,
                "key3": [c, n, g],             # JSON-safe
                "cluster": c,
                "node": n,
                "gpu": g,

                "tail_s": tail_total,
                "tail": tail_total,
                "queued": queued_total,
                "flop_backlog": flop_bg,

                "cluster_base_delay_s": base_delay,

                "tail_real": tail_real,
                "queued_real": queued_real,
                "pending_tail_s": pending_t,
                "pending_tasks_ct": pending_ct,

                "freq": cur_freq,
                "rates": rates_norm,

                "P_static_W": P_idle_W,
                "P_max_W": P_max_W,
                "C_p": C_p,
                "power_exp": power_exp,
                "phi_power": float(spec.get("phi_power", 0.0) or 0.0),
                "phi_power_exp": float(spec.get("_phi_exp", 1.0) or 1.0),
            }

        if len(gpus) < 2:
            print(f"[WARN] Optimizer snapshot has only {len(gpus)} GPU(s) with valid rate tables.")

        
        # ---- link caps ---------------------------------------------------
        links = _build_cluster_link_caps(cfg)
        

        # ---- horizon from deadlines (+ pad) ---------------------------------------
        now = float(_now(state))
        now_for_slack = now
        slacks = []
        rows_by_task = getattr(state, "rows_by_task", {}) or {}
        if isinstance(rows_by_task, dict):
            for (_job, _task), rows in rows_by_task.items():
                for r in (rows or []):
                    td = r.get("Task_Deadline")
                    if td not in ("", None):
                        try:
                            slacks.append(max(0.0, float(td) - now_for_slack))
                        except Exception:
                            pass

        opt_defs = (cfg.get("Optimizer_Defaults") or {})
        optimizer_power      = cfg.get("optimizer_power", {}) or {}
        optimizer_latency    = cfg.get("optimizer_latency", {}) or {}
        optimizer_efficiency = cfg.get("optimizer_efficiency", {}) or {}

        ts = (cfg.get("Task_Settings") or {})
        fps_present = (ts.get("SERVICE_ARRIVAL_RATE_fps", 30) not in ("", 0, None))

        # ---- fixed choices from state/store ---------------------------------------
        def _base_tid_safe(tid: str) -> str:
            try:
                return base_tid(tid)
            except Exception:
                return str(tid)

        mode = str(getattr(state, "freq_mode", "adaptive")).lower()

        # ---- pinned_assignments (fixed only) ----
        pinned_assignments: Dict[str, Dict[str, str]] = {}
        if mode == "fixed":
            xj = getattr(state, "x_j", {}) or {}
            if isinstance(xj, dict):
                for k, v in xj.items():
                    if not (isinstance(k, (tuple, list)) and len(k) >= 2):
                        continue
                    if not (isinstance(v, (tuple, list)) and len(v) == 3):
                        continue
                    tid = _base_tid_safe(str(k[1]))
                    c, n, g = map(str, v)
                    pinned_assignments.setdefault(tid, {"Cluster": c, "Node": n, "GPU": g})

            if getattr(self, "store", None):
                for tidb, rec in (self.store.assignments or {}).items():
                    pinned_assignments.setdefault(str(tidb), {
                        "Cluster": str(rec.get("Cluster", "")),
                        "Node":    str(rec.get("Node", "")),
                        "GPU":     str(rec.get("GPU", "")),
                    })

        # ---- pinned_frequencies (fixed only) ----
        pinned_frequencies: Dict[str, str] = {}
        if mode == "fixed":
            for key3, fmap in (rates_all.items() if isinstance(rates_all, dict) else []):
                if not isinstance(fmap, dict) or not fmap:
                    continue
                try:
                    fastest = max(
                        (fk for fk, rv in fmap.items() if float(rv) > 0.0),
                        key=lambda fk: float(fmap[fk])
                    )
                except Exception:
                    def _k2f(k):
                        try: return float(str(k))
                        except Exception: return -1.0
                    fastest = max(fmap, key=_k2f)

                c, n, g = map(str, key3)
                pinned_frequencies[f"{c}-{n}-{g}"] = str(fastest)

        if getattr(state, "debug_snapshot", False):
            print("[DEBUG _snap] pinned_frequencies =", pinned_frequencies)

        # # Assignments:
        # #  - ADAPTIVE: leave empty; decide() will inject just the job's pin for the current base task.
        # #  - FIXED:    it's okay to seed from state.x_j (and optionally the store).
        # if mode == "fixed":
        #     pinned_assignments = {
        #         _base_tid_safe(tid): {"Cluster": c, "Node": n, "GPU": g}
        #         for (_, tid), (c, n, g) in (getattr(state, "x_j", {}) or {}).items()
        #     }
        #     if getattr(self, "store", None):
        #         for tidb, rec in (self.store.assignments or {}).items():
        #             pinned_assignments.setdefault(tidb, {
        #                 "Cluster": rec.get("Cluster", ""),
        #                 "Node":    rec.get("Node",    ""),
        #                 "GPU":     rec.get("GPU",     "")
        #             })
        # else:
        #     # ADAPTIVE: do not pass any assignment pins from mirrors or store
        #     pinned_assignments = {}

        # # Frequencies:
        # #  - FIXED: pin fastest step per GPU
        # #  - ADAPTIVE: keep empty so solver is free
        # pinned_frequencies: Dict[str, str] = {}
        # if mode == "fixed":
        #     for key3, fmap in (state.gpu_catalog.rates or {}).items():
        #         if not fmap:
        #             continue
        #         # fastest by achieved service rate (robust if keys are not numeric)
        #         try:
        #             # choose among positive-rate steps only, if present
        #             fastest = max(
        #                 (fk for fk, rv in fmap.items() if float(rv) > 0.0),
        #                 key=lambda fk: float(fmap[fk])
        #             )
        #         except Exception:
        #             # fallback: compare by numeric freq key
        #             def _k2f(k):
        #                 try:
        #                     return float(str(k))
        #                 except Exception:
        #                     return -1.0
        #             fastest = max(fmap, key=_k2f)

        #         c, n, g = map(str, key3)
        #         pinned_frequencies[f"{c}-{n}-{g}"] = str(fastest)               
        # else:
        #     pinned_frequencies = {}
        # print("[DEBUG _snap] pinned_frequencies =", pinned_frequencies)

        # # # ---------------- Objective & freq penalty ----------------
        # # if mode == "fixed":
        # #     objective = "latency_fixed"
        # #     freq_penalty_weight = 0.0          # pure min-latency
        # # else:
        # #     objective = "latency_adaptive"
        # #     # tiny bias against high freq; config can override
        # #     freq_penalty_weight = float(
        # #         optimizer_latency.get("freq_penalty_weight", 0.0)
        # #     )


        # ---- meta -----------------------------------------------------------
        meta_cfg = (cfg.get("GPU_Specs_Meta") or {})
        meta = {
            "freq_units":       str(meta_cfg.get("freq_units", "MHz")),
            "cp_units":         str(meta_cfg.get("cp_units", "FLOPs_per_s_per_MHz")),
            "cp_freq_unit":     _unit_from_cp_units(meta_cfg.get("cp_units", "FLOPs_per_s_per_MHz")),
            "phi_power_units":  str(meta_cfg.get("phi_power_units", "W_per_MHz")),
            "phi_freq_unit":    _phi_freq_unit(meta_cfg.get("phi_power_units", "W_per_MHz")),
            "phi_power_exp":    float(meta_cfg.get("phi_power_exp", 1.0)),
            "rate_units":       str(meta_cfg.get("rate_units", "FLOPs_per_s")),
            "scale_gflops":     1e9,
        }

        theta_pw = float(opt_defs.get("theta_penalty_weight", 1e4))
        eff_mode        = str(opt_defs.get("eff_mode", "weighted")).lower()
        eff_beta        = float(opt_defs.get("eff_tradeoff_beta", 1e-8))
        eff_theta_init  = float(opt_defs.get("eff_theta_init", 1e-6))
        eff_max_iter    = int(opt_defs.get("eff_max_iter", 5))
        eff_tol         = float(opt_defs.get("eff_tol", 1e-4))
        eff_balance_w   = float(opt_defs.get("eff_balance_weight", 0.0))
        eff_lb_w        = float(opt_defs.get("eff_load_balance_weight", eff_balance_w))

        snapshot_defaults = {
            "cluster_share_cap":       float(opt_defs.get("cluster_share_cap", 1.0)),
            "min_active_clusters":     int(opt_defs.get("min_active_clusters", 1)),
            "use_arrival_rates":       bool(opt_defs.get("use_arrival_rates", fps_present)),
            "share_penalty_weight":    float(opt_defs.get("share_penalty_weight", 0.0)),
            "min_gpu_service_time_s":  float(opt_defs.get("min_gpu_service_time_s", 0.0)),
            "link_cap_penalty_weight": float(opt_defs.get("link_cap_penalty_weight",
                                            opt_defs.get("link_penalty_weight", 1e5))),
            "theta_penalty_weight":    theta_pw,
            "eff_mode":                eff_mode,
            "eff_tradeoff_beta":       eff_beta,
            "eff_theta_init":          eff_theta_init,
            "eff_max_iter":            eff_max_iter,
            "eff_tol":                 eff_tol,
            "eff_balance_weight":      eff_balance_w,
            "eff_load_balance_weight": eff_lb_w,
        }

        gpu_cap_penalty_w  = float(opt_defs.get("gpu_cap_penalty_weight", opt_defs.get("link_penalty_weight", 1e-2)))
        link_cap_penalty_w = float(opt_defs.get("link_cap_penalty_weight", opt_defs.get("link_penalty_weight", 1e5)))

        min_freq_global = float(opt_defs.get("min_freq_global", 1.0))
        min_freq_map: Dict[str, float] = {}
        for k, v in (opt_defs.get("min_freq_map", {}) or {}).items():
            try:
                if isinstance(k, (list, tuple)) and len(k) == 3:
                    c, n, g = map(str, k)
                else:
                    parts = str(k).split("-")
                    c, n, g = map(str, parts[:3])
                min_freq_map[f"{c}-{n}-{g}"] = float(v)
            except Exception:
                pass

        # ---- Theta_Min ----
        theta_cfg = (cfg.get("Theta_Min") or {})
        theta_min_per_gpu: Dict[str, float] = {}
        for gid, ginfo in gpus.items():
            theta_min_per_gpu[gid] = 0.0

        for k, v in (theta_cfg.get("per_gpu") or {}).items():
            parts = str(k).split("-")
            if len(parts) == 3:
                theta_min_per_gpu[f"{parts[0]}-{parts[1]}-{parts[2]}"] = _safe_float(v, 0.0)

        theta_min_per_cluster: Dict[str, float] = {}
        for c2, v in (theta_cfg.get("per_cluster") or {}).items():
            theta_min_per_cluster[str(c2)] = _safe_float(v, 0.0)

        theta_min_global = _safe_float(theta_cfg.get("global"), None)

        # ---- load balancing history ----
        hist = getattr(state, "gpu_jobs_served", {}) or {}
        gpu_jobs_served = {str(k): int(v) for k, v in hist.items() if v not in (None, "")}

        # # lambda_bg
        # lb_raw = getattr(state, "lambda_bg", {}) or {}
        # lambda_bg = {str(k): float(v) for k, v in lb_raw.items() if v not in (None, "")}

        # # utilization_guess from lambda_bg
        # utilization_guess = {}
        # for (c, n, g), ginfo in gpus.items():
        #     gid = ginfo.get("gid", f"{c}-{n}-{g}")
        #     lam_bg = float(lambda_bg.get(gid, 0.0) or 0.0)
        #     if lam_bg <= 0.0:
        #         continue
        #     rates_norm = ginfo.get("rates") or {}
        #     if not rates_norm:
        #         continue
        #     curr_freq = ginfo.get("freq") or ""
        #     curr_freq = str(curr_freq) if curr_freq is not None else ""
        #     if curr_freq and curr_freq in rates_norm:
        #         cap_rate = float(rates_norm[curr_freq])
        #     else:
        #         cap_rate = max(float(v) for v in rates_norm.values())
        #     if cap_rate <= 0.0:
        #         continue
        #     rho_est = lam_bg / cap_rate
        #     rho_est = max(0.0, min(1.0, rho_est))
        #     utilization_guess[gid] = rho_est

        # ---------------- Model B: lambda_bg is derived from current backlog ----------------
        horizon_s = float(getattr(state, "util_tau_s", 0.5) or 0.5)
        horizon_s = max(1e-6, horizon_s)
        lambda_bg = {}   # do NOT compute here (decide() overwrites before solve)

        # ---------------- util_guess should come from util EWMA (not from lambda_bg) ----------------
        util_raw = getattr(state, "util_ewma", {}) or {}        # gid -> 0..1
        utilization_guess = {str(k): float(v) for k, v in util_raw.items() if v not in (None, "")}

        freq_map_str = {
            f"{c}-{n}-{g}": str(f)
            for (c, n, g), f in (getattr(state, "gpu_to_freq", {}) or {}).items()
            if f not in ("", None)
        }        
        prev_assignments = {}
        prev_frequencies = {}
        if getattr(self, "store", None):
            for tb, rec in (self.store.assignments or {}).items():
                if isinstance(rec, dict) and {"Cluster","Node","GPU"} <= set(rec.keys()):
                    prev_assignments[str(tb)] = {
                        "Cluster": str(rec["Cluster"]),
                        "Node":    str(rec["Node"]),
                        "GPU":     str(rec["GPU"]),
                    }
            for gid, f in (self.store.frequencies or {}).items():
                prev_frequencies[str(gid)] = str(f)

        # --- assigned: stringify (job,task) -> "job|task" for JSON safety ---
        assigned_raw = getattr(state, "x_j", {}) or {}
        assigned_str: Dict[str, Dict[str, str]] = {}

        if isinstance(assigned_raw, dict):
            for k, v in assigned_raw.items():
                # k should be (job_id, tid)
                if isinstance(k, (tuple, list)) and len(k) >= 2:
                    job_id = str(k[0])
                    tid    = str(k[1])
                    key    = f"{job_id}|{tid}"
                else:
                    key = str(k)

                # v should be (c,n,g)
                if isinstance(v, (tuple, list)) and len(v) == 3:
                    c, n, g = map(str, v)
                    assigned_str[key] = {"Cluster": c, "Node": n, "GPU": g}
        return {
            "now": now,
            "gpus": gpus,
            "assigned": assigned_str,
            "freq_map": freq_map_str,
            "links": links,
            "R0_RC_Links": links,
            "cluster_base_delay_s": cluster_delay_map,
            "pinned_assignments": pinned_assignments,
            "pinned_frequencies": pinned_frequencies,
            "meta": meta,
            "min_freq_global": min_freq_global,
            "min_freq_map": min_freq_map,
            "link_cap_penalty_weight": link_cap_penalty_w,
            "gpu_cap_penalty_weight":  gpu_cap_penalty_w,
            "freq_tiebreak_eps": float(opt_defs.get("freq_tiebreak_eps", 0.0)),
            "theta_min_per_gpu": dict(theta_min_per_gpu),
            "theta_min_per_cluster": theta_min_per_cluster,
            "theta_min_global":      theta_min_global,
            **snapshot_defaults,
            "Optimizer_Defaults": {**opt_defs, "theta_penalty_weight": theta_pw},
            "Theta_Min": theta_cfg,
            "gpu_jobs_served": gpu_jobs_served,
            "lambda_bg":       lambda_bg,
            "utilization_guess": utilization_guess,
            "optimizer_power":      optimizer_power,
            "optimizer_latency":    optimizer_latency,
            "optimizer_efficiency": optimizer_efficiency,
            "freq_mode": str(getattr(state, "freq_mode", "adaptive")).lower(),
        }


    def decide(self, state: "SimState", pkt: Dict[str, Any]) -> Tuple[Tuple[str, str, str], str]:
        """
        Core policy:
        - If this job already has a pinned GPU/freq => reuse, never re-run optimizer.
        - Otherwise, run optimizer once for this job, pin GPU+freq, and remember it.
        - Always reserve "pending load" so _snapshot() will show claimed work and
        the next job will see that GPU as busy.
        """

        # ------------------------------------------------------------
        # 0. Make sure the shared bookkeeping dicts EXIST (do NOT reset)
        # ------------------------------------------------------------
        if not hasattr(state, "pending_service_time") or state.pending_service_time is None:
            state.pending_service_time = {}      # (C,N,G) -> claimed seconds
        if not hasattr(state, "pending_tasks") or state.pending_tasks is None:
            state.pending_tasks = {}             # (C,N,G) -> claimed task count
        if not hasattr(state, "job_assignment") or state.job_assignment is None:
            state.job_assignment = {}            # job_id -> (gpu_key, freq) OR (gpu_key,)
        if not hasattr(state, "job_freq_plan") or state.job_freq_plan is None:
            state.job_freq_plan = {}             # job_id -> {"gpu":(...), "freq":..., "src":...}
        if not hasattr(state, "x_job") or state.x_job is None:
            state.x_job = {}                     # job_id -> gpu_key
        if not hasattr(state, "gpu_freq_plan_fixed") or state.gpu_freq_plan_fixed is None:
            state.gpu_freq_plan_fixed = {}       # ('C','N','G') -> (freq, reason)
        # per-GPU job history for load balancing
        if not hasattr(state, "gpu_jobs_served") or state.gpu_jobs_served is None:
            state.gpu_jobs_served = {}           # "C-N-G" -> int
        # for logging/provenance, upstream code expects this later
        if not hasattr(state, "job_gpu_src") or state.job_gpu_src is None:
            state.job_gpu_src = {}               # job_id -> "optimizer"/"reuse-job"/...
        # Reservation latch: reserve ONCE per (Job, Task_ID)
        if not hasattr(state, "reserved_tasks") or state.reserved_tasks is None:
            state.reserved_tasks = set()   # {(job_id, task_id)}
        if not hasattr(state, "flop_backlog") or state.flop_backlog is None:
            state.flop_backlog = {}   # gid -> backlog FLOPs


        # ------------------------------------------------------------
        # 1. Helpers
        # ------------------------------------------------------------
        J = _norm_job_id(pkt)                 # normalized job_id (e.g. "AR_J0")
        task_id = str(pkt.get("Task_ID", "")) # exact task id
        reserve_key = (str(J), str(task_id))
        tid_base = base_tid(task_id)
        key_task = (str(J), str(task_id))  # exact Task_ID (NOT base_tid)

        def _reserve_once(gpu_key, freq_to_return, work_flops):
            if key_task in state.reserved_tasks:
                return
            _reserve_claim(gpu_key, freq_to_return, work_flops)
            print("[DBG] flop_backlog keys:", list(getattr(state, "flop_backlog", {}).keys())[:4])
            print("[DBG] lambda_bg preview:", snap.get("lambda_bg", {}))

            state.reserved_tasks.add(key_task)

        def _reserve_claim(gpu_key: Tuple[str, str, str], freq_choice: str, work_flops: float):
            """
            Fairness reservation: mark that <gpu_key> has <work_flops> more FLOPs
            coming at <freq_choice>. This feeds _snapshot() so the *next* job
            sees that GPU as busier (pending tail / queued).
            """

            key3 = _norm_gkey(gpu_key)
            wf = float(work_flops or 0.0)
            if wf > 0.0:
                state.flop_backlog[key3] = float(state.flop_backlog.get(key3, 0.0)) + wf
                _dbg(state, f"[RESERVE] key3={key3} +{wf} FLOPs -> backlog={state.flop_backlog.get(key3,0.0)}")

            # convert to estimated service time at that freq
            fmap = (self.catalog.rates or {}).get(gpu_key, {}) or {}
            run_key = _freq_key_match_in_map(fmap, str(freq_choice)) or str(freq_choice)

            rate = 0.0
            if run_key in fmap:
                try:
                    rate = float(fmap[run_key])  # FLOPs/sec
                except Exception:
                    rate = 0.0

            est_service_s = 0.0
            if rate > 0.0 and (work_flops or 0.0) > 0.0:
                est_service_s = float(work_flops) / rate
            
            state.pending_service_time[key3] = state.pending_service_time.get(key3, 0.0) + est_service_s
            state.pending_tasks[key3] = state.pending_tasks.get(key3, 0) + 1


        def _reuse_gpu_and_freq_for_job(job_id: str) -> Tuple[Tuple[str,str,str], str]:
            """
            Resolve (gpu_key=('C','N','G'), freq_str) for a job that's already pinned.
            Falls back sensibly if freq info isn't explicit.
            """
            gpu_key_for_job = None

            # 1) job_assignment might store either:
            # (gpu_key, freq) OR gpu_key (3-tuple)
            ja = state.job_assignment.get(job_id)
            if isinstance(ja, (tuple, list)):
                # case A: ((C,N,G), freq)
                if len(ja) == 2 and isinstance(ja[0], (tuple, list)) and len(ja[0]) == 3:
                    gpu_key_for_job = tuple(map(str, ja[0]))
                    pinned_freq     = str(ja[1])
                # case B: (C,N,G) only
                elif len(ja) >= 3:
                    gpu_key_for_job = tuple(map(str, ja[:3]))
                    pinned_freq     = ""
                else:
                    gpu_key_for_job = None
                    pinned_freq     = ""
            else:
                gpu_key_for_job = None
                pinned_freq     = ""

            # 2) fallback to x_job if not found
            if not gpu_key_for_job and job_id in state.x_job:
                gpu_key_for_job = tuple(map(str, state.x_job[job_id]))
                pinned_freq     = ""

            if not gpu_key_for_job:
                raise RuntimeError(f"No GPU assignment recorded yet for job {job_id}")

            # now choose a freq
            fmap = ((self.catalog.rates or {}).get(gpu_key_for_job, {}) or {})
            freq = ""

            # priority 1: if job_freq_plan[J] exists
            plan = state.job_freq_plan.get(job_id, {})
            if isinstance(plan, dict) and "freq" in plan:
                freq = str(plan["freq"])
            elif isinstance(plan, str):
                freq = plan

            # priority 2: pinned_freq from job_assignment
            if not freq:
                freq = pinned_freq

            # priority 3: current hardware freq
            if not freq:
                cur_map = getattr(state, "gpu_to_freq", {}) or {}
                freq = str(cur_map.get(gpu_key_for_job, "")) or ""

            # priority 4: default freq for that gpu
            if not freq:
                df = (self.catalog.defaults or {}).get(gpu_key_for_job, "")
                if df in fmap:
                    freq = df

            # priority 5: fastest step
            if not freq and fmap:
                try:
                    freq = max(fmap, key=lambda fk: float(fmap[fk]))
                except Exception:
                    # fallback if values aren't numeric
                    freq = max(fmap, key=lambda fk: float(fk))

            if not freq:
                raise RuntimeError(f"Could not resolve a frequency for GPU {gpu_key_for_job} in reuse path.")

            return gpu_key_for_job, str(freq)

        def _task_total_flops_once() -> float:
            """
            Decide() is called per PACKET; Workload_FLOPs is per TASK in the pipeline.
            Compute a task-level FLOPs value robustly.
            """
            # Try direct field first
            try:
                w = float(pkt.get("Workload_FLOPs", 0.0) or 0.0)
            except Exception:
                w = 0.0
            if w > 0.0:
                return w

            # Fallback: take max across pregen packets for this (Job, Task_ID)
            try:
                tk_pkts = state.pregen_by_task.get((J, task_id), [pkt])
                vals = []
                for p in tk_pkts:
                    try:
                        vals.append(float(p.get("Workload_FLOPs", 0.0) or 0.0))
                    except Exception:
                        pass
                return float(max(vals) if vals else 0.0)
            except Exception:
                return 0.0

        # ------------------------------------------------------------
        # 2. REUSE PATH: job already pinned → skip optimizer
        # ------------------------------------------------------------
        if J in state.job_assignment:
            gpu_key, freq_to_return = _reuse_gpu_and_freq_for_job(J)
            pkt.setdefault("GPU_Decision_Source", "reuse-job")
            # Workload_FLOPs is per-task in the pipeline → reserve it ONCE
            claimed_flops = _task_total_flops_once()
            _reserve_once(gpu_key, freq_to_return, claimed_flops)

            return gpu_key, str(freq_to_return)


        # ------------------------------------------------------------
        # 3. FIRST-TIME PATH: this job has NO assignment yet
        # ------------------------------------------------------------
        # Build aggregate "new_task" for this Task_ID
        key      = (J, task_id)
        tk_pkts  = state.pregen_by_task.get(key, [pkt])

        vals = []
        for p in tk_pkts:
            try:
                vals.append(float(p.get("Workload_FLOPs", 0.0) or 0.0))
            except Exception:
                pass
        total_flops = max(vals) if vals else float(pkt.get("Workload_FLOPs", 0.0) or 0.0)

        ul_total_kB = (
            sum(int(p.get("Packet_Size_KB", 0)) for p in tk_pkts)
            or int(pkt.get("Task_UL_Total_kB", 0))
        )
        dl_total_kB = int(pkt.get("Task_DL_Total_kB", 0))
        deadline    = float(pkt.get("Task_Deadline", float("inf")))

        # Lambda derivation (tasks/sec)
        try:
            lam = float(pkt.get("Lambda", pkt.get("Task_Arrival_Rate", 0)) or 0.0)
        except Exception:
            lam = 0.0
        if lam <= 0.0:
            cfg = getattr(self, "config", {}) or {}
            ts  = (cfg.get("Task_Settings") or {}) if isinstance(cfg, dict) else {}
            fps_raw    = ts.get("SERVICE_ARRIVAL_RATE_fps", ts.get("ARRIVAL_RATE_fps", 30))
            stride_raw = ts.get("STRIDE", ts.get("Task_Stride", ts.get("stride", 1)))
            try:
                fps = float(fps_raw or 0.0)
            except Exception:
                fps = 0.0
            try:
                stride = int(stride_raw or 1)
            except Exception:
                stride = 1
            if fps > 0.0:
                lam = fps / max(1, stride)

        if lam <= 0.0:
            raise RuntimeError(
                "Streaming-only mode: positive Lambda required. "
                "Provide pkt['Lambda'] or Task_Settings.SERVICE_ARRIVAL_RATE_fps (and STRIDE)."
            )

        new_task = {
            "Task_ID":         task_id,
            "Job_ID":          J,
            "Workload_FLOPs":  int(total_flops),
            "Task_Deadline":   float(deadline),
            "UL_Total_kB":     int(ul_total_kB),
            "DL_Total_kB":     int(dl_total_kB),
            "Lambda":          float(lam),
        }

        # ------------------------------------------------------------
        # 4. Snapshot system state (this includes pending_* so far)
        # ------------------------------------------------------------
        snap = self._snapshot(state)
        snap["use_arrival_rates"] = True  # benign flag; actual weights/ρ caps stay from snapshot
        # Option 1: lambda_bg comes ONLY from flop_backlog (single source of truth)
        fb = getattr(state, "flop_backlog", {}) or {}
        horizon_s = float(getattr(state, "util_tau_s", 0.5) or 0.5)
        horizon_s = max(1e-6, horizon_s)

        lambda_bg = {}
        for raw_key, backlog_flops in fb.items():
            # raw_key is usually (c,n,g) tuple in sim
            if isinstance(raw_key, (tuple, list)) and len(raw_key) == 3:
                gid = f"{raw_key[0]}-{raw_key[1]}-{raw_key[2]}"
            else:
                gid = str(raw_key)
            lambda_bg[gid] = float(backlog_flops or 0.0) / horizon_s  # FLOPs/s

        snap["lambda_bg"] = lambda_bg


        # ------------------------------------------------------------
        # 5. Choose optimizer entry function and call it
        # ------------------------------------------------------------
        mod = self._pick_module(getattr(state, "objective", None))
        if not mod:
            raise RuntimeError(f"No optimizer module configured for objective '{state.objective}'.")

        entry = (
            "solve_incremental" if hasattr(mod, "solve_incremental") else
            "solve"             if hasattr(mod, "solve")             else
            "solve_latency"     if hasattr(mod, "solve_latency")     else
            "solve_power"       if hasattr(mod, "solve_power")       else
            "solve_efficiency"  if hasattr(mod, "solve_efficiency")  else
            None
        )
        if entry is None:
            raise RuntimeError(f"No compatible entrypoint in {mod} for objective {state.objective}")

        opt_fn = getattr(mod, entry)

        if str(getattr(state, "objective", "")).lower() == "power":
            snap["pinned_frequencies"] = {}

        # build kwargs for optimizer but filter to allowed signature
        kwargs = dict(
            pinned_assignments = (snap.get("pinned_assignments") or {}).copy(),
            pinned_frequencies =  snap.get("pinned_frequencies") or {},
            objective          =  str(getattr(state, "objective", "")).lower(),
        )
        sig = inspect.signature(opt_fn)
        allowed = {k: v for k, v in kwargs.items() if k in sig.parameters}

        ans = opt_fn(snap, [new_task], **allowed)
        self.last_solution = ans

        if not (
            isinstance(ans, dict)
            and ("assignments" in ans)
            and ("frequencies" in ans)
        ):
            raise RuntimeError("Optimizer returned empty or invalid result.")

        # parse assignment for THIS task/job
        rec = (
            ans["assignments"].get(task_id)
            or ans["assignments"].get(base_tid(task_id))
        )
        if not rec:
            raise RuntimeError(f"No assignment for {task_id} in optimizer result.")

        gpu_key = (str(rec["Cluster"]), str(rec["Node"]), str(rec["GPU"]))
        gid     = f"{gpu_key[0]}-{gpu_key[1]}-{gpu_key[2]}"

        # merge frequencies / freq_plan into a single lookup
        def _norm_gid(gid_like):
            if isinstance(gid_like, (tuple, list)) and len(gid_like) == 3:
                return f"{gid_like[0]}-{gid_like[1]}-{gid_like[2]}"
            s = str(gid_like).replace(",", "-").replace(" ", "")
            parts = s.split("-")
            return "-".join(parts[:3]) if len(parts) >= 3 else s

        freqs_out = ans.get("frequencies") or {}
        freq_plan = ans.get("freq_plan") or {}
        freqs = { _norm_gid(k): str(v) for k, v in freqs_out.items() }
        for k, v in freq_plan.items():
            freqs[_norm_gid(k)] = str(v)

        planned_freq = freqs.get(gid, str(rec.get("Frequency", "")).strip())

        pkt["GPU_Decision_Source"] = "optimizer"

        # ------------------------------------------------------------
        # 6. Apply / lock DVFS, and persist job-wide pin
        # ------------------------------------------------------------
        fm = str(getattr(state, "freq_mode", "adaptive")).lower()
        key3 = _norm_gkey(gpu_key)  # canonical tuple key
        fmap_here = (self.catalog.rates or {}).get(gpu_key, {}) or {}

        if fm == "fixed":
            # force fastest step and lock it
            try:
                fastest = max(fmap_here, key=lambda fk: float(fmap_here[fk]))
            except Exception:
                fastest = max(fmap_here, key=lambda fk: float(fk))

            # # ----Actual frequency changes happen only in on_arrive_gpu and on_finish_gpu via apply_boundary_upshift_only.
            # cur_hw = (getattr(state, "gpu_to_freq", {}) or {}).get((key3), "")
            # cur_norm = _freq_key_match_in_map(fmap_here, cur_hw) or ""
            # if cur_norm != fastest:
            #     set_gpu_freq(
            #         state,
            #         gpu_key,
            #         fastest,
            #         reason="fixed-max",
            #         fmap=fmap_here,
            #         when=_now(state),
            #         origin="optimizer/fixed-max-ensure",
            #         force_log=True,
            #         override_lock=True,
            #     )

            state.gpu_freq_plan_fixed[key3] = (fastest, "fixed-max")
            freq_to_return = fastest

        else:
            # adaptive mode → allow retunes; normalize planned_freq first
            try:
                state.gpu_freq_plan_fixed.pop(key3, None)
            except Exception:
                pass

            run_key = _freq_key_match_in_map(fmap_here, str(planned_freq)) or str(planned_freq)
            if not run_key:
                # prefer min for power/eff adaptive, else prefer max for latency-ish
                obj_l = str(getattr(state, "objective", "") or "").lower()
                prefer = "min" if ("power" in obj_l or "eff" in obj_l) else "max"
            freq_to_return = run_key

            # if the optimizer's freq differs from hw, apply + DVFS epoch bump on upshift
            cur_hw = (getattr(state, "gpu_to_freq", {}) or {}).get((key3), "")
            cur_key = _freq_key_match_in_map(fmap_here, str(cur_hw)) or ""
            if freq_to_return and freq_to_return != cur_key:
                try:
                    def _R(k):
                        try:
                            return float(fmap_here.get(k, -1.0))
                        except Exception:
                            return -1.0
                    if _R(freq_to_return) > _R(cur_key):
                        _dvfs_epoch_bump(state, gpu_key)
                except Exception:
                    pass

        # ------------------------------------------------------------
        # 7. Persist this job’s pin so FUTURE packets/tasks skip optimizer
        # ------------------------------------------------------------
        state.job_assignment[J] = (gpu_key, str(freq_to_return))
        state.x_job[J] = gpu_key

        state.job_freq_plan[J] = {
            "gpu":  gpu_key,
            "freq": str(freq_to_return),
            "src":  "optimizer",
        }

        state.job_gpu_src[J] = "optimizer"

        # bump per-GPU job count exactly once per job
        gid_str = f"{gpu_key[0]}-{gpu_key[1]}-{gpu_key[2]}"
        # bump exactly once per job
        if not hasattr(state, "gpu_jobs_served"):
            state.gpu_jobs_served = {}
        # state.gpu_jobs_served[gid_str] = state.gpu_jobs_served.get(gid_str, 0) + 1
        state.gpu_jobs_served[gid_str] = int(state.gpu_jobs_served.get(gid_str, 0) or 0) + 1


        # ------------------------------------------------------------
        # 8. Store in incremental cache by base_tid for this task
        # ------------------------------------------------------------
        if self.store and getattr(self.store, "mode", "readwrite") != "readonly":
            self.store.set(tid_base, {
                "Cluster":   gpu_key[0],
                "Node":      gpu_key[1],
                "GPU":       gpu_key[2],
                "Frequency": str(freq_to_return),
            })
            self.store.set_freq(gid, str(freq_to_return))

        # ------------------------------------------------------------
        # 9. Reserve load for fairness so next job sees this GPU as busy
        # ------------------------------------------------------------
        claimed_flops = float(new_task.get("Workload_FLOPs", 0.0) or 0.0)
        _reserve_once(gpu_key, freq_to_return, claimed_flops)

        # ------------------------------------------------------------
        # 10. Return decision
        # ------------------------------------------------------------
        return gpu_key, str(freq_to_return)


# =============================
# Sim State & helpers
# =============================

@dataclass
class SimState:
    fel: FEL
    net: NetPerCluster
    stations: Dict[str, Station]
    gpu_catalog: GPUCatalog
    rng: random.Random

    scheduler: Optional[Any] = None

    logs: List[Dict[str, Any]] = field(default_factory=list)
    dvfs_log: List[Dict[str, Any]] = field(default_factory=list)

    agg: Dict[Tuple[str,str], Dict[str, Any]] = field(default_factory=dict)
    ports_per_cluster: Dict[str, int] = field(default_factory=dict)
    queue_capacity: int = 0
    admission_policy: str = "soft"
    dropped_tasks: Dict[Tuple[str,str], str] = field(default_factory=dict)
    admitted_tasks: Dict[Tuple[str,str], Dict[str,Any]] = field(default_factory=dict)
    pregen_by_task: Dict[Tuple[str,str], List[Dict[str,Any]]] = field(default_factory=dict)

    master_rows: List[Dict[str, Any]] = field(default_factory=list)
    row_index: Dict[Tuple[str, str, str, str], Dict[str, Any]] = field(default_factory=dict)
    rows_by_task: Dict[Tuple[str,str], List[Dict[str,Any]]] = field(default_factory=lambda: defaultdict(list))
    task_times: Dict[Tuple[str,str], Dict[str,float]] = field(default_factory=dict)
    R0_RC_Links: Dict[str, Dict[str, float]] = field(default_factory=dict)

    x_j: Dict[Tuple[str,str], Tuple[str,str,str]] = field(default_factory=dict)
    delta: Dict[Tuple[str,str,str], str] = field(default_factory=dict)
    gpu_to_freq: Dict[Tuple[str,str,str], str] = field(default_factory=dict)
    task_name_map: Dict[Tuple[str,str], str] = field(default_factory=dict)
    gpu_util: Dict[Tuple[str, str, str], "UtilRec"] = field(default_factory=dict)

    job_assignment: Dict[str, Tuple[Tuple[str,str,str], str]] = field(default_factory=dict)
    active_jobs: DefaultDict[Tuple[str,str,str], Set[str]] = field(default_factory=lambda: defaultdict(set))
    load_by_gpu: DefaultDict[Tuple[str, str, str], float] = field(
        default_factory=
        lambda: defaultdict(float)
    )
    # NEW: DVFS-safe FLOP backlog
    flop_backlog: DefaultDict[Tuple[str, str, str], float] = field(
        default_factory=lambda: defaultdict(float)
    )

    util_ewma: DefaultDict[Tuple[str, str, str], float] = field(
        default_factory=lambda: defaultdict(float)
    )
    gpu_jobs_served: Dict[str, int] = field(default_factory=dict) 
    debug_dvfs: bool = False 
    util_tau_s: float = 0.5          # EWMA smoothing time constant (sec) – default

    objective: str = "latency"
    assigner: str = "heuristic"
    optimizer: Optional["OptimizerFacade"] = None
    assignment_store: Optional[AssignmentStore] = None

    # stores per-job frequency and its decision source
    job_freq_plan: Dict[str, Dict[str, str]] = field(default_factory=dict)
    # Example entry: {"AR_J1": {"freq": "1500", "src": "optimizer"}}

    # Per-job provenance: where the GPU decision came from (optimizer/random/least-load/store_restore/...)
    job_gpu_src: Dict[str, str] = field(default_factory=dict)

def _util_init(state: "SimState"):
    if not hasattr(state, "gpu_util"):
        state.gpu_util = {}

def st_name(prefix: str, c: str, port: int) -> str:
    return f"{prefix}:{c}:P{int(port)}"

def get_station(state: SimState, prefix: str, c: str, port: int,
                svc_fn: Callable[[Dict[str, Any]], float]) -> Station:
    name = st_name(prefix, c, port)
    st = state.stations.get(name)
    if st is None:
        st = Station(name=name, service_time_fn=svc_fn)
        st.capacity = state.queue_capacity if state.queue_capacity > 0 else None
        state.stations[name] = st
    return st

def pick_port(state: SimState, prefix: str, c: str, n_ports: int, svc_fn: Callable[[Dict[str, Any]], float]) -> int:
    best_port, best_score = 1, None
    for p in range(1, int(n_ports) + 1):
        st = get_station(state, prefix, c, p, svc_fn)
        score = (len(st.q) + (1 if st.busy else 0))
        if best_score is None or score < best_score:
            best_port, best_score = p, score
    return best_port


@dataclass
class JobTracker:
    # # expected counts
    # tasks_per_job: dict = field(default_factory=lambda: defaultdict(int))
    # dl_packets_per_task: dict = field(default_factory=lambda: defaultdict(int))
    # # remaining counters
    # tasks_remaining: dict = field(default_factory=lambda: defaultdict(int))
    # dl_packets_remaining: dict = field(default_factory=lambda: defaultdict(int))
    # # timing
    # job_min_arrival: dict = field(default_factory=lambda: defaultdict(lambda: float("inf")))
    # job_completion_time: dict = field(default_factory=dict)
    # task_completion_time: dict = field(default_factory=dict)

    def __init__(self):
        # counts indexed by job / base task id
        self.tasks_per_job        = defaultdict(int)          # job -> tasks (by base id)
        self.tasks_remaining      = defaultdict(int)          # job -> remaining tasks
        self.dl_packets_per_task  = defaultdict(int)          # base_tid -> expected DL packets
        self.dl_packets_remaining = defaultdict(int)          # base_tid -> remaining DL packets

        # timing
        self.job_min_arrival      = defaultdict(lambda: math.inf)  # earliest UL arrival (s)

        # summaries
        self.job_completion_time  = {}     # optional
        self.task_completion_time = {}     # optional
        self.summary              = {}     # filled by recompute_from_rows() 

    def index_pregen_rows(self, rows):
        """
        Call once before scheduling: 'rows' should be the pre-generated rows you
        already load in run_sim() (UL side is enough).
        """
        seen = set()  # (job, base_tid)
        for r in rows:
            jid  = str(r.get("Job_ID",""))
            tid  = str(r.get("Task_ID",""))
            if not jid or not tid:
                continue
            dir_low = str(r.get("Direction","")).lower()
            tbase   = base_tid(tid)

            if dir_low == "uplink":
                key = (jid, tbase)
                if key not in seen:
                    self.tasks_per_job[jid] += 1
                    seen.add(key)
                # earliest UL arrival (seconds)
                ta = r.get("Packet_Arrival_Time")
                if ta not in ("", None):
                    try:
                        self.job_min_arrival[jid] = min(self.job_min_arrival[jid], float(ta))
                    except Exception:
                        pass

            elif dir_low == "downlink":
                # one DL row = one DL packet for that base task
                self.dl_packets_per_task[tbase] += 1

        # initialize remaining counters
        for jid, n in self.tasks_per_job.items():
            self.tasks_remaining[jid] = int(n)
        for tbase, n in self.dl_packets_per_task.items():
            self.dl_packets_remaining[tbase] = int(n)

    def on_dl_packet_depart(self, job_id: str, task_id: str, t: float):
        """Called on every DL egress; decrements per-task DL remaining and may close a task/job."""
        tbase = base_tid(task_id)
        if self.dl_packets_remaining.get(tbase, 0) > 0:
            self.dl_packets_remaining[tbase] -= 1
            if self.dl_packets_remaining[tbase] == 0:
                # base task finished; decrement job’s remaining task count if tracked
                if self.tasks_remaining.get(job_id, 0) > 0:
                    self.tasks_remaining[job_id] -= 1

    def recompute_from_rows(self, rows):
        """Rebuild per-job completion from final rows (seconds)."""
        from collections import defaultdict
        exp = defaultdict(int)
        done = defaultdict(int)
        last_end = defaultdict(float)

        for r in rows:
            if str(r.get("Direction","")).lower() != "downlink":
                continue
            jid = str(r.get("Job_ID",""))
            if not jid:
                continue
            exp[jid] += 1
            te = r.get("R0_DL_IN_exit")
            if te not in ("", None):
                try:
                    tef = float(te)
                    done[jid] += 1
                    last_end[jid] = max(last_end[jid], tef)
                except Exception:
                    pass

        self.summary = {}
        for jid in set(exp) | set(done):
            comp = int(done.get(jid, 0) >= exp.get(jid, 0) > 0)
            endt = last_end.get(jid, 0.0)
            self.summary[jid] = {
                "expected_dl_packets": exp.get(jid, 0),
                "completed_dl_packets": done.get(jid, 0),
                "completed": comp,
                "end_time": endt,
            }
            # keep legacy fields in sync so summary_rows() works
            if comp:
                self.job_completion_time[jid] = float(endt)

    def summary_rows(self):
        rows = []
        # prefer jobs we’ve seen in pregen (stable ordering), fall back to summary keys
        job_ids = list(sorted(self.tasks_per_job)) or list(sorted(self.summary))
        for jid in job_ids:
            start = float(self.job_min_arrival.get(jid, float("nan")))
            s = self.summary.get(jid, {})
            done = int(s.get("completed", 0))
            end  = float(s.get("end_time", float("nan"))) if done else float("nan")
            dur  = (end - start) if done and math.isfinite(start) and math.isfinite(end) else float("nan")
            rows.append({
                "Job_ID": jid,
                "Job_Start_Sim_s": start,
                "Job_End_Sim_s": end,
                "Job_Duration_Sim_s": dur,
                "Num_Tasks": int(self.tasks_per_job.get(jid, 0)),
                "Completed": done
            })
        return rows

def write_job_completion_csv(path, tracker: JobTracker, extra_cols: dict | None = None):
    extra_cols = extra_cols or {}
    with open(path, "w", newline="") as f:
        base_fields = [
            "Job_ID", "Job_Start_Sim_s", "Job_End_Sim_s", "Job_Duration_Sim_s",
            "Num_Tasks", "Completed"
        ]
        fieldnames = base_fields + [k for k in extra_cols.keys() if k not in base_fields]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in tracker.summary_rows():
            row = dict(r)
            row.update(extra_cols)
            w.writerow(row)


def write_sim_duration_csv(path, logical_s, wall_s):
    with open(path, "w", newline="") as f:
        f.write("logical_sim_duration_s,wall_runtime_s\n")
        f.write(f"{logical_s:.6f},{wall_s:.6f}\n")

def _ensure_dir(path: str) -> None:
    d = _os.path.dirname(path)
    if d:
        _os.makedirs(d, exist_ok=True)


def write_rows_csv(path: str, rows: list[dict], hdr: list[str]) -> None:
    """
    Generic CSV writer:
      - ensures directory exists
      - writes header once
      - streams rows with writerows()
    """
    _ensure_dir(path)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=hdr)
        w.writeheader()
        if rows:
            w.writerows({k: r.get(k, "") for k in hdr} for r in rows)
# ============================================================================

# =============================
# Event Handlers
# =============================

def _norm_gpu_key(gk):
    """Normalize any GPU key into ('C#','N#','G#')."""
    def _pref(val, lead):
        s = str(val).strip()
        su = s.upper()

        patterns = {
            "C": r"^C(?:LUSTER)?\s*(\d+)$",
            "N": r"^N(?:ODE)?\s*(\d+)$",
            "G": r"^G(?:PU)?\s*(\d+)$",
        }
        m = re.match(patterns[lead], su)
        if m:
            return f"{lead}{int(m.group(1))}"

        if su.isdigit():
            return f"{lead}{int(su)}"

        m = re.search(r"(\d+)", su)
        if m:
            return f"{lead}{int(m.group(1))}"

        raise RuntimeError(f"Invalid {lead}-component in GPU key: {val!r}")

    # tuple/list: accept >=3
    if isinstance(gk, (tuple, list)) and len(gk) >= 3:
        return (_pref(gk[0], "C"), _pref(gk[1], "N"), _pref(gk[2], "G"))

    # dict: accept common variants
    if isinstance(gk, dict):
        def _get(d, *keys, default=""):
            for k in keys:
                if k in d:
                    return d.get(k, default)
            return default

        return (
            _pref(_get(gk, "Cluster", "cluster", "C", "c"), "C"),
            _pref(_get(gk, "Node", "node", "N", "n"), "N"),
            _pref(_get(gk, "GPU", "gpu", "G", "g"), "G"),
        )

    # string: support "-", "," "/", and whitespace
    s = str(gk).strip()
    if s:
        for ch in ("—", "–", "‒", "−", "/", ","):
            s = s.replace(ch, "-")

        parts = [p.strip() for p in re.split(r"[-\s]+", s) if p.strip()]
        if len(parts) >= 3:
            return (_pref(parts[0], "C"), _pref(parts[1], "N"), _pref(parts[2], "G"))

        mC = re.search(r'(?i)c(?:luster)?\s*(\d+)', s)
        mN = re.search(r'(?i)n(?:ode)?\s*(\d+)', s)
        mG = re.search(r'(?i)g(?:pu)?\s*(\d+)', s)
        if mC and mN and mG:
            return (f"C{int(mC.group(1))}", f"N{int(mN.group(1))}", f"G{int(mG.group(1))}")

        nums = re.findall(r'(\d+)', s)
        if len(nums) >= 3:
            return (f"C{int(nums[0])}", f"N{int(nums[1])}", f"G{int(nums[2])}")

    raise RuntimeError(f"GPU key from decision path is invalid: {gk!r}")

# # helper for safe catalog access
def _rates_for(state, gkey):
    gc = getattr(state, "gpu_catalog", None)
    return (gc.rates.get(gkey, {}) if gc and getattr(gc, "rates", None) else {}) or {}

def _defaults_for(state, gkey):
    gc = getattr(state, "gpu_catalog", None)
    return (gc.defaults.get(gkey, "") if gc and getattr(gc, "defaults", None) else "")

def _ensure_assignment(state: "SimState", pkt: Dict[str, Any]) -> Tuple[Optional[Tuple[str, str, str]], str]:
    """
    PLAN-only:
    - Decide / reuse the GPU and PLANNED frequency for this job.
    - Do NOT apply DVFS here. DVFS must be done in on_arrive_gpu() only.
    - Stamp packet with Planned_Frequency and HW_Frequency for debugging.
    - Return (gpu_key, planned_freq_str).
    """

    job_id   = _norm_job_id(pkt)
    task_id  = str(pkt.get("Task_ID", ""))
    tid_base = base_tid(task_id)

    # lazy init state
    if not hasattr(state, "job_assignment") or state.job_assignment is None: state.job_assignment = {}
    if not hasattr(state, "x_job") or state.x_job is None:                  state.x_job = {}
    if not hasattr(state, "job_freq_plan") or state.job_freq_plan is None:  state.job_freq_plan = {}
    if not hasattr(state, "x_j") or state.x_j is None:                      state.x_j = {}
    if not hasattr(state, "job_gpu_src") or state.job_gpu_src is None:      state.job_gpu_src = {}
    if not hasattr(state, "_opt_printed_jobs") or state._opt_printed_jobs is None:
        state._opt_printed_jobs = set()

    freq_mode = str(getattr(state, "freq_mode", "adaptive") or "adaptive").lower()

    # ---------- helpers ----------
    def _pick_fastest_freq(fmap: Dict[str, Any]) -> str:
        if not fmap:
            return ""
        # prefer by achieved service rate
        try:
            return max(fmap, key=lambda fk: float(fmap[fk]))
        except Exception:
            # fallback: by numeric key
            try:
                return max(fmap, key=lambda fk: float(fk))
            except Exception:
                return next(iter(fmap.keys()))


    def _current_hw_freq_key(state, gpu_key: Tuple[str, str, str]) -> str:
        # canonical tuple key ('C','N','G')
        key3 = _norm_gkey(gpu_key)
        # read raw HW value
        gmap = getattr(state, "gpu_to_freq", {}) or {}
        raw_hw = gmap.get(key3, "")
        
        # legacy string-key fallback (defensive)
        if not raw_hw:
            raw_hw = gmap.get(f"{key3[0]}-{key3[1]}-{key3[2]}", "")

        if not raw_hw:
            return ""

        # normalize against this GPU's freq map
        fmap_here = _rates_for(state, key3) or {}
        return _freq_key_match_in_map(fmap_here, str(raw_hw)) or str(raw_hw)



    def _record_job_pin(jid: str, gpu_key: Tuple[str, str, str], freq: str, src: str) -> None:
        state.job_assignment[jid] = (gpu_key, str(freq))
        state.x_job[jid] = gpu_key
        state.x_j[(jid, tid_base)] = gpu_key
        state.job_gpu_src[jid] = src
        state.job_freq_plan[jid] = {"gpu": gpu_key, "freq": str(freq), "src": src}

    # def _emit_opt_line_once(jid: str, tid: str, gpu_key_now: Tuple[str, str, str], src: str) -> None:
    #     src_norm = str(src or "").lower()
    #     if not (src_norm == "optimizer" or src_norm.startswith("optimizer")):
    #         return
    #     if jid in state._opt_printed_jobs:
    #         return
    #     if to01(pkt.get("Task_Is_First", 0)) != 1:
    #         return
    #     if to01(pkt.get("Is_First", 0)) != 1:
    #         return

    #     c, n, g = map(str, gpu_key_now)
    #     planned = str(pkt.get("Planned_Frequency", "") or "")
    #     hw = _current_hw_freq_key(state, gpu_key_now) or ""
    #     print(f"[OPT] {tid} -> {c}/{n}/{g} planned={planned} hw={hw}")
    #     state._opt_printed_jobs.add(jid)

    def _emit_opt_line_once(jid: str, tid: str, gpu_key_now: Tuple[str, str, str], src: str) -> None:
        src_norm = str(src or "").lower()
        if not (src_norm == "optimizer" or src_norm.startswith("optimizer")):
            return
        if jid in state._opt_printed_jobs:
            return
        if to01(pkt.get("Task_Is_First", 0)) != 1:
            return
        if to01(pkt.get("Is_First", 0)) != 1:
            return

        c, n, g = map(str, gpu_key_now)
        ng = _norm_gkey(gpu_key_now)

        # planned freq (normalize if possible)
        planned_raw = str(pkt.get("Planned_Frequency", "") or "")
        fmap_here = _rates_for(state, ng) or {}
        planned = _freq_key_match_in_map(fmap_here, planned_raw) or planned_raw or ""

        # ---- BOOTSTRAP HW FIRST (so the first log isn't blank) ----
        state.gpu_to_freq = getattr(state, "gpu_to_freq", None) or {}
        if ng not in state.gpu_to_freq:
            # initialize HW to planned (or leave empty if planned is empty)
            if planned:
                state.gpu_to_freq[ng] = planned

        # now read hw
        hw = _current_hw_freq_key(state, gpu_key_now) or ""

        print(f"[OPT] {tid} -> {c}/{n}/{g} planned={planned} hw={hw}")
        state._opt_printed_jobs.add(jid)



    def _refresh_pkt_freq_fields(pkt, gpu_key, planned_freq):
        fmap_here = _rates_for(state, _norm_gkey(gpu_key)) or {}
        key3 = _norm_gkey(gpu_key)
        raw_hw = (getattr(state, "gpu_to_freq", {}) or {}).get(key3, "")
        if not raw_hw:
            raw_hw = (getattr(state, "gpu_to_freq", {}) or {}).get(f"{key3[0]}-{key3[1]}-{key3[2]}", "")
        hw_key = _freq_key_match_in_map(fmap_here, raw_hw) or ""
        pkt["Planned_Frequency"] = str(planned_freq or "")
        pkt["HW_Frequency"] = hw_key or str(planned_freq or "")


    def _reuse_job_gpu_and_freq(jid: str) -> Tuple[Optional[Tuple[str, str, str]], str]:
        memo = state.job_assignment.get(jid) or state.job_assignment.get(tid_base)

        gpu_key = None
        pinned_freq = ""

        if isinstance(memo, (tuple, list)):
            if len(memo) == 2 and isinstance(memo[0], (tuple, list)) and len(memo[0]) == 3:
                gpu_key = _norm_gpu_key(memo[0])
                pinned_freq = str(memo[1] or "")
            elif len(memo) == 3:
                gpu_key = _norm_gpu_key(memo)
        elif isinstance(memo, tuple) and len(memo) == 3:
            gpu_key = _norm_gpu_key(memo)

        if not gpu_key:
            return None, ""

        fmap_here = _rates_for(state, gpu_key) or {}

        # reuse order: job plan -> pinned -> default -> fastest
        freq_out = ""

        plan = state.job_freq_plan.get(jid) or {}
        if isinstance(plan, dict) and tuple(plan.get("gpu", ())) == gpu_key:
            freq_out = _freq_key_match_in_map(fmap_here, str(plan.get("freq", "") or "")) or ""

        if not freq_out and pinned_freq:
            freq_out = _freq_key_match_in_map(fmap_here, pinned_freq) or ""

        if not freq_out:
            df = _defaults_for(state, gpu_key)
            if df in fmap_here:
                freq_out = df

        if not freq_out:
            obj_l = str(getattr(state, "objective", "") or "").lower()
            prefer_min = ("power" in obj_l) or ("eff" in obj_l)
            if freq_mode == "adaptive" and prefer_min:
                freq_out = _pick_min_freq_key(fmap_here)
            else:
                freq_out = _pick_fastest_freq(fmap_here)


        return gpu_key, str(freq_out)

    # ---------- HARD GATE ----------
    should_optimize = (to01(pkt.get("Task_Is_First", 0)) == 1 and to01(pkt.get("Is_First", 0)) == 1)

    # ---------- REUSE ----------
    gpu_key_reuse, freq_reuse = _reuse_job_gpu_and_freq(job_id)
    if gpu_key_reuse:
        c, n, g_id = gpu_key_reuse
        pkt["Assigned_Cluster"]   = c
        pkt["Assigned_Node"]      = n
        pkt["Assigned_GPU"]       = g_id
        pkt["Assigned_Frequency"] = str(freq_reuse)  # planned/pinned
        pkt["Freq_Mode"]          = freq_mode
        pkt["GPU_Decision_Source"]= pkt.get("GPU_Decision_Source") or state.job_gpu_src.get(job_id, "") or "reuse-job"

        # pkt["Planned_Frequency"]  = str(freq_reuse or "")
        # pkt["HW_Frequency"]       = str(_current_hw_freq_key(state, gpu_key_now) or "")
        _refresh_pkt_freq_fields(pkt, gpu_key_reuse, freq_reuse)


        _finalize_provenance(state, pkt, job_id=job_id)
        return gpu_key_reuse, str(freq_reuse)

    # ---------- DEFER until true first packet ----------
    if not should_optimize:
        # We still need an assignment to route/queue the packet correctly.
        # Do NOT run optimizer, but DO make an assignment.
        # Simplest: run optimizer anyway (it will be latched once per job).
        should_optimize = True

    # ---------- NEW JOB: choose GPU + planned freq ----------
    gpu_key_new: Optional[Tuple[str, str, str]] = None
    freq_suggest = ""
    decision_src = ""
    EMPTY_GPU = ("", "", "")

    # store restore (optional)
    if getattr(state, "resume_from_store", False) and getattr(state, "assignment_store", None):
        if state.assignment_store.has(tid_base):
            rec = state.assignment_store.get(tid_base)
            gpu_key_new = _norm_gpu_key((str(rec["Cluster"]), str(rec["Node"]), str(rec["GPU"])))
            freq_suggest = str(rec.get("Frequency", "") or "")
            decision_src = "store_restore"

    if gpu_key_new is None:
        if getattr(state, "optimizer", None):
            chosen_gpu, chosen_freq = state.optimizer.decide(state, pkt)

           # ---- DROP / NO-ASSIGN GUARD (authoritative) ----
            def _is_empty_gpu_key(x) -> bool:
                if x is None:
                    return True
                if not isinstance(x, (tuple, list)) or len(x) < 3:
                    return True
                c, n, g = x[0], x[1], x[2]
                return (str(c).strip() == "" or str(n).strip() == "" or str(g).strip() == "")

            if _is_empty_gpu_key(chosen_gpu):
                pkt["GPU_Decision_Source"] = pkt.get("GPU_Decision_Source", "drop-no-assignment")
                if not isinstance(getattr(state, "dropped_tasks", None), dict):
                    state.dropped_tasks = {}

                state.dropped_tasks[(job_id, task_id)] = pkt.get("GPU_Decision_Source", "drop-no-assignment")
                pkt["Dropped"] = True
                return None, ""

            # -------------------------------------
            gpu_key_new = _norm_gpu_key(chosen_gpu)
            freq_suggest = str(chosen_freq or "")
            decision_src = pkt.get("GPU_Decision_Source", "optimizer") or "optimizer"

        else:
            chosen_gpu, chosen_freq = pick_best_by_least_load(state, pkt)
            # optional: same guard for heuristic path (recommended)
            if (not isinstance(chosen_gpu, (tuple, list))) or len(chosen_gpu) < 3 or \
            (str(chosen_gpu[0]).strip()=="" or str(chosen_gpu[1]).strip()=="" or str(chosen_gpu[2]).strip()==""):
                pkt["GPU_Decision_Source"] = "drop-heuristic-no-gpu"
                if not isinstance(getattr(state, "dropped_tasks", None), dict):
                    state.dropped_tasks = {}

                state.dropped_tasks[(job_id, task_id)] = pkt.get("GPU_Decision_Source", "drop-no-assignment")
                pkt["Dropped"] = True
                return None, ""

            gpu_key_new = _norm_gpu_key(chosen_gpu)
            freq_suggest = str(chosen_freq or "")
            decision_src = "least-load"

    assert gpu_key_new is not None

    fmap_new = _rates_for(state, gpu_key_new) or {}
    desired_key = _freq_key_match_in_map(fmap_new, freq_suggest) or ""

    def _pick_min_freq_key(fmap: Dict[str, Any]) -> str:
        if not fmap:
            return ""
        try:
            return min(map(str, fmap.keys()), key=lambda k: float(k))
        except Exception:
            return sorted(map(str, fmap.keys()))[0]

    # if optimizer didn't give a usable key, fall back
    if not desired_key:
        df = _defaults_for(state, gpu_key_new)
        if df in fmap_new:
            desired_key = df
        else:
            obj_l = str(getattr(state, "objective", "") or "").lower()
            prefer_min = ("power" in obj_l) or ("eff" in obj_l)

            # Only force min-first when you’re actually in adaptive mode
            if freq_mode == "adaptive" and prefer_min:
                desired_key = _pick_min_freq_key(fmap_new)
            else:
                desired_key = _pick_fastest_freq(fmap_new)


    _record_job_pin(job_id, gpu_key_new, desired_key, decision_src)

    # FIXED mode can still force immediately (if you want),
    # but if you truly want ALL DVFS in on_arrive_gpu, remove this too.
    if freq_mode == "fixed":
        fmap_here = _rates_for(state, gpu_key_new) or {}
        fastest = _pick_fastest_freq(fmap_here) or desired_key
        c, n, g_id = gpu_key_new

        pkt["Assigned_Cluster"]   = c
        pkt["Assigned_Node"]      = n
        pkt["Assigned_GPU"]       = g_id
        pkt["Assigned_Frequency"] = fastest
        pkt["Freq_Mode"]          = "fixed"
        pkt["GPU_Decision_Source"]= decision_src or "fixed"

        # pkt["Planned_Frequency"]  = fastest
        # pkt["HW_Frequency"]       = str(_current_hw_freq_key(state, gpu_key_new) or "")
        _refresh_pkt_freq_fields(pkt, gpu_key_new, fastest)
        _emit_opt_line_once(job_id, task_id, gpu_key_new, pkt["GPU_Decision_Source"])
        _finalize_provenance(state, pkt, job_id=job_id)
        return gpu_key_new, fastest

    # ADAPTIVE: plan only, do NOT read hw as assigned
    hw_now = _current_hw_freq_key(state, gpu_key_new) or ""

    c, n, g_id = gpu_key_new
    pkt["Assigned_Cluster"]   = c
    pkt["Assigned_Node"]      = n
    pkt["Assigned_GPU"]       = g_id
    pkt["Assigned_Frequency"] = desired_key      # <-- planned is assigned
    pkt["Freq_Mode"]          = freq_mode
    pkt["GPU_Decision_Source"]= decision_src or "optimizer"

    # pkt["Planned_Frequency"]  = desired_key
    # pkt["HW_Frequency"]       = str(hw_now)
    _refresh_pkt_freq_fields(pkt, gpu_key_new, desired_key)
    _emit_opt_line_once(job_id, task_id, gpu_key_new, pkt["GPU_Decision_Source"])
    _finalize_provenance(state, pkt, job_id=job_id)
    return gpu_key_new, desired_key

def _is_adaptive_mode(mode) -> bool:
    """
    Returns True for any spelling of the adaptive DVFS mode, e.g.
    'adaptive', 'freq-adaptive', 'freqadaptive'.
    """
    m = str(mode or "").lower().replace("_", "-")
    return ("adapt" in m)

def _finalize_ul_row(pkt: dict, t: float) -> None:
    pkt["Direction"] = "Uplink"
    pkt["Is_First"]       = to01(pkt.get("Is_First", 0))
    pkt["Is_Last"]        = to01(pkt.get("Is_Last", 0))
    pkt["Task_Is_First"]  = to01(pkt.get("Task_Is_First", 0))
    pkt["Task_Is_Last"]   = to01(pkt.get("Task_Is_Last", 0))
    start = float(pkt.get("R0_UL_IN_entry", pkt.get("Packet_Arrival_Time", t)))
    end   = float(pkt.get("R0_UL_EG_exit", t))
    pkt["overall_start_time"]     = start
    pkt["overall_end_time"]       = round(end,6)
    pkt["Total_Completion_Time"]  = round(end - start, 6)
    pkt["Task_Deadline_Violation"] = 0
    pkt["Task_Status"] = "UL_Completed"

def dl_gaps_for_task(state: "SimState", n: int, mode: str = "simultaneous"):
    # "simultaneous": all DL packets arrive at gpu_exit
    # "staggered": tiny increasing gaps to mimic serialization
    if n <= 0: return []
    if (mode or "simultaneous").lower() == "staggered":
        # use Python RNG that's already seeded
        gaps = [state.rng.uniform(5e-5, 2.5e-4) for _ in range(n)]
        # cumulative
        s = 0.0; out = []
        for g in gaps: s += g; out.append(s)
        return out
    return [0.0]*n

def on_arrive_r0_ul_in(state: SimState, t: float, pkt: Dict[str, Any]) -> None:
    gpu_key, planned = _ensure_assignment(state, pkt)

    # DROP path: _ensure_assignment decided this task/job is dropped
    if pkt.get("Dropped") is True:
        pkt["Direction"] = "Uplink"
        pkt.setdefault("R0_UL_IN_entry", float(pkt.get("Packet_Arrival_Time", t)))
        state.logs.append(pkt)
        return

    # DEFER path: no assignment yet (not true first packet)
    if gpu_key is None:
        # retry latch (prevents infinite loop if flags are broken)
        tries = int(pkt.get("_ensure_retry", 0) or 0)
        if tries >= 5:
            pkt["Task_Status"] = "Dropped_Defer_Exceeded_R0_UL"
            pkt["Dropped"] = True
            pkt["Direction"] = "Uplink"
            pkt.setdefault("R0_UL_IN_entry", float(pkt.get("Packet_Arrival_Time", t)))
            state.logs.append(pkt)
            return

        pkt["_ensure_retry"] = tries + 1
        # re-inject soon; the real first packet should pin the job quickly
        state.fel.schedule(t + 1e-6, Ev.ARRIVE_R0_UL_IN, {"packet": pkt})
        return

    # from here on, assignment MUST exist
    pkt["Direction"] = "Uplink"

    c = str(pkt.get("Assigned_Cluster", "")).strip()
    if not c:
        pkt["Task_Status"] = "Dropped_No_Assignment_R0_UL"
        pkt["Dropped"] = True
        pkt.setdefault("R0_UL_IN_entry", float(pkt.get("Packet_Arrival_Time", t)))
        state.logs.append(pkt)
        return

    n_ports = int(state.ports_per_cluster.get(c, 1))
    port = int(pkt.get("UL_Port") or pick_port(state, "R0_UL", c, n_ports, lambda p: state.net.ul_time(p)))
    pkt["UL_Port"] = port

    ul_st = get_station(state, "R0_UL", c, port, lambda p: state.net.ul_time(p))

    pkt.setdefault("R0_UL_IN_entry", float(pkt.get("Packet_Arrival_Time", t)))

    if not ul_st.busy:
        ul_st.busy = True
        pkt["R0_UL_service_start"] = round(t, 6)
        pkt["R0_UL_queue_delay"]   = round(t - float(pkt["R0_UL_IN_entry"]), 6)

        svc = float(ul_st.service_time_fn(pkt))
        pkt["R0_UL_service_time"]  = round(svc, 6)

        state.fel.schedule(t + svc, Ev.DEPART_R0_UL_EG, {"packet": pkt})
        ul_st.tail_time = t + svc
    else:
        if ul_st.capacity and (len(ul_st.q) >= ul_st.capacity):
            pkt["Task_Status"] = "Dropped_Q_Full_R0_UL"
            pkt.setdefault("R0_UL_IN_entry", float(pkt.get("Packet_Arrival_Time", t)))
            pkt["overall_start_time"] = float(pkt["R0_UL_IN_entry"])
            pkt["overall_end_time"]   = float(t)
            pkt["Total_Completion_Time"] = 0.0
            pkt["Task_Deadline_Violation"] = 1
            _stamp_assignment_for_drop(state, pkt)
            state.logs.append(pkt)
            return

        enqueue_fcfs(ul_st, pkt, "R0_UL_IN_entry", t)

def on_depart_r0_ul_eg(state: SimState, t: float, pkt: Dict[str, Any]):
    c = str(pkt["Assigned_Cluster"])
    port = int(pkt.get("UL_Port", 1))

    ul_st = state.stations[st_name("R0_UL", c, port)]
    pkt["R0_UL_EG_exit"] = round(t,6)

    # Stamp start ONLY from the first UL packet (P0) and only once.
    if to01(pkt.get("Is_First", 0)) == 1 or str(pkt.get("Packet_id","")).endswith("_P0"):
        k = (_norm_job_id(pkt), base_tid(str(pkt["Task_ID"])))
        times = state.task_times.setdefault(k, {})
        if "start" not in times:
            times["start"] = float(pkt["R0_UL_IN_entry"])

    _finalize_ul_row(pkt, t)

    if ul_st.q:
        nxt = ul_st.q.popleft()
        _q_update(ul_st, t, len(ul_st.q))
        nxt["R0_UL_service_start"] = t
        nxt["R0_UL_queue_delay"] = round(t - float(nxt["R0_UL_IN_entry"]), 6)
        svc = ul_st.service_time_fn(nxt)
        nxt["R0_UL_service_time"] = round(float(svc), 6)
        state.fel.schedule(t + svc, Ev.DEPART_R0_UL_EG, {"packet": nxt})
        ul_st.busy = True
        ul_st.tail_time = t + svc
    else:
        ul_st.busy = False

    d = state.net.ul_prop(pkt)
    pkt["R0_UL_prop_delay"] = d
    if getattr(state.net, "fast_rc_ul", False):
        state.fel.schedule(t + d, Ev.ARRIVE_GPU, {"packet": pkt})
    else:
        state.fel.schedule(t + d, Ev.ARRIVE_RC_UL_IN, {"packet": pkt})

def on_arrive_rc_ul_in(state: SimState, t: float, pkt: Dict[str, Any]):
    """
    RC_UL is modelled as a free / wire-speed hop.

    We only stamp timing fields for completeness and forward
    directly to the GPU stage.
    """
    # Optional: keep a port number for logging, but it has no effect
    c = str(pkt.get("Assigned_Cluster", ""))
    n_ports = int(state.ports_per_cluster.get(c, 1))
    pkt["RC_UL_Port"] = int(
        pkt.get("RC_UL_Port") or pick_port(state, "RC_UL", c, n_ports, lambda p: 0.0)
    )

    # Zero-delay RC_UL
    now = round(t, 5)
    pkt["RC_UL_IN_entry"] = now
    pkt["RC_UL_IN_exit"]  = now
    pkt["RC_UL_IN_delay"] = 0.0
    pkt["RC_UL_service_start"] = now
    pkt["RC_UL_service_time"]  = 0.0

    # Immediately continue to GPU
    state.fel.schedule(t, Ev.ARRIVE_GPU, {"packet": pkt})

def _objective_flags(state) -> tuple[bool, bool, bool]:
    """
    Returns: (is_latency_obj, is_power_obj, is_eff_obj)
    """
    obj = str(getattr(state, "objective", "")).lower()
    is_latency_obj = ("latency" in obj)          # latency / min-latency / etc.
    is_power_obj   = ("power" in obj)        # power / min-power
    is_eff_obj     = ("efficiency" in obj)          # efficiency / max-efficiency
    return is_latency_obj, is_power_obj, is_eff_obj

def _resolve_service_freq(state, gkey, fmap, job_id=None):
    """
    Return a concrete frequency key for this GPU.

    ADAPTIVE:
      fixed-lock > job plan > "" (do NOT invent)
    NON-ADAPTIVE:
      fixed-lock > job plan > active HW > catalog default > fastest step
    """
    freq_mode_now = str(getattr(state, "freq_mode", "adaptive") or "adaptive").lower()

    # 1) fixed lock
    fixed_map = getattr(state, "gpu_freq_plan_fixed", {}) or {}
    fixed = None
    try:
        fixed = (fixed_map.get(_norm_gkey(gkey)) or fixed_map.get(gkey) or (None, ""))[0]
    except Exception:
        fixed = None
    if fixed:
        return str(fixed)

    # 2) per-job optimizer plan
    if job_id is not None:
        plan = (getattr(state, "job_freq_plan", {}) or {}).get(str(job_id))
        if plan:
            if isinstance(plan, (tuple, list)) and len(plan) >= 2:
                # allow either (gpu_key, freq, ...) or (freq, ...)
                if len(plan) >= 2 and isinstance(plan[0], (tuple, list)) and len(plan[0]) == 3:
                    if tuple(map(str, plan[0])) == tuple(map(str, gkey)) and plan[1]:
                        return str(plan[1])
                elif plan[0] and not isinstance(plan[0], (tuple, list, dict)):
                    # defensive: if someone stored (freq, reason)
                    return str(plan[0])
            elif isinstance(plan, dict):
                pf = str(plan.get("freq", "") or plan.get("Frequency", "") or "")
                if pf:
                    return pf

        if freq_mode_now == "adaptive":
            return ""  # ADAPTIVE: no plan -> don't invent

    # NON-ADAPTIVE below
    ng = _norm_gkey(gkey)

    # 3) current active HW freq
    active = (getattr(state, "gpu_to_freq", {}) or {}).get(ng, "") or (getattr(state, "gpu_to_freq", {}) or {}).get(gkey, "")
    if active:
        return _freq_key_match_in_map(fmap, str(active)) or str(active)


    # 4) catalog default (safe)
    cat = getattr(state, "gpu_catalog", None)
    defaults_map = getattr(cat, "defaults", {}) or {}
    default = defaults_map.get(tuple(map(str, gkey)), defaults_map.get(ng, ""))
    if default:
        return str(default)

    # 5) fastest in fmap (robust)
    if fmap:
        # prefer by service rate
        try:
            return max(fmap.items(), key=lambda kv: float(kv[1]))[0]
        except Exception:
            # fallback by numeric key
            def _knum(k):
                try:
                    return float(str(k))
                except Exception:
                    return -1.0
            try:
                return max(fmap.keys(), key=_knum)
            except Exception:
                # final deterministic fallback
                try:
                    return sorted(map(str, fmap.keys()))[-1]
                except Exception:
                    return ""

    return ""


def on_arrive_gpu(state: "SimState", t: float, pkt: Dict[str, Any]) -> None:
    """
    UL packet hits the GPU stage.
    Aggregate per (Job_ID, Task_ID); when last UL packet arrives, form the task
    and either dispatch immediately (if GPU idle) or enqueue (if busy).

    IMPORTANT:
    - We treat work at TASK granularity (Task_ID).
    - We log [RUN] exactly once per task *when that task actually begins service*.
    """

    # ---------- aggregation of UL packets per task ----------
    key = (str(pkt["Job_ID"]), str(pkt["Task_ID"]))
    info = state.agg.setdefault(key, {"packets": []})
    info["packets"].append(pkt)

    # only proceed when we've received the last packet of that task
    is_last = (to01(pkt.get("Is_Last")) == 1) or (to01(pkt.get("Task_Is_Last")) == 1)
    if not is_last:
        return
    
    # --- release reservation latch (safe): compute ids BEFORE discard ---
    try:
        job_id  = str(_norm_job_id(pkt))                 # same normalization as decide()
    except Exception:
        job_id  = str(pkt.get("Job_ID", ""))

    task_id = str(pkt.get("Task_ID", ""))                # exact Task_ID (NOT base_tid)

    # # discard only if latch exists
    # try:
    #     if hasattr(state, "reserved_tasks") and state.reserved_tasks is not None:
    #         state.reserved_tasks.discard((job_id, task_id))
    # except Exception:
    #     pass

    
    # ---------- idempotence guard: prevent double-finalization ----------
    if info.get("_finalized", False):
        return
    info["_finalized"] = True

    # ---------- resolve GPU key ----------
    c = str(pkt["Assigned_Cluster"])
    n = str(pkt["Assigned_Node"])
    g = str(pkt["Assigned_GPU"])
    gkey  = (c, n, g)
    ngkey = _norm_gkey(gkey)
    gname = f"GPU:{c}:{n}:{g}"

    _dbg(
        state,
        f"[ARRIVE_GPU] t={t:.6f} GPU={gkey} Task={pkt['Task_ID']} "
        f"Job={pkt['Job_ID']} (last UL pkt)"
    )

    # Assigned_Frequency: fallback to current hw freq
    assigned_freq = str(
        pkt.get("Assigned_Frequency", "") or
        (getattr(state, "gpu_to_freq", {}) or {}).get(ngkey, "")
    )

    first_pkt  = info["packets"][0] if info["packets"] else {}
    work_flops = float(first_pkt.get("Workload_FLOPs", 0))

    # ---------- build task object ----------
    task = {
        "Job_ID":             pkt["Job_ID"],
        "Task_ID":            pkt["Task_ID"],
        "Packets":            list(info["packets"]),
        "Assigned_Cluster":   c,
        "Assigned_Node":      n,
        "Assigned_GPU":       g,
        "Assigned_Frequency": assigned_freq,
        "Workload_FLOPs":     work_flops,
        "Task_DL_Total_kB":   int(pkt.get("Task_DL_Total_kB", 0)),
        "gpu_ready_time":     t,
    }

    # inherit provenance if missing
    for k in ("Freq_Mode", "Freq_Decision_Source", "GPU_Decision_Source"):
        if not task.get(k):
            task[k] = first_pkt.get(k, "")

    # ------------------------------------------------------------
    # Planned_Frequency MUST be stored for BOTH dispatched AND queued tasks
    # (so boundary DVFS (finish->next) can use it reliably)
    # ------------------------------------------------------------
    try:
        fmap_tmp = _rates_for(state, gkey)  # gkey is (c,n,g) tuple
        planned_raw = str(first_pkt.get("Planned_Frequency", "") or "")
        planned_key = _freq_key_match_in_map(fmap_tmp, planned_raw) if planned_raw else ""
        task["Planned_Frequency"] = planned_key
    except Exception:
        task["Planned_Frequency"] = ""
    
    # clear aggregation bucket
    del state.agg[key]

    # # ---------- FLOP backlog: DVFS-safe queue metric ----------
    # if not hasattr(state, "flop_backlog") or state.flop_backlog is None:
    #     state.flop_backlog = {}      # (C,N,G) -> FLOPs
    # state.flop_backlog[ngkey] = float(state.flop_backlog.get(ngkey, 0.0)) + float(work_flops)

    # ------------------------------------------------------------------
    # INFLOW (λ) ACCOUNTING – only for the least-load heuristic strategy
    # ------------------------------------------------------------------
    # keep flops_inflow_per_gpu consistent with backlog
    try:
        if getattr(state, "dvfs_controller", "heuristic") == "heuristic":
            heur = str(getattr(state, "heuristic",
                        getattr(state, "scheduling_strategy", ""))).lower()
            if heur in ("least-load", "leastload", "load"):
                update_flops_inflow(state)
    except Exception:
        pass

    # ---------- ensure Station exists ----------
    st = state.stations.get(gname)
    if st is None:
        MIN_SVC = float(
            ((getattr(state, "config", {}) or {}).get("Optimizer_Defaults", {}) or {})
            .get("min_gpu_service_time_s", 0.0)
        )

        def gpu_svc(p: Dict[str, Any]) -> float:
            rate = float(get_gpu_rate(state, p))
            work = int(p.get("Workload_FLOPs", 0))
            p["service_rate"] = rate
            svc = (work / rate) if rate > 0.0 else 1e9
            if MIN_SVC > 0.0:
                svc = max(MIN_SVC, svc)
            return float(svc)

        st = Station(name=gname, service_time_fn=gpu_svc)
        state.stations[gname] = st
        _dbg(state, f"[ARRIVE_GPU] init Station for {gkey}")

    if not hasattr(st, "q"):
        st.q = deque()
    if not hasattr(st, "busy"):
        st.busy = False
    if not hasattr(st, "jobs_present"):
        st.jobs_present = Counter()
    if not hasattr(st, "current_job_for_freq"):
        st.current_job_for_freq = None
    if not hasattr(st, "tail_time"):
        st.tail_time = t

    # Ensure DVFS epoch bookkeeping exists
    if not hasattr(state, "_dvfs_epoch") or not isinstance(state._dvfs_epoch, dict):
        state._dvfs_epoch = {}
    if not hasattr(state, "_dvfs_epoch_applied") or not isinstance(state._dvfs_epoch_applied, dict):
        state._dvfs_epoch_applied = {}
    state._dvfs_epoch.setdefault(ngkey, 0)
    state._dvfs_epoch_applied.setdefault(ngkey, -1)

    if getattr(state, "debug_dvfs", False):
        _dbg_dump_gpu_catalog(state, gkey, tag="DVFS")

    # ---------- residency accounting ----------

    freq_mode_now = getattr(state, "freq_mode", "adaptive")
    is_adaptive   = _is_adaptive_mode(freq_mode_now)

    job_id = str(task["Job_ID"])
    st.jobs_present[job_id] = st.jobs_present.get(job_id, 0) + 1

    was_idle = (not st.busy)  

    if was_idle and st.jobs_present[job_id] == 1:
        # real busy-period start: GPU was idle and a new job arrives
        if is_adaptive:
            _dbg(
                state,
                f"[ARRIVE_GPU] Job {job_id} became resident on {gkey} "
                f"(jobs_present=1) -> bump dvfs epoch"
            )
            _dvfs_epoch_bump(state, ngkey)
        else:
            _dbg(
                state,
                f"[ARRIVE_GPU] Job {job_id} became resident on {gkey} "
                f"(jobs_present=1) [fixed-mode: no epoch bump]"
            )
    else:
        _dbg(
            state,
            f"[ARRIVE_GPU] Job {job_id} now has {st.jobs_present[job_id]} "
            f"active task(s) on {gkey}"
        )



    # ============================================================
    # CASE 1: GPU is idle → DISPATCH NOW (start of busy period)
    # ============================================================
    if not st.busy:
        st.busy = True

        prev_owner = st.current_job_for_freq
        new_owner  = job_id
        st.current_job_for_freq = new_owner
        # DO NOT lock before DVFS
        st.in_service_job = None

        _dbg(
            state,
            f"[DISPATCH@ARRIVE] GPU={gkey} was idle. "
            f"Dispatch Task={task['Task_ID']} Job={job_id}. "
            f"Owner change {prev_owner or 'None'} -> {new_owner}"
        )

        # entry timestamps
        task["gpu_entry_time"]   = round(t, 6)
        task["gpu_queue_delay"]  = 0.0

        base_key = (str(pkt["Job_ID"]), base_tid(str(pkt["Task_ID"])))
        tt = state.task_times.setdefault(base_key, {})
        tt["gpu_entry"] = min(tt.get("gpu_entry", t), float(t))

        # ------------------ DVFS at busy-period start ------------------
        # ------------------ decide desired frequency (PLAN) ------------------
        fmap = _rates_for(state, gkey)

        hw_cur  = (getattr(state, "gpu_to_freq", {}) or {}).get(ngkey, "")
        cur_key = _freq_key_match_in_map(fmap, hw_cur) or hw_cur or ""

        planned_key = str(task.get("Planned_Frequency", "") or "")

        is_least_load = (
            getattr(state, "is_least_load", False)
            and getattr(state, "dvfs_controller", "") == "heuristic"
        )

        if is_least_load:
            desired_raw = choose_freq_for(state, gkey, task)
            desired_key = _freq_key_match_in_map(fmap, desired_raw) if desired_raw else ""
            c, n, g = gkey
            print(f"[LL] {task['Task_ID']} -> {c}/{n}/{g} @ {desired_key} (least-load)")
        else:
            desired_key = planned_key
            if not desired_key:
                desired_raw = _resolve_service_freq(state, gkey, fmap, job_id=new_owner)
                desired_key = _freq_key_match_in_map(fmap, desired_raw) if desired_raw else ""

        # -------------------------------------------------------
        # DVFS PLACE (A): idle → right before service starts
        # (NO OTHER DVFS CALLS IN on_arrive_gpu)
        # -------------------------------------------------------
        st.in_service_job = None  # boundary open
        if desired_key:
            apply_boundary_upshift_only(
                state, gkey, desired_key,
                t=t, fmap=fmap,
                trigger_job_id=job_id,
                trigger_task_id=task["Task_ID"],
                reason="idle_to_start",
            )

        st.in_service_job = job_id  # service starts now

                 

        # stamp the true hardware freq after any DVFS we just did
        fmap       = _rates_for(state, gkey)
        hw_key_now = (getattr(state, "gpu_to_freq", {}) or {}).get(ngkey, "")
        hw_norm    = _freq_key_match_in_map(fmap, hw_key_now) or hw_key_now or "-"

        if not hw_norm or hw_norm == "-":
            try:
                hw_norm = max(fmap, key=lambda fk: float(fk))
            except Exception:
                try:
                    hw_norm = max(fmap, key=lambda fk: float(fmap[fk]))
                except Exception:
                    hw_norm = hw_key_now or "-"
        task["Service_Frequency"] = hw_norm
        # print(f"[GPU] {task['Task_ID']} start on {c}/{n}/{g} "
        #     f"planned={task.get('Planned_Frequency','')} "
        #     f"svc={task.get('Service_Frequency','')} "
        #     f"hw={hw_key_now}")


        _dbg(
            state,
            f"[DISPATCH@ARRIVE] final Service_Frequency={task['Service_Frequency']} "
            f"(hw={hw_key_now}) mode={freq_mode_now}"
        )

        # utilization EWMA: GPU busy now
        _util_touch(
            state, gkey,
            now=float(t),
            tau_s=float(getattr(state, "util_tau_s", 0.5)),
            busy_flag=1,
        )

        # compute service time and schedule finish
        svc = float(st.service_time_fn(task))
        if svc <= 0.0 or not math.isfinite(svc):
            svc = 1e-6
        task["gpu_service_time"] = round(svc, 6)
        tt["gpu_service"] = max(
            float(tt.get("gpu_service", 0.0)),
            float(task["gpu_service_time"])
        )

        _dbg(state, f"[SVC] {gkey} Job={job_id} Task={task['Task_ID']} "
             f"freq_mode={freq_mode_now} "
             f"Service_Frequency={task['Service_Frequency']} "
             f"Assigned_Frequency={task.get('Assigned_Frequency')} "
             f"rate={get_gpu_rate(state, task)}")


        finish_t = t + svc
        _dbg(
            state,
            f"[DISPATCH@ARRIVE] SVC={svc:.6f}s FINISH@{finish_t:.6f} "
            f"Task={task['Task_ID']} Job={job_id} GPU={gkey}"
        )
        state.fel.schedule(finish_t, Ev.FINISH_GPU, {"task": task, "st": gname})
        st.tail_time = finish_t

        # load + active_jobs bookkeeping – ONLY when service starts
        state.load_by_gpu = getattr(state, "load_by_gpu", {})
        state.load_by_gpu[gkey] = float(state.load_by_gpu.get(gkey, 0.0)) + float(svc)

        state.active_jobs = getattr(
            state,
            "active_jobs",
            __import__("collections").defaultdict(set)
        )
        state.active_jobs[gkey].add(job_id)
        state.dvfs_started = True

        # [RUN] LOGGING
        try:
            first_flag = int(first_pkt.get("Is_First", 0))
        except Exception:
            first_flag = 0
        if first_flag == 1 and getattr(state, "debug_run_log", False):
            print(
                f"[RUN] {task['Task_ID']} servicing on {c}/{n}/{g} "
                f"@ {task['Service_Frequency']}"
            )
        return

    # ============================================================
    # CASE 2: GPU busy → ENQUEUE (NO DVFS CHANGES HERE)
    # ============================================================
    task["gpu_queue_enqueue_time"] = t
    st.q.append(task)
    _q_update(st, t, len(st.q))

    _dbg(
        state,
        f"[ENQUEUE] GPU={gkey} Task={task['Task_ID']} Job={job_id} "
        f"q_len={len(st.q)} tail_time={getattr(st,'tail_time',None)}"
    )

    # NOTE: we no longer touch load_by_gpu here.
    # load_by_gpu is "claimed svc time of *running* tasks only";
    # queued work is represented via flop_backlog[gkey].

    state.active_jobs = getattr(
        state,
        "active_jobs",
        __import__("collections").defaultdict(set)
    )
    state.active_jobs[gkey].add(job_id)


def on_finish_gpu(state: "SimState", t: float, payload: Dict[str, Any]):
    """
    Called when the GPU completes service of `task`.
    - finalize timing
    - update jobs_present / utilization
    - optionally upshift once if next job wants higher freq
    - dispatch next task
    - schedule DL traffic
    """

    task  = payload["task"]
    gname = payload["st"]
    st    = state.stations[gname]

    c = str(task.get("Assigned_Cluster"))
    n = str(task.get("Assigned_Node"))
    g = str(task.get("Assigned_GPU"))
    key3 = _norm_gkey((c, n, g))  # ('C1','N2','G1')

    finished_job_id  = str(task.get("Job_ID"))
    finished_task_id = str(task.get("Task_ID"))

    task["gpu_exit_time"] = round(t, 6)

    _dbg(
        state,
        f"[FINISH_GPU] t={t:.6f} GPU={key3} "
        f"Task={finished_task_id} Job={finished_job_id} done"
    )
    # release lock at finish boundary so boundary DVFS can run
    st.in_service_job = None


    # ---------- load accounting: subtract used service time ----------
    try:
        used = float(task.get("gpu_service_time", 0.0))
        state.load_by_gpu = getattr(state, "load_by_gpu", {})
        prev_used = float(state.load_by_gpu.get(key3, 0.0))
        state.load_by_gpu[key3] = max(0.0, prev_used - used)
    except Exception:
        pass

    # ---------- FLOP backlog: remove this task's work ----------
    try:
        work_flops = float(task.get("Workload_FLOPs", 0.0) or 0.0)
        fb = getattr(state, "flop_backlog", None)
        if fb is None:
            fb = {}
        # ensure dict-like
        if not isinstance(fb, dict):
            fb = dict(fb)
        # # subtract from normalized key first
        # prev = float(fb.get(key3, 0.0) or 0.0)
        # fb[key3] = max(0.0, prev - work_flops)
        k_raw = (c, n, g)
        if k_raw != key3 and k_raw in fb:
            fb[k_raw] = max(0.0, float(fb.get(k_raw,0.0)) - work_flops)

        state.flop_backlog = fb
    except Exception:
        pass

    try:
        if getattr(state, "reserved_tasks", None) is not None:
            tid0 = str(task.get("Task_ID",""))
            parts = tid0.split("_")
            J_alt = "_".join(parts[:2]) if len(parts) >= 2 else tid0
            state.reserved_tasks.discard((J_alt, finished_task_id))
    except Exception:
        pass

    # ---------- helpers (UL-only / DL synth) ----------
    def finalize_ul_only(base_key_local: tuple):
        times = state.task_times.setdefault(base_key_local, {})
        t0 = float(times.get("start", task.get("gpu_ready_time", t)))
        t1 = float(t)
        for r in state.rows_by_task.get(base_key_local, []):
            r["overall_start_time"]    = t0
            r["overall_end_time"]      = t1
            r["Total_Completion_Time"] = t1 - t0

    def _synthesize_dl_rows(task_local, base_key_local):
        job_id_local, task_base = base_key_local
        total_kb = float(task_local.get("DL_Total_kB",
                                        task_local.get("Task_DL_Total_kB", 0.0)))
        if total_kb <= 0:
            return []

        cfg_network = getattr(state, "cfg", {}).get("Network", {}) if hasattr(state, "cfg") else {}
        pkt_kb = float(task_local.get("DL_Packet_kB", 0.0)) or float(cfg_network.get("DL_Packet_kB", 0.0))
        n_pkts = int(task_local.get("DL_Num_Packets")
                     or (math.ceil(total_kb / pkt_kb) if pkt_kb > 0 else 1))
        n_pkts = max(1, n_pkts)
        base_sz = total_kb / n_pkts

        rows = []
        for i in range(1, n_pkts + 1):
            rows.append({
                "Direction": "Downlink",
                "Job_ID": str(task_local.get("Job_ID")),
                "Task_ID": f"{task_base}_D",
                "Packet_id": f"{task_base}_D_P{i}",
                "Packet_Size_KB": base_sz,
                "Task_Deadline": task_local.get("Task_Deadline"),

                "Assigned_Cluster":     task_local.get("Assigned_Cluster"),
                "Assigned_Node":        task_local.get("Assigned_Node"),
                "Assigned_GPU":         task_local.get("Assigned_GPU"),
                "Assigned_Frequency":   task_local.get("Assigned_Frequency"),
                "Service_Frequency":    task_local.get("Service_Frequency"),
                "Freq_Mode":            task_local.get("Freq_Mode", ""),
                "Freq_Decision_Source": task_local.get("Freq_Decision_Source", ""),
                "GPU_Decision_Source":  task_local.get("GPU_Decision_Source", ""),

                "gpu_entry_time":   "",
                "gpu_service_time": "",
                "gpu_exit_time":    "",
                "gpu_queue_delay":  "",
            })
        task_rows_local = state.rows_by_task.setdefault(base_key_local, [])
        task_rows_local.extend(rows)
        if hasattr(state, "all_rows"):
            state.all_rows.extend(rows)
        return rows

    # ---------- residency accounting ----------
    if not hasattr(st, "jobs_present"):
        st.jobs_present = Counter()
    if finished_job_id in st.jobs_present:
        st.jobs_present[finished_job_id] -= 1
        if st.jobs_present[finished_job_id] <= 0:
            _dbg(
                state,
                f"[FINISH_GPU] Job {finished_job_id} no longer has active tasks on {key3} "
                f"(jobs_present=0 → removing)"
            )
            del st.jobs_present[finished_job_id]
        else:
            _dbg(
                state,
                f"[FINISH_GPU] Job {finished_job_id} still resident on {key3}: "
                f"{st.jobs_present[finished_job_id]} task(s) remain"
            )
    else:
        _dbg(
            state,
            f"[FINISH_GPU] WARN: Job {finished_job_id} not in jobs_present on {key3}"
        )

    # utilization touch @finish → mark GPU momentarily idle at this instant
    _util_touch(
        state, key3,
        now=float(t),
        tau_s=float(getattr(state, "util_tau_s", 0.5)),
        busy_flag=0,
    )

    # ================== DISPATCH NEXT TASK ==================
    if getattr(st, "q", None):
        # after finishing current task
        st.in_service_job = None   # release at boundary
        nxt = st.q.popleft()
        _q_update(st, t, len(st.q))
        st.busy = True

        next_job_id = str(nxt.get("Job_ID", ""))

        # freq_mode_raw = str(getattr(state, "freq_mode", "adaptive"))
        # if _is_adaptive_mode(freq_mode_raw):
        #     _dvfs_epoch_bump(state, ngkey)

        prev_owner = getattr(st, "current_job_for_freq", None)
        st.current_job_for_freq = next_job_id
       
        # # update lock holder to the next job
        # st.in_service_job = next_job_id

        # ==========================================================
        # BOUNDARY DVFS (FINISH -> NEXT DISPATCH)
        # Applies optimizer/planned freq for nxt so hw tracks planned.
        # - latency: upshift-only
        # - power/eff: allow up or down
        # ==========================================================
        freq_mode_raw = str(getattr(state, "freq_mode", "adaptive"))
        is_adaptive2 = _is_adaptive_mode(freq_mode_raw)

        fmap2      = _rates_for(state, key3)
        ngkey = _norm_gkey(key3)
        hw_cur2    = (getattr(state, "gpu_to_freq", {}) or {}).get(ngkey, "")
        cur_key2   = _freq_key_match_in_map(fmap2, hw_cur2) or hw_cur2 or ""

        # prefer per-task planned key (stored at arrive)
        planned2 = str(nxt.get("Planned_Frequency", "") or "")
        planned2 = _freq_key_match_in_map(fmap2, planned2) if planned2 else ""

        desired2 = planned2
        if not desired2:
            desired_raw2 = _resolve_service_freq(state, key3, fmap2, job_id=next_job_id)
            desired2 = _freq_key_match_in_map(fmap2, desired_raw2) if desired_raw2 else ""

        def _rate2(k: str) -> float:
            try:
                return float(fmap2.get(k, -1.0))
            except Exception:
                return -1.0

        # -------------------------------------------------------
        # DVFS PLACE (B): finish → next start
        # (NO OTHER DVFS CALLS IN on_finish_gpu boundary dispatch)
        # -------------------------------------------------------
        st.in_service_job = None  # boundary open
        if desired2:
            apply_boundary_upshift_only(
                state, key3, desired2,
                t=t, fmap=fmap2,
                trigger_job_id=next_job_id,
                trigger_task_id=str(nxt.get("Task_ID", "")),
                reason="finish_to_next_start",
            )

        # NOW service truly begins
        st.in_service_job = next_job_id

        _dbg(
            state,
            f"[DISPATCH@FINISH] GPU={key3} "
            f"Next Task={nxt.get('Task_ID')} Job={next_job_id} "
            f"Owner change {prev_owner or 'None'} -> {next_job_id} "
            f"q_len_after_pop={len(st.q)}"
        )

        # timing stamps
        nxt["gpu_entry_time"] = round(t, 6)
        enq = float(nxt.get("gpu_queue_enqueue_time", t))
        nxt["gpu_queue_delay"] = round(t - enq, 6)

        base_key_nxt = (str(nxt.get("Job_ID")), base_tid(str(nxt.get("Task_ID"))))
        tt_nxt = state.task_times.setdefault(base_key_nxt, {})
        tt_nxt["gpu_entry"] = min(tt_nxt.get("gpu_entry", t), float(t))
        
        # # NOW the next job truly owns the GPU (lock-holder)
        # st.in_service_job = next_job_id

        # # ensure jobs_present entry exists (no increment here; arrive_gpu already did it)
        # st.jobs_present[next_job_id] = st.jobs_present.get(next_job_id, 0)
        _dbg(
            state,
            f"[DISPATCH@FINISH] jobs_present[{next_job_id}] now "
            f"{st.jobs_present[next_job_id]} on {key3}"
        )


        freq_mode_raw = str(getattr(state, "freq_mode", "adaptive"))
        freq_mode_now = freq_mode_raw.lower()
        fmap2      = _rates_for(state, key3)
        # After any freq actions, read actual HW freq and stamp for nxt
        hw_key_now = (getattr(state, "gpu_to_freq", {}) or {}).get(ngkey, "")
        hw_norm    = _freq_key_match_in_map(fmap2, hw_key_now) or hw_key_now or "-"
        nxt["Service_Frequency"] = hw_norm

        _dbg(
            state,
            f"[DISPATCH@FINISH] final Service_Frequency={nxt['Service_Frequency']} "
            f"(hw={hw_key_now}) mode={freq_mode_now}"
        )

        # GPU goes busy for nxt
        _util_touch(
            state, key3,
            now=float(t),
            tau_s=float(getattr(state, "util_tau_s", 0.5)),
            busy_flag=1,
        )

        # compute service time for nxt under the (possibly) updated freq
        svc = float(st.service_time_fn(nxt))
        if svc <= 0.0 or not math.isfinite(svc):
            svc = 1e-6
        nxt["gpu_service_time"] = round(svc, 6)
        tt_nxt["gpu_service"] = max(float(tt_nxt.get("gpu_service", 0.0)), svc)

        finish_next = t + svc
        _dbg(
            state,
            f"[DISPATCH@FINISH] SVC={svc:.6f}s FINISH@{finish_next:.6f} "
            f"Task={nxt.get('Task_ID')} Job={next_job_id} GPU={key3}"
        )
        state.fel.schedule(finish_next, Ev.FINISH_GPU, {"task": nxt, "st": gname})
        st.tail_time = finish_next

        # track active_jobs
        state.active_jobs = getattr(
            state,
            "active_jobs",
            __import__("collections").defaultdict(set)
        )
        state.active_jobs[key3].add(next_job_id)

    else:
        # GPU idle (queue empty)
        _dbg(state, f"[DISPATCH@FINISH] GPU={key3} idle (no queued tasks)")
        st.busy = False
        st.current_job_for_freq = None
        st.in_service_job = None
        # IMPORTANT: no idle DVFS retunes.
        # policy: upshift-only, and only when a new job arrives.


        # # ---- DOWNSHIFT STARTS - (NOT NEEDED - COMMENT FROM HERE) --------
        # # ==========================================================
        # # OPTIONAL IDLE DOWNSHIFT
        # # Only for power/eff objectives. Never for latency.
        # # ==========================================================
        # is_latency_obj, is_power_obj, is_eff_obj = _objective_flags(state)
        # freq_mode_now = str(getattr(state, "freq_mode", "adaptive"))

        # if _is_adaptive_mode(freq_mode_now) and (is_power_obj or is_eff_obj) and (not is_latency_obj):
        #     fmap_idle = _rates_for(state, gkey)

        #     # lowest perf bin = min service rate
        #     try:
        #         min_key = min(fmap_idle, key=lambda k: float(fmap_idle[k]))
        #     except Exception:
        #         min_key = next(iter(fmap_idle), "")

        #     hw_idle = (getattr(state, "gpu_to_freq", {}) or {}).get(ngkey, "")
        #     cur_idle = _freq_key_match_in_map(fmap_idle, hw_idle) or hw_idle or ""

        #     if min_key and min_key != cur_idle:
        #         safe_set_gpu_freq(
        #             state,
        #             gkey,
        #             min_key,
        #             reason="idle_downshift",
        #             fmap=fmap_idle,
        #             when=t,
        #             force_log=True,
        #             override_lock=True,   # must override since no owner
        #             trigger_job_id=str(finished_job_id),
        #             trigger_task_id=str(finished_task_id),
        #         )
        # # ---- DOWNSHIFT ENDS (NOT NEEDED - COMMENT TILL HERE) ---------

        # # >>> APPLY PENDING DVFS CHANGE AT IDLE (one window per frequency) <<<
        # # Only apply gpu_freq_pending for heuristic controller (least-load path)
        # if getattr(state, "dvfs_controller", "heuristic") == "heuristic":
        #     pend = getattr(state, "gpu_freq_pending", {}) or {}
        #     want = pend.get(gkey)

        #     # confirm idle
        #     is_idle_now = (not getattr(st, "busy", False)) and (len(getattr(st, "q", [])) == 0)

        #     if want and is_idle_now:
        #         catalog = getattr(state, "gpu_catalog", None)
        #         rates   = (catalog.rates if catalog else {}) or {}
        #         fmap_all = getattr(state.gpu_catalog, "rates", {}) if catalog else {}
        #         fmap_g   = fmap_all.get(gkey, {})

        #         fb = _freq_bins(state, gkey)
                

        #         cur_hw = _gpu_hw_freq_key(state, gkey, fmap=fmap_g)

        #         def _as_idx(freq_key: str) -> int:
        #             try:
        #                 return fb.index(freq_key)
        #             except ValueError:
        #                 return 0

        #         idx_cur  = _as_idx(cur_hw)
        #         idx_want = _as_idx(str(want))

        #         if idx_want < idx_cur:
        #             _dbg(
        #                 state,
        #                 f"[DVFS-pending-idle] skip downshift {gkey}: "
        #                 f"{cur_hw}->{want} at t={t:.6f}"
        #             )
        #         elif idx_want == idx_cur:
        #             # same bin → just extend existing window
        #             win_map = getattr(state, "dvfs_active_windows", {}) or {}
        #             win = win_map.get(_norm_gkey(gkey))
        #             if isinstance(win, dict):
        #                 win["end"] = float(t)
        #         else:
        #             fmap = fmap_g or (rates.get(gkey) or {})
        #             try:
        #                 safe_set_gpu_freq(
        #                     state,
        #                     gkey,
        #                     want,
        #                     reason="heuristic-idle-apply",
        #                     fmap=fmap,
        #                     when=float(t),
        #                     trigger_job_id="-",
        #                     trigger_task_id="-",
        #                 )
        #                 _dbg(
        #                     state,
        #                     f"[DVFS-idle-apply] GPU={gkey} {cur_hw or '-'} "
        #                     f"-> {want} at t={t:.6f}"
        #                 )
        #             except Exception:
        #                 pass

        #         # clear pending
        #         try:
        #             del pend[gkey]
        #         except KeyError:
        #             pass
        #         state.gpu_freq_pending = pend



    # ================== LOG / DOWNLINK (unchanged) ==================
    pkts = task["Packets"]
    # base_key = (str(pkts[0]["Job_ID"]), base_tid(str(pkts[0]["Task_ID"])))
    base_key = (str(_norm_job_id(pkts[0])), base_tid(str(pkts[0]["Task_ID"])))

    tt = state.task_times.setdefault(base_key, {})
    t_entry = float(task.get("gpu_entry_time", t))
    t_exit  = float(task.get("gpu_exit_time",  t))
    svc_fin = float(task.get("gpu_service_time", max(1e-6, t_exit - t_entry)))
    if svc_fin <= 0.0 or not math.isfinite(svc_fin):
        svc_fin = max(1e-6, t_exit - t_entry)

    tt["gpu_entry"]   = min(float(tt.get("gpu_entry", t_entry)), t_entry)
    tt["gpu_exit"]    = max(float(tt.get("gpu_exit",  t_exit)),  t_exit)
    tt["gpu_service"] = max(float(tt.get("gpu_service", 0.0)),   svc_fin)

    plan        = state.job_freq_plan.get(finished_job_id)
    freq_mode_s = str(getattr(state, "freq_mode", "adaptive"))

    job_gpu_src_map = getattr(state, "job_gpu_src", {}) or {}
    gpu_src = task.get("GPU_Decision_Source") or job_gpu_src_map.get(finished_job_id, "")

    plan_src = (
        plan[2] if isinstance(plan, tuple) and len(plan) >= 3
        else plan.get("src", "") if isinstance(plan, dict) else ""
    )

    freq_src = (
        task.get("Freq_Decision_Source")
        or plan_src
        or ("locked" if freq_mode_s.lower().startswith("fixed") else "")
        or ("optimizer/current" if gpu_src == "optimizer"
            else (str(getattr(state, "_last_freq_reason", "")) or "dvfs-helper"))
    )

    gpu_fields = {
        "gpu_entry_time":       t_entry,
        "gpu_service_time":     svc_fin,
        "gpu_exit_time":        round(t_exit, 6),
        "gpu_queue_delay":      task.get("gpu_queue_delay", ""),
        "Assigned_Cluster":     task.get("Assigned_Cluster"),
        "Assigned_Node":        task.get("Assigned_Node"),
        "Assigned_GPU":         task.get("Assigned_GPU"),
        "Assigned_Frequency":   task.get("Assigned_Frequency"),
        "Service_Frequency":    task.get("Service_Frequency", ""),
        "Freq_Mode":            freq_mode_s,
        "GPU_Decision_Source":  gpu_src,
        "Freq_Decision_Source": freq_src,
    }

    for r in state.rows_by_task.get(base_key, []):
        if str(r.get("Direction", "")).lower() == "uplink":
            r.update(gpu_fields)

    job_id_dl, task_base = base_key

    def _is_dl_row(r):
        d = str(r.get("Direction","")).lower()
        tid = str(r.get("Task_ID",""))
        pid = str(r.get("Packet_id",""))
        return (d == "downlink") or tid.endswith("_D") or ("_D_" in pid)

    task_rows = state.rows_by_task.get((job_id_dl, task_base), [])
    dl_rows = [r for r in task_rows if _is_dl_row(r)]
    for r in dl_rows:
        if str(r.get("Direction","")).lower() != "downlink":
            r["Direction"] = "Downlink"

    if not dl_rows:
        dl_rows = _synthesize_dl_rows(task, base_key)

    # UL-only workflow
    if not dl_rows:
        _dbg(
            state,
            f"[FINISH_GPU] UL-only Task={finished_task_id} Job={finished_job_id} "
            f"finalizing + checking job cleanup"
        )
        finalize_ul_only(base_key)
        _clear_job_records_if_done(state, finished_job_id)
        return

    # propagate provenance to DL rows (clear runtime timing)
    for dl_row in dl_rows:
        dl_row.setdefault("Assigned_Cluster",    task.get("Assigned_Cluster"))
        dl_row.setdefault("Assigned_Node",       task.get("Assigned_Node"))
        dl_row.setdefault("Assigned_GPU",        task.get("Assigned_GPU"))
        dl_row.setdefault("Assigned_Frequency",  task.get("Assigned_Frequency"))
        dl_row.setdefault(
            "Service_Frequency",
            task.get("Service_Frequency", task.get("Assigned_Frequency"))
        )
        dl_row["Freq_Mode"]            = gpu_fields["Freq_Mode"]
        dl_row["GPU_Decision_Source"]  = gpu_fields["GPU_Decision_Source"]
        dl_row["Freq_Decision_Source"] = gpu_fields["Freq_Decision_Source"]
        dl_row.update({
            "gpu_entry_time":   "",
            "gpu_service_time": "",
            "gpu_exit_time":    "",
            "gpu_queue_delay":  "",
        })

    mode = (state.__dict__.get("dl_arrival_mode") or "simultaneous")
    gaps = dl_gaps_for_task(state, len(dl_rows), mode=mode)

    _dbg(
        state,
        f"[FINISH_GPU] Task {finished_task_id} of Job {finished_job_id} "
        f"produced {len(dl_rows)} DL packet(s); enqueue to RC (mode={mode})"
    )

    for i, dl_row in enumerate(dl_rows):
        planned = float(t) + float(gaps[i]) + EPS
        dl_row["planned_RC_DL_IN_entry"] = planned
        dl_row["Packet_Arrival_Time"]    = planned
        state.fel.schedule(planned, Ev.ARRIVE_RC_DL_IN, {"packet": dl_row})

    # cleanup if that job is globally done
    _clear_job_records_if_done(state, finished_job_id)


def _inherit_assignment_for_dl(state: SimState, pkt: Dict[str,Any]) -> None:
    """Ensure DL row has assignment + DVFS provenance. Copy from UL; else use job plan."""
    needed = (
        "Assigned_Cluster","Assigned_Node","Assigned_GPU","Assigned_Frequency",
        "Freq_Mode","Freq_Decision_Source","GPU_Decision_Source","Service_Frequency",
    )
    if all(pkt.get(k) not in ("", None, "") for k in needed):
        return
    key = (str(pkt["Job_ID"]), base_tid(str(pkt["Task_ID"])))
    # Prefer UL row
    for r in state.rows_by_task.get(key, []):
        if str(r.get("Direction","")).lower() != "uplink":
            continue
        for kk in needed:
            if r.get(kk) not in ("", None, "") and pkt.get(kk) in ("", None, ""):
                pkt[kk] = r[kk]
        break
    # Fallback: per-job frequency plan
    plan = state.job_freq_plan.get(pkt.get("Job_ID"))
    if plan and any(pkt.get(k) in ("", None, "") for k in needed):
        if isinstance(plan, tuple):
            gk = plan[0]
            fstep = str(plan[1]) if len(plan) > 1 else ""
            src = str(plan[2]) if len(plan) > 2 else ""
        else:
            gk = plan.get("gpu"); gk = _norm_gpu_key(gk) if gk else None
            fstep = str(plan.get("freq","") or plan.get("Frequency",""))
            src = str(plan.get("src","") or plan.get("source",""))
        if gk:
            pkt.setdefault("Assigned_Cluster", gk[0])
            pkt.setdefault("Assigned_Node",    gk[1])
            pkt.setdefault("Assigned_GPU",     gk[2])
        if fstep:
            pkt.setdefault("Assigned_Frequency", fstep)
            pkt.setdefault("Service_Frequency",  fstep)
        if src:
            pkt.setdefault("Freq_Decision_Source", src)
            pkt.setdefault("Freq_Mode", "adaptive" if src != "locked" else "fixed")
            pkt.setdefault("GPU_Decision_Source", pkt.get("GPU_Decision_Source",""))

def on_arrive_rc_dl_in(state: SimState, t: float, pkt: Dict[str, Any]):
    pkt["Direction"] = "Downlink"
    normalize_packet_flags(pkt)
    _inherit_assignment_for_dl(state, pkt)
    _finalize_provenance(state, pkt, job_id=str(pkt.get("Job_ID")))

    job  = str(pkt["Job_ID"])
    base = base_tid(str(pkt["Task_ID"]))
    t_gpu = float(state.task_times.get((job, base), {}).get("gpu_exit", t))

    planned = pkt.get("planned_RC_DL_IN_entry", pkt.get("Packet_Arrival_Time", t))
    planned = float(planned) if planned not in ("", None) else float(t)
    t_in = max(planned, float(t), t_gpu)  # clamp

    pkt["Packet_Arrival_Time"] = round(t_in,6)
    pkt["RC_DL_IN_entry"]      = round(t_in,6)
    if pkt.get("Task_Arrival_Time") in ("", None) and to01(pkt.get("Is_First", 0)) == 1:
        pkt["Task_Arrival_Time"] = t_in

    c = str(pkt["Assigned_Cluster"])
    n_ports = int(state.ports_per_cluster.get(c, 1))
    port = int(pkt.get("RC_DL_Port") or
               pick_port(state, "RC_DL", c, n_ports, lambda p: state.net.rc_dl_time(p)))
    pkt["RC_DL_Port"] = port

    st = get_station(state, "RC_DL", c, port, lambda p: state.net.rc_dl_time(p))
    if not st.busy:
        st.busy = True
        pkt["RC_DL_service_start"] = round(t_in,6)
        svc = st.service_time_fn(pkt)
        pkt["RC_DL_service_time"] = round(float(svc),6)
        state.fel.schedule(t_in + svc, Ev.DEPART_RC_DL_IN, {"packet": pkt})
        st.tail_time = t_in + svc
    else:
        if st.capacity and (len(st.q) >= st.capacity):
            pkt["Task_Status"] = "Dropped_Q_Full_RC_DL"
            _stamp_assignment_for_drop(state, pkt)
            state.logs.append(pkt); return
        enqueue_fcfs(st, pkt, "RC_DL_IN_entry", t_in)

def on_depart_rc_dl_in(state: SimState, t: float, pkt: Dict[str, Any]):
    c = str(pkt["Assigned_Cluster"])
    port = int(pkt.get("RC_DL_Port", 1))
    st = state.stations[st_name("RC_DL", c, port)]
    pkt["RC_DL_IN_exit"] = round(t,6)
    pkt["RC_DL_IN_delay"] = round(float(pkt["RC_DL_service_start"]) - float(pkt["RC_DL_IN_entry"]), 6)

    if st.q:
        nxt = st.q.popleft()
        _q_update(st, t, len(st.q))
        nxt["RC_DL_service_start"] = round(t,6)
        svc = st.service_time_fn(nxt)
        nxt["RC_DL_service_time"] = round(float(svc),6)
        state.fel.schedule(t + svc, Ev.DEPART_RC_DL_IN, {"packet": nxt})
        st.busy = True
        st.tail_time = t + svc
    else:
        st.busy = False

    d = state.net.dl_prop(pkt)
    pkt["R0_DL_prop_delay"] = d
    state.fel.schedule(t + d, Ev.ARRIVE_R0_DL_IN, {"packet": pkt})

def on_arrive_r0_dl_in(state: SimState, t: float, pkt: Dict[str, Any]):
    """
    R0 downlink ingress.
    R0_DL is modeled as an instantaneous hop:
      - no queue
      - no service time
      - no additional delay beyond the RC->R0 propagation (already applied).
    We only stamp trivial fields so logs stay consistent, then finish the task.
    """
    # We can still pick/record a port for bookkeeping if you like.
    c = str(pkt["Assigned_Cluster"])
    n_ports = int(state.ports_per_cluster.get(c, 1))
    port = pick_port(state, "R0_DL", c, n_ports, lambda p: 0.0)  # always 0-time
    pkt["R0_DL_Port"] = int(port)

    now = round(float(t), 5)

    # Stamp dummy queue / service fields (zero-cost hop)
    pkt["R0_DL_IN_entry"]      = now
    pkt["R0_DL_IN_exit"]       = now
    pkt["R0_DL_IN_delay"]      = 0.0
    pkt["R0_DL_service_start"] = now
    pkt["R0_DL_service_time"]  = 0.0

    # Immediately treat as completed R0_DL
    _finish_r0_dl(state, now, pkt)


def _deadline_or_inf(x):
    try:
        d = float(x)
        # Treat missing/0/negative deadlines as "no deadline"
        return float("inf") if d <= 0 else d
    except (TypeError, ValueError):
        return float("inf")


def _finish_r0_dl(state: SimState, t: float, pkt: Dict[str, Any]) -> None:
    """
    Final DL packet completion hook.

    - Stamps task start/end and per-row overall timing.
    - Evaluates deadline violation and assigns task status.
    - Mirrors that status into state.task_times so fast-export can work.
    - Handles job/GPU bookkeeping and JobTracker notification.
    """

    def _to_float(x, default: float = 0.0) -> float:
        try:
            return float(x)
        except Exception:
            return default

    # --- Detect final task packet ------------------------------------------------
    is_last = (
        to01(pkt.get("Is_Last", 0)) == 1 or
        to01(pkt.get("Task_Is_Last", 0)) == 1
    )

    if is_last:
        # ----------------------------------------------------------------------
        # 1. Look up logical task key (Job_ID, base_Task_ID)
        # ----------------------------------------------------------------------
        job_id  = str(pkt["Job_ID"])
        task_id = str(pkt["Task_ID"])
        base    = base_tid(task_id)
        k       = (job_id, base)

        # state.task_times entry for this task (for fast-export)
        rec = state.task_times.setdefault(k, {})
        rec_end_prev = rec.get("end")

        # --- Determine task-level start and end ---------------------------------
        # end is this DL completion time
        t1 = float(t)
        rec["end"] = t1

        t0 = rec.get("start")
        if t0 is None:
            # fall back to first UL entry for this task if start not set
            t0 = _to_float(pkt.get("R0_UL_IN_entry"), t1)
            rec["start"] = t0

        # ----------------------------------------------------------------------
        # 2. Stamp timing + deadline violation + status on all packet rows
        #    for this logical task
        # ----------------------------------------------------------------------
        EPS = 1e-9  # tolerance for floating comparisons

        rows = state.rows_by_task.get(k, [])
        any_status = False
        max_viol = 0
        final_status = ""

        for r in rows:
            # timing
            r["overall_start_time"]    = t0
            r["overall_end_time"]      = t1
            r["Total_Completion_Time"] = t1 - t0

            # deadline (may be None / inf)
            deadline = _deadline_or_inf(r.get("Task_Deadline"))
            t1_num   = float(t1)
            violated = 1 if (t1_num - deadline) > EPS else 0
            max_viol = max(max_viol, violated)

            # classify UL-first vs DL-final rows
            d          = str(r.get("Direction", "")).lower()
            is_ul_first = (d == "uplink"   and to01(r.get("Is_First", 0)) == 1)
            is_dl_last  = (d == "downlink" and to01(r.get("Is_Last",  0)) == 1)

            if is_dl_last:
                # Only the terminal DL row carries the task outcome
                r["Task_Deadline_Violation"] = violated
                r["Task_Status"] = "Violated" if violated else "Completed"
                any_status = True
                final_status = r["Task_Status"]

            elif is_ul_first:
                # UL-first row gets a "UL_Completed" marker, no violation here
                r["Task_Deadline_Violation"] = 0
                r["Task_Status"] = "UL_Completed"
                any_status = True
                # don't override final_status if we later see DL-LAST

            else:
                # All other packets: no outcome flags; keep status if already set
                r.setdefault("Task_Deadline_Violation", 0)

        # ----------------------------------------------------------------------
        # 3. Mirror status + violation into state.task_times for fast-export
        # ----------------------------------------------------------------------
        # If no packet row ended up with a status (shouldn't happen, but be safe),
        # derive a conservative fallback.
        if not any_status:
            if max_viol:
                final_status = "Violated"
            else:
                final_status = "Completed"

        rec["deadline_violation"] = int(max_viol)
        rec["status"] = final_status

        # Optional debug:
        # print(f"[TS-DBG] task_times[{k}] -> start={rec['start']}, "
        #       f"end={rec['end']}, dv={rec['deadline_violation']}, "
        #       f"status={rec['status']}")

        # ----------------------------------------------------------------------
        # 4. Emit synthetic final DL row for aggregators (unchanged)
        # ----------------------------------------------------------------------
        if to01(pkt.get("Is_Last", 0)) == 1:
            _emit_final_dl_row(state, t, pkt)

    # --- Normal job/GPU bookkeeping ---------------------------------------------
    try:
        c2  = str(pkt.get("Assigned_Cluster"))
        n2  = str(pkt.get("Assigned_Node"))
        g2  = str(pkt.get("Assigned_GPU"))
        jid = str(pkt.get("Job_ID"))

        if hasattr(state, "active_jobs") and (c2, n2, g2) in state.active_jobs:
            if to01(pkt.get("Task_Is_Last", pkt.get("Is_Last", 0))) == 1:
                state.active_jobs[(c2, n2, g2)].discard(jid)
    except Exception:
        pass

    # --- Increment DL departure count ------------------------------------------
    state.ev_count["dl_depart"] += 1

    # --- JobTracker integration -------------------------------------------------
    try:
        state.job_tracker.on_dl_packet_depart(
            str(pkt["Job_ID"]),
            str(pkt["Task_ID"]),
            float(t),
        )
    except Exception:
        pass

    # --- Clear job records if job is complete ----------------------------------
    if to01(pkt.get("Is_Last", 0)) == 1:
        _clear_job_records_if_done(state, pkt.get("Job_ID"))


def _emit_final_dl_row(state: SimState, t: float, pkt: Dict[str, Any]):
    """Emit a single DL final-egress row so aggregators can pick completion cleanly."""
    job_id  = str(pkt.get("Job_ID"))
    task_id = str(pkt.get("Task_ID"))
    base    = base_tid(task_id)

    row = {
        "Strategy": getattr(state, "strategy", None),
        "Stage": "R0_Downlink_Egress",      # final egress; match aggregator filter
        "Is_Last": 1,
        "Timestamp": round(float(t), 5),    # Task_Complete_Time
        "Job_ID": job_id,
        "Task_ID": task_id,
        "Assigned_Cluster": str(pkt.get("Assigned_Cluster")),
        "Assigned_Node":    str(pkt.get("Assigned_Node")),
        "Assigned_GPU":     str(pkt.get("Assigned_GPU")),
    }

    # ----- single-emission guard per BASE task ID -----
    key_full = (job_id, task_id)
    key_base = (job_id, base)

    if not hasattr(state, "_final_emitted"):
        state._final_emitted = set()
    if key_full in state._final_emitted or key_base in state._final_emitted:
        return
    state._final_emitted.add(key_full)
    state._final_emitted.add(key_base)

    # preferred logging path
    log_fn = getattr(state, "log_row", None)
    if callable(log_fn):
        log_fn(row)
        if hasattr(state, "rows_by_task"):
            state.rows_by_task.setdefault(key_base, []).append(row)
        return

    # Fallbacks
    if hasattr(state, "rows_by_task"):
        state.rows_by_task.setdefault(key_base, []).append(row)

    if hasattr(state, "rows"):
        state.rows.append(row)


# === Helpers ===
def _ensure_parent(path: str | Path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)

def build_snapshot_from_config(cfg: dict, *, seed: int) -> dict:
    """
    Minimal snapshot builder used only for pregen.
    If you already have a true snapshot builder, call it here instead.
    """
    snap = {
        "seed": seed,
        "Job_Settings": cfg.get("Job_Settings", {}),
        "System": cfg.get("System", {}),
    }
    return snap

def generate_arrivals_from_config(cfg: dict, *, seed: int) -> list[dict]:
    """
    Return a *list of dicts* (arrivals) the same shape the simulator expects
    when running without a pregen. Keep the keys you already consume later
    (e.g., Task_ID, UL_Total_kB, DL_Total_kB, Workload_FLOPs, Task_Deadline, Lambda, etc.)
    This minimal version just delegates to the existing random generator if present.
    """
    # If you already have a generator, call it; else, return an empty list to keep the pipeline alive.
    gen = globals().get("pre_generate_from_config", None)
    if callable(gen):
        return gen(cfg, seed) or []
    return []

def write_pregen(path: str | Path, snapshot_or_arrivals: dict | list[dict]) -> None:
    """
    Accepts either a 'snapshot' dict (from build_snapshot_from_config) *or*
    a flat list of arrival rows. Writes a CSV so it’s easy to reuse with --pregen-in.
    """
    _ensure_parent(path)
    # If we got a dict (snapshot), try to extract/flatten arrivals if present
    if isinstance(snapshot_or_arrivals, dict) and "arrivals" in snapshot_or_arrivals:
        rows = snapshot_or_arrivals["arrivals"]
    else:
        rows = snapshot_or_arrivals

    # If we somehow got nothing, still write a header-only CSV so downstream won’t crash.
    rows = rows or []
    cols = set()
    for r in rows:
        cols.update(r.keys())
    cols = sorted(cols) if cols else ["Task_ID", "UL_Total_kB", "DL_Total_kB", "Workload_FLOPs", "Task_Deadline", "Lambda"]

    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        if rows:
            w.writerows(rows)

def load_pregen(path: str | Path) -> list[dict]:
    """
    Read the CSV written by write_pregen(...) and return list[dict] rows.
    """
    out = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            out.append(row)
    return out

def build_pregen_summary(snapshot: dict | list[dict]) -> pd.DataFrame:
    """
    Tiny smoke/sanity DF used only when creating a pregen with --out.
    It’s fine if this is minimal; plots don’t use it.
    """
    if isinstance(snapshot, dict) and "arrivals" in snapshot:
        rows = snapshot["arrivals"]
    elif isinstance(snapshot, list):
        rows = snapshot
    else:
        rows = []

    n = len(rows)
    lam = None
    try:
        # Try to read Lambda from config-shaped dict
        if isinstance(snapshot, dict):
            lam = (
                snapshot.get("Job_Settings", {})
                        .get("Job_Arrivals", {})
                        .get("Lambda_per_s", None)
            )
    except Exception:
        pass

    df = pd.DataFrame([{
        "count": n,
        "Lambda_per_s": lam,
    }])
    return df

# --- tictok -------------------------------------------------
class TicTok:
    def __init__(self): self.t0 = time.perf_counter()
    def tic(self, label): setattr(self, label, time.perf_counter())
    def tok(self, label):
        now = time.perf_counter()
        start = getattr(self, label, self.t0)
        print(f"[TICTOK] {label}: {now - start:.3f}s"); return now - start


# ==Helper==
def prepare_gurobi_dir(sub: str = "latency", base: str = "gurobi_cl") -> str:
    """
    Create a writable, unique directory for Gurobi dumps/logs:
      base/sub/YYYYmmdd-HHMMSS_PID/

    Also sets OS env var GUROBI_DUMP_DIR to that path.
    """
    base_path = Path(base)

    # If a file named 'gurobi_cl' blocks directory creation, move or remove it.
    if base_path.exists() and not base_path.is_dir():
        backup = base_path.with_suffix(base_path.suffix + ".bak")
        try:
            base_path.rename(backup)
            print(f"[WARN] '{base_path}' was a file; moved to '{backup}'.")
        except Exception:
            base_path.unlink(missing_ok=True)
            print(f"[WARN] '{base_path}' was a file; removed.")

    # Unique subdir per invocation
    stamp = time.strftime("%Y%m%d-%H%M%S")
    uniq  = f"{stamp}_{_os.getpid()}"  # or _os.getpid() if you used an alias
    d = base_path / sub / uniq
    d.mkdir(parents=True, exist_ok=True)

    _os.environ["GUROBI_DUMP_DIR"] = str(d)
    return str(d)

def _sniff_lambda_from_pregen(p: str):
    if not p:
        return None
    try:
        import csv
        with open(p, newline="") as f:
            r = csv.DictReader(f)
            for row in r:
                v = row.get("Lambda_per_s") or row.get("lambda_per_s")
                if v not in (None, ""):
                    return float(v)
                break
    except Exception:
        pass
    return None

def _load_module(ref):
    """Accepts module object, module name, or .py path → returns a loaded module."""
    if ref is None:
        return None
    if not isinstance(ref, str):
        return ref  # already a module

    # .py file path?
    if ref.endswith(".py") or _os.path.sep in ref:
        p = _Path(ref)
        if not p.is_file():
            # allow bare filename without path next to the simulator
            here = _Path(_os.path.abspath(_os.path.dirname(__file__)))
            cand = here / ref
            if cand.is_file():
                p = cand
        name = p.stem
        spec = importlib.util.spec_from_file_location(name, str(p))
        mod = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(mod)
        return mod

    # bare module name
    return importlib.import_module(ref)

# =============================
# Runner
# =============================
def run_sim(
    cfg: dict,
    seed: int,
    pregen: Optional[List[Dict[str, Any]]] = None,
    admission: str = "soft",
    objective: str = "latency",
    opt_modules: Optional[Dict[str, str]] = None,
    freq_policy: Optional[str] = None,
    assigner: str = "optimizer",
    inc_store_path: Optional[str] = None,
    inc_mode: str = "off",
    freq_mode: str = "adaptive",
    fixed_frequency: str = "",
    heuristic: str = "least-load",
    fixed_freq_map: Optional[Dict[tuple, str]] = None,
    strategy_name: Optional[str] = None,
    run_tag: Optional[str] = None,
    job_completion_out: Optional[str] = None,
    fast_decider: bool = False,
):
    # -----------------------------
    # GPU catalog construction
    # -----------------------------
    rates = gpu_rates_from_cluster_config(cfg)
    types, defaults = gpu_type_and_defaults(cfg)
    catalog = GPUCatalog(rates=rates, types=types, defaults=defaults)

    # Make the type map visible to power code
    tmap = cfg.setdefault("GPU_Type_Map", {})

    def _insert(c, n, g, gtype):
        tmap[(c, n, g)] = gtype
        tmap[f"{c}-{n}-{g}"] = gtype
        tmap[f"{c}/{n}/{g}"] = gtype
        tmap[f"{c}:{n}:{g}"] = gtype
        tmap[f"{c},{n},{g}"] = gtype

    for (c, n, g), gtype in (types or {}).items():
        c, n, g = str(c), str(n), str(g)
        _insert(c, n, g, str(gtype))

    cfg["GPU_Default_Freq"] = {k: str(v) for k, v in (defaults or {}).items()}
    catalog.power_specs = cfg.get("GPU_Specs", {})

    # -----------------------------
    # Network + FEL + RNG
    # -----------------------------
    per_cl = per_cluster_links(cfg)
    ports_map = ports_per_cluster_from_config(cfg)
    fast_rc_ul = fast_rc_ul_from_config(cfg)
    fast_rc_dl = bool(cfg.get("Network_Settings", {}).get("FAST_RC_DL", False))
    fast_r0_dl = fast_r0_dl_from_config(cfg)
    net = NetPerCluster(per_cl, fast_rc_ul=fast_rc_ul, fast_rc_dl=fast_rc_dl, fast_r0_dl=fast_r0_dl)

    _seed_salt = f"{assigner}|{objective}|{freq_mode}|{heuristic}|{freq_policy or ''}"
    try:
        salt_int = int.from_bytes(_hl.sha256(_seed_salt.encode()).digest()[:8], "big")
    except Exception:
        salt_int = sum(ord(ch) for ch in _seed_salt)
    seed_eff = (int(seed) ^ (salt_int & 0x7FFFFFFF)) & 0x7FFFFFFF

    rng = random.Random(seed_eff)
    fel = FEL()
    qcap = int(cfg.get("Network_Settings", {}).get("Queue_Capacity", 0))
    sim_start_time_s = 0.0

    # -----------------------------
    # Incremental store
    # -----------------------------
    store = None
    inc_mode = str(inc_mode or "off").lower()

    if inc_mode != "off" and not (inc_store_path or "").strip():
        print("[WARN] inc_mode is enabled but inc_store_path is missing. Forcing inc_mode='off'.")
        inc_mode = "off"

    if inc_mode != "off":
        store = AssignmentStore.from_optimizer_json(inc_store_path)
        store.mode = inc_mode
        store.cfg = cfg
        if store.mode != "readonly":
            _reconcile_store_freqs(store, cfg)

    fast_decider = bool(fast_decider or cfg.get("FAST_DECIDER", False))

    # -----------------------------
    # Load optimizer modules
    # -----------------------------
    loaded_modules = {}
    for k, v in (opt_modules or {}).items():
        try:
            loaded_modules[k] = _load_module(v)
        except Exception as e:
            raise RuntimeError(f"Failed to import optimizer module for '{k}' from {v!r}: {e}")

    # -----------------------------
    # Build optimizer object
    # -----------------------------
    optimizer = None
    if assigner == "optimizer":
        if fast_decider:
            if FastPolicy is None:
                raise RuntimeError(
                    "FastPolicy not found. Put fast_decider.py next to the simulator, "
                    "or call run_sim(..., fast_decider=False)."
                )
            optimizer = FastPolicy(
                config=cfg,
                modules=loaded_modules,
                objective=objective,
                catalog=catalog,
                store=store,
            )
        else:
            optimizer = OptimizerFacade(
                catalog=catalog,
                objective=objective,
                modules=loaded_modules,
                store=store,
            )

    print("[SIM] optimizer class =", type(optimizer).__name__ if optimizer else None, "fast_decider=", fast_decider)

    global GLOBAL_OPT
    GLOBAL_OPT = optimizer

    # -----------------------------
    # Create state (EARLY)
    # -----------------------------
    state = SimState(
        fel=fel,
        net=net,
        stations={},
        scheduler=None,
        gpu_catalog=catalog,
        rng=rng,
        ports_per_cluster=ports_map,
        queue_capacity=qcap,
        admission_policy=admission,
        assigner=assigner,
        objective=objective,
        optimizer=optimizer,
        assignment_store=store,
    )

    # Keep both object and plain dict for helpers
    state.cfg = cfg
    state.cfg_plain = cfg

    # ---- logging knobs ----
    logging_cfg = (cfg.get("Logging") or {})
    state.light_packet_log = bool(logging_cfg.get("LIGHT_PACKET_LOG", True))

    # ---- core flags ----
    state.sim_started = False
    state.allow_optimizer_solve = False

    # ---- util horizon ----
    state.util_tau_s = float((state.cfg.get("Scheduler") or {}).get("UTIL_TAU_S", 0.5))

    # ---- global λ_fps (DVFS inflow model) ----
    ts = (cfg.get("Task_Settings") or {}) if isinstance(cfg, dict) else {}
    fps_raw = (ts.get("SERVICE_ARRIVAL_RATE_fps") or ts.get("ARRIVAL_RATE_fps") or ts.get("Lambda_per_s") or 0.0)
    stride_raw = (ts.get("STRIDE") or ts.get("Task_Stride") or ts.get("stride") or 1)
    try:
        fps = float(fps_raw)
    except Exception:
        fps = 0.0
    try:
        stride = int(stride_raw)
    except Exception:
        stride = 1
    stride = max(1, stride)
    state.lambda_fps = fps / stride

    # Debug flags
    state.debug_dvfs = False
    state.debug_dvfs_logger = False
    state.debug_cluster_delay = False
    state.debug_task_status = True

    # Required containers
    state.pending_service_time = defaultdict(float)  # {(C,N,G): seconds}
    state.pending_tasks = defaultdict(int)           # {(C,N,G): count}

    if not hasattr(state, "dvfs_active_windows"):
        state.dvfs_active_windows = {}

    if not hasattr(state, "gpu_to_freq"):        state.gpu_to_freq = {}
    if not hasattr(state, "dvfs_window_log"):    state.dvfs_window_log = []
    if not hasattr(state, "last_freq_change"):   state.last_freq_change = {}
    if not hasattr(state, "dvfs_min_window_s"):  state.dvfs_min_window_s = 0.02
    if not hasattr(state, "dvfs_cooldown_s"):    state.dvfs_cooldown_s  = 0.01

    # attach store/optimizer onto state
    if store is not None:
        state.store = store
    if optimizer is not None:
        state.optimizer = optimizer
    state.assignment_store = optimizer.store if optimizer and hasattr(optimizer, "store") else store

    state.assigned_counts = getattr(state, "assigned_counts", _dd(int))
    if not hasattr(state, "dvfs_logger"):
        state.dvfs_logger = DVFSLogger()


    # 1) heuristic first
    state.heuristic = str(heuristic).lower()

    # 2) canonicalize freq_mode BEFORE dvfs_controller / warmup
    freq_mode_raw = str(freq_mode or cfg.get("Frequency_Mode", "adaptive"))
    m = freq_mode_raw.lower().replace("_", "-")
    if m in ("adaptive", "freq-adaptive", "freqadaptive"):
        m = "adaptive"
    elif m in ("fixed", "freq-fixed", "freqfixed"):
        m = "fixed"
    state.freq_mode = m
    state.Frequency_Mode = freq_mode_raw  # original label for logs

    # 3) decide DVFS controller BEFORE is_least_load / warmup
    if state.freq_mode == "fixed":
        state.dvfs_controller = "fixed"
    elif optimizer is not None and getattr(optimizer, "controls_dvfs", True):
        state.dvfs_controller = "optimizer"
    else:
        state.dvfs_controller = "heuristic"

    # 4) now compute is_least_load 
    state.is_least_load = (
        state.heuristic in ("least-load", "leastload", "load")
        and getattr(state, "dvfs_controller", "") == "heuristic"
    )

    if not hasattr(state, "use_balanced_util"):
        state.use_balanced_util = True

    # least-load safety floor
    ll_cfg = (cfg.get("LeastLoad") or {})
    lam_floor = float(ll_cfg.get("lambda_floor", 5.0))
    min_ll_idx = int(ll_cfg.get("min_freq_index", 1))
    if state.is_least_load:
        lam = float(getattr(state, "lambda_fps", 0.0))
        state.min_ll_freq_index = min_ll_idx if lam >= lam_floor else 0

    # fixed knobs
    state.fixed_frequency = (fixed_frequency or "").strip()
    state.fixed_freq_map = fixed_freq_map or {}

    # strategy/tag stamps
    _strategy_fallback = f"{assigner}_{heuristic}_{freq_mode}".replace("/", "-").replace(" ", "_")
    strategy_name = (strategy_name or _strategy_fallback)
    _run_tag_fallback = f"{'opt' if assigner=='optimizer' else 'heur'}_{objective}_freq-{freq_mode}_adm-{admission}_seed{seed}"
    run_tag = (run_tag or _run_tag_fallback).replace("/", "-").replace(" ", "_")
    state.strategy_name = strategy_name
    state.run_tag = run_tag

    # 5) warmup LAST (state now has heuristic/freq_mode/dvfs_controller)
    if optimizer is not None:
        warm = getattr(optimizer, "warmup", None)
        if callable(warm):
            try:
                sig = inspect.signature(warm)
                if len(sig.parameters) == 0:
                    warm()
                else:
                    warm(state)
            except Exception:
                # warmup must never kill the run
                pass

    state.sim_started = True
    state.allow_optimizer_solve = True

    # ---- misc DVFS/heuristic state ----
    if not hasattr(state, "gpu_freq_plan_fixed"):
        state.gpu_freq_plan_fixed = {}

    def _gkey_tuple_to_str(key):
        return f"{key[0]}-{key[1]}-{key[2]}"
    state._gkey_tuple_to_str = _gkey_tuple_to_str

    state.tail                 = getattr(state, "tail", {})
    state.pending_tail         = getattr(state, "pending_tail", {})
    state.assigned_counts      = getattr(state, "assigned_counts", {})
    state.current_freq         = getattr(state, "current_freq", {})
    state.freq_bins_by_gpu     = getattr(state, "freq_bins_by_gpu", {})
    state.dvfs_window          = getattr(state, "dvfs_window", 0.5)
    state.alpha_penalty        = getattr(state, "alpha_penalty", 0.002)
    state.beta_penalty         = getattr(state, "beta_penalty",  0.005)
    state.Cp                   = getattr(state, "Cp", {})
    state.flops_inflow_per_gpu = getattr(state, "flops_inflow_per_gpu", {})
    state._last_arrival_gpu_ts = getattr(state, "_last_arrival_gpu_ts", {})

    if not hasattr(state, "freq_steps_by_gpu"):
        state.freq_steps_by_gpu = {}

    def _freq_key_to_float(k):
        try:
            return float(str(k).rstrip(".0"))
        except Exception:
            return float("inf")

    for gkey, fmap in (catalog.rates or {}).items():
        if fmap:
            state.freq_steps_by_gpu[gkey] = sorted(fmap.keys(), key=_freq_key_to_float)

    state.gpu_candidates = [
        tuple(map(str, k))
        for k, fmap in (state.gpu_catalog.rates or {}).items()
        if fmap
    ]

    # restore HW from store only for NON-optimizer runs (ok)
    restore_hw = bool(cfg.get("Restore_HW_From_Store", False))
    is_optimizer = (assigner == "optimizer")
    if restore_hw and store and getattr(store, "frequencies", None) and (not is_optimizer):
        for gid, f in store.frequencies.items():
            parts = gid.split("-")
            if len(parts) == 3:
                state.gpu_to_freq[(parts[0], parts[1], parts[2])] = str(f)

    state.dl_arrival_mode = (cfg.get("Network_Settings", {}).get("DL_ARRIVAL_MODE", "simultaneous")).lower()

    state.freq_policy = str(cfg.get("Scheduler", {}).get("FREQ_POLICY", "min_deadline")).lower()
    if freq_policy is not None:
        state.freq_policy = str(freq_policy).lower()

    # --- If freq-mode is FIXED, lock GPUs immediately ---
    if state.freq_mode == "fixed":
        for key, fmap in (state.gpu_catalog.rates or {}).items():
            chosen = (
                (state.fixed_freq_map or {}).get(key, "") or
                state.fixed_frequency or
                ((state.gpu_catalog.defaults or {}).get(key, "") if (state.gpu_catalog.defaults or {}).get(key, "") in fmap else "") or
                (max(fmap, key=lambda k: float(fmap[k])) if fmap else "")
            )
            if not chosen:
                continue

            safe_set_gpu_freq(
                state, key, chosen,
                reason="fixed-lock",
                fmap=fmap,
                when=0.0,
                force_log=True,
                override_lock=True,
                trigger_job_id="init",
                trigger_task_id="init",
            )

            state.gpu_freq_plan_fixed[gkey_from_any(key)] = (chosen, "fixed-lock")
            if not hasattr(state, "_fixed_applied"):
                state._fixed_applied = set()
            state._fixed_applied.add(gkey_from_any(key))

    # ----- DVFS init seeding ----
    try:
        if not getattr(state, "_dvfs_seeded", False):
            freq_mode_now  = str(getattr(state, "freq_mode", "adaptive")).lower()
            objective_now  = str(getattr(state, "objective", "") or "").lower()
            latency_mode   = any(k in objective_now for k in ("latency", "min_latency", "opt_latency"))

            disable_seed = (assigner == "optimizer")
            disable_seed = disable_seed or (freq_mode_now == "adaptive" and latency_mode)

            if disable_seed:
                state._dvfs_seeded = True
            else:
                rates_map   = getattr(state.gpu_catalog, "rates", {}) or {}
                gpu_to_freq = getattr(state, "gpu_to_freq", {}) or {}

                for gkey, fmap in rates_map.items():
                    c, n, g = map(str, gkey)
                    ng = (c, n, g)

                    if freq_mode_now == "fixed":
                        continue
                    if not fmap:
                        continue

                    cur = str(gpu_to_freq.get(ng, "")).strip()
                    if cur and cur in fmap:
                        f0 = cur
                    else:
                        keys = list(fmap.keys())
                        try:
                            freqs_sorted = sorted(keys, key=lambda fk: float(fk))
                        except Exception:
                            freqs_sorted = sorted(keys, key=lambda fk: float(fmap[fk]))

                        if len(freqs_sorted) <= 2:
                            f0 = freqs_sorted[-1]
                        else:
                            f0 = freqs_sorted[len(freqs_sorted)//2]

                    safe_set_gpu_freq(
                        state, ng, f0,
                        reason="init_seed",
                        fmap=fmap,
                        when=0.0,
                        force_log=True,
                        override_lock=True,
                        trigger_job_id="init",
                        trigger_task_id="init",
                    )

                state._dvfs_seeded = True

    except Exception as e:
        print(f"[WARN] DVFS init seeding failed: {e}")
        state._dvfs_seeded = True

    # -----------------------------
    # Pregen rows + indexing
    # -----------------------------
    rows = pregen if pregen is not None else pre_generate_from_config(cfg, seed)
    state.master_rows = rows

    state.job_tracker = JobTracker()
    state.job_tracker.index_pregen_rows(rows)

    state.row_index = {}
    state.rows_by_task = defaultdict(list)
    state.task_times = {}
    state.pregen_by_task = {}

    # schedule arrivals + build rows_by_task
    for r in rows:
        if "Packet_Arrival_Time" not in r or r["Packet_Arrival_Time"] in ("", None):
            if "Packet_Arrival_Time_s" in r and r["Packet_Arrival_Time_s"] not in ("", None):
                r["Packet_Arrival_Time"] = float(r["Packet_Arrival_Time_s"])
            elif "Packet_Arrival_Time_ms" in r and r["Packet_Arrival_Time_ms"] not in ("", None):
                r["Packet_Arrival_Time"] = float(r["Packet_Arrival_Time_ms"]) / 1000.0
            else:
                r["Packet_Arrival_Time"] = 0.0

        t_arr = float(r["Packet_Arrival_Time"])
        r["R0_UL_IN_entry"] = t_arr

        if str(r.get("Direction", "")).lower() == "uplink":
            fel.schedule(t_arr, Ev.ARRIVE_R0_UL_IN, {"packet": r})

        # normalize_packet_flags(r)

        k4 = (str(r["Job_ID"]), str(r["Task_ID"]), str(r["Packet_id"]), str(r["Direction"]))
        state.row_index[k4] = r

        k_task = (str(r["Job_ID"]), base_tid(str(r["Task_ID"])))
        state.rows_by_task[k_task].append(r)

    # build pregen_by_task with BOTH raw + normalized Job_ID keys
    for r in rows:
        if str(r.get("Direction", "")).lower() != "uplink":
            continue
        r.setdefault("Assigned_Cluster", "C1")
        r.setdefault("Assigned_Node", "N1")
        r.setdefault("Assigned_GPU", "G1")
        r.setdefault("Assigned_Frequency", "")

        jid_raw = str(r.get("Job_ID", ""))
        try:
            jid_norm = str(_norm_job_id(r))
        except Exception:
            jid_norm = jid_raw

        tid_full = str(r.get("Task_ID", ""))  # keep exact Task_ID (same as decide())
        state.pregen_by_task.setdefault((jid_raw,  tid_full), []).append(r)
        state.pregen_by_task.setdefault((jid_norm, tid_full), []).append(r)

    # after scheduling pregen
    state.ev_count = {"total": 0, "dl_depart": 0}
    print(f"[SIM] queued events after pregen = {fel.size()}")

    # --- Schedule first DVFS retune per GPU (heuristic + adaptive ONLY) ---
    if state.dvfs_controller == "heuristic" and state.freq_mode == "adaptive":
        heur_now = str(getattr(state, "heuristic", "")).lower()
        if heur_now in ("least-load", "leastload", "load"):
            start_t = sim_start_time_s + float(getattr(state, "dvfs_window", 0.5))
            for gkey in (state.gpu_catalog.rates or {}).keys():
                state.fel.schedule(start_t, Ev.DVFS_RETUNE, {"gpu": gkey})

    # -----------------------------
    # FEL loop
    # -----------------------------
    def handle_event(t, etype, payload):
        now = float(t)
        state.t = now
        state.now = now
        et = Ev(etype)
        state.ev_count["total"] += 1

        if et == Ev.ARRIVE_R0_UL_IN:
            on_arrive_r0_ul_in(state, t, payload["packet"])
        elif et == Ev.DEPART_R0_UL_EG:
            on_depart_r0_ul_eg(state, t, payload["packet"])
        elif et == Ev.ARRIVE_RC_UL_IN:
            on_arrive_rc_ul_in(state, t, payload["packet"])
        elif et == Ev.ARRIVE_GPU:
            on_arrive_gpu(state, t, payload["packet"])
        elif et == Ev.FINISH_GPU:
            on_finish_gpu(state, t, payload)
        elif et == Ev.DVFS_RETUNE:
            dvfs_retune_event(state, payload["gpu"], t)
        elif et == Ev.ARRIVE_RC_DL_IN:
            on_arrive_rc_dl_in(state, t, payload["packet"])
        elif et == Ev.DEPART_RC_DL_IN:
            on_depart_rc_dl_in(state, t, payload["packet"])
        elif et == Ev.ARRIVE_R0_DL_IN:
            on_arrive_r0_dl_in(state, t, payload["packet"])
        else:
            return

    print(f"[SIM] starting FEL drain with {fel.size()} queued events")
    run_until_empty(fel, handle_event)
    print(f"[SIM] finished FEL drain, last_t={state.fel.t:.6f}s")

    # -----------------------------
    # Post-processing 
    # -----------------------------
    def _pidx(pid):
        try:
            return int(str(pid).split("_P")[-1])
        except Exception:
            return 0

    rows = state.master_rows

    if not state.light_packet_log:
        rows_sorted = sorted(
            rows,
            key=lambda r: (
                str(r.get("Job_ID", "")),
                base_tid(str(r.get("Task_ID", ""))),
                0 if str(r.get("Direction", "")).lower() == "uplink" else 1,
                _pidx(r.get("Packet_id", "")),
            ),
        )

        time_like = {
            "Job_Start_Time", "Job_End_Time", "Task_Arrival_Time", "Task_Deadline",
            "Packet_Arrival_Time", "R0_UL_IN_entry", "R0_UL_service_start", "R0_UL_service_time",
            "R0_UL_EG_exit", "R0_UL_queue_delay", "R0_UL_prop_delay",
            "gpu_entry_time", "gpu_service_time", "gpu_exit_time", "gpu_queue_delay",
            "RC_DL_IN_entry", "RC_DL_service_start", "RC_DL_service_time", "RC_DL_IN_exit",
            "RC_DL_IN_delay", "R0_DL_prop_delay", "R0_DL_IN_entry", "R0_DL_service_start",
            "R0_DL_service_time", "R0_DL_IN_exit", "R0_DL_IN_delay",
            "overall_start_time", "overall_end_time", "Total_Completion_Time",
        }

        for r in rows_sorted:
            for k in time_like:
                v = r.get(k)
                if v not in ("", None):
                    try:
                        r[k] = round(float(v), 6)
                    except (TypeError, ValueError):
                        pass

        must_dl = (
            "RC_DL_IN_entry", "RC_DL_service_start", "RC_DL_service_time", "RC_DL_IN_exit",
            "RC_DL_IN_delay", "R0_DL_prop_delay", "R0_DL_IN_entry", "R0_DL_service_start",
            "R0_DL_service_time", "R0_DL_IN_exit", "R0_DL_IN_delay",
        )
        for r in rows_sorted:
            if str(r.get("Direction", "")).lower() == "downlink":
                for k in must_dl:
                    if r.get(k) in ("", None):
                        r[k] = 0.0

        rows_out = rows_sorted
    else:
        rows_out = rows

    global __dvfs_log_buffer__
    try:
        __dvfs_log_buffer__ = _collect_dvfs_events(state)
        global __violations_log_buffer__
        try:
            __violations_log_buffer__ = dict(state.dropped_tasks)
        except Exception:
            __violations_log_buffer__ = {}
    except Exception:
        __dvfs_log_buffer__ = []

    # Memory cleanup
    try:
        if getattr(state, "scheduler", None) is not None:
            if hasattr(state.scheduler, "store"):
                state.scheduler.store = None
            state.scheduler = None
    except Exception:
        pass

    try:
        state.assignment_store = None
    except Exception:
        pass

    try:
        state.dvfs_logger = None
    except Exception:
        pass

    try:
        state.row_index = {}
        state.rows_by_task = {}
        state.pregen_by_task = {}
    except Exception:
        pass

    try:
        state.master_rows = None
    except Exception:
        pass

    try:
        state.fel = None
        state.net = None
    except Exception:
        pass

    try:
        state.gpu_catalog = None
        state.cfg = None
    except Exception:
        pass

    return rows_out, state

# =============================
# CLI
# =============================

def main():
    ap = argparse.ArgumentParser(
        description="FEL simulator with pre-generation & dynamic config (live optimizer)"
    )
    ap.add_argument("--config", default="config.json", help="Path to config.json")
    ap.add_argument("--seed", type=int, default=41)
    ap.add_argument("--freq-policy", choices=["fastest", "default", "min_deadline", "balanced"], default=None)
    ap.add_argument("--admission", type=str, default="soft", choices=["soft", "hard"], help="Deadline admission policy")
    ap.add_argument("--assign-key", choices=["base", "full"], default="base",
                    help="Task keying: 'base' drops _U/_D, 'full' keeps the suffix.")
    ap.add_argument("--objective", type=str, default="latency", choices=["latency", "power", "efficiency"])
    ap.add_argument("--assigner", type=str, default="optimizer", choices=["optimizer", "heuristic"])
    ap.add_argument("--heuristic", type=str, default="least-load", choices=["least-load", "random"])
    ap.add_argument("--freq-mode", type=str, default="adaptive", choices=["adaptive", "fixed"])
    ap.add_argument("--fixed-frequency", type=str, default="")
    ap.add_argument("--fixed-freq-map", dest="fixed_freq_map", default=None,
                    help="Path to JSON mapping 'C-N-G' -> freq for per-GPU fixed DVFS.")
    ap.add_argument(
        "--strategy",
        choices=[
            "least-load_adaptive", "least-load_fixed",
            "random_adaptive", "random_fixed",
            "opt_latency_adaptive", "opt_latency_fixed",
            "opt_power_adaptive", "opt_power_fixed",
            "opt_efficiency_adaptive", "opt_efficiency_fixed",
        ],
        default="least-load_adaptive",
    )
    ap.add_argument("--out", type=str, default=None)

    # --- output location ---
    ap.add_argument("--out-dir", default="runs",
                    help="Directory where this run writes CSV/plots (default: runs)")
    ap.add_argument("--prefix", default="exp",
                    help="Filename prefix (e.g., exp1_L3.5__least-load_fixed)")

    # --- convenience toggle ---
    ap.add_argument("--enable-adaptive-freq", action="store_true",
                    help="Enable adaptive/adjust frequency mode for the solver")

    # --- Lambda override and batch interface ---
    ap.add_argument("--lambda", dest="lambda_per_s", type=float, default=None,
                    help="Override Job_Arrivals.Lambda_per_s from config.json")

    ap.add_argument("--multi", dest="multi", type=str, default=None,
                    help="Batch runs. Format: 'L:3.5,4.0;runs:5;seed0:41;outdir:runs;prefix:exp1'")

    ap.add_argument("--lambdas", type=str, default=None,
                    help="Comma-separated list, e.g., '3.5,4.0'")
    ap.add_argument("--runs", type=int, default=1,
                    help="Number of runs per Lambda (default: 1)")
    ap.add_argument("--seed0", type=int, default=41,
                    help="Base seed; run i uses seed0+i (default: 41)")

    ap.add_argument("--pregen-out", type=str, default=None)
    ap.add_argument("--pregen-in", type=str, default=None,
                    help="Read a pregenerated CSV and reuse it for all strategies")
    ap.add_argument("--export-opt-array", type=str, default=None)
    ap.add_argument("--incremental-store", type=str, default=None)
    ap.add_argument("--incremental-mode", choices=["off", "readonly", "readwrite"], default="off")

    # Optimizer module flags (name or .py path)
    ap.add_argument("--opt-module", type=str, default=None, help="[Legacy] Single module used for all objectives")
    ap.add_argument("--opt-module-latency", dest="opt_module_latency", default=None,
                    help="Optimizer module for the 'latency' objective.")
    ap.add_argument("--opt-module-power", dest="opt_module_power", default=None,
                    help="Optimizer module for the 'power' objective.")
    ap.add_argument("--opt-module-efficiency", dest="opt_module_efficiency", default=None,
                    help="Optimizer module for the 'efficiency' objective.")

    ap.add_argument("--fast-decider", action="store_true",
                    help="Use the high-performance FastPolicy wrapper (tiny snapshots + O(1) reuse)")
    ap.add_argument("--fast-export", action="store_true",
                    help="Skip global sort, vectorize rounding, and write Parquet + a light CSV index.")

    args = ap.parse_args()
    energy_rows = []

    # ---- pregen guards ----
    if args.pregen_in and args.pregen_out:
        ap.error("Use only one: --pregen-in OR --pregen-out (not both).")

    if args.lambda_per_s is not None and args.pregen_in:
        print("[WARN] --lambda is ignored because --pregen-in is provided.")

    OUT_DIR = Path(args.out_dir)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PREFIX = (args.prefix or "").strip() or "exp"

    def build_outfile(tag: str, ext: str = "csv") -> str:
        return str(OUT_DIR / f"{PREFIX}__{tag}.{ext}")

    def save_csv(df, tag: str, *, index: bool = False) -> str:
        p = build_outfile(tag, "csv")
        df.to_csv(p, index=index)
        print(f"[SIM] wrote {p}")
        return p

    # normalize pregen paths relative to out_dir
    if args.pregen_out:
        if not _os.path.isabs(args.pregen_out):
            args.pregen_out = str(OUT_DIR / Path(args.pregen_out).name)
        Path(args.pregen_out).parent.mkdir(parents=True, exist_ok=True)

    if args.pregen_in and not _os.path.isabs(args.pregen_in):
        args.pregen_in = str(OUT_DIR / Path(args.pregen_in).name)

    # ---- parse multi string (single definition) ----
    def _parse_multi(s: str):
        parts = dict(kv.split(":", 1) for kv in s.split(";") if ":" in kv)
        Ls = [float(x) for x in parts.get("L", "").split(",") if x]
        runs = int(parts.get("runs", 1))
        seed0 = int(parts.get("seed0", 41))
        outdir = parts.get("outdir", args.out_dir)
        prefix = parts.get("prefix", args.prefix)
        return Ls, runs, seed0, outdir, prefix

    # -------- fixed per-GPU freq map (if provided) --------
    fixed_freq_map = None
    if args.fixed_freq_map:
        try:
            with open(args.fixed_freq_map, "r", encoding="utf-8") as f:
                raw = json.load(f)
            fixed_freq_map = {}
            if isinstance(raw, dict):
                for k, v in raw.items():
                    kk = str(k).strip()
                    for sep in ("-", ",", ":", "/"):
                        kk = kk.replace(sep, " ")
                    parts = [p for p in kk.split() if p]
                    if len(parts) >= 3:
                        fixed_freq_map[(parts[0], parts[1], parts[2])] = str(v)
        except Exception as e:
            print(f"[WARN] Failed to load --fixed-freq-map '{args.fixed_freq_map}': {e}")
            fixed_freq_map = None

    # -------- Strategy → effective knobs --------
    eff_assigner = args.assigner
    eff_objective = args.objective
    eff_freq_mode = args.freq_mode
    eff_heuristic = args.heuristic

    s = (args.strategy or "").strip()
    if s.startswith("least-load"):
        eff_assigner = "heuristic"
        eff_heuristic = "least-load"
        eff_freq_mode = "fixed" if s.endswith("fixed") else "adaptive"
    elif s.startswith("random"):
        eff_assigner = "heuristic"
        eff_heuristic = "random"
        eff_freq_mode = "fixed" if s.endswith("fixed") else "adaptive"
    elif s.startswith("opt_latency"):
        eff_assigner = "optimizer"
        eff_objective = "latency"
        eff_freq_mode = "fixed" if s.endswith("fixed") else "adaptive"
    elif s.startswith("opt_power"):
        eff_assigner = "optimizer"
        eff_objective = "power"
        eff_freq_mode = "fixed" if s.endswith("fixed") else "adaptive"
    elif s.startswith("opt_efficiency"):
        eff_assigner = "optimizer"
        eff_objective = "efficiency"
        eff_freq_mode = "fixed" if s.endswith("fixed") else "adaptive"

    # Optimizer modules required only if using optimizer
    opt_modules = {
        "latency": args.opt_module_latency or args.opt_module,
        "power": args.opt_module_power or args.opt_module,
        "efficiency": args.opt_module_efficiency or args.opt_module,
    }
    opt_modules = {k: v for k, v in opt_modules.items() if v}

    if eff_assigner == "optimizer" and not opt_modules:
        raise RuntimeError("No optimizer module specified. Use --opt-module or per-objective flags.")

    # Update args to reflect effective knobs (used for tags)
    args.assigner = eff_assigner
    args.objective = eff_objective
    args.freq_mode = eff_freq_mode
    args.heuristic = eff_heuristic

    tag = build_run_tag(args)
    strategy_name = (args.strategy or f"{eff_assigner}_{eff_heuristic}_{eff_freq_mode}")

    # -------- Paths --------
    _, pregen_path, export_path, inc_store_path = resolve_paths(args)

    # Ensure GUROBI dump dirs exist
    if eff_assigner == "optimizer":
        _os.environ["GUROBI_DUMP_DIR"] = prepare_gurobi_dir(eff_objective)
    else:
        if "GUROBI_DUMP_DIR" in _os.environ:
            del _os.environ["GUROBI_DUMP_DIR"]

    # =========================
    # MULTI-RUN PATH
    # =========================
    if args.multi or args.lambdas:
        if args.multi:
            Ls, R, S0, outdir, prefix = _parse_multi(args.multi)
        else:
            Ls = [float(x) for x in (args.lambdas or "").split(",") if x]
            R, S0, outdir, prefix = args.runs, args.seed0, args.out_dir, args.prefix

        OUTDIR = Path(outdir)
        OUTDIR.mkdir(parents=True, exist_ok=True)
        sim_path = _os.path.abspath(sys.argv[0])

        extras_parts = []
        if args.opt_module_latency:
            extras_parts += ["--opt-module-latency", args.opt_module_latency]
        if args.opt_module_power:
            extras_parts += ["--opt-module-power", args.opt_module_power]
        if args.opt_module_efficiency:
            extras_parts += ["--opt-module-efficiency", args.opt_module_efficiency]
        if args.enable_adaptive_freq:
            extras_parts += ["--enable-adaptive-freq"]
        extras_parts += ["--admission", args.admission]
        if args.fast_decider:
            extras_parts += ["--fast-decider"]
        EXTRAS = " ".join(extras_parts)

        for L in Ls:
            for r in range(1, R + 1):
                seed = S0 + (r - 1)

                # write a temp config overriding Lambda
                cfg2 = json.load(open(args.config, "r", encoding="utf-8"))
                js = cfg2.setdefault("Job_Settings", {})
                ja = js.setdefault("Job_Arrivals", {})
                ja["Lambda_per_s"] = float(L)

                tmp_cfg = OUTDIR / f"tmp_config_L{L}_r{r}.json"
                tmp_cfg.write_text(json.dumps(cfg2, indent=2))

                child_prefix = f"{prefix}_L{L}__run{r}"
                pregen_path = OUTDIR / f"{child_prefix}_pregen.csv"

                # 1) pregen (only if missing)
                if not pregen_path.exists():
                    pregen_cmd = [
                        sys.executable, sim_path,
                        "--config", str(tmp_cfg),
                        "--seed", str(seed),
                        "--pregen-out", str(pregen_path),
                    ]
                    env = {**_os.environ, "MPLBACKEND": "Agg", "PYTHONUNBUFFERED": "1"}
                    print(">", " ".join(pregen_cmd))
                    res = subprocess.run(pregen_cmd, check=False, capture_output=True, text=True, env=env)
                    if res.returncode != 0 or not pregen_path.exists():
                        print(f"[ERR] pregen failed for {child_prefix}")
                        print(res.stdout)
                        print(res.stderr)
                        sys.exit(res.returncode if res.returncode else 2)

                # 2) run ALL strategies using the SAME tmp config + SAME pregen
                EXTRAS_SEEDED = f"{EXTRAS} --seed {seed}"
                all_cmd = [
                    sys.executable, "run_all_strategies.py",
                    "--sim", sim_path,
                    "--out-dir", str(OUTDIR),
                    "--prefix", child_prefix,
                    "--pregen-in", str(pregen_path),
                    # FIX: use the per-run config that contains Lambda override
                    "--config", str(tmp_cfg),
                    "--extras", EXTRAS_SEEDED,
                ]

                env = {**_os.environ, "MPLBACKEND": "Agg", "PYTHONUNBUFFERED": "1"}
                print(">", " ".join(all_cmd))
                res = subprocess.run(all_cmd, check=False, env=env, capture_output=True, text=True)
                if res.returncode != 0:
                    print(f"[ERR] run_all_strategies failed for {child_prefix} (code {res.returncode})")
                    print(res.stdout)
                    print(res.stderr)
                    sys.exit(res.returncode)

        sys.exit(0)

    # --- incremental store sanity ---
    if args.incremental_mode != "off" and not (args.incremental_store or "").strip():
        print("[WARN] incremental_mode is enabled but --incremental-store is missing. Disabling incremental mode.")
        args.incremental_mode = "off"
        args.incremental_store = None

    # If a store path is provided, normalize it under OUT_DIR unless absolute
    if args.incremental_store:
        if not _os.path.isabs(args.incremental_store):
            args.incremental_store = str(OUT_DIR / Path(args.incremental_store).name)
        Path(args.incremental_store).parent.mkdir(parents=True, exist_ok=True)


    # =========================
    # SINGLE-RUN PATH
    # =========================
    cfg = json.load(open(args.config, "r", encoding="utf-8"))

    def _infer_fmax_mhz_from_spec(spec: dict) -> float | None:
        if "f_max" in spec:
            try:
                return float(spec["f_max"])
            except Exception:
                return None
        freqs = spec.get("freqs", [])
        try:
            vals = [float(x) for x in freqs]
            return max(vals) if vals else None
        except Exception:
            return None

    def calibrate_phi_power_in_cfg(cfg: dict) -> None:
        meta = cfg.get("GPU_Specs_Meta", {}) or {}
        default_exp = float(meta.get("power_exp", 1.0))
        for gtype, spec in (cfg.get("GPU_Specs") or {}).items():
            e = float(spec.get("power_exp", default_exp))
            if "phi_power" in spec:
                spec["_phi_exp"] = e
                continue
            Pst = spec.get("P_static_W", spec.get("P_static", None))
            Pmx = spec.get("P_max_W", spec.get("P_max", None))
            fmx = _infer_fmax_mhz_from_spec(spec)
            if Pst is None or Pmx is None or fmx is None:
                continue
            Pst, Pmx, fmx = float(Pst), float(Pmx), float(fmx)
            spec["phi_power"] = max(0.0, (Pmx - Pst) / (max(fmx, 1e-9) ** e))
            spec["_phi_exp"] = e

    calibrate_phi_power_in_cfg(cfg)

    # PREGEN-ONLY mode
    if args.pregen_out:
        if args.lambda_per_s is not None:
            cfg.setdefault("Job_Settings", {}).setdefault("Job_Arrivals", {})["Lambda_per_s"] = float(args.lambda_per_s)

        pregen_rows = pre_generate_from_config(cfg, args.seed)
        lam = float(cfg["Job_Settings"]["Job_Arrivals"]["Lambda_per_s"])

        write_pregen_csv(args.pregen_out, pregen_rows, lam=lam)

        if args.out:
            smoke = build_pregen_summary({"arrivals": pregen_rows})
            smoke.to_csv(args.out, index=False)
            print(f"[SIM] wrote pregen smoke -> {args.out}")

        print(f"[SIM] wrote pregen -> {args.pregen_out}")
        return

    # Obtain arrivals for the actual run
    if args.pregen_in:
        arrivals = load_pregen(args.pregen_in)
    else:
        if args.lambda_per_s is not None:
            cfg.setdefault("Job_Settings", {}).setdefault("Job_Arrivals", {})["Lambda_per_s"] = float(args.lambda_per_s)
        arrivals = generate_arrivals_from_config(cfg, seed=args.seed)

    # Run simulation
    logs, sim_state = run_sim(
        cfg,
        args.seed,
        pregen=arrivals,
        admission=args.admission,
        objective=eff_objective,
        opt_modules=opt_modules,
        freq_policy=args.freq_policy,
        assigner=eff_assigner,
        inc_store_path=(args.incremental_store if args.incremental_mode != "off" else None),
        inc_mode=args.incremental_mode,
        freq_mode=eff_freq_mode,
        fixed_frequency=args.fixed_frequency,
        fixed_freq_map=fixed_freq_map,
        heuristic=eff_heuristic,
        strategy_name=strategy_name,
        run_tag=build_run_tag(args),
        fast_decider=args.fast_decider,
    )


    # ---- Correct λ for this run (precedence: CLI → pregen → config) ----
    lam_val = (
        float(args.lambda_per_s) if args.lambda_per_s is not None
        else (_sniff_lambda_from_pregen(args.pregen_in) if args.pregen_in else None)
    )
    if lam_val is None:
        lam_val = float(cfg["Job_Settings"]["Job_Arrivals"]["Lambda_per_s"])

    for r in logs:
        r["Lambda_per_s"] = lam_val
        r.setdefault("Strategy", (args.strategy or "unknown"))

    strat_short = (args.strategy or "unknown").strip()
    mode_tag = "freq-fixed" if eff_freq_mode == "fixed" else "freq-adaptive"
    adm_tag = f"adm-{args.admission}"
    Ltag = str(lam_val).replace(".", "_")
    seed_val = int(args.seed)

    def _stamp(rows):
        out = []
        for r in (rows or []):
            rr = dict(r)
            rr["Strategy"] = strat_short
            rr["Freq_Mode"] = mode_tag
            rr["Admission"] = adm_tag
            rr["Lambda_per_s"] = lam_val
            rr["Seed"] = seed_val
            out.append(rr)
        return out

    if args.fast_export:
        task_rows_raw = build_task_summaries_from_state(sim_state)
    else:
        task_rows_raw = build_task_summaries_from_packet_rows(logs)

    task_rows = _stamp(task_rows_raw)

    tag_base = f"task_packets_summary_{strat_short}_{mode_tag}_{adm_tag}_L{Ltag}_seed{seed_val}"
    main_csv_path = build_outfile(tag_base, "csv")
    write_task_summary_csv(main_csv_path, task_rows)
    print(f"[OK] wrote {len(task_rows)} task-summary rows to {main_csv_path}")

    # ---- DVFS windows export ----
    try:
        main_rows = logs or []

        # 1) Build run_end from task completion + any DVFS window timestamps
        end_candidates = []

        # task-level completion stamps
        for r in main_rows:
            t_end = _to_float(r.get("overall_end_time"))
            if t_end is not None and math.isfinite(t_end):
                end_candidates.append(t_end)

        # also include any DVFS window start/end times we've already logged
        for w in getattr(sim_state, "dvfs_window_log", []):
            t_start = _to_float(w.get("start"))
            t_end   = _to_float(w.get("end"))
            if t_start is not None and math.isfinite(t_start):
                end_candidates.append(t_start)
            if t_end is not None and math.isfinite(t_end):
                end_candidates.append(t_end)

        run_end = max(end_candidates) if end_candidates else 0.0
        if not end_candidates:
            print("[DVFS][WARN] No overall_end_time or DVFS timestamps; run_end=0.0")

        # # 2) Finalize any still-open DVFS windows at sim end
        # try:
        #     for gkey_str, active_win in list(getattr(sim_state, "dvfs_active_windows", {}).items()):
        #         parts = gkey_str.split("-")
        #         if len(parts) == 3:
        #             key_tuple = (parts[0], parts[1], parts[2])
        #             _close_dvfs_window(sim_state, key_tuple, run_end)
        #     sim_state.dvfs_active_windows = {}
        # except Exception as e:
        #     print("[DVFS][finalize][WARN]", e)

        # 2) Finalize any still-open DVFS windows at sim end
        try:
            aw = getattr(sim_state, "dvfs_active_windows", {}) or {}
            for gkey, active_win in list(aw.items()):

                # gkey may be ("C1","N2","G1") OR "C1-N2-G1"
                if isinstance(gkey, (tuple, list)) and len(gkey) == 3:
                    key_tuple = (str(gkey[0]), str(gkey[1]), str(gkey[2]))
                else:
                    s = str(gkey)
                    parts = s.split("-")
                    if len(parts) == 3:
                        key_tuple = (parts[0], parts[1], parts[2])
                    else:
                        # last resort: try normalizer if available
                        key_tuple = _norm_gkey(gkey)

                _close_dvfs_window(sim_state, key_tuple, run_end)

            sim_state.dvfs_active_windows = {}
        except Exception as e:
            print("[DVFS][finalize][WARN]", e)


        # 3) dvfs_events is now the finalized per-window log
        dvfs_events = list(getattr(sim_state, "dvfs_window_log", []))

        # 3b) Coalesce contiguous same-freq windows (safety net)
        dvfs_events = merge_adjacent_same_freq_windows(dvfs_events)

        # 4) Enrich windows with active/idle breakdown
        #    NOTE: compute_active_idle_from_logs(...) expects a list of windows shaped like dvfs_events
        windows = compute_active_idle_from_logs(main_rows, dvfs_events) or []

        # 5) Energy + per-GPU totals
        try:
            energy_rows, power_totals = compute_power_windows(
                cfg,
                windows,
                rows=main_rows or []
            )
            # Stamp Strategy / Freq_Mode / Admission / Lambda / Seed into energy_rows
            energy_rows = _stamp(energy_rows)

            ewin_csv = build_outfile(f"{tag_base}_dvfs_energy_windows")
            write_dvfs_energy_windows_csv(ewin_csv, energy_rows or [])
            print(f"[DVFS] energy rows: {len(energy_rows)} -> {ewin_csv}")

        except Exception as e:
            print(f"[WARN] DVFS energy export failed: {e}")

    except Exception as e:
        print(f"[WARN] DVFS windows export failed: {e}")

    stats = getattr(sim_state, "admit_stats", {})
    print("Admitted jobs:", stats.get("admitted", 0))
    print("Dropped jobs:",  stats.get("dropped", 0))
    for rec in stats.get("dropped_jobs", []):
        print(rec)


    # # Force a collection at process end
    # gc.collect()


if __name__ == "__main__":
    main()

# ===RUN COMMAND ====

# python queuing_system_simulation_incremental.py \
#   --config config.json \
#   --multi "L:3.5,4.0;runs:5;seed0:41;outdir:runs;prefix:exp1" \
#   --opt-module-latency    live_min_latency_optimizer_incremental \
#   --opt-module-power      live_min_power_optimizer_incremental \
#   --opt-module-efficiency live_max_efficiency_optimizer_incremental \
#   --admission soft 

# # Adaptive freq ON
# python queuing_system_simulation_incremental.py \
#   --config config.json \
#   --multi "L:3.5,4.0;runs:5;seed0:41;outdir:runs;prefix:exp1" \
#   --enable-adaptive-freq \
#   --opt-module-latency    live_min_latency_optimizer_incremental \
#   --opt-module-power      live_min_power_optimizer_incremental \
#   --opt-module-efficiency live_max_efficiency_optimizer_incremental \
#   --admission soft

# python queuing_system_simulation_incremental.py \
#   --config config.json \
#   --multi "L:3.5;runs:1;seed0:41;outdir:runs;prefix:exp1" \
#   --enable-adaptive-freq \
#   --opt-module-latency    live_min_latency_optimizer_incremental \
#   --opt-module-power      live_min_power_optimizer_incremental \
#   --opt-module-efficiency live_max_efficiency_optimizer_incremental \
#   --admission soft

# python queuing_system_simulation_incremental.py \
#   --config config.json \
#   --multi "L:3.5,4.0;runs:5;seed0:41;outdir:runs;prefix:exp1" \
#   --enable-adaptive-freq \
#   --fast-decider \
#   --opt-module-latency    live_min_latency_optimizer_incremental \
#   --opt-module-power      live_min_power_optimizer_incremental \
#   --opt-module-efficiency live_max_efficiency_optimizer_incremental \
#   --admission soft

# python queuing_system_simulation_incremental.py \
#   --config config.json \
#   --multi "L:4;runs:1;seed0:41;outdir:runs;prefix:exp1" \
#   --enable-adaptive-freq \
#   --fast-decider \
#   --opt-module-latency    live_min_latency_optimizer_incremental \
#   --opt-module-power      live_min_power_optimizer_incremental \
#   --opt-module-efficiency live_max_efficiency_optimizer_incremental \
#   --admission soft

# == Execute Plot Creation ======
# python plot_from_task_packets.py --runs runs --prefix exp1 --out plots_lambda_compare
