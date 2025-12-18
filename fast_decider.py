"""
fast_decider.py — high‑performance drop‑in for the streaming simulator

This module keeps the external optimizer (live_min_latency_optimizer_incremental)
but eliminates per‑event Python overhead and maximizes Gurobi reuse.

It provides a class FastPolicy with the same public surface you use:
  - warmup(state)
  - decide(state, pkt) -> ((C,N,G), freq)

Key ideas:
  • Zero‑copy snapshots: build the tiniest snapshot the optimizer needs.
  • Array first: represent GPUs as compact arrays, not dicts.
  • Reuse everything: model cache (already in optimizer) + our own signature cache.
  • Fast paths: common case is “job already pinned” → O(1) return.
  • Minimal conversions: avoid repeated float()/str() churn and chained .get().
  • Tight helpers: localize hot lookups to Python locals.
  • Tuned Gurobi params only once; pass only when changed.

Drop‑in usage:
    policy = FastPolicy(config, modules={"latency": live_min_latency_optimizer_incremental})
    policy.warmup(state)
    gpu_key, freq = policy.decide(state, pkt)
"""
from __future__ import annotations
from typing import Any, Dict, Tuple, Optional, List
import inspect
import math, csv, os

try:
    import live_min_latency_optimizer_incremental as inc
    import live_min_power_optimizer_incremental as minpwr
    import live_max_efficiency_optimizer_incremental as maxeff
except ImportError:
    from . import live_min_latency_optimizer_incremental as inc
    from . import live_min_power_optimizer_incremental as minpwr
    from . import live_max_efficiency_optimizer_incremental as maxeff


# --------------------------------------
# Small, allocation‑free utilities
# --------------------------------------
__all__ = ["FastPolicy"]

_S = str
_F = float
DROP_GPU = ("", "", "")
DROP_FREQ = ""

def _I(x) -> int:
    """Fast-ish, but tolerant int conversion.

    Accepts ints, floats, numeric strings like '15' or '15.0'.
    Returns 0 on anything weird/None.
    """
    if isinstance(x, int):
        return x
    if isinstance(x, float):
        return int(x)
    if x is None:
        return 0
    try:
        return int(x)
    except (TypeError, ValueError):
        try:
            return int(float(x))
        except (TypeError, ValueError):
            return 0

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

def _norm_gkey(gkey):
    if isinstance(gkey, (tuple, list)) and len(gkey) >= 3:
        return (str(gkey[0]), str(gkey[1]), str(gkey[2]))
    s = str(gkey).replace(",", "-").replace(" ", "")
    parts = s.split("-")
    if len(parts) >= 3:
        return (parts[0], parts[1], parts[2])
    return ("", "", "")

# Normalizes a GPU 3‑tuple to a canonical key used by catalog
def _gid(c: _S, n: _S, g: _S) -> _S:
    return f"{c}-{n}-{g}"

# Returns the best matching frequency key present in fmap for a given choice
def _match_freq(fmap: Dict[_S, float], want: _S) -> _S:
    want = _S(want or "").strip()
    if not fmap:
        return _S("")
    if want and want in fmap:
        return want

    # numeric nearest-neighbor if possible (tie-break: smaller numeric key)
    try:
        w = _F(want)
        best = None
        best_d = 1e99
        best_num = None
        for k in fmap.keys():
            try:
                kn = _F(k)
                d = abs(kn - w)
                if (d < best_d) or (d == best_d and (best_num is None or kn < best_num)):
                    best_d = d
                    best = _S(k)
                    best_num = kn
            except Exception:
                pass
        if best is not None:
            return best
    except Exception:
        pass

    # deterministic fallback: smallest numeric key else lexicographic
    keys = [_S(k) for k in fmap.keys()]
    if not keys:
        return _S("")
    try:
        return min(keys, key=lambda k: _F(k))
    except Exception:
        return sorted(keys)[0]

def _needs_power_fields(obj: str) -> bool:
    o = (obj or "").lower()
    return ("power" in o) or ("min-power" in o) or ("efficiency" in o) or ("max-efficiency" in o)

def _norm_job(J: str) -> str:
    J = str(J or "")
    return J.strip()

def _norm_job_from_pkt(pkt) -> str:
    jid = pkt.get("Job_ID") or pkt.get("job_id")
    if jid:
        return _norm_job(jid)
    tid = str(pkt.get("Task_ID", ""))
    parts = tid.split("_")
    return _norm_job("_".join(parts[:2]) if len(parts) >= 2 else tid)

def _append_drop_to_csv(job_id: str, task_id: str, reason: str,
                        path: str = "dropped_jobs.csv") -> None:
    """Append a dropped job entry to CSV. Create file with header if needed."""
    file_exists = os.path.isfile(path)

    try:
        with open(path, "a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["Job_ID", "Task_ID", "Reason"])
            writer.writerow([job_id, task_id, reason])
    except Exception as e:
        print(f"[WARN] Failed to write dropped job to CSV: {e}")

def _get_od_value(state, snap, key, default=None):
    """
    Priority: snapshot[key] (if explicitly set) > snapshot.Optimizer_Defaults[key]
              > self.config.Optimizer_Defaults[key] (if available via state.policy/config)
              > default
    """
    # 1) direct snapshot override
    if isinstance(snap, dict) and key in snap and snap[key] is not None:
        return snap[key]

    # 2) snapshot Optimizer_Defaults
    od = (snap.get("Optimizer_Defaults") or {}) if isinstance(snap, dict) else {}
    if key in od and od[key] is not None:
        return od[key]

    # 3) config Optimizer_Defaults (if reachable)
    cfg = (
        getattr(state, "cfg", None)
        or getattr(state, "cfg_plain", None)
        or getattr(state, "config", None)  # keep as last fallback
        or getattr(getattr(state, "policy", None), "config", None)
    )

    od2 = (cfg.get("Optimizer_Defaults") or {}) if isinstance(cfg, dict) else {}
    if key in od2 and od2[key] is not None:
        return od2[key]

    return default

def _pick_start_freq_min(state, *, snap: dict, gid: str, fmap_here: dict) -> str:
    """
    Pick the minimum starting frequency key (string) for this GPU, respecting
    min_freq_map / min_freq_global with correct precedence:

      1) snap["min_freq_map"][gid]
      2) snap["min_freq_map"][gpu_type]  (optional if we can infer type)
      3) snap["min_freq_global"]
      4) snap["Optimizer_Defaults"]["min_freq_map"][gid]
      5) snap["Optimizer_Defaults"]["min_freq_map"][gpu_type]
      6) snap["Optimizer_Defaults"]["min_freq_global"]
      7) fallback: numeric min key in fmap_here

    Returns "" only if fmap_here is empty.
    """
    snap = snap if isinstance(snap, dict) else {}
    od = (snap.get("Optimizer_Defaults") or {}) if isinstance(snap.get("Optimizer_Defaults"), dict) else {}

    if not fmap_here:
        return ""

    def _f(x, default=None):
        try:
            if x in (None, ""):
                return default
            return float(x)
        except Exception:
            return default

    def _freq_keys_sorted():
        out = []
        for k in fmap_here.keys():
            try:
                out.append((float(k), str(k)))
            except Exception:
                pass
        out.sort()
        return out

    # Fallback: absolute minimum existing key
    keys_sorted = _freq_keys_sorted()
    if not keys_sorted:
        # fmap has non-numeric keys; just use stable sorted order
        return sorted(map(str, fmap_here.keys()))[0]

    min_key = keys_sorted[0][1]

    # --- optional: infer gpu_type if available ---
    gpu_type = ""
    try:
        # common patterns; adjust to actual catalog/state fields if different
        cat = getattr(state, "gpu_catalog", None)
        if cat is not None:
            # (a) explicit mapping by gid
            tmap = getattr(cat, "types", None)
            if isinstance(tmap, dict):
                gpu_type = str(tmap.get(gid, "") or "")
    except Exception:
        gpu_type = ""

    # --- read thresholds with correct precedence (snap overrides OD) ---
    snap_min_map = snap.get("min_freq_map", {}) or {}
    od_min_map   = od.get("min_freq_map", {}) or {}

    # pick threshold value if present
    thr = None
    if gid in snap_min_map:
        thr = _f(snap_min_map.get(gid), None)
    elif gpu_type and gpu_type in snap_min_map:
        thr = _f(snap_min_map.get(gpu_type), None)
    else:
        thr = _f(snap.get("min_freq_global", None), None)

    if thr is None:
        if gid in od_min_map:
            thr = _f(od_min_map.get(gid), None)
        elif gpu_type and gpu_type in od_min_map:
            thr = _f(od_min_map.get(gpu_type), None)
        else:
            thr = _f(od.get("min_freq_global", None), None)

    # If no threshold configured anywhere, return absolute minimum key
    if thr is None or thr <= 0:
        return min_key

    # Choose the minimum frequency key >= threshold
    for fk, ks in keys_sorted:
        if fk + 1e-12 >= thr:
            return ks

    # Threshold above all available freqs -> fall back to minimum key (safe)
    return min_key

def choose_fallback_freq(
    state,
    fmap_here: dict,
    key3,
    *,
    planned_freq: str = "",
    prefer: str = "",
    snap: dict = None,   
) -> str:
    """
    Deterministic fallback when planned_freq is empty/invalid.

    - If prefer is explicitly ("min"/"max"): pick min/max FREQUENCY KEY.
      For prefer="min", respect Optimizer_Defaults via _pick_start_freq_min():
        1) min_freq_map[gid]
        2) min_freq_global
        3) numeric min key in fmap_here

    - If prefer is not provided: derive from objective/strategy/freq_mode,
      then use rate-based selection with stable tie-breaks.
    """
    snap = snap if isinstance(snap, dict) else {}
    planned_freq = str(planned_freq or "").strip()

    # 0) planned already valid
    if planned_freq and planned_freq in fmap_here:
        return planned_freq
    if not fmap_here:
        return ""

    def _freqnum(k):
        try:
            return float(k)
        except Exception:
            return None

    def _as_gid_from_key3(k3) -> str:
        if isinstance(k3, (tuple, list)) and len(k3) == 3:
            return _gid(str(k3[0]), str(k3[1]), str(k3[2]))
        return str(k3)

    # 1) catalog default (only if it’s actually in fmap)
    cat = getattr(state, "gpu_catalog", None)
    defaults = (getattr(cat, "defaults", {}) or {}) if cat else {}

    df = ""
    try:
        if isinstance(key3, (tuple, list)) and len(key3) == 3:
            k3t = (str(key3[0]), str(key3[1]), str(key3[2]))
            df = defaults.get(k3t, "") or ""
        else:
            df = defaults.get(key3, "") or ""
    except Exception:
        df = ""

    df = str(df or "").strip()
    if df and df in fmap_here:
        return df

    prefer = str(prefer or "").strip().lower()

    # ============================================================
    # A) EXPLICIT prefer="min"/"max"  -> FREQUENCY-BASED SELECTION
    # ============================================================
    if prefer in ("min", "max"):
        if prefer == "min":
            gid = _as_gid_from_key3(key3)
            # single source of truth: _get_od_value reads from snap["Optimizer_Defaults"] first
            return _pick_start_freq_min(state, snap=snap, gid=gid, fmap_here=fmap_here) or ""
        # prefer == "max"
        try:
            return max(map(str, fmap_here.keys()), key=lambda k: float(k))
        except Exception:
            keys = sorted(map(str, fmap_here.keys()))
            return keys[-1] if keys else ""

    # ============================================================
    # B) NO explicit prefer -> derive from objective/strategy/freq_mode
    # ============================================================
    if not prefer:
        obj = str(snap.get("objective", getattr(state, "objective", "")) or "").lower()
        strat = str(snap.get("strategy", getattr(state, "strategy", "")) or "").lower()
        fmode = str(snap.get("freq_mode", getattr(state, "freq_mode", "")) or "").lower()


        if ("lat" in obj) or ("lat" in strat):
            prefer = "max"
        elif ("power" in obj) or ("efficiency" in obj) or ("power" in strat) or ("efficiency" in strat):
            prefer = "min"
        else:
            prefer = "min"

        if fmode == "fixed":
            prefer = "max"

    # deterministic pick by FREQUENCY KEY (not by rate)
    try:
        keys = list(map(str, fmap_here.keys()))
        if prefer == "max":
            return max(keys, key=lambda k: float(k))
        else:
            gid = _as_gid_from_key3(key3)
            return _pick_start_freq_min(state, snap=snap, gid=gid, fmap_here=fmap_here) or min(keys, key=lambda k: float(k))
    except Exception:
        keys = sorted(map(str, fmap_here.keys()))
        return keys[-1] if (prefer == "max" and keys) else (keys[0] if keys else "")


def build_pinned_assignments(state):
    pins = {}
    ja = getattr(state, "job_assignment", {}) or {}
    for jid, rec in ja.items():
        # rec may be ((C,N,G), freq) or (C,N,G)
        if isinstance(rec, (tuple, list)):
            if len(rec) == 2 and isinstance(rec[0], (tuple, list)) and len(rec[0]) == 3:
                gpu = tuple(map(str, rec[0]))
            elif len(rec) >= 3:
                gpu = tuple(map(str, rec[:3]))
            else:
                continue
            pins[str(jid)] = {"Cluster": gpu[0], "Node": gpu[1], "GPU": gpu[2]}
    return pins

# --------------------------------------
# FastPolicy
# --------------------------------------
class FastPolicy:
    __slots__ = (
        "config",
        "_modules",
        "objective",
        "_opt_warmed_local",
        "_sig_cache",
        "catalog",
        "store",
        "calls_total",
        "calls_first_time",
        # optional debug latch
        "_dbg_power_once",
    )

    def __init__(self, config: Dict[str, Any], modules: Dict[_S, Any], *, objective: _S = "latency", catalog=None, store=None):
        self.config = config or {}
        self._modules = modules or {}
        self.objective = (objective or "latency").lower()
        self._opt_warmed_local = False
        self._sig_cache = {}
        self.catalog = catalog if catalog is not None else None
        self.store = store
        self.calls_total = 0
        self.calls_first_time = 0
        self._dbg_power_once = False

    # -----------------------------
    # Module pick (hot path safe)
    # -----------------------------
    def _pick_module(self, objective: Optional[str]) -> Optional[Any]:
        raw = str(objective or self.objective or "latency").lower()

        if "lat" in raw:
            obj = "latency"
        elif "power" in raw:
            obj = "power"
        elif "eff" in raw:
            obj = "efficiency"
        else:
            obj = raw  # last resort direct match once

        return (
            self._modules.get(obj)
            or self._modules.get(raw)
            or self._modules.get("latency")
            or self._modules.get("power")
            or self._modules.get("efficiency")
        )

    # -----------------------------
    # Warmup: build static model once
    # -----------------------------
    def warmup(self, state=None) -> None:
        """
        Warmup: build/compile the optimizer's static model once so the first real solve is fast.

        Contract:
        - snapshot["_warmup"] == True => build-only, no decisions, no prints, no DVFS side effects.
        - MUST NOT run a real solve on dummy tasks.
        """
        if getattr(self, "_opt_warmed_local", False):
            return
        if state is not None and getattr(state, "_opt_warmed", False):
            return

        warmed_ok = False

        try:
            # ---- objective selection (state wins; fallback to self.objective) ----
            state_obj = (str(getattr(state, "objective", "") or "").lower() if state is not None else "")
            if "lat" in state_obj:
                obj = "latency"
            elif "power" in state_obj:
                obj = "power"
            elif "eff" in state_obj:
                obj = "efficiency"
            else:
                base = str(getattr(self, "objective", "latency") or "latency").lower()
                if "lat" in base:
                    obj = "latency"
                elif "power" in base:
                    obj = "power"
                elif "eff" in base:
                    obj = "efficiency"
                else:
                    obj = "latency"

            mod = self._pick_module(obj)
            if not mod:
                return

            # ---- build a runtime-shaped snapshot (do NOT handcraft) ----
            snap = self._snapshot_fast(state) if state is not None else (self._snapshot_fast_dummy())
            snap["_warmup"] = True

            # ---- call "build-only" path: empty task list ----
            if hasattr(mod, "solve_incremental"):
                mod.solve_incremental(snap, [], objective=obj)
            elif hasattr(mod, "solve"):
                mod.solve(snap, [], objective=obj)
            else:
                return

            warmed_ok = True

        except Exception:
            warmed_ok = False

        finally:
            if warmed_ok:
                self._opt_warmed_local = True
                if state is not None:
                    state._opt_warmed = True


    # --------------------------------------
    # Snapshot builder — ultra compact
    # --------------------------------------
    def _snapshot_fast(self, state) -> Dict[_S, Any]:
        cfg = self.config
        meta_cfg = cfg.get("GPU_Specs_Meta", {}) or {}
        od = cfg.get("Optimizer_Defaults", {}) or {}

        optimizer_power      = (cfg.get("optimizer_power")      or {})
        optimizer_latency    = (cfg.get("optimizer_latency")    or {})
        optimizer_efficiency = (cfg.get("optimizer_efficiency") or {})

        state_obj = str(getattr(state, "objective", "") or "").lower()

        if "lat" in state_obj:
            obj = "latency"
        elif "power" in state_obj:
            obj = "power"
            if obj == "power" and not optimizer_power:
                print("[WARN] optimizer_power config missing — power penalties disabled")
        elif "eff" in state_obj:
            obj = "efficiency"
        else:
            base = str(getattr(self, "objective", "latency") or "latency").lower()
            if "lat" in base:
                obj = "latency"
            elif "power" in base:
                obj = "power"
            elif "eff" in base:
                obj = "efficiency"
            else:
                obj = "latency"
        
        want_power = _needs_power_fields(obj)

        # freq mode is separate
        mode = str(getattr(state, "freq_mode", "adaptive") or "adaptive").lower()

        # cluster delay map
        cd_raw = (cfg.get("Cluster_Delay_Base_s") or {})
        cluster_delay_map: Dict[_S, float] = {}
        for cid, v in cd_raw.items():
            try:
                cluster_delay_map[_S(cid)] = _F(v)
            except Exception:
                continue

        gpu_to_freq = getattr(state, "gpu_to_freq", {}) or {}
        pending_service_time = getattr(state, "pending_service_time", {}) or {}
        pending_tasks = getattr(state, "pending_tasks", {}) or {}

        gpus: Dict[_S, Dict[_S, Any]] = {}

        # handy: Cluster_Config type lookup (config structure)
        cluster_cfg = (cfg.get("Cluster_Config") or {})

        for key3, fmap in (state.gpu_catalog.rates or {}).items():
            c, n, g = key3
            st = (getattr(state, "stations", {}).get(f"GPU:{c}:{n}:{g}")
                if hasattr(state, "stations") else None)

            if not fmap:
                continue

            rates_norm: Dict[_S, float] = {}
            for fk, rv in fmap.items():
                try:
                    r = _F(rv)
                    if r > 0.0:
                        rates_norm[_S(fk)] = r
                except Exception:
                    continue
            if not rates_norm:
                continue

            tail_real = (_F(getattr(st, "tail_time", getattr(st, "queue_time", 0.0)) or 0.0)
                        if st is not None else 0.0)
            queued_real = 0
            if st is not None:
                if getattr(st, "qlen", None) is not None:
                    queued_real = _I(st.qlen or 0)
                else:
                    q_obj = getattr(st, "q", None)
                    queued_real = _I(len(q_obj)) if q_obj is not None else 0


            key3t = (c, n, g)
            tail_total = max(0.0, tail_real + _F(pending_service_time.get(key3t, 0.0)))
            queued_total = queued_real + _I(pending_tasks.get(key3t, 0))

            base_delay = float(cluster_delay_map.get(_S(c), 0.0))
            gid = _gid(c, n, g)

            ginfo: Dict[_S, Any] = {
                "gid": gid,
                # "key3": (c, n, g),
                "key3": [c, n, g],
                "tail": tail_total,
                "tail_s": tail_total,
                "queued": queued_total,
                "freq": _S(gpu_to_freq.get(key3, "") or ""),
                "rates": rates_norm,
                "cluster": _S(c),
                "cluster_base_delay_s": base_delay,
            }

            if want_power:
                gpu_specs = cfg.get("GPU_Specs") or {}
                default_exp = float((cfg.get("GPU_Specs_Meta") or {}).get("phi_power_exp", 1.0))

                # Correct type lookup: Cluster_Config[C][N][G]["type"] OR state.gpu_catalog.types
                gtype = ""
                try:
                    if getattr(getattr(state, "gpu_catalog", None), "types", None):
                        gtype = (state.gpu_catalog.types or {}).get((c, n, g), "") or ""
                except Exception:
                    pass
                if not gtype:
                    gtype = (((cluster_cfg.get(str(c), {}) or {})
                            .get(str(n), {}) or {})
                            .get(str(g), {}) or {}).get("type", "") or ""

                spec = gpu_specs.get(gtype, {}) if gtype else {}

                p_idle = float(spec.get("P_static_W", spec.get("P_st", 0.0)))
                p_max  = float(spec.get("P_max_W", 0.0))
                pexp   = float(spec.get("power_exp", default_exp))

                ginfo.update({
                    "type": _S(gtype),
                    "P_idle_W": p_idle,
                    "P_max_W": p_max,
                    "power_exp": pexp,
                })

            gpus[_S(gid)] = ginfo

        links = (getattr(state, "R0_RC_Links", None) or cfg.get("R0_RC_Links") or {})
        


        # lambda_bg: derived live from current backlog
        horizon_s = float(getattr(state, "util_tau_s", 0.5) or 0.5)
        horizon_s = max(1e-6, horizon_s)

        # fb = getattr(state, "flop_backlog", {}) or {}    # (C,N,G) -> FLOPs
        # horizon_s = max(1e-6, float(getattr(state, "util_tau_s", 0.5) or 0.5))

        # lambda_bg = {}
        # for gid_str, ginfo in gpus.items():              # gpus dict
        #     c, n, g = ginfo["key3"]
        #     key = (c, n, g)
        #     W = float(fb.get(key, 0.0) or 0.0)
        #     lambda_bg[gid_str] = max(0.0, W) / horizon_s

        # snap["lambda_bg"] = lambda_bg


        # utilization_guess: take EWMA directly (0..1)
        util_raw = getattr(state, "util_ewma", {}) or {}
        utilization_guess = { _S(k): _F(v) for k, v in util_raw.items() if v not in (None, "") }

        lambda_bg = {}   # do NOT compute here (decide() overwrites before solve)


        # freq penalty weight depends on objective AND mode (but does not change obj!)
        if mode == "fixed":
            freq_penalty_weight = 0.0
        else:
            if "latency" in obj:
                freq_penalty_weight = float(optimizer_latency.get("freq_penalty_weight", 0.0))
            elif "power" in obj:
                freq_penalty_weight = float(optimizer_power.get("freq_penalty_weight", 0.0))
            else:
                freq_penalty_weight = float(optimizer_efficiency.get("freq_penalty_weight", 0.0))

        meta: Dict[_S, Any] = {
            "freq_units": _S(meta_cfg.get("freq_units", "MHz")),
            "cp_units": _S(meta_cfg.get("cp_units", "FLOPs_per_s_per_MHz")),
            "rate_units": _S(meta_cfg.get("rate_units", "FLOPs_per_s")),
            "scale_gflops": 1e9,
        }
        if want_power:
            meta.update({
                "phi_power_units": _S(meta_cfg.get("phi_power_units", "W_per_MHz")),
                "phi_power_exp": float(meta_cfg.get("phi_power_exp", 1.0)),
            })

        snap: Dict[_S, Any] = {
            "_warmup": False,
            "now": _F(_now(state)),
            "gpus": gpus,
            "links": links,
            "R0_RC_Links": links,
            "cluster_base_delay_s": cluster_delay_map,
            "pinned_assignments": {},
            "pinned_frequencies": {},
            "meta": meta,
            "Optimizer_Defaults": od,

            # ✅ objective is the REAL one
            "objective": obj,

            # ✅ keep separate flags
            "freq_mode": mode,
            "latency_mode": ("latency" in obj),
            "freq_penalty_weight": float(freq_penalty_weight),

            "gpu_jobs_served": dict(getattr(state, "gpu_jobs_served", {}) or {}),
            "lambda_bg": lambda_bg,
            "utilization_guess": utilization_guess,
            "optimizer_power": optimizer_power,
            "optimizer_latency": optimizer_latency,
            "optimizer_efficiency": optimizer_efficiency,
            "use_arrival_rates": bool(od.get("use_arrival_rates", True)),
            "pwl_points": int(od.get("pwl_points", 2)),
            "pwl_rho_max": _F(od.get("pwl_rho_max", 0.0)),
            "link_cap_penalty_weight": _F(od.get("link_cap_penalty_weight", od.get("link_penalty_weight", 0.0))),
            "gpu_cap_penalty_weight": _F(od.get("gpu_cap_penalty_weight", od.get("link_penalty_weight", 0.0))),
            "freq_tiebreak_eps": _F(od.get("freq_tiebreak_eps", 0.0)),
        }

        snap["solver"] = {
            "mip_gap": _F(od.get("mip_gap", 0.02)),
            "time_limit_s": _F(od.get("time_limit_s", 0.3)),
            "threads": _I(od.get("threads", 0)),
        }
        return snap


    # --------------------------------------
    # Decide — hot path with O(1) reuse
    # --------------------------------------
    # def decide(self, state, pkt: Dict[_S, Any]) -> Tuple[Tuple[_S, _S, _S], _S]:
    #     # Ensure bookkeeping dicts exist (do NOT reset)
    #     if not hasattr(state, "pending_load") or state.pending_load is None:
    #         state.pending_load = {}
    #     if not hasattr(state, "pending_service_time") or state.pending_service_time is None:
    #         state.pending_service_time = {}
    #     if not hasattr(state, "pending_tasks") or state.pending_tasks is None:
    #         state.pending_tasks = {}
    #     if not hasattr(state, "job_assignment") or state.job_assignment is None:
    #         state.job_assignment = {}
    #     if not hasattr(state, "job_freq_plan") or state.job_freq_plan is None:
    #         state.job_freq_plan = {}
    #     if not hasattr(state, "x_job") or state.x_job is None:
    #         state.x_job = {}
    #     if not hasattr(state, "gpu_freq_plan_fixed") or state.gpu_freq_plan_fixed is None:
    #         state.gpu_freq_plan_fixed = {}
    #     if not hasattr(state, "job_gpu_src") or state.job_gpu_src is None:
    #         state.job_gpu_src = {}

    #     if not hasattr(state, "gpu_jobs_served") or state.gpu_jobs_served is None:
    #         state.gpu_jobs_served = {}  # gid -> int
    #     # if not hasattr(state, "lambda_bg") or state.lambda_bg is None:
    #     #     state.lambda_bg = {}        # gid -> FLOPs/s
    #     # if not hasattr(state, "lambda_bg_jobs") or state.lambda_bg_jobs is None:
    #     #     state.lambda_bg_jobs = {}   # Job_ID -> {"gid": str, "rate": float}
    #     if not hasattr(state, "dropped_jobs") or state.dropped_jobs is None:
    #         state.dropped_jobs = set()

    #     # ---- REUSE / DROP-BY-JOB PATH ---------------------------------------
    #     jid = pkt.get("Job_ID") or pkt.get("job_id")
    #     if jid:
    #         J = _norm_job_from_pkt(pkt)
    #     else:
    #         tid0 = str(pkt.get("Task_ID", ""))
    #         parts = tid0.split("_")
    #         J = "_".join(parts[:2]) if len(parts) >= 2 else tid0

    #     if J in state.dropped_jobs:
    #         pkt["GPU_Decision_Source"] = "drop-job-already-dropped"
    #         return DROP_GPU, ""
    #         # return None, _S("")

    #     if J in state.job_assignment:
    #         gpu_key, freq_to_return = self._reuse_gpu_and_freq_for_job(state, J)
    #         pkt.setdefault("GPU_Decision_Source", "reuse-job")

    #         self._reserve_claim(
    #             state,
    #             gpu_key,
    #             freq_to_return,
    #             _F(pkt.get("Workload_FLOPs", 0.0) or 0.0),
    #         )

    #         gid = _gid(*gpu_key)
    #         # try:
    #         #     state.gpu_jobs_served[gid] = int(state.gpu_jobs_served.get(gid, 0) or 0) + 1
    #         # except Exception:
    #         #     state.gpu_jobs_served[gid] = 1

    #         return gpu_key, _S(freq_to_return)

    #     self.calls_first_time += 1

    #     # ---- FIRST-TIME PATH -----------------------------------------------
    #     task_id = _S(pkt.get("Task_ID", ""))
    #     key = (J, task_id)
    #     pkts = getattr(state, "pregen_by_task", {}).get(key, [pkt])

    #     total_flops = 0
    #     ul_total_kB = 0
    #     for p in pkts:
    #         total_flops += _I(p.get("Workload_FLOPs", 0) or 0)
    #         ul_total_kB += _I(p.get("Packet_Size_KB", 0) or 0)

    #     if not ul_total_kB:
    #         ul_total_kB = _I(pkt.get("Task_UL_Total_kB", 0) or 0)
    #     dl_total_kB = _I(pkt.get("Task_DL_Total_kB", 0) or 0)
    #     deadline = _F(pkt.get("Task_Deadline", float("inf")))

    #     lam = pkt.get("Lambda") or pkt.get("Task_Arrival_Rate")
    #     lam = _F(lam or 0.0)
    #     if lam <= 0.0:
    #         ts = (self.config.get("Task_Settings") or {})
    #         fps = _F(ts.get("SERVICE_ARRIVAL_RATE_fps", ts.get("ARRIVAL_RATE_fps", 30)) or 0.0)
    #         stride = _I(ts.get("STRIDE", ts.get("stride", 1)) or 1)
    #         if fps > 0.0:
    #             lam = fps / max(1, stride)
    #     if lam <= 0.0:
    #         raise RuntimeError("Streaming-only mode: positive Lambda required.")

    #     new_task = {
    #         "Task_ID": task_id,
    #         "Job_ID": J,
    #         "Workload_FLOPs": total_flops,
    #         "Task_Deadline": deadline,
    #         "UL_Total_kB": ul_total_kB,
    #         "DL_Total_kB": dl_total_kB,
    #         "Lambda": lam,
    #     }

    #     snap = self._snapshot_fast(state)
    #     snap["pinned_assignments"]  = build_pinned_assignments(state)

    #     mod = self._pick_module(getattr(state, "objective", self.objective))
    #     if not mod:
    #         raise RuntimeError(f"No optimizer module configured for objective '{self.objective}'.")

    #     entry = (
    #         "solve_incremental" if hasattr(mod, "solve_incremental")
    #         else "solve" if hasattr(mod, "solve")
    #         else None
    #     )
    #     if not entry:
    #         raise RuntimeError("No compatible entrypoint in optimizer module.")

    #     opt_fn = getattr(mod, entry)
    #     kws = {
    #         "pinned_assignments": (snap.get("pinned_assignments") or {}),
    #         "pinned_frequencies": (snap.get("pinned_frequencies") or {}),
    #         "objective": _S(getattr(state, "objective", "")).lower(),
    #     }

    #     sig = self._sig_cache.get(id(opt_fn))
    #     if sig is None:
    #         sig = inspect.signature(opt_fn)
    #         self._sig_cache[id(opt_fn)] = sig
    #     allowed = {k: v for k, v in kws.items() if k in sig.parameters}

    #     ans = opt_fn(snap, [new_task], **allowed)
    #     if not (isinstance(ans, dict) and ("assignments" in ans) and ("frequencies" in ans)):
    #         raise RuntimeError("Optimizer returned empty or invalid result.")

    #     assigns = ans.get("assignments") or {}
    #     status_str = str(ans.get("status", "")).lower()
    #     if not hasattr(state, "admit_stats") or state.admit_stats is None:
    #         state.admit_stats = {
    #             "admitted": 0,
    #             "dropped": 0,
    #             "dropped_jobs": [],
    #             "dropped_job_ids": set(),
    #         }

    #     def _record_drop(reason: str) -> None:
    #         st = state.admit_stats
    #         st.setdefault("dropped_job_ids", set())
    #         if J not in st["dropped_job_ids"]:
    #             st["dropped"] += 1
    #             st["dropped_jobs"].append({"Job_ID": J, "Task_ID": task_id, "reason": reason})
    #             st["dropped_job_ids"].add(J)
    #             try:
    #                 _append_drop_to_csv(J, task_id, reason)
    #             except Exception:
    #                 pass
    #         state.dropped_jobs.add(J)

    #     if (not assigns) or status_str in ("infeasible", "no_solution", "noop"):
    #         _record_drop(status_str or "no_assignment")
    #         pkt["GPU_Decision_Source"] = f"drop-{status_str or 'no_assignment'}"
    #         return DROP_GPU, ""

    #     rec = assigns.get(task_id) or assigns.get(J)
    #     if not rec and len(assigns) == 1:
    #         rec = next(iter(assigns.values()))
    #     if not rec:
    #         _record_drop(f"no_assignment_for_{task_id}")
    #         pkt["GPU_Decision_Source"] = "drop-no-assignment"
    #         return DROP_GPU, ""

    #     st = state.admit_stats
    #     st.setdefault("admitted_job_ids", set())
    #     if J not in st["admitted_job_ids"]:
    #         st["admitted"] += 1
    #         st["admitted_job_ids"].add(J)


    #     gpu_key = (_S(rec["Cluster"]), _S(rec["Node"]), _S(rec["GPU"]))
    #     gid = _gid(*gpu_key)
    #     freqs = ans.get("frequencies") or {}
    #     planned_freq = _S(freqs.get(gid) or rec.get("Frequency", ""))
    #     key3 = _norm_gkey(gpu_key)
    #     fmap_here = ((getattr(getattr(state,"gpu_catalog",None),"rates",{}) or {}).get(key3, {}) or {})
        
    #     if planned_freq:
    #         planned_freq = _match_freq(fmap_here, planned_freq)

    #     # if not planned_freq:
    #     #     df = (getattr(getattr(state,"gpu_catalog",None),"defaults",{}) or {}).get(key3, "")
    #     #     if df and df in fmap_here:
    #     #         planned_freq = df
    #     #     elif fmap_here:
    #     #         # safer deterministic fallback: slowest
    #     #         planned_freq = min(fmap_here, key=lambda fk: float(fmap_here[fk]))
    #     #     else:
    #     #         planned_freq = _S("")

    #     if not planned_freq:
    #         key3 = _norm_gkey(gpu_key)                # ('C','N','G')
    #         planned_freq = choose_fallback_freq(state, fmap_here, key3, planned_freq=planned_freq)

    #     # DO NOT retune here (DVFS must happen only at GPU boundary sites)
    #     pkt["Planned_Frequency"] = _S(planned_freq)  # used later by on_arrive_gpu/on_finish_gpu


    #     # cur_hw = (getattr(state, "gpu_to_freq", {}) or {}).get(key3, "")
    #     # cur_norm = _match_freq(fmap_here, _S(cur_hw)) if cur_hw else ""
    #     # if planned_freq and planned_freq != cur_norm:
    #     #     set_gpu_freq_fn = globals().get("safe_set_gpu_freq") \
    #     #        or globals().get("set_gpu_freq") \
    #     #        or getattr(state, "set_gpu_freq", None)
    #     #     if set_gpu_freq_fn:
    #     #         set_gpu_freq_fn(
    #     #             state,
    #     #             gpu_key,
    #     #             planned_freq,
    #     #             reason="optimizer-retune",
    #     #             fmap=fmap_here,
    #     #             when=getattr(getattr(state, "fel", None), "t", 0.0),
    #     #             origin="fast_decider",
    #     #             trigger_job_id=J,
    #     #             trigger_task_id=task_id,
    #     #         )

    #     state.job_assignment[J] = (gpu_key, _S(planned_freq))
    #     state.x_job[J] = gpu_key
    #     state.job_freq_plan[J] = {"gpu": gpu_key, "freq": _S(planned_freq), "src": "optimizer"}
    #     state.job_gpu_src[J] = "optimizer"

    #     if getattr(self, "store", None) and getattr(self.store, "mode", "readwrite") != "readonly":
    #         try:
    #             self.store.set(J, {"Cluster": gpu_key[0], "Node": gpu_key[1], "GPU": gpu_key[2], "Frequency": _S(planned_freq)})
    #             self.store.set_freq(gid, _S(planned_freq))
    #         except Exception:
    #             pass

    #     self._reserve_claim(state, gpu_key, planned_freq, _F(new_task.get("Workload_FLOPs", 0.0) or 0.0))
    #     pkt["GPU_Decision_Source"] = "optimizer"

    #     # # background FLOP/s add (once per job)
    #     # try:
    #     #     job_rate_flops = float(total_flops) * float(lam)
    #     # except Exception:
    #     #     job_rate_flops = 0.0

    #     # if job_rate_flops > 0.0:
    #     #     state.lambda_bg[gid] = float(state.lambda_bg.get(gid, 0.0) or 0.0) + job_rate_flops
    #     #     state.lambda_bg_jobs[J] = {"gid": gid, "rate": job_rate_flops}

    #     try:
    #         state.gpu_jobs_served[gid] = int(state.gpu_jobs_served.get(gid, 0) or 0) + 1
    #     except Exception:
    #         state.gpu_jobs_served[gid] = 1

    #     return gpu_key, _S(planned_freq)

    def decide(self, state, pkt: Dict[_S, Any]) -> Tuple[Tuple[_S, _S, _S], _S]:
        # Ensure bookkeeping dicts exist (do NOT reset)
        if not hasattr(state, "flop_backlog") or state.flop_backlog is None:
            state.flop_backlog = {}   # gid -> backlog FLOPs
        if not hasattr(state, "pending_service_time") or state.pending_service_time is None:
            state.pending_service_time = {}
        if not hasattr(state, "pending_tasks") or state.pending_tasks is None:
            state.pending_tasks = {}
        if not hasattr(state, "job_assignment") or state.job_assignment is None:
            state.job_assignment = {}
        if not hasattr(state, "job_freq_plan") or state.job_freq_plan is None:
            state.job_freq_plan = {}
        if not hasattr(state, "x_job") or state.x_job is None:
            state.x_job = {}
        if not hasattr(state, "gpu_freq_plan_fixed") or state.gpu_freq_plan_fixed is None:
            state.gpu_freq_plan_fixed = {}
        if not hasattr(state, "job_gpu_src") or state.job_gpu_src is None:
            state.job_gpu_src = {}
        if not hasattr(state, "gpu_jobs_served") or state.gpu_jobs_served is None:
            state.gpu_jobs_served = {}  # gid -> int
        if not hasattr(state, "dropped_jobs") or state.dropped_jobs is None:
            state.dropped_jobs = set()
        # Reservation latch: reserve ONCE per (Job, Task_ID)
        if not hasattr(state, "reserved_tasks") or state.reserved_tasks is None:
            state.reserved_tasks = set()  # {(J, task_id_pkt)}
            # one-time per-GPU latch: only do "start-low" once per gid
        if not hasattr(state, "low_first_done") or state.low_first_done is None:
            state.low_first_done = set()   # {gid strings}


        # ---- Resolve normalized job id J ----
        if pkt.get("Job_ID") is not None or pkt.get("job_id") is not None:
            J = _norm_job_from_pkt(pkt)
        else:
            tid0 = str(pkt.get("Task_ID", ""))
            parts = tid0.split("_")
            J = "_".join(parts[:2]) if len(parts) >= 2 else tid0

        task_id_pkt = _S(pkt.get("Task_ID", ""))  # exact Task_ID
        key_task = (str(J), str(task_id_pkt))
        # print("[DBG] Task_ID seen by decide():", pkt.get("Task_ID"))

        # ---- WARMUP GUARD (add HERE) ----
        if not getattr(state, "allow_optimizer_solve", True):
            pkt["GPU_Decision_Source"] = "warmup-skip"
            return DROP_GPU, ""

        # ---- Drop-by-job ----
        if J in state.dropped_jobs:
            pkt["GPU_Decision_Source"] = "drop-job-already-dropped"
            return DROP_GPU, ""

        def _task_total_flops_once() -> float:
            """
            Workload_FLOPs is PER-TASK in the pipeline, but decide() is called PER-PACKET.
            Reserve once per (J, Task_ID). If pkt has task FLOPs, use it; otherwise infer.
            """
            try:
                w = float(pkt.get("Workload_FLOPs", 0.0) or 0.0)
            except Exception:
                w = 0.0
            if w > 0.0:
                return w
            try:
                print("---Fallback in FastPolicy.decider---")
                pkts = getattr(state, "pregen_by_task", {}).get((J, task_id_pkt), [pkt])
                vals = []
                for p in pkts:
                    try:
                        vals.append(float(p.get("Workload_FLOPs", 0.0) or 0.0))
                    except Exception:
                        pass
                return float(max(vals) if vals else 0.0)
            except Exception:
                return 0.0

        def _reserve_if_needed(gpu_key, freq_to_return):
            if key_task in state.reserved_tasks:
                return
            total_flops = _task_total_flops_once()
            self._reserve_claim(state, gpu_key, freq_to_return, total_flops)
            state.reserved_tasks.add(key_task)

        def _canon_objective(raw: str) -> str:
            s = (raw or "").lower()
            if "lat" in s:  return "latency"
            if "power" in s: return "power"
            if "eff" in s:   return "efficiency"
            return "latency"
        
        # ============================================================
        # REUSE PATH: job already pinned → skip optimizer
        # ============================================================
        if J in state.job_assignment:
            gpu_key, freq_to_return = self._reuse_gpu_and_freq_for_job(state, J)

            # normalize for reservation (critical)
            rates_map = (getattr(getattr(state, "gpu_catalog", None), "rates", {}) or {})
            fmap = rates_map.get(gpu_key) or rates_map.get(_norm_gkey(gpu_key), {}) or {}
            if fmap:
                freq_to_return = _match_freq(fmap, _S(freq_to_return)) or _S(freq_to_return)

            pkt.setdefault("GPU_Decision_Source", "reuse-job")
            _reserve_if_needed(gpu_key, freq_to_return)
            return gpu_key, _S(freq_to_return)


        self.calls_first_time += 1

        # ============================================================
        # FIRST-TIME PATH
        # ============================================================
        task_id = task_id_pkt
        key = (J, task_id)
        pkts = getattr(state, "pregen_by_task", {}).get(key, [pkt])

        total_flops = _task_total_flops_once()

        ul_total_kB = 0
        for p in pkts:
            ul_total_kB += _I(p.get("Packet_Size_KB", 0) or 0)

        if not ul_total_kB:
            ul_total_kB = _I(pkt.get("Task_UL_Total_kB", 0) or 0)
        dl_total_kB = _I(pkt.get("Task_DL_Total_kB", 0) or 0)
        deadline = _F(pkt.get("Task_Deadline", float("inf")))

        lam = pkt.get("Lambda") or pkt.get("Task_Arrival_Rate")
        lam = _F(lam or 0.0)
        if lam <= 0.0:
            ts = (self.config.get("Task_Settings") or {})
            fps = _F(ts.get("SERVICE_ARRIVAL_RATE_fps", ts.get("ARRIVAL_RATE_fps", 30)) or 0.0)
            stride = _I(ts.get("STRIDE", ts.get("stride", 1)) or 1)
            if fps > 0.0:
                lam = fps / max(1, stride)
        if lam <= 0.0:
            raise RuntimeError("Streaming-only mode: positive Lambda required.")

        new_task = {
            "Task_ID": task_id,
            "Job_ID": J,
            "Workload_FLOPs": total_flops,
            "Task_Deadline": deadline,
            "UL_Total_kB": ul_total_kB,
            "DL_Total_kB": dl_total_kB,
            "Lambda": lam,
        }

        snap = self._snapshot_fast(state)
        snap["pinned_assignments"] = build_pinned_assignments(state)

        # Overwrite lambda_bg for optimizer (single source of truth: flop_backlog)
        fb = getattr(state, "flop_backlog", {}) or {}
        horizon_s = float(getattr(state, "util_tau_s", 0.5) or 0.5)
        horizon_s = max(1e-6, horizon_s)

        lambda_bg = {}
        for raw_key, backlog_flops in fb.items():
            if isinstance(raw_key, (tuple, list)) and len(raw_key) == 3:
                gid = f"{raw_key[0]}-{raw_key[1]}-{raw_key[2]}"
            else:
                gid = str(raw_key)
            lambda_bg[gid] = float(backlog_flops or 0.0) / horizon_s  # FLOPs/s
        snap["lambda_bg"] = lambda_bg

        # ---- pick optimizer module ----
        objective_raw = _S(getattr(state, "objective", self.objective) or "")
        objective = _canon_objective(objective_raw)
        mod = self._pick_module(objective)
        if not mod:
            raise RuntimeError(f"No optimizer module configured for objective '{objective}'.")

        entry = (
            "solve_incremental" if hasattr(mod, "solve_incremental")
            else "solve" if hasattr(mod, "solve")
            else None
        )
        if not entry:
            raise RuntimeError("No compatible entrypoint in optimizer module.")
        opt_fn = getattr(mod, entry)

        kws = {
            "pinned_assignments": (snap.get("pinned_assignments") or {}),
            "pinned_frequencies": (snap.get("pinned_frequencies") or {}),
            "objective": objective,
        }
        sig = self._sig_cache.get(id(opt_fn))
        if sig is None:
            sig = inspect.signature(opt_fn)
            self._sig_cache[id(opt_fn)] = sig
        allowed = {k: v for k, v in kws.items() if k in sig.parameters}

        ans = opt_fn(snap, [new_task], **allowed)
        if not (isinstance(ans, dict) and ("assignments" in ans) and ("frequencies" in ans)):
            raise RuntimeError("Optimizer returned empty or invalid result.")

        assigns = ans.get("assignments") or {}
        status_str = str(ans.get("status", "")).lower()

        if not hasattr(state, "admit_stats") or state.admit_stats is None:
            state.admit_stats = {
                "admitted": 0,
                "dropped": 0,
                "dropped_jobs": [],
                "dropped_job_ids": set(),
            }

        def _record_drop(reason: str) -> None:
            st = state.admit_stats
            st.setdefault("dropped_job_ids", set())
            if J not in st["dropped_job_ids"]:
                st["dropped"] += 1
                st["dropped_jobs"].append({"Job_ID": J, "Task_ID": task_id, "reason": reason})
                st["dropped_job_ids"].add(J)
                try:
                    _append_drop_to_csv(J, task_id, reason)
                except Exception:
                    pass
            state.dropped_jobs.add(J)

        if (not assigns) or status_str in ("infeasible", "no_solution", "noop"):
            _record_drop(status_str or "no_assignment")
            pkt["GPU_Decision_Source"] = f"drop-{status_str or 'no_assignment'}"
            return DROP_GPU, ""

        rec = assigns.get(task_id) or assigns.get(J)
        if not rec and len(assigns) == 1:
            rec = next(iter(assigns.values()))
        if not rec:
            _record_drop(f"no_assignment_for_{task_id}")
            pkt["GPU_Decision_Source"] = "drop-no-assignment"
            return DROP_GPU, ""

        st = state.admit_stats
        st.setdefault("admitted_job_ids", set())
        if J not in st["admitted_job_ids"]:
            st["admitted"] += 1
            st["admitted_job_ids"].add(J)

        gpu_key = (_S(rec["Cluster"]), _S(rec["Node"]), _S(rec["GPU"]))
        gid = _gid(*gpu_key)

        rates_map = (getattr(getattr(state, "gpu_catalog", None), "rates", {}) or {})
        fmap_here = rates_map.get(gpu_key) or {}
        if not fmap_here:
            try:
                fmap_here = rates_map.get(_norm_gkey(gpu_key)) or {}
            except Exception:
                fmap_here = {}
        key3 = _norm_gkey(gpu_key)

        # ============================================================
        # FREQUENCY SELECTION
        # ============================================================
        strategy  = _S(getattr(state, "strategy", "") or "").lower()
        freq_mode = _S(getattr(state, "freq_mode", "") or snap.get("freq_mode", "") or "").lower()

        # treat as adaptive if either knob says adaptive
        is_adaptive = (freq_mode == "adaptive") or strategy.endswith("adaptive") or ("adaptive" in strategy)
        
        # ============================================================
        # FREQUENCY SELECTION (OPTIMIZER-ONLY)
        # ============================================================
        freqs = ans.get("frequencies") or {}

        # optimizer-provided freq (prefer 'frequencies[gid]', then rec['Frequency'])
        planned_freq = _S(freqs.get(gid) or rec.get("Frequency", "")).strip()

        # normalize to a key that exists in fmap
        if planned_freq:
            planned_freq = _match_freq(fmap_here, planned_freq) or ""

        # pinned frequency (if you want pins to override optimizer, keep this; else remove it)
        pinmap = (snap.get("pinned_frequencies") or {})
        pinned_here = ""
        if gid in pinmap and pinmap.get(gid) not in (None, ""):
            pinned_here = _match_freq(fmap_here, _S(pinmap.get(gid))) or ""

        # # If you want STRICT optimizer-only, comment out the next two lines.
        # if pinned_here:
        #     planned_freq = pinned_here

        # final fallback if optimizer didn't return a valid key
        if not planned_freq:
            planned_freq = choose_fallback_freq(state, fmap_here, key3, planned_freq="", prefer="", snap=snap)
        if not planned_freq:
            planned_freq = choose_fallback_freq(state, fmap_here, key3, planned_freq="", prefer="min", snap=snap)

        pkt["GPU_Decision_Source"] = "optimizer"
        pkt["Planned_Frequency"] = _S(planned_freq)


        # # only force LOW-FIRST for adaptive power/efficiency
        # force_low_first = (
        #     is_adaptive
        #     and (("power" in objective) or ("efficiency" in objective))
        #     and (gid not in state.low_first_done)
        # )

        # # (A) pinned wins if present
        # pinned_here = ""
        # if gid in pinmap and pinmap.get(gid) not in (None, ""):
        #     pinned_here = _match_freq(fmap_here, _S(pinmap.get(gid))) or ""

        # planned_freq = ""
        # if force_low_first:
        #     planned_freq = pinned_here
        #     if not planned_freq:
        #         planned_freq = _pick_start_freq_min(state, snap=snap, gid=gid, fmap_here=fmap_here)

        #         # planned_freq = choose_fallback_freq(
        #         #     state, fmap_here, key3,
        #         #     planned_freq="",
        #         #     prefer="min",         
        #         #     snap=snap
        #         # )
        #     pkt["GPU_Decision_Source"] = f"optimizer-gpu-only-start-low:{objective}"

        # else:
        #     # allow optimizer frequency (fixed mode / latency-adaptive / etc.)
        #     planned_freq = _S(freqs.get(gid) or rec.get("Frequency", "")).strip()
        #     if planned_freq:
        #         planned_freq = _match_freq(fmap_here, planned_freq) or ""
        #     if not planned_freq:
        #         planned_freq = choose_fallback_freq(state, fmap_here, key3, planned_freq="", prefer="", snap=snap)

        #     pkt["GPU_Decision_Source"] = "optimizer"

        # if planned_freq and fmap_here:
        #     planned_freq = _match_freq(fmap_here, _S(planned_freq)) or ""

        # # final safety
        # if not planned_freq:
        #     planned_freq = choose_fallback_freq(state, fmap_here, key3, planned_freq="", prefer="min", snap=snap)

        # if force_low_first:
        #     state.low_first_done.add(gid)

        # pkt["Planned_Frequency"] = _S(planned_freq)

        # ---- persist job plan (this is what reuse path will latch) ----
        state.job_assignment[J] = (gpu_key, _S(planned_freq))
        state.x_job[J] = gpu_key
        state.job_freq_plan[J] = {
            "gpu": gpu_key,
            "freq": _S(planned_freq),
            "src": ("optimizer"),
            # "src": ("optimizer-gpu-only-start-low" if force_low_first else "optimizer"),
            "objective": objective,
            "freq_mode": freq_mode,
            "strategy": strategy,
        }
        state.job_gpu_src[J] = "optimizer"

        if getattr(self, "store", None) and getattr(self.store, "mode", "readwrite") != "readonly":
            try:
                self.store.set(J, {"Cluster": gpu_key[0], "Node": gpu_key[1], "GPU": gpu_key[2], "Frequency": _S(planned_freq)})
                self.store.set_freq(gid, _S(planned_freq))
            except Exception:
                pass

        _reserve_if_needed(gpu_key, planned_freq)

        try:
            state.gpu_jobs_served[gid] = int(state.gpu_jobs_served.get(gid, 0) or 0) + 1
        except Exception:
            state.gpu_jobs_served[gid] = 1

        # Debug
        print("[DBG] objective:", objective, "freq_mode:", freq_mode, "gid:", gid, "optimizer_freq:", freqs.get(gid), "planned_freq:", planned_freq)
        # print("[DBG] pinned_here:", pinned_here, "used_optimizer_freq:", (not force_low_first))

        return gpu_key, _S(planned_freq)

    # --------------------------------------
    # internals
    # --------------------------------------

    def _reserve_claim(self, state, gpu_key, freq_choice, work_flops: float) -> None:
        """
        Reservation: add this task's FLOPs into the GPU backlog so snapshot->lambda_bg works.
        Must be called ONCE per (Job_ID, Task_ID).
        """
        # ---- ensure fields exist ----
        if not hasattr(state, "flop_backlog") or state.flop_backlog is None:
            state.flop_backlog = {}
        if not hasattr(state, "pending_service_time") or state.pending_service_time is None:
            state.pending_service_time = {}
        if not hasattr(state, "pending_tasks") or state.pending_tasks is None:
            state.pending_tasks = {}

        key3 = _norm_gkey(gpu_key)  # ('C1','N2','G1') always

        wf = float(work_flops or 0.0)
        if wf > 0.0:
            state.flop_backlog[key3] = float(state.flop_backlog.get(key3, 0.0)) + wf
            # print(f"[RESERVE(FastPolicy)] key3={key3} +{wf} FLOPs -> backlog={state.flop_backlog.get(key3,0.0)}")

        rate = 0.0
        try:
            fmap = (getattr(getattr(state, "gpu_catalog", None), "rates", None)
                    or getattr(self.catalog, "rates", None)
                    or {})
            fmap = (fmap.get(key3) or fmap.get(gpu_key) or {})
            run_key = _match_freq(fmap, str(freq_choice)) or str(freq_choice)
            rate = float(fmap.get(run_key, 0.0) or 0.0)  # FLOPs/s
        except Exception:
            rate = 0.0

        est_service_s = (wf / rate) if (wf > 0.0 and rate > 0.0) else 0.0
        state.pending_service_time[key3] = float(state.pending_service_time.get(key3, 0.0)) + est_service_s
        state.pending_tasks[key3] = int(state.pending_tasks.get(key3, 0) or 0) + 1


    def _reuse_gpu_and_freq_for_job(self, state, J: _S) -> Tuple[Tuple[_S, _S, _S], _S]:
        ja = state.job_assignment.get(J)
        if isinstance(ja, (tuple, list)):
            if len(ja) == 2 and isinstance(ja[0], (tuple, list)) and len(ja[0]) == 3:
                gpu_key = (_S(ja[0][0]), _S(ja[0][1]), _S(ja[0][2]))
                pinned_freq = _S(ja[1])
            elif len(ja) >= 3:
                gpu_key = (_S(ja[0]), _S(ja[1]), _S(ja[2]))
                pinned_freq = _S("")
            else:
                raise RuntimeError(f"Invalid job_assignment entry for {J}")
        else:
            raise RuntimeError(f"No GPU assignment recorded yet for job {J}")

        # --- fetch fmap robustly (supports tuple keys or normalized keys) ---
        rates = (getattr(getattr(state, "gpu_catalog", None), "rates", {}) or {})
        fmap = rates.get(gpu_key) or {}
        if not fmap:
            try:
                key3 = _norm_gkey(gpu_key)
                fmap = rates.get(key3) or {}
            except Exception:
                pass
        if not fmap:
            raise RuntimeError(f"GPU {gpu_key} has no rate table in reuse path.")

        def _norm_in_fmap(candidate: _S) -> _S:
            if not candidate:
                return _S("")
            return _match_freq(fmap, _S(candidate))

        # --- mode/objective detection ---
        fmode = str(getattr(state, "freq_mode", "") or "").lower()
        obj   = str(getattr(state, "objective", "") or "").lower()
        strat = str(getattr(state, "strategy", "") or "").lower()

        # adaptive power/eff strategies must start low unless explicitly pinned
        adaptive_start_low = (fmode == "adaptive") and (
            ("power" in obj) or ("efficiency" in obj) or
            ("power" in strat) or ("efficiency" in strat)
        )

        # ---- candidates (pinned wins first) ----
        plan = state.job_freq_plan.get(J, {})
        cand_plan = _norm_in_fmap(plan.get("freq") if isinstance(plan, dict) else plan)
        cand_pinned = _norm_in_fmap(pinned_freq)

        # # If adaptive-start-low: only honor explicit pins.
        # # Otherwise reuse planned/current/def as before.
        # if adaptive_start_low:
        #     if cand_pinned:
        #         return gpu_key, _S(cand_pinned)

        #     # Start LOW deterministically (uses Optimizer_Defaults min_freq_map/min_freq_global/min key)
        #     try:
        #         key3 = _norm_gkey(gpu_key)
        #     except Exception:
        #         key3 = gpu_key
        #     freq = choose_fallback_freq(state, fmap, key3, planned_freq="", prefer="min", snap=None)
        #     if not freq:
        #         raise RuntimeError(f"Could not resolve a min frequency for GPU {gpu_key} in adaptive reuse path.")
        #     return gpu_key, _S(freq)

        # ---- legacy reuse behavior (non-adaptive or latency-ish) ----
        cur_raw = ((getattr(state, "gpu_to_freq", {}) or {}).get(gpu_key, ""))
        cand_cur = _norm_in_fmap(cur_raw)

        defaults = getattr(getattr(state, "gpu_catalog", None), "defaults", {}) or {}
        cand_def = _norm_in_fmap(defaults.get(gpu_key, "")) or _norm_in_fmap(defaults.get(getattr(locals().get("key3", None), "__str__", lambda: "")(), ""))

        # deterministic fastest fallback
        try:
            key3 = _norm_gkey(gpu_key)
        except Exception:
            key3 = gpu_key
        cand_fastest = choose_fallback_freq(state, fmap, key3, planned_freq="", prefer="max", snap=None)

        # Prefer the stored planned freq (optimizer) first
        freq = cand_plan or cand_pinned or cand_cur or cand_def or cand_fastest

        # objective = str(getattr(state, "objective", "") or "").lower()
        # strategy  = str(getattr(state, "strategy", "") or "").lower()
        # freq_mode = str(getattr(state, "freq_mode", "") or "").lower()
        # is_adaptive = (freq_mode == "adaptive") or strategy.endswith("adaptive") or ("adaptive" in strategy)
        # force_low_first = is_adaptive and (("power" in objective) or ("efficiency" in objective))

        # if force_low_first:
        #     cand_min = choose_fallback_freq(state, fmap, key3, planned_freq="", prefer="min", snap=None)
        #     freq = cand_cur or cand_plan or cand_pinned or cand_def or cand_min
        # else:
        #     cand_fastest = choose_fallback_freq(state, fmap, key3, planned_freq="", prefer="max", snap=None)
        #     freq = cand_plan or cand_pinned or cand_cur or cand_def or cand_fastest

        if not freq:
            raise RuntimeError(f"Could not resolve a frequency for GPU {gpu_key} in reuse path.")

        return gpu_key, _S(freq)

