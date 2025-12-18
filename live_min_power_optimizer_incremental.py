# live_min_power_optimizer_incremental_fast.py
# --------------------------------------------------------------
# Incremental MIN-POWER optimizer (Gurobi), STREAMING-ONLY (one task),
# reworked to mirror the runtime optimizations from min-latency.
#
# Key speedups vs. old min-power:
#  - Static model cache across calls (keeps GPU/frequency structure & PWLs)
#  - Frequency de-duplication by equal effective RATE (prunes dominated bins)
#  - Tight MILP with indicators only where needed; no z[t,g,f]
#  - Single-task formulation (streaming arrival), tiny variable footprint
#  - Tuned Gurobi params (aggressive presolve; no costly logging)
#  - Optional downlink handled via a switch to avoid extra rows
#  - TicTok timings to profile end-to-end
#
# Objective (pass k uses utilization U_g^(k)):
#   Minimize  sum_g ( P_static[g] * a_active[g] + sum_f y[g,f] * dyn_power(g,f,U_g) )
#   dyn_power(g,f,U) = U * C_dyn[g] * f ** dyn_exp[g]
#
# Constraints (streaming, one task t0 with rate lambda0 and FLOPS0):
#   1) Exactly one GPU chosen:  sum_g x[g] = 1
#   2) If GPU active ⇒ one frequency: sum_f y[g,f] = a_active[g]
#   3) Assignment implies activation: a_active[g] >= x[g]
#   4) Define Lambda_gpu[g] = x[g] * (lambda0 * FLOPS0)
#   5) Queue cost Q_gpu[g] via PWL(Lambda_gpu[g]) — only to tighten rho constraint
#   6) Soft rho cap: Lambda_gpu[g] <= rho_cap * sum_f y[g,f]*RATE[g,f] + s_rho[g]
#   7) (Optional) UL/DL aggregate rates per cluster for capacity soft caps
#
# Two-pass scheme for utilization U_g:
#   pass-1: use prior U guess (from snapshot or 0) ⇒ solve; get assignment
#   pass-2: recompute U from snapshot+assignment (or use provided util map) ⇒ resolve
# --------------------------------------------------------------
from typing import Dict, Any, List, Tuple
import copy as _copy
import math, os, json, pathlib, re
import time
from power_shared import build_power_params
from typing import Optional

try:
    import gurobipy as gp
    from gurobipy import GRB
    _HAS_GUROBI = True
except Exception as e:
    _HAS_GUROBI = False
    _IMPORT_ERR = e

eps = 1e-9
# _DASHES = '‑‐‒–—−-'
_DASHES = ["—", "–", "−", "‒", "-", "﹘", "﹣", "－"]


def _dump_dir() -> Optional[pathlib.Path]:
    d = os.getenv("GUROBI_DUMP_DIR", "").strip()
    if not d:
        return None
    p = pathlib.Path(d)
    p.mkdir(parents=True, exist_ok=True)
    return p

def _dump_model(m, name: str):
    p = _dump_dir()
    if not p: return
    try:
        m.write(str(p / f"{name}.lp"))
        m.write(str(p / f"{name}.mps"))
    except Exception:
        pass

def _dump_iis(m, name: str):
    p = _dump_dir()
    if not p: return
    try:
        m.computeIIS()
        m.write(str(p / f"{name}.iis"))
    except Exception:
        pass

def _dump_solution_obj(m, name: str):
    p = _dump_dir()
    if not p: return
    try:
        m.write(str(p / f"{name}.sol"))
    except Exception:
        pass

def _dump_json(payload: dict, name: str):
    p = _dump_dir()
    if not p: return
    try:
        (p / f"{name}.json").write_text(json.dumps(payload, indent=2))
    except Exception:
        pass

# ----------------- small utils -----------------
def _f(x, default=0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)

def _f(x, d=0.0):
    try:
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return d
        return v
    except Exception:
        return d

def _q(x, eps=1e-12):
    """Quantize floats for signatures."""
    x = _f(x, 0.0)
    if x == 0.0:
        return 0.0
    s = 1.0 if x > 0 else -1.0
    m = abs(x)
    # 12 digits total-ish, stable order
    return s * round(m, 12)

def _gid(g):
    return f"{g[0]}-{g[1]}-{g[2]}"

def _norm_gpu_key(k):
    if isinstance(k, tuple) and len(k) == 3:
        return tuple(map(str, k))
    if isinstance(k, dict):
        return (str(k.get("Cluster","")), str(k.get("Node","")), str(k.get("GPU","")))
    if isinstance(k, str):
        s = k.strip().replace(",","-")
        parts = [p.strip() for p in s.split("-") if p.strip()]
        if len(parts) == 3:
            return (parts[0], parts[1], parts[2])
    return (str(k), "N?", "G?")

def _norm_cid(k: str) -> str:
    s = str(k).strip()
    if s.startswith("C") and s[1:].isdigit():
        return s
    if s.isdigit():
        return f"C{s}"
    return s or "C?"

# ----------------- timing -----------------

class TicTok:
    def __init__(self, enabled=True):
        self.enabled = bool(enabled)
        self._stack = []
        self.records = []  # (label, dt)
        self.tags = {}
    def tic(self, label: str):
        if not self.enabled: return
        self._stack.append((label, time.perf_counter()))
    def tok(self, label: Optional[str] = None):
        if not self.enabled or not self._stack: return 0.0
        lab, t0 = self._stack.pop()
        dt = time.perf_counter() - t0
        self.records.append((label or lab, dt))
        return dt
    def set(self, k, v):
        if self.enabled: self.tags[k] = v
    def dump(self):
        if not self.enabled: return "(tictok disabled)"
        return {"times": self.records, "tags": self.tags}


def _get_tictok(snapshot, default_enabled=False) -> TicTok:
    OD = snapshot.get("Optimizer_Defaults", {}) or {}
    enabled = bool(snapshot.get("tictok", OD.get("tictok", default_enabled)))
    return TicTok(enabled=enabled)

# ----------------- PWL for queue tightening (same as latency) -----------------
def _add_safe_pwl(m, xvar, yvar, R, *, rho_max=0.90, K=6, name="PWL"):
    """
    Adds a numerically safe PWL: if R is ~0, avoid building a degenerate PWL.
    """
    R = float(R)
    if R <= 1e-6:
        # No capacity → we don't use PWL tightening; just pin yvar to 0.
        m.addConstr(yvar == 0.0, name=f"{name}:y0")
        return
    xs, ys = sanitize_pwl(*_build_pwl_points(R, rho_max=rho_max, K=K))
    # Ensure strictly increasing support; if too narrow, make a tiny 2-point segment.
    if len(xs) < 2 or (xs[-1] - xs[0]) < 1e-6:
        eps = max(1e-6, R * rho_max * 1e-6)
        xs = [0.0, eps]
        # linearized tiny slope around rho≈0 ⇒ cost ≈ rho
        ys = [0.0, eps / R]  # since rho = lam/R, and 1/(1-ρ)-1 ≈ ρ for small ρ
    m.addGenConstrPWL(xvar, yvar, xs, ys, name=name)

def _build_pwl_points(R, rho_max=0.99, K=2):
    # convex proxy for 1/(1-ρ) - 1 near 0≤ρ<rho_max
    R = max(1e-6, float(R))
    xs = [0.0]
    ys = [0.0]
    for k in range(1, K+1):
        rho = rho_max * (k / K)
        lam = rho * R
        cost = (1.0/(1.0 - rho) - 1.0)
        xs.append(lam)
        ys.append(cost)
    return xs, ys

def sanitize_pwl(xs, ys, *, rtol=1e-9, atol=1e-9):
    xs_c, ys_c = list(xs), list(ys)
    if len(xs_c) != len(ys_c):
        n = min(len(xs_c), len(ys_c))
        xs_c, ys_c = xs_c[:n], ys_c[:n]
    if len(xs_c) <= 2:
        return xs_c, ys_c
    xs_m, ys_m = [xs_c[0]], [ys_c[0]]
    for i in range(1, len(xs_c)):
        if xs_c[i] - xs_m[-1] >= max(atol, rtol*max(1.0, abs(xs_c[i]), abs(xs_m[-1]))):
            xs_m.append(xs_c[i]); ys_m.append(ys_c[i])
        else:
            xs_m.append(xs_m[-1] + max(atol, rtol)); ys_m.append(ys_c[i])
    return list(xs_m), list(ys_m)

# ----------------- frequency pruning -----------------

def _prune_freqs(F, RATE, *, tol=1e-9):
    """Keep at most one freq per distinct effective RATE[g][f]."""
    def _fnum(s, default=float("-inf")):
        try:    return float(s)
        except: return default

    F2 = {}
    for g, flist in F.items():
        pairs = [(float(RATE[g][f]), f) for f in flist]
        pairs.sort(key=lambda t: (round(t[0], 9), _fnum(t[1])))
        kept = []
        for r, f in pairs:
            if not kept:
                kept.append((r, f)); continue
            r0, f0 = kept[-1]
            if abs(r - r0) <= tol * max(1.0, abs(r), abs(r0)):
                if _fnum(f) > _fnum(f0):  # equal rate ⇒ keep numerically highest label
                    kept[-1] = (r, f)
            else:
                kept.append((r, f))
        F2[g] = [f for _, f in kept]
    return F2

# ----- GPU tails (seconds) -----
def _tail_seconds(info: Optional[dict]) -> float:
    if info is None:
        return 0.0
    try:
        if "tail_s" in info:
            return _f(info["tail_s"], 0.0)          # seconds
        if "tail_ms" in info:
            return _f(info["tail_ms"], 0.0) / 1e3   # ms -> s
        if "tail" in info:
            # NEW: treat 'tail' as seconds (from snapshot / legacy collectors)
            return _f(info["tail"], 0.0)
    except Exception:
        pass
    return 0.0


# ----------------- static model cache -----------------
_MODEL_CACHE = {}

def _static_signature(G, F, RATE, R0_RC_Links, rho, Kpts, *, use_dl=False):
    g_key = tuple(sorted(
        (_gid(g),
         tuple(sorted(F[g])),
         tuple(sorted((str(f), _q(RATE[g][f])) for f in F[g])))
        for g in G))
    clusters = sorted({g[0] for g in G})
    link_key = tuple(sorted(
        (c,
         _q(_f((R0_RC_Links.get(c) or {}).get("ul_rate_kBps"), 0.0)),
         _q(_f((R0_RC_Links.get(c) or {}).get("dl_rate_kBps"), 0.0)))
        for c in clusters))
    # minpower_linear_f -> power optimizer, power model is linear in frequency
    return ("minpwr_linear_f", g_key, link_key, _q(rho), int(Kpts), bool(use_dl))

def _warm_build_only(snapshot):
    # Build "G, F, RATE, R0_RC_Links" from snapshot only (no tasks)
    G, F, RATE = [], {}, {}
    for gk, info in (snapshot.get("gpus") or {}).items():
        g = _norm_gpu_key(gk)
        rates = info.get("rates") or {}
        if not rates:
            continue
        G.append(g)
        F[g] = [str(k) for k in rates.keys()]
        RATE[g] = {str(f): _f(rates[f], 0.0) for f in rates}

    links_src = snapshot.get("R0_RC_Links") or snapshot.get("links") or {}
    R0_RC_Links = {}
    for k, d in links_src.items():
        c = _norm_cid(k)
        R0_RC_Links[c] = {
            "ul_rate_kBps": _f(d.get("ul_rate_kBps"), 0.0),
            "dl_rate_kBps": _f(d.get("dl_rate_kBps"), 0.0),
        }

    OD = snapshot.get("Optimizer_Defaults", {}) or {}
    rho_max = float(snapshot.get("pwl_rho_max", OD.get("pwl_rho_max", 0.99)))
    Kpts    = int(snapshot.get("pwl_points",  OD.get("pwl_points", 2)))

    if G:
        _ensure_static(snapshot, G, F, RATE, R0_RC_Links, rho_max, Kpts)


def _ensure_static(snapshot, G, F, RATE, R0_RC_Links, rho, Kpts):
    """
    Build or reuse the static MILP structure.

    FIXED: rho and Kpts must come from the caller (min-power optimizer),
    not re-read from snapshot.
    """
    use_dl = bool(snapshot.get("use_downlink", False))
    tt = _get_tictok(snapshot, default_enabled=False)

    # --- use rho and Kpts exactly as passed ---
    rho_max = float(rho)
    Kpts    = int(Kpts)

    # correct signature with caller-provided rho_max/Kpts
    base_sig = _static_signature(G, F, RATE, R0_RC_Links, rho_max, Kpts,
                                 use_dl=use_dl)
    sig = ("pstatic_v2", base_sig)        # <-- bump version so old cache is invalid

    if _MODEL_CACHE.get("sig") == sig and _MODEL_CACHE.get("m") is not None:
        return _MODEL_CACHE

    if not _HAS_GUROBI:
        raise RuntimeError(f"Gurobi not available: {_IMPORT_ERR}")

    m = gp.Model("minpower")
    m.Params.OutputFlag    = 0
    m.Params.Presolve      = 2
    m.Params.Heuristics    = 0.10
    m.Params.Cuts          = 0
    m.Params.MIPFocus      = 1
    m.Params.NumericFocus  = 1
    m.Params.PreQLinearize = 1
    m.Params.NonConvex     = 0
    m.Params.Threads       = int(snapshot.get("gurobi_threads",
                             snapshot.get("threads", 0)) or 0)

    # --- static vars ---
    a_active = {g: m.addVar(vtype=GRB.BINARY, name=f"a_active[{_gid(g)}]") for g in G}
    y = {
        g: {
            f: m.addVar(vtype=GRB.BINARY, name=f"y[{_gid(g)},{f}]")
            for f in F[g]
        }
        for g in G
    }
    Q_gpu      = {g: m.addVar(lb=0.0, name=f"Q_gpu[{_gid(g)}]") for g in G}
    Lambda_gpu = {g: m.addVar(lb=0.0, name=f"Lambda_gpu[{_gid(g)}]") for g in G}
    s_rho      = {g: m.addVar(lb=0.0, name=f"s_rho[{_gid(g)}]") for g in G}

    clusters = sorted({g[0] for g in G})
    lam_ul_c = {c: m.addVar(lb=0.0, name=f"lam_ul[{c}]") for c in clusters}
    U_c      = {c: m.addVar(lb=0.0, name=f"U_ul[{c}]")  for c in clusters}
    s_ul_cap = {c: m.addVar(lb=0.0, name=f"s_ul_cap[{c}]") for c in clusters}

    lam_dl_c, D_c, s_dl_cap = {}, {}, {}
    if use_dl:
        lam_dl_c = {c: m.addVar(lb=0.0, name=f"lam_dl[{c}]") for c in clusters}
        D_c      = {c: m.addVar(lb=0.0, name=f"D_dl[{c}]")  for c in clusters}
        s_dl_cap = {c: m.addVar(lb=0.0, name=f"s_dl_cap[{c}]") for c in clusters}

    P_static_W, phi_power, exp_const, freq_value = build_power_params(snapshot, G, F, RATE)

    # print("[PWRDBG] phi_power:", { _gid(g): float(phi_power[g]) for g in G })
    # print("[PWRDBG] exp_const:", { _gid(g): float(exp_const[g]) for g in G })
    # print("[PWRDBG] P_static_W:", { _gid(g): float(P_static_W[g]) for g in G })

    # for g in G:
    #     print("[PWRDBG] freqs for", _gid(g), ":", [float(freq_value[f]) for f in F[g]])


    Q_rho, Pdyn_gf, p_static_W, fmax_g = {}, {}, {}, {}

    for g in G:
        gid      = _gid(g)
        e_g      = float(exp_const[g])
        phi_g    = float(phi_power[g])          # W / MHz^e
        p_static = float(P_static_W[g])
        p_static_W[g] = p_static

        # max freq for this GPU (for logging/tiebreak bias)
        f_max = max(float(freq_value[f]) for f in F[g])
        fmax_g[g] = f_max

        Q_rho[g], Pdyn_gf[g] = {}, {}
        for f in F[g]:
            fnum = float(freq_value[f])

            q  = m.addVar(lb=0.0, ub=rho_max, name=f"rho[{gid},{f}]")
            pd = m.addVar(lb=0.0, name=f"Pdyn[{gid},{f}]")
            Q_rho[g][f]   = q
            Pdyn_gf[g][f] = pd

            coef = phi_g * (fnum ** e_g)

            xs = [rho_max * k / (Kpts - 1) for k in range(Kpts)] if Kpts > 1 else [0.0, rho_max]
            ys = [coef * x for x in xs]

            m.addGenConstrPWL(q, pd, xs, ys, name=f"PWL_power[{gid},{f}]")

            # existing gate
            m.addConstr(q <= rho_max * y[g][f], name=f"gate_rho[{gid},{f}]")

            # recommended extra tightening (optional)
            m.addConstr(pd <= (coef * rho_max) * y[g][f], name=f"gate_pdyn[{gid},{f}]")


        m.addConstr(gp.quicksum(y[g][f] for f in F[g]) == a_active[g],
                    name=f"C_onefreq[{gid}]")


    # --- link PWLs + soft caps for UL/DL ---
    for c in clusters:
        ulR = _f((R0_RC_Links.get(c) or {}).get("ul_rate_kBps"), 0.0)
        _add_safe_pwl(
            m, lam_ul_c[c], U_c[c], ulR,
            rho_max=rho_max, K=Kpts, name=f"PWL_ul[{c}]",
        )
        if use_dl:
            dlR = _f((R0_RC_Links.get(c) or {}).get("dl_rate_kBps"), 0.0)
            _add_safe_pwl(
                m, lam_dl_c[c], D_c[c], dlR,
                rho_max=rho_max, K=Kpts, name=f"PWL_dl[{c}]",
            )

    # --- precomputed upper bounds for Big-M delay linking (mirror of min-latency) ---
    QMAX = {}
    for g in G:
        ymax = 0.0
        for f in F[g]:
            xs, ys = sanitize_pwl(
                *_build_pwl_points(
                    max(1e-6, float(RATE[g][f])),
                    rho_max=rho_max,
                    K=Kpts,
                )
            )
            if ys:
                ymax = max(ymax, max(ys))
        QMAX[g] = ymax

    UDMAX = {}
    for c in clusters:
        Lrec = (R0_RC_Links.get(c) or {})
        r_ul = max(1e-6, _f(Lrec.get("ul_rate_kBps"), 1.0))
        xsU, ysU = sanitize_pwl(
            *_build_pwl_points(
                r_ul,
                rho_max=rho_max,
                K=Kpts,
            )
        )
        if use_dl:
            r_dl = max(1e-6, _f(Lrec.get("dl_rate_kBps"), 1.0))
            xsD, ysD = sanitize_pwl(
                *_build_pwl_points(
                    r_dl,
                    rho_max=rho_max,
                    K=Kpts,
                )
            )
            UDMAX[c] = (
                max(ysU) if ysU else 0.0,
                max(ysD) if ysD else 0.0,
            )
        else:
            UDMAX[c] = (max(ysU) if ysU else 0.0, 0.0)

    _MODEL_CACHE.clear()
    _MODEL_CACHE.update({
        "sig": sig, "m": m,
        "a_active": a_active, "y": y,
        "Q_gpu": Q_gpu, "Lambda_gpu": Lambda_gpu, "s_rho": s_rho,
        "clusters": clusters, "lam_ul_c": lam_ul_c, "U_c": U_c, "s_ul_cap": s_ul_cap,
        "use_dl": use_dl, "lam_dl_c": lam_dl_c, "D_c": D_c, "s_dl_cap": s_dl_cap,
        "Q_rho": Q_rho, "Pdyn_gf": Pdyn_gf, "P_static_W": p_static_W, "fmax_g": fmax_g,
        "QMAX": QMAX, "UDMAX": UDMAX,
    })

    return _MODEL_CACHE


# ----------------- data collection -----------------

def _collect(snapshot: Dict[str, Any], new_task: List[Dict[str, Any]]):

    # tasks (streaming only; expect single t0)
    T = [str(t.get("Task_ID")) for t in new_task]
    FLOPS = {str(t.get("Task_ID")): _f(t.get("Workload_FLOPs"), 0.0) for t in new_task}
    DEADL = {str(t.get("Task_ID")): _f(t.get("Task_Deadline"), float("inf")) for t in new_task}
    ULKB  = {str(t.get("Task_ID")): _f(t.get("UL_Total_kB"), 0.0) for t in new_task}
    DLKB  = {str(t.get("Task_ID")): _f(t.get("DL_Total_kB"), 0.0) for t in new_task}

    # ---- Arrival rates (tasks/s) ----
    LAMBDA = {}
    for t in new_task:
        tid = str(t.get("Task_ID"))
        lam_raw = (t.get("Lambda") or t.get("lambda") or t.get("arrival_rate") or t.get("Task_Arrival_Rate"))
        lam = _f(lam_raw, 0.0)
        if lam <= 0.0:
            raise RuntimeError(f"Streaming-only: Lambda missing/zero for task {tid}.")
        LAMBDA[tid] = lam

    # ---- GPUs (normalize keys; scale rates to GFLOP/s) ----
    G, F, RATE, TAIL, GPUINFO = [], {}, {}, {}, {}
    for gk, info in (snapshot.get("gpus") or {}).items():
        try:
            g = _norm_gpu_key(gk)
        except Exception:
            continue

        rates = info.get("rates") or {}
        if not rates: continue

        G.append(g)
        F[g] = [str(k) for k in rates.keys()]
        RATE[g] = {str(f): _f(rates[f], 0.0) for f in rates}
        # TAIL[g] = _f(info.get("tail"), 0.0)
        TAIL[g] = _tail_seconds(info)

        GPUINFO[g] = dict(info)


    # links
    R0_RC_Links = {}
    links_src = snapshot.get("R0_RC_Links") or snapshot.get("links") or {}
    for k, d in links_src.items():
        c = _norm_cid(k)
        R0_RC_Links[c] = {
            "ul_rate_kBps": _f(d.get("ul_rate_kBps", d.get("r0_c")), 0.0),
            # "dl_rate_kBps": _f(d.get("dl_rate_kBps", d.get("rc_rate_kBps", d.get("rc_0"))), 0.0),
            "dl_rate_kBps": _f(d.get("dl_rate_kBps", d.get("rc_0", 0.0)), 0.0),
            "ul_prop_s":    _f(d.get("ul_prop_s", d.get("uplink_prop_delay")), 0.0),
            "dl_prop_s":    _f(d.get("dl_prop_s", d.get("downlink_prop_delay")), 0.0),
        }
    
    # # prune duplicate-rate freqs
    # F = _prune_freqs(F, RATE)

    # print("G",G)
    # print("F",F)
    # print("R0_RC_Links",R0_RC_Links)
    # print("RATE",RATE)
    # print("FLOPS",FLOPS)
    # print("T",T)
    # print("TAIL",TAIL)
    # print("LAMBDA",LAMBDA)
    # print("DEADL",DEADL)
    # print("GPUINFO",GPUINFO)


    return T, G, F, RATE, TAIL, FLOPS, DEADL, ULKB, DLKB, GPUINFO, LAMBDA, R0_RC_Links

# ----------------- MILP core (STREAMING-ONLY) -----------------

def _solve_power(snapshot: Dict[str, Any], new_task: List[Dict[str, Any]], *,
                 util_guess: Optional[Dict[str, float]] = None,
                 pinned_assignments: Optional[Dict[str, Dict[str, str]]] = None,
                 pinned_frequencies: Optional[Dict[str, str]] = None) -> Dict[str, Any]:

    T, G, F, RATE, TAIL, FLOPS, DEADL, ULKB, DLKB, GPUINFO, LAMBDA, R0_RC_Links = _collect(snapshot, new_task)
    
    if not T or not G:
        raise RuntimeError("min-power optimizer: empty task or GPU set")

    t0     = sorted(T)[0]
    lam_t0 = _f(LAMBDA.get(t0), 0.0)
    if lam_t0 <= 0.0:
        raise RuntimeError(f"Streaming-only: Lambda missing/zero for task {t0}.")
    now = _f(snapshot.get("now"), 0.0)

    OD = snapshot.get("Optimizer_Defaults", {}) or {}

    # ---- objective-specific config block (where JSON lives) ----
    OBJCFG = (snapshot.get("optimizer_power") or {})   # for power solver

    def _cfg(key, default=None):
        """Snapshot overrides objective block overrides Optimizer_Defaults."""
        v = snapshot.get(key, None)
        if v is not None:
            return v
        v = OBJCFG.get(key, None)
        if v is not None:
            return v
        v = OD.get(key, None)
        if v is not None:
            return v
        return default

    def _od(key, default):
        v = _cfg(key, None)
        return float(default) if v is None else float(v)

    
    # --- core knobs ---
    rho_cap   = float(_cfg("rho_cap", 0.60))
    rho_max = float(_cfg("pwl_rho_max", 0.99))
    Kpts    = int(_cfg("pwl_points", 2) or 2)
    cap_pen_w = float(_cfg("link_penalty_weight", 0.0))
    deadline_w = float(_cfg("deadline_penalty_weight", 0.0))
    # background util weight (for future use if you want)
    bg_util_weight = float(_cfg("bg_util_weight", 0.0))
    # tie-breaker: prefer *lower* freqs, but take from Optimizer_Defaults
    freq_eps_cfg = float(_cfg("freq_tiebreak_eps", 0.2))
    freq_eps = abs(freq_eps_cfg)
    # ρ-slack penalty
    rho_pen_w = float(_cfg("rho_penalty", 1.0))
    # optional separate weight for the frequency bias
    freq_w = float(_cfg("freq_penalty_weight", 1e-1))
    
    # NEW: normalize background utilization map (per GPU)
    util_guess = util_guess or {}

    FlopsPerTask = float(FLOPS[t0])          # FLOPs / task
    lam_tasks    = float(lam_t0)             # tasks / s
    lam_flops    = lam_tasks * FlopsPerTask  # FLOPs / s


    # --- background FLOP/s load (lambda_bg) if present ---
    lb_raw = snapshot.get("lambda_bg", {}) or {}
    lambda_bg = {
        str(k): float(v)
        for k, v in lb_raw.items()
        if v not in (None, "")
    }
    
    # ====== DEBUG START ======
    is_warmup = bool(snapshot.get("_warmup", False))

    if not is_warmup and bool(snapshot.get("debug_solver", False)):
        print("[DBG_SOLVE] warmup=", is_warmup,
            "now=", snapshot.get("now"),
            "num_gpus=", len(snapshot.get("gpus", {}) or {}),
            "gpu_keys=", list((snapshot.get("gpus", {}) or {}).keys()))

    if not is_warmup and bool(snapshot.get("debug_solver", False)):
        print("=== FREQS SEEN BY OPT ===")
        for g in G:
            freqs = sorted([float(f) for f in F[g]])
            print(_gid(g), "min/max:", freqs[0], freqs[-1], "all:", freqs)

    if not is_warmup and bool(snapshot.get("debug_solver", False)):
        print("lambda_bg", lambda_bg)

    # ====== DEBUG END ======

    H = _ensure_static(snapshot, G, F, RATE, R0_RC_Links, rho_max, Kpts)
    m = H["m"]

    # --- IMPORTANT: reset frequency-selection bounds (cached model!) ---
    for g in G:
        if g not in H["y"]:
            continue
        for f in H["y"][g]:
            H["y"][g][f].LB = 0.0
            H["y"][g][f].UB = 1.0

    # ---- Remove dynamic vars/constraints from previous call (cached model safety) ----
    DYN_CONSTR_PREFIXES = (
        "C_assign[", "def_", "link_", "C_rho_soft[", "C_deadline[",
        "rho_lb[", "rho_eq[", "and1[", "and2[", "and3[", "def_D",
    )

    DYN_VAR_PREFIXES = (
        "x[", "z[", "D_overall[", "D_comp[", "D_sys[", "D_prop[", "ddl_slack["
    )

    # remove constraints
    for c in list(m.getConstrs()):
        if c.ConstrName.startswith(DYN_CONSTR_PREFIXES):
            m.remove(c)

    # remove lingering QCs
    if hasattr(m, "getQConstrs"):
        for qc in list(m.getQConstrs()):
            m.remove(qc)

    # remove variables (IMPORTANT)
    for v in list(m.getVars()):
        nm = v.VarName
        if nm.startswith(DYN_VAR_PREFIXES):
            m.remove(v)

    m.update()
    m.setObjective(0.0)


    # --- Apply min frequency constraints (snapshot overrides OD) ---
    minf_global = float(_cfg("min_freq_global", 0.0) or 0.0)
    minf_map    = _cfg("min_freq_map", {}) or {}

    for g in G:
        gid   = _gid(g)
        gtype = str(GPUINFO.get(g, {}).get("type", ""))

        # priority: by GPU id -> by type -> global
        if gid in minf_map:
            minf = float(minf_map[gid])
        elif gtype in minf_map:
            minf = float(minf_map[gtype])
        else:
            minf = float(minf_global)

        if minf > 0:
            for f in H["y"][g]:
                if float(f) < minf:
                    H["y"][g][f].UB = 0.0

    # If pinned frequencies conflict with minf, fail loudly
    if pinned_frequencies:
        for gk, ffix in pinned_frequencies.items():
            try:
                gg = _norm_gpu_key(gk)
            except Exception:
                continue
            if gg not in H["y"]:
                continue

            want = float(ffix)
            ggid = _gid(gg)
            gtype = str(GPUINFO.get(gg, {}).get("type", ""))

            if ggid in minf_map:
                mn = float(minf_map[ggid])
            elif gtype in minf_map:
                mn = float(minf_map[gtype])
            else:
                mn = float(minf_global)

            if mn > 0 and want < mn - 1e-9:
                raise RuntimeError(f"Pinned freq {want} < min_freq {mn} for {ggid}")



    # ---------------- dynamic vars ----------------
    x = {g: m.addVar(vtype=GRB.BINARY, name=f"x[{_gid(g)}]") for g in G}

    # z[g,f] = 1 iff task assigned to g AND freq f selected on g
    z = {
        g: {f: m.addVar(vtype=GRB.BINARY, name=f"z[{_gid(g)},{f}]")
            for f in H["y"][g]}
        for g in G
    }

    Dall   = m.addVar(lb=0.0, name=f"D_overall[{t0}]")
    Dcomp  = m.addVar(lb=0.0, name=f"D_comp[{t0}]")
    Dsys   = m.addVar(lb=0.0, name=f"D_sys[{t0}]")
    Dprop  = m.addVar(lb=0.0, name=f"D_prop[{t0}]")
    s_dead = m.addVar(lb=0.0, name=f"ddl_slack[{t0}]")

    # exactly one GPU for this task
    m.addConstr(gp.quicksum(x[g] for g in G) == 1, name=f"C_assign[{t0}]")
    for g in G:
        m.addConstr(H["a_active"][g] - x[g] >= 0, name=f"link_a_ge_x[{t0},{_gid(g)}]")

    # --- AND constraints: z = x AND y ---
    for g in G:
        gid = _gid(g)
        for f in H["y"][g]:
            m.addConstr(z[g][f] <= x[g],             name=f"and1[{t0},{gid},{f}]")
            m.addConstr(z[g][f] <= H["y"][g][f],     name=f"and2[{t0},{gid},{f}]")
            m.addConstr(z[g][f] >= x[g] + H["y"][g][f] - 1,
                        name=f"and3[{t0},{gid},{f}]")

    # ---------------- UL/DL aggregation by cluster ----------------
    use_dl = bool(H.get("use_dl", False))
    for c in H["clusters"]:
        m.addConstr(
            H["lam_ul_c"][c] == gp.quicksum(
                x[g] * float(lam_t0) * float(ULKB[t0]) for g in G if g[0] == c
            ),
            name=f"def_lam_ul[{c}]",
        )
        if use_dl and H["lam_dl_c"]:
            m.addConstr(
                H["lam_dl_c"][c] == gp.quicksum(
                    x[g] * float(lam_t0) * float(DLKB[t0]) for g in G if g[0] == c
                ),
                name=f"def_lam_dl[{c}]",
            )

    # ---------------- rho linking (FIXED) ----------------
    # task arrival in FLOPs/s (units-correct)
    lam_flops_task = float(lam_t0) * float(FLOPS[t0])   # FLOPs/s

    for g in G:
        gid = _gid(g)

        U_bg_raw = float((util_guess or {}).get(gid, 0.0))
        U_bg     = max(0.0, min(rho_max, bg_util_weight * U_bg_raw))

        lam_bg_g = float(lambda_bg.get(gid, 0.0))  # FLOPs/s

        for f in H["y"][g]:
            mu_f = max(1e-9, float(RATE[g][f]))  # FLOPs/s

            util_expr = U_bg + (lam_flops_task + lam_bg_g) / mu_f
            util_gf   = max(0.0, min(rho_max, util_expr))   
            over = max(0.0, util_expr - rho_cap)     # constant for this (g,f)
            # add a linear penalty directly (no new vars needed)
            H.setdefault("rho_over", {}).setdefault(g, {})[f] = over

            m.addConstr(
                H["Q_rho"][g][f] == util_gf * z[g][f],
                name=f"rho_eq[{t0},{gid},{f}]"
            )

        #use the same capped constants here as well (consistency)
        m.addConstr(
            gp.quicksum(
                (max(0.0, min(rho_max, U_bg + (lam_flops_task + lam_bg_g)
                            / max(1e-9, float(RATE[g][f])))))
                * z[g][f]
                for f in H["y"][g]
            )
            <= rho_cap + H["s_rho"][g],
            name=f"C_rho_soft[{t0},{gid}]",
        )



    # --- Delay decomposition with Big-M, mirroring min-latency (no bilinear terms) ---
    TAIL0 = {g: _tail_seconds(GPUINFO.get(g, {})) for g in G}
    Fl    = float(FLOPS[t0])
    UL    = float(ULKB[t0])
    DL    = float(DLKB[t0]) if (use_dl and H.get("D_c")) else 0.0


    # propagation per cluster
    tau_sum = {
        c: _f(R0_RC_Links.get(c, {}).get("ul_prop_s"), 0.0)
           + _f(R0_RC_Links.get(c, {}).get("dl_prop_s"), 0.0)
        for c in H["clusters"]
    }

    for g in G:
        c = g[0]
        Umax, Dmax = H["UDMAX"][c]

        # Big-M constants
        M_comp = Fl * H["QMAX"][g] + float(TAIL0[g])
        M_sys  = Umax * UL + ((Dmax * DL) if (use_dl and H.get("D_c")) else 0.0)
        M_prop = abs(float(tau_sum[c]))

        # Compute delay contributions for the *chosen* GPU g (x[g] = 1)
        m.addConstr(
            Dcomp >= Fl * gp.quicksum(H["Q_rho"][g][f] for f in H["y"][g])
                    + float(TAIL0[g]) - M_comp * (1 - x[g]),
            name=f"def_Dcomp_lin[{t0},{_gid(g)}]",
        )

        if use_dl and H.get("D_c"):
            m.addConstr(
                Dsys >= H["U_c"][c] * UL + H["D_c"][c] * DL - M_sys * (1 - x[g]),
                name=f"def_Dsys_lin[{t0},{_gid(g)}]",
            )
        else:
            m.addConstr(
                Dsys >= H["U_c"][c] * UL - M_sys * (1 - x[g]),
                name=f"def_Dsys_lin[{t0},{_gid(g)}]",
            )

        m.addConstr(
            Dprop >= tau_sum[c] - M_prop * (1 - x[g]),
            name=f"def_Dprop_lin[{t0},{_gid(g)}]",
        )

    m.addConstr(Dall == Dcomp + Dsys + Dprop, name=f"def_Doverall[{t0}]")

    # deadline
    Dt = _f(DEADL.get(t0), float("inf")) - now
    if math.isfinite(Dt) and Dt > 1e-9:
        m.addConstr(Dall <= Dt + s_dead, name=f"C_deadline[{t0}]")

    # print("[DEBUG] pinned_frequencies is None?", pinned_frequencies is None)
    # print("[DEBUG] pinned_frequencies =", pinned_frequencies)

    # if pinned_assignments and t0 in pinned_assignments:
    #     pin = pinned_assignments[t0]
    #     pin_g = (_norm_cid(pin["Cluster"]), str(pin["Node"]), str(pin["GPU"]))
    #     for g in G:
    #         if g != pin_g:
    #             m.addConstr(x[g] == 0, name=f"pin_x0[{t0},{_gid(g)}]")


    
    # pin freqs (bounds only)
    # pinned_frequencies = pinned_frequencies or {}
    if pinned_frequencies:
        for gk, ffix in (pinned_frequencies or {}).items():
            try:
                g = _norm_gpu_key(gk)
            except Exception:
                continue
            if g not in H["y"]:
                continue
            fbest = None
            want  = None
            try:
                want = float(ffix)
                bestd = 1e99
                for f in H["y"][g]:
                    try:
                        d = abs(float(f) - want)
                        if d < bestd:
                            bestd, fbest = d, f
                    except Exception:
                        pass
            except Exception:
                if str(ffix) in H["y"][g]:
                    fbest = str(ffix)
            if fbest is None:
                continue

            print("[PIN]", _gid(g), "->", fbest, "(requested:", ffix, ")")

            for f in H["y"][g]:
                val = 1.0 if f == fbest else 0.0
                H["y"][g][f].LB = val
                H["y"][g][f].UB = val

    if pinned_frequencies:
        print("[PIN] pinned_frequencies keys:", list(pinned_frequencies.keys())[:5])

    # # Objective: idle + dynamic + small penalties (+ monotone low-f bias)
    # idle_map = H.get("P_static_W", H.get("P_idle_W", {}))  # safe even if key missing
    # obj = gp.quicksum(
    #     float(idle_map.get(g, 0.0)) * H["a_active"][g]
    #     + gp.quicksum(H["Pdyn_gf"][g][f] * z[g][f] for f in H["y"][g])
    #     for g in G
    # )

    # # obj += gp.quicksum(
    # #     freq_eps * (float(f) / max(1e-9, float(H["fmax_g"][g]))) * H["y"][g][f]
    # #     for g in G for f in H["y"][g]
    # # )

    # # use freq_w * freq_eps so JSON actually matters
    # obj += freq_w * gp.quicksum(
    #     freq_eps * (float(f) / max(1e-9, float(H["fmax_g"][g]))) * z[g][f]
    #     for g in G for f in H["y"][g]
    # )


    # obj += cap_pen_w * gp.quicksum(H["s_ul_cap"][c] for c in H["clusters"])
    # if use_dl and H.get("s_dl_cap"):
    #     obj += cap_pen_w * gp.quicksum(H["s_dl_cap"][c] for c in H["clusters"])

    # obj += rho_pen_w * gp.quicksum(H["s_rho"][g] for g in G)

    # obj += (deadline_w * 1e-3) * s_dead
    
    # m.setObjective(obj, GRB.MINIMIZE)

    # -------------------------
    # SINGLE objective (min-power-adaptive) — no setObjectiveN()
    # -------------------------

    # If cached model was previously used with multi-objectives, clear safely
    try:
        if hasattr(m, "NumObj") and int(getattr(m, "NumObj", 1)) > 1:
            for i in range(int(m.NumObj)):
                m.setObjectiveN(0.0, index=i, priority=0, weight=0.0, name=f"clear_obj_{i}")
    except Exception:
        pass

    m.setObjective(0.0)

    # ---- weights (snapshot overrides OD) ----
    deadline_w  = _od("deadline_penalty_weight", 0.0)
    latency_w   = _od("latency_weight", 0.0)
    rho_pen_w   = _od("rho_penalty", 1.0)
    cap_pen_w   = _od("link_cap_penalty_weight", 1.0)

    freq_w      = _od("freq_penalty_weight", 1.0)
    freq_eps    = abs(_od("freq_tiebreak_eps", 1e-3))

    idle_map = H.get("P_static_W", H.get("P_idle_W", {}))

    # ---- POWER (CORRECT): idle + chosen dynamic power only ----
    power_expr = gp.quicksum(
        float(idle_map.get(g, 0.0)) * H["a_active"][g]
        + gp.quicksum(H["Pdyn_gf"][g][f] * z[g][f] for f in H["y"][g])  # <-- MUST use z
        for g in G
    )
    
    # ---- feasibility slack ----
    rho_slack = gp.quicksum(H["s_rho"][g] for g in G)
    cap_slack = gp.quicksum(H["s_ul_cap"][c] for c in H["clusters"])
    if use_dl and H.get("s_dl_cap"):
        cap_slack += gp.quicksum(H["s_dl_cap"][c] for c in H["clusters"])

    # ---- tiny frequency bias (prefer lower freq, only as tie-break) ----
    freq_bias = gp.LinExpr(0.0)
    if freq_w != 0.0 and freq_eps != 0.0:
        freq_bias = gp.quicksum(
            (freq_w * freq_eps)
            * (float(f) / max(1e-9, float(H["fmax_g"][g])))
            * z[g][f]
            for g in G for f in H["y"][g]
        )
    
    
    rho_over_w = float(_cfg("rho_over_penalty_weight", 0.0) or 0.0)

    rho_over_pen = gp.quicksum(
        float(H["rho_over"][g][f]) * z[g][f]
        for g in G for f in H["y"][g]
    )

    # ---- min-power-adaptive objective: real power + feasibility penalties ----
    obj = gp.LinExpr(0.0)
    # primary: power
    obj += power_expr
    # keep deadlines/latency in mind:
    # - Dall is the modeled end-to-end delay
    # - s_dead is slack for deadline constraint
    if latency_w != 0.0:
        obj += latency_w * Dall
    if deadline_w != 0.0:
        obj += deadline_w * s_dead
    # enforce feasibility softly (if caps/rho violated, pay penalty)
    if rho_pen_w != 0.0:
        obj += rho_pen_w * rho_slack
    if cap_pen_w != 0.0:
        obj += cap_pen_w * cap_slack
    # tie-break
    obj += freq_bias

    obj += rho_over_w * rho_over_pen

    m.setObjective(obj, GRB.MINIMIZE)

    m.update()
    

    # link caps
    for c in H["clusters"]:
        ulR = _f(R0_RC_Links.get(c, {}).get("ul_rate_kBps"), 0.0)
        m.addConstr(H["lam_ul_c"][c] <= ulR + H["s_ul_cap"][c], name=f"def_ulcap[{c}]")
        if use_dl and H.get("lam_dl_c"):
            dlR = _f(R0_RC_Links.get(c, {}).get("dl_rate_kBps"), 0.0)
            m.addConstr(H["lam_dl_c"][c] <= dlR + H["s_dl_cap"][c], name=f"def_dlcap[{c}]")

    # sanity: complain only if there are actual quadratic nonzeros
    if getattr(m, "NumQConstrs", 0) > 0 and getattr(m, "NumQNZs", 0) > 0:
        raise RuntimeError(
            f"Model has quadratic leftovers: "
            f"NumQConstrs={getattr(m,'NumQConstrs',0)}, NumQNZs={getattr(m,'NumQNZs',0)}"
        )

    # optional model dump (fast)
    debug_dump = bool(snapshot.get("debug_dump_models", False))
    dump_fmt   = (snapshot.get("dump_format") or "lp").lower()  # "mps" or "lp"

    m.optimize()
    # print(f"[DBG] Status={m.Status} IsMIP={m.IsMIP} NumQNZs={getattr(m,'NumQNZs',0)} NumQConstrs={getattr(m,'NumQConstrs',0)}")

    status_code = m.Status

    # 1) Handle hard failure / infeasibility WITHOUT raising
    if status_code == GRB.INFEASIBLE:
        # Optional: IIS only when debugging
        if bool(snapshot.get("debug_dump_models", False)):
            _dump_iis(m, "minpower_model")
        # Drop this task: no assignments, no freqs
        out = {
            "assignments": {},
            "frequencies": {},
            "status": "infeasible",
            "objective": None,
            "use_rate_form": True,
        }
        # _dump_json(out, "minpower_solution_assignments")
        return out

    if status_code not in (GRB.OPTIMAL, GRB.INTERRUPTED):
        # Unknown / numerical issue – also drop the task and continue
        out = {
            "assignments": {},
            "frequencies": {},
            "status": f"status_{status_code}",
            "objective": None,
            "use_rate_form": True,
        }
        # _dump_json(out, "minpower_solution_assignments")
        return out

    # 2) Normal case: we have a solution
    # _dump_solution_obj(m, "minpower_model_solution")

    pick = next((g for g in G if x[g].X > 0.5), None)

    if pick and (not is_warmup):
        # print("[DEBUG] chosen GPU:", _gid(pick))
        for f in H["y"][pick]:
            if H["y"][pick][f].X > 0.5:
                print("[DEBUG] chosen g:", _gid(pick), "chosen f:", f, "lambda_bg:", lambda_bg,
                        "rho:", H["Q_rho"][pick][f].X,
                        "Pdyn:", H["Pdyn_gf"][pick][f].X)


    # If solver "solved" but chose no GPU → treat as no-op, drop task
    if pick is None:
        out = {
            "assignments": {},
            "frequencies": {},
            "status": "noop",
            "objective": float(m.ObjVal) if m.SolCount > 0 else None,
            "use_rate_form": True,
        }
        # _dump_json(out, "minpower_solution_assignments")
        return out

    freqs = {}
    for g in G:
        if H["a_active"][g].X > 0.5:
            for f in H["y"][g]:
                if H["y"][g][f].X > 0.5:
                    freqs[_gid(g)] = str(f)
                    break

    status = {GRB.OPTIMAL: "optimal",
              GRB.INTERRUPTED: "interrupted"}.get(status_code, f"status_{status_code}")

    out = {
        "assignments": {t0: {"Cluster": pick[0], "Node": pick[1], "GPU": pick[2]}},
        "frequencies": freqs,
        "status": status,
        "objective": float(m.ObjVal) if m.SolCount > 0 else None,
        "use_rate_form": True,
    }
    # _dump_json(out, "minpower_solution_assignments")
    return out

# ----------------- FastPolicy entrypoints -----------------

def optimize_power(snapshot, new_task, catalog=None, store=None):
    """
    FastPolicy-compatible power entrypoint.
    Delegates to the existing incremental MILP solver.
    """
    return solve_incremental(snapshot, new_task)

def optimize(snapshot, new_task, objective="power", catalog=None, store=None):
    """
    Generic FastPolicy fallback entrypoint.
    """
    if (objective or "power").lower() == "power":
        return solve_incremental(snapshot, new_task)
    else:
        raise RuntimeError(f"Objective '{objective}' not supported in this optimizer.")

# ----------------- public entrypoints (two-pass U_g) -----------------
def solve_incremental(snapshot: Dict[str, Any],
                      new_tasks: List[Dict[str, Any]],
                      objective: str = "power",
                      pinned_assignments: Optional[Dict[str, Dict[str, str]]] = None,
                      pinned_frequencies: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    if snapshot.get("_warmup", False):
        _warm_build_only(snapshot)   # builds cached static MILP
        return {"status": "warmed", "assignments": {}, "frequencies": {}}
    
    # pass-1: use guess from snapshot (previous window / heuristic)
    util_guess = snapshot.get("utilization_guess", {}) or {}
    first = _solve_power(snapshot, new_tasks,
                         util_guess=util_guess,
                         pinned_assignments=pinned_assignments,
                         pinned_frequencies=pinned_frequencies)

    # compute U from snapshot after first assignment if available
    pick = None
    if first.get("assignments"):
        t0 = next(iter(first["assignments"]))
        a = first["assignments"][t0]
        pick = (a.get("Cluster"), a.get("Node"), a.get("GPU"))

    # # pass-2 load map:
    # #  - if caller provided snapshot["utilization"], use that
    # #  - otherwise, reuse util_guess (so we DON'T lose load info)
    # if pick and isinstance(snapshot.get("utilization"), dict):
    #     U2 = snapshot["utilization"].copy()
    # else:
    #     U2 = util_guess

    # # print("util_guess",util_guess)

    # second = _solve_power(snapshot, new_tasks,
    #                       util_guess=U2,
    #                       pinned_assignments=pinned_assignments,
    #                       pinned_frequencies=pinned_frequencies)
    # second["utilization"] = U2
    return first


# Simple alias

def solve(snapshot: Dict[str, Any], new_tasks: List[Dict[str, Any]], objective: str = "power") -> Dict[str, Any]:
    return solve_incremental(snapshot, new_tasks, objective=objective)

# legacy alias

def solve_power(snapshot: Dict[str, Any], new_tasks: List[Dict[str, Any]]):
    return solve_incremental(snapshot, new_tasks, objective="power")
