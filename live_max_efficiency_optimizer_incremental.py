# --------------------------------------------------------------
# Incremental MAX power-efficiency optimizer (Gurobi).
# Compatible with OptimizerFacade generic/incremental API.
#
# Objective (Eq. 12): maximize Σ_{g,f} ( Σ_t z[t,g,f] * (RATE[g,f]/W_t) ) / P[g,f]
#   with P[g,f] = P_static_W[g] + U_g * C_p[g] * f^(dyn_exp[g])  (U_g via 2-pass)
#
# Constraints mirror latency MILP:
#   C1: one frequency per GPU
#   C2: one GPU per task
#   C7: z[t,g,f] <= x[t,g], z[t,g,f] <= y[g,f], z[t,g,f] >= x[t,g]+y[g,f]-1
#   C3: ∑ x[t,g]*FLOPs[t]*λ_t  ≤ ∑ y[g,f]*RATE[g,f]        (rate form, if λ given)
#       or ∑ x[t,g]*FLOPs[t]   ≤ H * ∑ y[g,f]*RATE[g,f]    (batch form)
#   C4: per-cluster UL cap (optional)
#   C5: per-cluster DL cap (optional)
#   C6: tail[g] + (1/Rmin[g]) * ∑ x[tp,g]*FLOPs[tp] ≤ (deadline[t]-now)+M_g*(1-x[t,g])
#
# Public entrypoints:
#   solve_incremental(snapshot, new_tasks, objective="efficiency",
#                     pinned_assignments={}, pinned_frequencies={})
#   solve(snapshot, new_tasks, ...)          # alias
#   solve_efficiency(snapshot, new_tasks, ...)  # legacy alias
# --------------------------------------------------------------

import time
from contextlib import contextmanager
import re
import statistics
from typing import Dict, Any, List, Tuple, Optional
import copy as _copy
import time
import math, os, json, pathlib

try:
    import gurobipy as gp
    from gurobipy import GRB
    _HAS_GUROBI = True
except Exception as e:
    _HAS_GUROBI = False
    _IMPORT_ERR = e

EPS_FLOPS = 1e-6  # keep objective bounded if a task mistakenly has 0 FLOPs


# --------------- simple tictok profiler ---------------
class TicTok:
    def __init__(self, enabled: bool = True, logger=None):
        self.enabled = bool(enabled)
        self._t = None
        self.logger = logger or (lambda s: print(s))
    def tic(self, tag: str = ""):
        if not self.enabled: 
            return
        self._t = time.perf_counter()
        if tag:
            self.logger(f"[TICTOK] {tag} start")
    def tok(self, tag: str = "") -> float:
        if not self.enabled:
            return 0.0
        now = time.perf_counter()
        if self._t is None:
            self._t = now
            return 0.0
        dt = now - self._t
        if tag:
            self.logger(f"[TICTOK] {tag}: {dt*1000:.2f} ms")
        self._t = now
        return dt

# ----------------- helpers -----------------
def _dump_dir() -> Optional[pathlib.Path]:
    d = os.getenv("GUROBI_DUMP_DIR", "").strip()
    if not d:
        return None
    p = pathlib.Path(d)
    p.mkdir(parents=True, exist_ok=True)
    return p

def _dump_model(m, name: str):
    p = _dump_dir()
    if not p:
        return
    try:
        m.write(str(p / f"{name}.lp"))
        m.write(str(p / f"{name}.mps"))
    except Exception:
        pass

def _dump_iis(m, name: str):
    p = _dump_dir()
    if not p:
        return
    try:
        m.computeIIS()
        m.write(str(p / f"{name}.iis"))
    except Exception:
        pass

def _dump_solution_obj(m, name: str):
    p = _dump_dir()
    if not p:
        return
    try:
        m.write(str(p / f"{name}.sol"))
    except Exception:
        pass

def _dump_json(payload: dict, name: str):
    p = _dump_dir()
    if not p:
        return
    try:
        (p / f"{name}.json").write_text(json.dumps(payload, indent=2))
    except Exception:
        pass

# --- unit helpers ---

def _norm_gpu_key(k):
    """Return (Cluster, Node, GPU) from tuple/str/dict."""
    # Already a 3-tuple
    if isinstance(k, tuple) and len(k) == 3:
        return tuple(map(str, k))
    # A facade assignment dict
    if isinstance(k, dict):
        c = str(k.get("Cluster", ""))
        n = str(k.get("Node", ""))
        g = str(k.get("GPU", ""))
        return (c, n, g)
    # String variants: "C-N-G" or "C,N,G"
    if isinstance(k, str):
        s = k.strip()
        s = s.replace(",", "-")
        parts = [p.strip() for p in s.split("-") if p.strip() != ""]
        if len(parts) == 3:
            return (parts[0], parts[1], parts[2])
    # Fallback (shouldn't happen)
    return (str(k), "N?", "G?")

def _gid(g):  # stable printable id "C-N-G"
    c,n,gg = g
    return f"{c}-{n}-{gg}"

def _base_tid(s: str) -> str:
    s = str(s)
    return s[:-2] if s.endswith("_U") or s.endswith("_D") else s

def _f(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return float(default)

def _q(x, eps=1e-12):
    """Quantize floats for signatures."""
    x = _f(x, 0.0)
    if x == 0.0:
        return 0.0
    s = 1.0 if x > 0 else -1.0
    m = abs(x)
    # 12 digits total-ish, stable order
    return s * round(m, 12)

def _norm_cid(x) -> str:
    """
    Normalize anything ('C1', '1', 1, '(C1,N1,G1)', 'Cluster1') -> 'C1'.
    """
    s = str(x).strip()
    # tuple-ish like "('C1','N1','G1')" -> take first field
    if s.startswith("(") and "," in s:
        s = s[1:s.find(",")].strip().strip("'\"")
    # drop 'Cluster' prefix if any
    if s.lower().startswith("cluster"):
        s = s[len("cluster"):]
    s = s.strip()
    # already 'C#'
    if s and (s[0] in ("C", "c")) and s[1:].isdigit():
        return f"C{int(s[1:])}"
    # pure digit -> add 'C'
    if s.isdigit():
        return f"C{int(s)}"
    # fallback: keep 'C?' if present, else return as-is
    return s if s else "C1"

def _link_for_cluster(R0_RC_Links: dict, cid_like) -> dict:
    """Safe fetch of per-cluster link record."""
    cid = str(cid_like)
    if cid in R0_RC_Links:
        return R0_RC_Links[cid]
    # Normalize "C1" <-> "1"
    if cid.startswith("C") and cid[1:].isdigit():
        alt = cid[1:]
        if alt in R0_RC_Links: 
            return R0_RC_Links[alt]
        if alt.isdigit() and int(alt) in R0_RC_Links:
            return R0_RC_Links[int(alt)]
    if cid.isdigit():
        c_alt = f"C{cid}"
        if c_alt in R0_RC_Links:
            return R0_RC_Links[c_alt]
    # fallback
    return next(iter(R0_RC_Links.values()), {"ul_rate_kBps":1e-6,"dl_rate_kBps":1e-6})

def power_exp_from_units(units: str, default: float = 3.0) -> float:
    if not units: return default
    m = re.search(r"(\d+)\s*$", str(units))
    return float(m.group(1)) if m else default

def _apply_solver_params(m, snapshot):
    # Core MILP tuning
    m.Params.NonConvex     = 0
    m.Params.Presolve      = 2
    m.Params.Aggregate     = 2
    m.Params.MIPFocus      = 1
    m.Params.Heuristics    = 0.2
    m.Params.Cuts          = 0
    m.Params.NumericFocus  = 1
    m.Params.ScaleFlag     = 2

def _build_pwl_points(R: float, rho_max: float = 0.9, K: int = 2):
    """Return breakpoints (x_k, y_k) for phi(x)=1/(R-x), denser near rho_max."""
    R = max(1e-9, float(R))
    xs, ys = [], []
    for k in range(K):
        u = rho_max * (k / (K - 1))**2  # denser near rho_max
        x = u * R
        y = 1.0 / max(1e-9, (R - x))
        xs.append(x)
        ys.append(y)
    return xs, ys

def sanitize_pwl(xs, ys, *, rtol: float = 1e-9, atol: float = 1e-9,
                 min_span: float = 1e-5):
    """
    Make PWL breakpoint lists safe for Gurobi:

      * enforce strictly increasing x-values
      * ensure overall span max(x) - min(x) >= min_span (>= 1e-6)
      * keep x/y lengths consistent

    This is used for both GPU and link PWLs, including the static model.
    """
    xs_c = list(xs)
    ys_c = list(ys)

    # keep same length
    if len(xs_c) != len(ys_c):
        n = min(len(xs_c), len(ys_c))
        xs_c, ys_c = xs_c[:n], ys_c[:n]

    # basic guards
    if not xs_c:
        return [0.0, 1.0], [0.0, 0.0]
    if len(xs_c) == 1:
        # single point: just extend a flat segment
        return [float(xs_c[0]), float(xs_c[0]) + min_span], [ys_c[0], ys_c[0]]

    # enforce strictly increasing with a small minimum step
    xs_m = [float(xs_c[0])]
    ys_m = [ys_c[0]]

    for i in range(1, len(xs_c)):
        x_i = float(xs_c[i])
        y_i = ys_c[i]

        # required minimum step from last kept x
        step_min = max(
            atol,
            rtol * max(1.0, abs(x_i), abs(xs_m[-1]))
        )
        if x_i <= xs_m[-1] + step_min:
            # bump this x slightly forward
            x_i = xs_m[-1] + step_min
        xs_m.append(x_i)
        ys_m.append(y_i)

    # ensure overall span is large enough for Gurobi
    span = xs_m[-1] - xs_m[0]
    if span < min_span:
        # Stretch the x-axis around xs_m[0] while preserving relative shape.
        base = xs_m[0]
        if span <= 0.0:
            # all points collapsed: create a small segment
            xs_m = [base, base + min_span]
            ys_m = [ys_m[0], ys_m[-1]]
        else:
            scale = min_span / span
            xs_m = [base + (x - base) * scale for x in xs_m]

    return xs_m, ys_m


def dbg_pwl(xs, tag):
    xs_sorted = sorted(xs)
    if len(xs_sorted) < 2: 
        print(f"[DBG2] {tag}: insufficient points"); return
    gaps = [xs_sorted[i+1] - xs_sorted[i] for i in range(len(xs_sorted)-1)]
    print(f"[DBG2] {tag}: min_gap={min(gaps):.3e}, max_gap={max(gaps):.3e}, span={xs_sorted[-1]-xs_sorted[0]:.3e}, n={len(xs_sorted)}")

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

# ----------------- θ bounds pre-computation -----------------

def compute_theta_bounds(snapshot: Dict[str, Any]) -> dict:
    """
    Compute theta_min / theta_max from config + capacities.

    θ(g,f) = R_eff(g,f) / W_ref  where R_eff is effective FLOPs/s
            for GPU g at DVFS step f.

    Returns a dict:
      {
        "theta_min_global": float,
        "theta_max_global": float,
        "per_gpu": {
            "C-N-G": {
               "theta_max": float,
               "by_freq": {"freq_label": theta}
            }, ...
        },
        "W_ref_used": float,
        "safety_factor": float,
      }
    """

    TH      = snapshot.get("Theta_Min", {}) or {}
    G       = tuple(snapshot.get("G") or [])
    F       = dict(snapshot.get("F") or {})
    RATE    = dict(snapshot.get("RATE") or {})
    GPUINFO = snapshot.get("GPUINFO") or {}
    FLOPS   = snapshot.get("FLOPS") or {}

    # --- reference workload selection (same idea as min-latency/min-power) ---
    safety     = float(TH.get("safety_factor", 0.6))
    use_med    = bool(TH.get("use_pending_tasks_median", True))
    pct        = int(TH.get("ref_workload_percentile", 50))
    W_ref_cfg  = float(TH.get("ref_workload_flops", 4.5e10))

    # Build a reference workload from current pending tasks if requested
    W_ref = W_ref_cfg
    if use_med and FLOPS:
        flops_vals = []
        for w in FLOPS.values():
            try:
                v = float(w)
                if v > 0.0 and math.isfinite(v):
                    flops_vals.append(v)
            except Exception:
                pass
        if flops_vals:
            flops_vals.sort()
            # percentile index
            k = max(0, min(len(flops_vals) - 1,
                           int(round((pct / 100.0) * (len(flops_vals) - 1)))))
            W_ref = float(flops_vals[k])

    if not math.isfinite(W_ref) or W_ref <= 0.0:
        W_ref = W_ref_cfg

    # --- helper accessors ---
    def _gid(g):
        return f"{g[0]}-{g[1]}-{g[2]}"

    def _cp(g):
        """Fallback Cp if we ever need CP * f as a crude rate proxy."""
        try:
            info = GPUINFO.get(g) or GPUINFO.get(_gid(g), {}) or {}
            v = float(info.get("C_p", 1.0))
            if not math.isfinite(v) or v <= 0.0:
                return 1.0
            return v
        except Exception:
            return 1.0

    def _num(x):
        """Extract first numeric substring from a frequency label."""
        try:
            return float(x)
        except Exception:
            m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", str(x))
            return float(m.group(0)) if m else None

    # Mirror the frequency pruning used in the model so θ is computed
    # on the same effective F/RATE domain.
    F_pruned = {g: tuple(F.get(g, ())) for g in G}
    F_pruned = _prune_freqs(F_pruned, RATE)

    per_gpu = {}
    theta_max_global = 0.0
    theta_min_candidate = float("inf")

    for g in G:
        freqs = F_pruned.get(g) or ()
        if not freqs:
            continue

        by_freq = {}
        for f in freqs:
            # tolerate RATE keyed by str or original object
            R = None
            if g in RATE:
                rm = RATE[g]
                if f in rm:
                    R = float(rm[f])
                else:
                    fs = str(f)
                    if fs in rm:
                        R = float(rm[fs])

            if R is None:
                # crude fallback: Cp * numeric_freq
                fv = _num(f) or 0.0
                R = _cp(g) * fv

            if R <= 0.0:
                continue

            theta = float(R) / float(W_ref)
            by_freq[str(f)] = theta

        if not by_freq:
            continue

        th_g_max = max(by_freq.values())
        per_gpu[_gid(g)] = {"theta_max": th_g_max, "by_freq": by_freq}

        theta_max_global = max(theta_max_global, th_g_max)
        theta_min_candidate = min(theta_min_candidate, th_g_max)

    if theta_min_candidate == float("inf"):
        theta_min_candidate = 0.0

    theta_min_global = safety * theta_min_candidate

    # Make sure we never accidentally report a floor above the max
    theta_min_global = min(theta_min_global, theta_max_global)

    return {
        "theta_min_global": theta_min_global,
        "theta_max_global": theta_max_global,
        "per_gpu": per_gpu,
        "W_ref_used": W_ref,
        "safety_factor": safety,
    }

# ------------------------------------------------------------------
# Static model cache (GPU/link PWL + y/Lambda/lam_ul/lam_dl, etc.)
# ------------------------------------------------------------------

_MODEL_CACHE: Dict[str, Any] = {}


def _static_signature(G, F, RATE, R0_RC_Links, rho_max, Kpts, *, use_dl: bool = True):
    """
    Signature for the static part of the max-efficiency model:
    depends ONLY on topology, capacities and PWL discretization.
    """
    g_key = tuple(sorted(
        (_gid(g),
         tuple(sorted(F[g])),
         tuple(sorted((str(f), _q(RATE[g][f])) for f in F[g])))
        for g in G
    ))

    clusters = sorted({g[0] for g in G})

    def _link_key_for(c: str):
        rec = _link_for_cluster(R0_RC_Links, c) or {}
        return (
            c,
            _q(_f(rec.get("ul_rate_kBps"), 0.0)),
            _q(_f(rec.get("dl_rate_kBps"), 0.0)),
        )

    link_key = tuple(sorted(_link_key_for(c) for c in clusters))

    return ("eff_v1", g_key, link_key, _q(rho_max), int(Kpts), bool(use_dl))


def _ensure_static(snapshot: Dict[str, Any],
                   G: Tuple[Tuple[str, str, str], ...],
                   F: Dict[Any, Tuple[str, ...]],
                   RATE: Dict[Any, Dict[str, float]],
                   R0_RC_Links: Dict[str, Dict[str, float]],
                   rho_max: float,
                   Kpts: int) -> Dict[str, Any]:
    """
    Build (or reuse) the static part of the max-efficiency model:
      - y[g,f], Lambda_g[g], lam_ul_c[c], lam_dl_c[c]
      - Q_g[g], U_c[c], D_c[c]
      - GPU PWL and link PWL constraints
    Everything here is independent of the current set of tasks.
    """
    use_dl = bool(snapshot.get("use_downlink", True))
    sig = _static_signature(G, F, RATE, R0_RC_Links, rho_max, Kpts, use_dl=use_dl)
    if _MODEL_CACHE.get("sig") == sig and _MODEL_CACHE.get("m") is not None:
        return _MODEL_CACHE

    if not _HAS_GUROBI:
        raise RuntimeError(f"Gurobi not available: {_IMPORT_ERR}")

    m = gp.Model("maxefficiency")
    # Keep logging off here; per-call OutputFlag is controlled later if needed.
    m.Params.OutputFlag = 0
    _apply_solver_params(m, snapshot)


    # --- static vars ---
    a_active = {g: m.addVar(vtype=GRB.BINARY, name=f"a_active[{_gid(g)}]") for g in G}
    y = {g: {f: m.addVar(vtype=GRB.BINARY,
                         name=f"y[{_gid(g)},{f}]")
             for f in F[g]}
         for g in G}

    Q_g      = {g: m.addVar(lb=0.0, name=f"Q_gpu[{_gid(g)}]") for g in G}
    Lambda_g = {g: m.addVar(lb=0.0, name=f"Lambda_gpu[{_gid(g)}]") for g in G}
    clusters = sorted({g[0] for g in G})
    lam_ul_c = {c: m.addVar(lb=0.0, name=f"lam_ul[{c}]") for c in clusters}
    U_c      = {c: m.addVar(lb=0.0, name=f"U_ul[{c}]")   for c in clusters}

    lam_dl_c: Dict[str, gp.Var] = {}
    D_c: Dict[str, gp.Var]      = {}
    if use_dl:
        lam_dl_c = {c: m.addVar(lb=0.0, name=f"lam_dl[{c}]") for c in clusters}
        D_c      = {c: m.addVar(lb=0.0, name=f"D_dl[{c}]")   for c in clusters}

    # Auxiliary for power linearization: Zlam[g,f] ~ Lambda_g[g] * y[g,f]
    Zlam = {
        g: {
            f: m.addVar(lb=0.0, name=f"Zlam[{_gid(g)},{f}]")
            for f in F[g]
        }
        for g in G
    }

    # --- static constraints: one-freq, PWL(Q_g), link PWLs, soft link caps ---
    BIGM = 1e6

    # GPU: one frequency, PWL queueing delay, and Lambda guard for rho_max
    for g in G:
        gid = _gid(g)

        # at most one DVFS bin per GPU (static)
        m.addConstr(
            gp.quicksum(y[g][f] for f in F[g]) == a_active[g],
            name=f"C_onefreq[{gid}]",
        )
        # m.addConstr(
        #     gp.quicksum(y[g][f] for f in F[g]) <= 1,
        #     name=f"C1_freq_one[{gid}]"
        # )

        # cap guard: Lambda_g <= rho_max * R(g,f) when y[g,f]=1
        m.addConstr(
            Lambda_g[g] <= gp.quicksum(
                y[g][f] * (rho_max * max(1e-6, float(RATE[g][f])))
                for f in F[g]
            ),
            name=f"cap_guard[{gid}]"
        )

        # PWL(Q_g) around Lambda_g[g]
        for f in F[g]:
            Q_gf = m.addVar(lb=0.0, name=f"Q_gpu_pwl[{gid},{f}]")
            xs, ys = _build_pwl_points(
                max(1e-6, float(RATE[g][f])),
                rho_max=rho_max,
                K=Kpts,
            )
            xs, ys = sanitize_pwl(xs, ys)
            m.addGenConstrPWL(Lambda_g[g], Q_gf, xs, ys,
                              name=f"PWL_gpu[{gid},{f}]")
            # link shared Q_g[g] to the PWL output under DVFS choice y[g,f]
            m.addConstr(Q_g[g] <= Q_gf + BIGM * (1 - y[g][f]),
                        name=f"link_Q_le[{gid},{f}]")
            m.addConstr(Q_g[g] >= Q_gf - BIGM * (1 - y[g][f]),
                        name=f"link_Q_ge[{gid},{f}]")

    # Tighter Big-M for Zlam
    Rmax_g = {
        g: float(rho_max) * max(1e-9, max(float(RATE[g][f]) for f in F[g]))
        for g in G
    }
    for g in G:
        gid = _gid(g)
        Mg = Rmax_g[g]
        for f in F[g]:
            m.addConstr(Zlam[g][f] <= Mg * y[g][f],
                        name=f"Zlam_ub_bin[{gid},{f}]")
            m.addConstr(Zlam[g][f] <= Lambda_g[g],
                        name=f"Zlam_ub_lam[{gid},{f}]")
            m.addConstr(Zlam[g][f] >= Lambda_g[g] - Mg * (1 - y[g][f]),
                        name=f"Zlam_lb[{gid},{f}]")

    # Link PWLs and soft caps (UL & optional DL)
    rho_guard = float(snapshot.get("pwl_rho_max", rho_max))
    s_ul_cap: Dict[str, gp.Var] = {}
    s_dl_cap: Dict[str, gp.Var] = {}

    for c in clusters:
        Lrec = _link_for_cluster(R0_RC_Links, c)
        r_ul = max(1e-6, _f(Lrec.get("ul_rate_kBps"), 0.0))
        xsU, ysU = sanitize_pwl(
            *_build_pwl_points(r_ul, rho_max=rho_guard, K=Kpts)
        )
        m.addGenConstrPWL(lam_ul_c[c], U_c[c], xsU, ysU, name=f"PWL_ul[{c}]")

        s_ul = m.addVar(lb=0.0, name=f"ul_cap_slack[{c}]")
        s_ul_cap[c] = s_ul
        m.addConstr(lam_ul_c[c] <= rho_guard * r_ul + s_ul,
                    name=f"C4_ul_cap[{c}]")

        if use_dl:
            r_dl = max(1e-6, _f(Lrec.get("dl_rate_kBps"), 0.0))
            xsD, ysD = sanitize_pwl(
                *_build_pwl_points(r_dl, rho_max=rho_guard, K=Kpts)
            )
            m.addGenConstrPWL(lam_dl_c[c], D_c[c], xsD, ysD,
                              name=f"PWL_dl[{c}]")

            s_dl = m.addVar(lb=0.0, name=f"dl_cap_slack[{c}]")
            s_dl_cap[c] = s_dl
            m.addConstr(lam_dl_c[c] <= rho_guard * r_dl + s_dl,
                        name=f"C5_dl_cap[{c}]")

    # --- pre-computed upper bounds used in Big-M linearizations ---
    QMAX: Dict[Any, float] = {}
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

    UDMAX: Dict[str, Tuple[float, float]] = {}
    for c in clusters:
        Lrec = _link_for_cluster(R0_RC_Links, c)
        r_ul = max(1e-6, _f(Lrec.get("ul_rate_kBps"), 0.0))
        xsU, ysU = sanitize_pwl(
            *_build_pwl_points(r_ul, rho_max=rho_guard, K=Kpts)
        )
        if use_dl:
            r_dl = max(1e-6, _f(Lrec.get("dl_rate_kBps"), 0.0))
            xsD, ysD = sanitize_pwl(
                *_build_pwl_points(r_dl, rho_max=rho_guard, K=Kpts)
            )
            UDMAX[c] = (max(ysU) if ysU else 0.0,
                        max(ysD) if ysD else 0.0)
        else:
            UDMAX[c] = (max(ysU) if ysU else 0.0, 0.0)

    _MODEL_CACHE.clear()
    _MODEL_CACHE.update({
        "sig":       sig,
        "m":         m, "a_active": a_active, 
        "y":         y,
        "Lambda_g":  Lambda_g,
        "Q_g":       Q_g,
        "clusters":  clusters,
        "lam_ul_c":  lam_ul_c,
        "U_c":       U_c,
        "lam_dl_c":  lam_dl_c,
        "D_c":       D_c,
        "Zlam":      Zlam,
        "s_ul_cap":  s_ul_cap,
        "s_dl_cap":  s_dl_cap,
        "QMAX":      QMAX,
        "UDMAX":     UDMAX,
        "use_dl":    use_dl,
    })
    return _MODEL_CACHE

# ----------------- data collection -----------------
def _collect(snapshot: Dict[str, Any], new_tasks: List[Dict[str, Any]]):
    """
    Normalize snapshot + new_tasks for the max-efficiency optimizer.

    This is aligned with the min-latency / min-power collectors:
      - single source of truth for FLOPS / RATE
      - consistent λ_j handling (STREAMING-ONLY: λ_j > 0 is enforced in _solve_eff)
    """

    # ---- tasks ----
    T = [str(t["Task_ID"]) for t in new_tasks]

    FLOPS = {str(t["Task_ID"]): _f(t.get("Workload_FLOPs"), 0.0)
             for t in new_tasks}
    DEADL = {str(t["Task_ID"]): _f(t.get("Task_Deadline"), float("inf"))
             for t in new_tasks}
    ULKB  = {str(t["Task_ID"]): _f(t.get("UL_Total_kB"), 0.0)
             for t in new_tasks}
    DLKB  = {str(t["Task_ID"]): _f(t.get("DL_Total_kB"), 0.0)
             for t in new_tasks}

    # ---- λ_j (arrival rate) ----
    # Global default λ (fps) if a task doesn't override it
    arrival_rate_fps = float(
        snapshot.get("SERVICE_ARRIVAL_RATE_fps",
                     snapshot.get("arrival_rate_fps", 0.0))
    )

    LAMBDA: Dict[str, float] = {}
    for t in new_tasks:
        tid = str(t["Task_ID"])
        lam_raw = (
            t.get("Lambda")
            or t.get("lambda")
            or t.get("arrival_rate")
            or t.get("Task_Arrival_Rate")
        )
        if lam_raw is None:
            lam_raw = arrival_rate_fps
        LAMBDA[tid] = _f(lam_raw, arrival_rate_fps)

    # ---- GPUs ----
    G: List[Tuple[str, str, str]] = []
    F: Dict[Any, Tuple[str, ...]] = {}
    RATE: Dict[Any, Dict[str, float]] = {}
    GPUINFO: Dict[Any, Dict[str, Any]] = {}

    EPS_RATE = 1e-12

    for gk, info in (snapshot.get("gpus") or {}).items():
        g = _norm_gpu_key(gk)  # ('C#','N#','G#') tuple

        rates = (info or {}).get("rates") or {}
        if not isinstance(rates, dict) or not rates:
            # skip GPUs with no usable rate map
            continue

        rate_map: Dict[str, float] = {}
        for fk, rv in rates.items():
            try:
                r = _f(rv, 0.0)
            except Exception:
                continue
            if r <= 0.0:
                continue
            rate_map[str(fk)] = max(EPS_RATE, float(r))

        if not rate_map:
            # nothing usable for this GPU
            continue

        def _fnum(s):
            s = str(s).lower().replace("ghz", "").replace("mhz", "")
            try:
                return float(
                    re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)[0]
                )
            except Exception:
                return float("inf")

        G.append(g)
        # sort bins by numeric frequency (ascending)
        F[g] = tuple(sorted(rate_map.keys(), key=_fnum))
        RATE[g] = rate_map
        GPUINFO[g] = dict(info)

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


    GPU_TAIL = {g: _tail_seconds(GPUINFO.get(g)) for g in G}

    # ---- links (R0↔Rc) ----
    R0_RC_Links = {}
    links_src = snapshot.get("R0_RC_Links") or snapshot.get("links") or {}
    for k, d in links_src.items():
        c = _norm_cid(k)
        R0_RC_Links[c] = {
            "ul_rate_kBps": _f(d.get("ul_rate_kBps", d.get("r0_c")), 0.0),
            "dl_rate_kBps": _f(d.get("dl_rate_kBps", d.get("rc_rate_kBps", d.get("rc_0"))), 0.0),
            "ul_prop_s":    _f(d.get("ul_prop_s", d.get("uplink_prop_delay")), 0.0),
            "dl_prop_s":    _f(d.get("dl_prop_s", d.get("downlink_prop_delay")), 0.0),
        }
    

    # # Final frequency pruning (use same helper used later for the model)
    # F = _prune_freqs(F, RATE)

    # print("G",G)
    # print("F",F)
    # print("R0_RC_Links",R0_RC_Links)
    # print("RATE",RATE)
    # print("FLOPS",FLOPS)
    # print("T",T)
    # print("GPU_TAIL",GPU_TAIL)
    # print("LAMBDA",LAMBDA)
    # print("DEADL",DEADL)
    # print("GPUINFO",GPUINFO)

    return (T, tuple(G), F, RATE, GPU_TAIL,
            FLOPS, DEADL, ULKB, DLKB, GPUINFO, LAMBDA, R0_RC_Links)


# ----------------- MILP core -----------------
def _solve_eff(snapshot: Dict[str, Any],
               new_tasks: List[Dict[str, Any]], *,
               pinned_assignments: Optional[Dict[str, Dict[str, str]]] = None,
               pinned_frequencies: Optional[Dict[str, str]] = None) -> Dict[str, Any]:

    if not _HAS_GUROBI:
        raise RuntimeError(f"Gurobi not available: {_IMPORT_ERR}")

    # -------- collect --------
    (T, G, F, RATE, GPU_TAIL, FLOPS, DEADL, ULKB, DLKB,
     GPUINFO, LAMBDA, R0_RC_Links) = _collect(snapshot, new_tasks)

    # incremental variant: exactly one new task
    if not T or not G:
        raise RuntimeError("efficiency optimizer: empty task or GPU set")
    if len(T) != 1:
        raise RuntimeError(f"max-eff (incremental) expects exactly one task, got {len(T)}")
    t0 = T[0]

    # # keep only GPUs that actually have at least one freq bin
    # G = tuple(g for g in G if F.get(g) and len(F[g]) > 0)
    # if not G:
    #     raise RuntimeError("efficiency optimizer: no usable GPUs after pruning")

    # # ---- per-GPU minimum frequency thresholds ----
    # min_freq_global = float(snapshot.get("min_freq_global", 0.0) or 0.0)
    # min_freq_map = snapshot.get("min_freq_map", {}) or {}

    # def _as_float_freq(s):
    #     try:
    #         # tolerate labels like "2505", "2505MHz", "2.5GHz"
    #         s_ = str(s).strip().lower().replace("ghz", "").replace("mhz", "")
    #         return float(s_)
    #     except Exception:
    #         return float("inf")  # keep unparseable bins

    # def _ok(g, f):
    #     gid = f"{g[0]}-{g[1]}-{g[2]}"
    #     thr = float(min_freq_map.get(gid, min_freq_global) or 0.0)
    #     if thr <= 0.0:
    #         return True
    #     return _as_float_freq(f) >= thr

    # # prune freq bins by threshold, but ensure at least one remains
    # for g in list(G):
    #     pruned = tuple(f for f in F[g] if _ok(g, f))
    #     if not pruned:
    #         # keep the closest-to-threshold bin
    #         pruned = (min(F[g], key=lambda fk: float(_as_float_freq(fk) or 0.0)),)
    #     F[g] = pruned


    # STREAMING-ONLY: task must have positive arrival rate λ
    if float(LAMBDA.get(t0, 0.0)) <= 0.0:
        raise RuntimeError(f"Streaming-only: Lambda missing/zero for task {t0}.")
    
    now = _f(snapshot.get("now"), 0.0)
    OD  = snapshot.get("Optimizer_Defaults", {}) or {}

    # ---------- θ floors from capacities (using CURRENT G/F/RATE/FLOPS) ----------
    tb = compute_theta_bounds({
        **snapshot,
        "G":     G,
        "F":     F,
        "RATE":  RATE,
        "GPUINFO": GPUINFO,
        "FLOPS": FLOPS,
    })

    # global θ floor (already includes safety_factor from Theta_Min)
    THETA_MIN_GLOBAL = float(tb.get("theta_min_global", 0.0))

    # safety factor used inside compute_theta_bounds (default 0.3 from config)
    safety = float(tb.get("safety_factor", 0.3))

    # scale per-GPU floors by the *same* safety factor
    theta_gpu_scale = safety          # set to 0.0 to effectively disable per-GPU θ floors

    THETA_MIN_GPU_MAP = {
        gid: theta_gpu_scale * float(rec.get("theta_max", 0.0))
        for gid, rec in tb.get("per_gpu", {}).items()
    }

    # -------- static model (GPU & link PWLs, y, Lambda_g, lam_*_c, Zlam, ...) --------
    rho_max = float(snapshot.get("pwl_rho_max", OD.get("pwl_rho_max", 0.75)))
    Kpts    = int(snapshot.get("pwl_points",   OD.get("pwl_points",   2)))

    H = _ensure_static(snapshot, G, F, RATE, R0_RC_Links, rho_max, Kpts)
    m = H["m"]

    # # per-call logging override
    # m.Params.OutputFlag = int(snapshot.get("solver_log", 0))
    
    y        = H["y"]
    Lambda_g = H["Lambda_g"]
    Q_g      = H["Q_g"]
    clusters = H["clusters"]
    lam_ul_c = H["lam_ul_c"]
    U_c      = H["U_c"]
    lam_dl_c = H["lam_dl_c"]
    D_c      = H["D_c"]
    Zlam     = H["Zlam"]
    s_ul_cap = H["s_ul_cap"]
    s_dl_cap = H["s_dl_cap"]
    QMAX     = H["QMAX"]
    UDMAX    = H["UDMAX"]
    use_dl   = bool(H.get("use_dl", True))

     # -------- background load lambda_bg (FLOPs/s already on each GPU) ----------
    lambda_bg_raw = snapshot.get("lambda_bg", {}) or {}
    lambda_bg_norm = {
        str(k): float(v)
        for k, v in (lambda_bg_raw.items() if hasattr(lambda_bg_raw, "items") else [])
        if v not in (None, "")
    }



    # ---- remove dynamic constraints from any previous call & reset objective ----
    for c in list(m.getConstrs()):
        if c.ConstrName.startswith((
            "C2_assign[", "link_x_to_y[",
            "z_le_x[", "z_le_y[", "z_ge_xplusy-1[",
            "def_Lambda_gpu_rate[", "def_lam_ul_rate[", "def_lam_dl_rate[",
            "C_theta_gpu[", "C_theta_global[",
            "C_rho_soft[",
            "def_lam_ul_total", "def_lam_dl_total",
            "share_ul[", "share_dl[",
            "link_a_jobs[", "lam_ul_on[", "lam_dl_on[",
            "min_active_clusters",
            "C6_deadline[",
            "C_comp_lin[", "C_sys_lin[", "C_prop_lin[", "C_all[",
            "C_pin_freq_lb[", "C_pin_freq_ub[",
        )):
            m.remove(c)

    if hasattr(m, "getQConstrs"):
        for qc in list(m.getQConstrs()):
            m.remove(qc)

    if m.getObjective() is not None:
        m.setObjective(0.0)

    # -------- dynamic variables (per call) --------
    # assignment vars for the single task
    x = {g: m.addVar(vtype=GRB.BINARY, name=f"x[{t0},{_gid(g)}]") for g in G}

    # Per-task delay decomposition
    Dcomp = m.addVar(lb=0.0, name=f"Dcomp[{t0}]")
    Dsys  = m.addVar(lb=0.0, name=f"Dsys[{t0}]")
    Dprop = m.addVar(lb=0.0, name=f"Dprop[{t0}]")
    Dall  = m.addVar(lb=0.0, name=f"Doverall[{t0}]")
    # deadline slack
    s_dead = m.addVar(lb=0.0, name=f"ddl_slack[{t0}]")

    # z = x ∧ y
    z: Dict[Tuple[Any, str], gp.Var] = {}
    for g in G:
        for f in F[g]:
            z[(g, f)] = m.addVar(vtype=GRB.BINARY,
                                 name=f"z[{t0},{_gid(g)},{f}]")

    obj_extra = gp.LinExpr(0.0)

    # -------- core constraints --------

    # C2: one GPU per task
    m.addConstr(gp.quicksum(x[g] for g in G) == 1, name=f"C2_assign[{t0}]")
    for g in G:
        m.addConstr(H["a_active"][g] - x[g] >= 0, name=f"link_a_ge_x[{t0},{_gid(g)}]")

    # Link assignments to active frequency: if any task uses g, some y must be 1
    for g in G:
        m.addConstr(
            x[g] <= gp.quicksum(y[g][f] for f in F[g]),
            name=f"link_x_to_y[{_gid(g)}]"
        )

    # z = x ∧ y
    for g in G:
        for f in F[g]:
            zgf = z[(g, f)]
            m.addConstr(zgf <= x[g],         name=f"z_le_x[{t0},{_gid(g)},{f}]")
            m.addConstr(zgf <= y[g][f],      name=f"z_le_y[{t0},{_gid(g)},{f}]")
            m.addConstr(zgf >= x[g] + y[g][f] - 1,
                        name=f"z_ge_xplusy-1[{t0},{_gid(g)},{f}]")

    # Rate definitions (STREAMING-ONLY) for this task only
    lam_t   = float(LAMBDA[t0])
    flops_t = float(FLOPS[t0])
    ulkb_t  = float(ULKB[t0])
    dlkb_t  = float(DLKB[t0])

    for g in G:
        m.addConstr(
            Lambda_g[g] == x[g] * flops_t * lam_t,
            name=f"def_Lambda_gpu_rate[{_gid(g)}]"
        )
    for c in clusters:
        m.addConstr(
            lam_ul_c[c] == gp.quicksum(
                x[g] * ulkb_t * lam_t for g in G if g[0] == c
            ),
            name=f"def_lam_ul_rate[{c}]"
        )
        if lam_dl_c:
            m.addConstr(
                lam_dl_c[c] == gp.quicksum(
                    x[g] * dlkb_t * lam_t for g in G if g[0] == c
                ),
                name=f"def_lam_dl_rate[{c}]"
            )

    # C3: GPU capacity (use Lambda_g directly, RATE is κ in FLOPs/s)
    for g in G:
        rhs = gp.quicksum(y[g][f] * float(RATE[g][f]) for f in F[g])
        m.addConstr(Lambda_g[g] <= rhs, name=f"C3_capacity_rate[{_gid(g)}]")

    # θ floors
    def _theta_lhs_for(g):
        return gp.quicksum(
            z[(g, f)] *
            (max(1e-9, float(RATE[g][f])) / max(1e-9, flops_t))
            for f in F[g]
        )

    theta_w = float(snapshot.get(
        "theta_penalty_weight",
        (snapshot.get("Optimizer_Defaults", {}) or {}).get("theta_penalty_weight", 1e4),
    ))

    # Per-GPU θ floors
    for g in G:
        gid = f"{g[0]}-{g[1]}-{g[2]}"
        th = float(THETA_MIN_GPU_MAP.get(gid, 0.0))
        if th > 0.0:
            s = m.addVar(lb=0.0, name=f"theta_slack[{t0},{_gid(g)}]")
            m.addConstr(
                _theta_lhs_for(g) + s >= th * x[g],
                name=f"C_theta_gpu[{t0},{_gid(g)}]"
            )
            obj_extra += theta_w * s

    # Single global θ floor
    if THETA_MIN_GLOBAL and float(THETA_MIN_GLOBAL) > 0.0:
        th = float(THETA_MIN_GLOBAL)
        s = m.addVar(lb=0.0, name=f"theta_slack_global[{t0}]")
        lhs_job = gp.quicksum(_theta_lhs_for(g) for g in G)
        m.addConstr(lhs_job + s >= th, name=f"C_theta_global[{t0}]")
        obj_extra += theta_w * s

   # ---- Cluster share (soft) & min-active clusters ----
    share_cap = float(OD.get("cluster_share_cap",
                            snapshot.get("cluster_share_cap", 1.0)))
    share_w   = float(OD.get("share_penalty_weight",
                            snapshot.get("share_penalty_weight", 0.0)))

    # ---- per-GPU utilization soft cap (includes background load) ----
    rho_cap_soft = float(snapshot.get("rho_cap", OD.get("rho_cap", 0.8)))
    rho_penalty  = float(snapshot.get("rho_penalty", OD.get("rho_penalty", 1e5)))


    s_rho = {}
    for g in G:
        gid = _gid(g)
        s_rho[g] = m.addVar(lb=0.0, name=f"s_rho[{gid}]")
        cap_g = gp.quicksum(y[g][f] * float(RATE[g][f]) for f in F[g])
        lambda_bg_g = float(lambda_bg_norm.get(gid, 0.0))

        # background + new load can exceed cap only via slack
        m.addConstr(
            lambda_bg_g + Lambda_g[g] <= rho_cap_soft * cap_g + s_rho[g],
            name=f"C_rho_soft[{gid}]"
        )

    # penalty term (will be subtracted from objective later)
    obj_extra += rho_penalty * gp.quicksum(s_rho[g] for g in G)

    # # ---- total utilization hard cap including background load ----
    # rho_cap_total = float(snapshot.get("rho_cap_total",
    #                                    OD.get("rho_cap_total", 0.0)))
    # if rho_cap_total > 0.0:
    #     for g in G:
    #         gid = _gid(g)
    #         cap_g = gp.quicksum(y[g][f] * float(RATE[g][f]) for f in F[g])
    #         lambda_bg_g = float(lambda_bg_norm.get(gid, 0.0))
    #         # background + this-job load must not exceed rho_cap_total * capacity
    #         m.addConstr(
    #             lambda_bg_g + Lambda_g[g] <= rho_cap_total * cap_g,
    #             name=f"C_rho_total[{gid}]"
    #         )

    if share_cap < 1.0 - 1e-6 and share_w > 0.0:
        lam_ul_total = m.addVar(lb=0.0, name="lam_ul_total")
        lam_dl_total = m.addVar(lb=0.0, name="lam_dl_total")
        m.addConstr(lam_ul_total == gp.quicksum(lam_ul_c[c] for c in clusters),
                    name="def_lam_ul_total")
        if lam_dl_c:
            m.addConstr(lam_dl_total == gp.quicksum(lam_dl_c[c] for c in clusters),
                        name="def_lam_dl_total")
        s_ul_share = {c: m.addVar(lb=0.0, name=f"share_ul_slack[{c}]")
                      for c in clusters}
        s_dl_share = {c: m.addVar(lb=0.0, name=f"share_dl_slack[{c}]")
                      for c in clusters}
        for c in clusters:
            m.addConstr(lam_ul_c[c] <= share_cap * lam_ul_total + s_ul_share[c],
                        name=f"share_ul[{c}]")
            if lam_dl_c:
                m.addConstr(lam_dl_c[c] <= share_cap * lam_dl_total + s_dl_share[c],
                            name=f"share_dl[{c}]")
            obj_extra += share_w * (s_ul_share[c] +
                                    (s_dl_share[c] if lam_dl_c else 0.0))

    Kmin = int(snapshot.get("min_active_clusters", 1))
    if Kmin > 1:
        a_clust = {c: m.addVar(vtype=GRB.BINARY, name=f"a_clust[{c}]")
                   for c in clusters}
        for c in clusters:
            jobs_in_c = gp.quicksum(x[g] for g in G if g[0] == c)
            m.addConstr(jobs_in_c <= 1 * a_clust[c], name=f"link_a_jobs[{c}]")
            BIG = 1e9
            m.addConstr(lam_ul_c[c] <= BIG * a_clust[c], name=f"lam_ul_on[{c}]")
            if lam_dl_c:
                m.addConstr(lam_dl_c[c] <= BIG * a_clust[c], name=f"lam_dl_on[{c}]")
        m.addConstr(gp.quicksum(a_clust[c] for c in clusters)
                    >= min(Kmin, len(clusters)),
                    name="min_active_clusters")

    # ---- link cap penalties (use static slacks) ----
    w_link = float(OD.get("link_cap_penalty_weight",
                          snapshot.get("link_cap_penalty_weight", 1.0)))
    obj_extra += w_link * (
        gp.quicksum(s_ul_cap[c] for c in s_ul_cap) +
        gp.quicksum(s_dl_cap[c] for c in s_dl_cap)
    )

    # ---- soft deadline for this task ----
    deadline_w = float(OD.get("deadline_penalty_weight",
                              snapshot.get("deadline_penalty_weight", 1e4)))
    Dt = _f(DEADL.get(t0), float("inf")) - now
    if math.isfinite(Dt) and Dt > 1e-9:
        m.addConstr(Dall <= Dt + s_dead, name=f"C6_deadline[{t0}]")
        obj_extra += deadline_w * s_dead

    # ---- compose per-task delays via Big-M (no quadratic constraints) ----
    tau_sum = {
        c: _f(_link_for_cluster(R0_RC_Links, c).get("ul_prop_s"), 0.0) +
           _f(_link_for_cluster(R0_RC_Links, c).get("dl_prop_s"), 0.0)
        for c in clusters
    }

    TAIL0 = {g: float(GPU_TAIL[g]) for g in G}
    for g in G:
        c = g[0]
        Umax, Dmax = UDMAX[c]
        M_comp = flops_t * QMAX[g] + TAIL0[g]
        M_sys  = ulkb_t * Umax + ((dlkb_t * Dmax) if use_dl else 0.0)
        M_prop = abs(float(tau_sum[c]))

        m.addConstr(
            Dcomp >= flops_t * Q_g[g] + TAIL0[g] - M_comp * (1 - x[g]),
            name=f"C_comp_lin[{t0},{_gid(g)}]"
        )
        if use_dl and D_c:
            m.addConstr(
                Dsys >= U_c[c] * ulkb_t + D_c[c] * dlkb_t
                        - M_sys * (1 - x[g]),
                name=f"C_sys_lin[{t0},{_gid(g)}]"
            )
        else:
            m.addConstr(
                Dsys >= U_c[c] * ulkb_t - M_sys * (1 - x[g]),
                name=f"C_sys_lin[{t0},{_gid(g)}]"
            )
        m.addConstr(
            Dprop >= tau_sum[c] - M_prop * (1 - x[g]),
            name=f"C_prop_lin[{t0},{_gid(g)}]"
        )

    m.addConstr(Dall == Dcomp + Dsys + Dprop, name=f"C_all[{t0}]")

    # -------- pinned DVFS (conditional on being assigned) --------
    if pinned_frequencies:
        for gid, ffix in (pinned_frequencies or {}).items():
            gk = _norm_gpu_key(gid)
            if gk in G and F[gk]:
                ffix = str(ffix)
                valid = {str(f) for f in F[gk]}

                if ffix not in valid:
                    def _ff(x_):
                        try:
                            return float(
                                str(x_).lower()
                                .replace("ghz", "")
                                .replace("mhz", "")
                            )
                        except Exception:
                            return None

                    target = _ff(ffix)
                    if target is not None:
                        cands = [k for k in valid if _ff(k) is not None]
                        if cands:
                            ffix = min(
                                cands, key=lambda k: abs(_ff(k) - target)
                            )
                    if ffix not in valid:
                        ffix = max(F[gk], key=lambda k: float(RATE[gk][k]))
                        print(f"[WARN] pinned_frequencies[{gid}] invalid; using '{ffix}'")

                for f in F[gk]:
                    fname = str(f)
                    if fname == ffix:
                        m.addConstr(
                            y[gk][f] >= x[gk],
                            name=f"C_pin_freq_lb[{_gid(gk)},{fname}]",
                        )
                    else:
                        m.addConstr(
                            y[gk][f] <= 1 - x[gk],
                            name=f"C_pin_freq_ub[{_gid(gk)},{fname}]",
                        )
            else:
                print(
                    f"[MILP][WARN] Ignoring pinned frequency for {gid} "
                    f"(GPU not in G or no F[g])"
                )

        # -------- objective ingredients --------
    # throughput (FLOPs/s for this job)
    served = gp.LinExpr(flops_t * lam_t)

    def _p_static_of(GPUINFO, g) -> float:
        info = GPUINFO.get(g, {}) or {}
        for k in ("P_static_W", "P_static", "P_idle_W",
                  "p_static_w", "p_idle_w", "P_st", "P_st_W"):
            if k in info:
                try:
                    v = float(info[k])
                    if v > 0:
                        return v
                except Exception:
                    pass
        return 60.0

    P_static_W = {g: _p_static_of(GPUINFO, g) for g in G}

    def power_exp_of(g):
        info = GPUINFO.get(g, {}) or {}
        meta = snapshot.get("GPU_Specs_Meta", {}) or {}
        p = info.get("power_exp", meta.get("power_exp", 1.0))
        try:
            return max(1.0, float(p))
        except Exception:
            return 1.0

    def _num_freq(x_):
        try:
            return float(str(x_).lower().replace("ghz", "").replace("mhz", ""))
        except Exception:
            m_ = re.search(r"[\d.]+", str(x_))
            return float(m_.group(0)) if m_ else 0.0

    freq_value: Dict[Any, float] = {}
    for g in G:
        for f in F[g]:
            if f not in freq_value:
                freq_value[f] = _num_freq(f)

    exp_const = {g: power_exp_of(g) for g in G}
    phi = {
        g: max(0.0, float((GPUINFO.get(g, {}) or {}).get("phi_power", 0.0)))
        for g in G
    }
    k = {
        g: {
            f: phi[g] * (float(freq_value[f]) ** (exp_const[g] - 1.0))
            for f in F[g]
        }
        for g in G
    }

    obj_static  = gp.quicksum(
        P_static_W[g] * gp.quicksum(y[g][f] for f in F[g]) for g in G
    )
    obj_dynamic = gp.quicksum(
        k[g][f] * Zlam[g][f] for g in G for f in F[g]
    )
    power_total = obj_static + obj_dynamic

    latency_sur = Dall

    # -------- weights for weighted objective --------
    w_power = float(snapshot.get("eff_power_weight",
                                OD.get("eff_power_weight", 1.0)))
    w_lat   = float(snapshot.get("eff_latency_weight",
                                 OD.get("eff_latency_weight", 1.0)))
    w_load  = float(snapshot.get("eff_bg_load_weight",
                                 OD.get("eff_bg_load_weight", 0.3)))
    w_hist  = float(snapshot.get("eff_load_balance_weight",
                                 OD.get("eff_load_balance_weight", 0.3)))

    # ----- per-GPU load term from lambda_bg ----------
    if w_load > 0.0 and lambda_bg_norm:
        cost_load = gp.quicksum(
            float(lambda_bg_norm.get(_gid(g), 0.0)) * x[g]
            for g in G
        )
    else:
        cost_load = gp.LinExpr(0.0)

    # ----- history term: prefer GPUs that have served fewer jobs ----------
    hist_used = snapshot.get("gpu_jobs_served", {}) or {}
    hist_norm = {
        str(k): float(v)
        for k, v in (hist_used.items() if hasattr(hist_used, "items") else [])
        if v not in (None, "")
    }
    if w_hist > 0.0 and hist_norm:
        cost_hist = gp.quicksum(
            (float(hist_norm.get(_gid(g), 0.0)) ** 2) * x[g]
            for g in G
        )
    else:
        cost_hist = gp.LinExpr(0.0)

    # ----- explicit penalty for low frequencies (Fix C) -----
    freq_pen_w = float(snapshot.get(
        "freq_penalty_weight",
        OD.get("freq_penalty_weight", 0.0),
    ))

    freq_penalty = gp.LinExpr(0.0)
    if freq_pen_w > 0.0:
        # compute max numeric freq per GPU
        fmax = {
            g: max(freq_value[f] for f in F[g])
            for g in G
        }
        # penalize distance from fmax for chosen bins
        freq_penalty = gp.quicksum(
            freq_pen_w * (fmax[g] - freq_value[f]) * y[g][f]
            for g in G for f in F[g]
        )


    # -------- final single weighted objective: throughput + fairness --------
    obj_main = (
        served
        - w_power * power_total
        - w_lat * latency_sur
        - w_load * cost_load
        - w_hist * cost_hist
        - freq_penalty
    )

    # Small tiebreaker: prefer higher raw compute if all else equal
    obj_tie = -1e-6 * gp.quicksum(
        float(RATE[g][f]) * y[g][f] for g in G for f in F[g]
    )

    m.setObjective(obj_main + obj_tie - obj_extra, GRB.MAXIMIZE)
    m.optimize()

    status_code = m.Status
    if status_code == GRB.INFEASIBLE:
        if bool(snapshot.get("debug_dump_models", False)):
            _dump_iis(m, "efficiency_model")
            _dump_model(m, "max_eff_infeasible", write_mps=True)
        out = {
            "assignments":      {},
            "frequencies":      {},
            "task_frequencies": {},
            "status":           "infeasible",
            "objective":        None,
        }
        _dump_json(out, "efficiency_solution_assignments")
        return out

    if status_code not in (GRB.OPTIMAL, GRB.INTERRUPTED):
        out = {
            "assignments":      {},
            "frequencies":      {},
            "task_frequencies": {},
            "status":           f"status_{status_code}",
            "objective":        None,
        }
        _dump_json(out, "efficiency_solution_assignments")
        return out

    if m.SolCount == 0:
        out = {
            "assignments":      {},
            "frequencies":      {},
            "task_frequencies": {},
            "status":           "no_solution",
            "objective":        None,
        }
        _dump_json(out, "efficiency_solution_assignments")
        return out

    # _dump_solution_obj(m, "efficiency_model_solution")

    # -------- read solution --------
    assign: Dict[str, Dict[str, str]] = {}
    for g in G:
        if x[g].X > 0.5:
            c, n, gname = g
            assign[t0] = {"Cluster": c, "Node": n, "GPU": gname}
            break

    if not assign:
        out = {
            "assignments":      {},
            "frequencies":      {},
            "task_frequencies": {},
            "status":           "noop",
            "objective":        None,
        }
        _dump_json(out, "efficiency_solution_assignments")
        return out

    freqs_gid: Dict[str, str] = {}
    freq_plan: Dict[Tuple[str, str, str], str] = {}
    for g in G:
        chosen = None
        for f in F[g]:
            if y[g][f].X > 0.5:
                chosen = str(f)
                break
        if chosen:
            gid = _gid(g)
            freqs_gid[gid] = chosen
            freq_plan[g] = chosen

    task_freq: Dict[str, str] = {}
    for g in G:
        for f in F[g]:
            if z[(g, f)].X > 0.5:
                task_freq[str(t0)] = str(f)
                break

    status = {
        GRB.OPTIMAL: "optimal",
        GRB.INTERRUPTED: "interrupted",
    }.get(m.Status, f"status_{m.Status}")

    out = {
        "assignments":      assign,
        "frequencies":      freqs_gid,
        "task_frequencies": task_freq,
        "status":           status,
        "objective":        None,
    }

    # power / throughput diagnostics
    P_static = float(
        gp.quicksum(
            P_static_W[g] * gp.quicksum(y[g][f] for f in F[g]) for g in G
        ).getValue()
    )
    P_dyn = float(
        gp.quicksum(k[g][f] * Zlam[g][f] for g in G for f in F[g]).getValue()
    )

    served_val = float(served.getValue())
    power_val  = float(power_total.getValue())

    out["served_load"] = served_val
    out["power_total"] = power_val

    if out["power_total"] > 0.0:
        out["objective"] = out["served_load"] / out["power_total"]
    else:
        out["objective"] = None

    # _dump_json(out, "efficiency_solution_assignments")
    return out


# ----------------- FastPolicy entrypoints -----------------
def optimize_efficiency(snapshot, new_task, catalog=None, store=None):
    tasks_list = new_task if isinstance(new_task, list) else [new_task]
    return solve_incremental(snapshot, tasks_list, objective="efficiency")


def optimize(snapshot, new_task, objective="efficiency", catalog=None, store=None):
    obj = (objective or "efficiency").lower()
    if obj in ("efficiency", "max_efficiency", "eff"):
        tasks_list = new_task if isinstance(new_task, list) else [new_task]
        return solve_incremental(snapshot, tasks_list, objective="efficiency")

    
# ----------------- public entrypoint -----------------
def solve_incremental(snapshot: Dict[str, Any],
              new_tasks: List[Dict[str, Any]], objective: str = "efficiency",
              pinned_assignments: Optional[Dict[str, Dict[str, str]]] = None,
              pinned_frequencies: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """
    Efficiency optimizer:
      - 'weighted' mode: single MILP solve (served - beta*power - gamma*latency).
      - 'dinkelbach' mode: outer loop updating theta, each step solves the same MILP.
    """
    pinned_assignments  = pinned_assignments  or snapshot.get("pinned_assignments", {})  or {}
    pinned_frequencies  = pinned_frequencies  or snapshot.get("pinned_frequencies", {})  or {}
    mode = str((snapshot.get("eff_mode") or "weighted")).lower()

    if mode != "dinkelbach":
        # Single pass
        return _solve_eff(snapshot, new_tasks,
                          pinned_assignments=pinned_assignments,
                          pinned_frequencies=pinned_frequencies)

    # ----- Dinkelbach outer loop (ratio served/power) -----
    theta   = float(snapshot.get("eff_theta_init", 1e-2))
    max_it  = int(snapshot.get("eff_max_iter", 1))
    tol_rel = float(snapshot.get("eff_tol", 1e-3))

    sol = None
    for _ in range(max_it):
        snap_i = _copy.deepcopy(snapshot)
        snap_i["eff_mode"]  = "dinkelbach"
        snap_i["eff_theta"] = theta

        sol = _solve_eff(snap_i, new_tasks,
                         pinned_assignments=pinned_assignments,
                         pinned_frequencies=pinned_frequencies)

        served = float(sol.get("served_load", 0.0))
        power  = float(sol.get("power_total", 0.0))
        if power <= 1e-9:
            break

        theta_new = served / power
        if abs(theta_new - theta) <= tol_rel * max(1.0, theta):
            theta = theta_new
            break
        theta = theta_new

    if sol is not None:
        sol["eff_theta_final"] = theta
    return sol


def solve(snapshot: Dict[str, Any],
          new_tasks: List[Dict[str, Any]],
          objective: str = "efficiency") -> Dict[str, Any]:
    # Streaming-only pipeline uses the incremental two-pass method
    return solve_incremental(snapshot, new_tasks, objective=objective)

# legacy alias
def solve_efficiency(snapshot: Dict[str, Any], new_tasks: List[Dict[str, Any]]):
    return solve_incremental(snapshot, new_tasks, objective="efficiency")