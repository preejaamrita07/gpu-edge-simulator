# live_min_latency_optimizer_incremental.py
# --------------------------------------------------------------
# Incremental MIN-LATENCY optimizer (Gurobi), compatible with facade.
# --------------------------------------------------------------

from typing import Dict, Any, List, Tuple, Optional
import copy as _copy
import math, os, json, pathlib, re
import time
from contextlib import contextmanager

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


# ---------- Static model cache ----------
_MODEL_CACHE = {"sig": None, "m": None, "H": None}
_REBUILDS = {"solves": 0}
REBUILD_EVERY = 250   # pick 100–1000 based on load

def _clear_static_cache():
    _MODEL_CACHE.clear()

class TicTok:
    """Lightweight timings collector."""
    def __init__(self, enabled=True):
        self.enabled = bool(enabled)
        self._stack = []   # [(label, t0)]
        self.records = []  # [(label, dt_seconds)]
        self.tags = {}     # optional key->value metadata

    def tic(self, label: str):
        if not self.enabled: return
        self._stack.append((label, time.perf_counter()))

    def tok(self, label: str = None):
        if not self.enabled or not self._stack: return 0.0
        lab, t0 = self._stack.pop()
        dt = time.perf_counter() - t0
        self.records.append((label or lab, dt))
        return dt

    @contextmanager
    def section(self, label: str):
        self.tic(label)
        try:
            yield
        finally:
            self.tok(label)

    def add(self, label: str, seconds: float):
        if not self.enabled: return
        self.records.append((label, float(seconds)))

    def summary_str(self, sort_by='time', precision=3, total=True):
        if not self.enabled: return "(tictok disabled)"
        if sort_by == 'time':
            rows = sorted(self.records, key=lambda x: x[1], reverse=True)
        else:
            rows = list(self.records)
        lines = []
        w = max([len(l) for l, _ in rows] + [6])
        for lab, dt in rows:
            lines.append(f"{lab:<{w}}  {dt:.{precision}f}s")
        if total:
            lines.append("-" * (w + 10))
            lines.append(f"{'TOTAL':<{w}}  {sum(dt for _, dt in rows):.{precision}f}s")
        return "\n".join(lines)

def _get_tictok(container, default_enabled=True):
    """
    Retrieve or create a TicTok collector from a holder object (self/state/snapshot).
    - If container has attribute/key '_tictok', reuse it.
    - Enable if container has 'tictok' truthy flag, else default_enabled.
    """
    tt = None
    # dict-like
    if isinstance(container, dict):
        tt = container.get("_tictok")
        if tt is None:
            tt = TicTok(enabled=bool(container.get("tictok", default_enabled)))
            container["_tictok"] = tt
        return tt
    # object-like
    tt = getattr(container, "_tictok", None)
    if tt is None:
        enabled = bool(getattr(container, "tictok", default_enabled))
        tt = TicTok(enabled=enabled)
        setattr(container, "_tictok", tt)
    return tt

# ---------------- Solver parameters tuned for MILP ----------------
def _apply_solver_params(m, snapshot):
    # Core MILP tuning
    m.Params.NonConvex     = 0
    m.Params.Presolve      = 2
    m.Params.Aggregate     = 2
    m.Params.MIPFocus      = 1
    m.Params.Heuristics    = 0.2
    m.Params.Cuts          = 2
    m.Params.NumericFocus  = 2
    m.Params.ScaleFlag     = 2
    # User overrides
    gap = snapshot.get("mip_gap", snapshot.get("solver", {}).get("mip_gap"))
    if gap is not None:
        try: m.Params.MIPGap = float(gap)
        except: pass
    tl = snapshot.get("time_limit", snapshot.get("solver", {}).get("time_limit_s"))
    if tl is not None:
        try: m.Params.TimeLimit = float(tl)
        except: pass
    threads = snapshot.get("threads", snapshot.get("solver", {}).get("threads"))
    if threads:
        try: m.Params.Threads = int(threads)
        except: pass

# ----------------- helpers & plumbing -----------------
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

def _canon_cng(C: str, N: str, G: str) -> tuple:
    import re as _re
    def canon_one(x: str, prefix: str) -> str:
        x = (x or '').strip()
        m = _re.match(rf'(?i){prefix}\s*(\d+)$', x)
        if m:
            return f'{prefix.upper()}{int(m.group(1))}'
        if _re.fullmatch(r'\d+', x):
            return f'{prefix.upper()}{int(x)}'
        alias = {'C': 'cluster', 'N': 'node', 'G': 'gpu'}[prefix.upper()]
        m2 = _re.match(rf'(?i){alias}\s*(\d+)$', x)
        if m2:
            return f'{prefix.upper()}{int(m2.group(1))}'
        return x or f'{prefix.upper()}?'
    return (canon_one(str(C), 'C'), canon_one(str(N), 'N'), canon_one(str(G), 'G'))

def _q(x, k=9):  # quantize floats for a stable signature
    return round(float(x), k)

def _norm_gpu_key(k):
    """Normalize GPU key into canonical (C#, N#, G#) tuple of strings.
    Accepts tuples/lists, dicts with Cluster/Node/GPU, and strings like:
    'C1-N1-G2', 'C1–N1–G2', 'C1, N1, G2', 'C1 N1 G2', '1-2-3', 'Cluster1-Node2-GPU3'.
    Raises ValueError if unrecognized.
    """
    if isinstance(k, (list, tuple)) and len(k) == 3:
        C, N, G = (str(x).strip() for x in k)
        return _canon_cng(C, N, G)

    if isinstance(k, dict):
        C = str(k.get('Cluster', '')).strip()
        N = str(k.get('Node', '')).strip()
        G = str(k.get('GPU', '')).strip()
        return _canon_cng(C, N, G)

    s = str(k).strip()
    if s:
        for d in _DASHES:
            s = s.replace(d, '-')
        s = re.sub(r'[\s,]+', '-', s)
        parts = [p for p in s.split('-') if p]
        if len(parts) == 3:
            return _canon_cng(parts[0], parts[1], parts[2])
        mC = re.search(r'(?i)c\s*(\d+)', s)
        mN = re.search(r'(?i)n\s*(\d+)', s)
        mG = re.search(r'(?i)g\s*(\d+)', s)
        if mC and mN and mG:
            return (f'C{int(mC.group(1))}', f'N{int(mN.group(1))}', f'G{int(mG.group(1))}')
        nums = re.findall(r'(\d+)', s)
        if len(nums) >= 3:
            return (f'C{int(nums[0])}', f'N{int(nums[1])}', f'G{int(nums[2])}')
    raise ValueError(f"Unrecognized GPU key format: {k!r}")

def _norm_cid(k):
    s = str(k).upper().strip()
    if s.startswith("R0-RC"):
        s = "C" + s.split("R0-RC", 1)[1]
    elif s.startswith("RC"):
        s = "C" + s[2:]
    elif not s.startswith("C"):
        try: s = "C" + str(int(s))
        except: s = "C" + s
    return s

def _extract_rates(info: dict) -> Dict[str, float]:
    """
    Return a normalized frequency->rate map for a GPU.

    Accepts any of these fields:
      - "rates"            (preferred)
      - "service_rates"
      - "freqs"
      - "Service_Rates"
      - "SERVICE_RATES"

    Keeps only positive, numeric rates and coerces keys to str.
    Example output: {"1500": 3.2e12, "2300": 5.1e12}
    """
    candidates = ["rates", "service_rates", "freqs", "Service_Rates", "SERVICE_RATES"]

    rates = None
    for name in candidates:
        r = info.get(name)
        if isinstance(r, dict) and r:
            rates = r
            break

    if not isinstance(rates, dict):
        return {}

    out: Dict[str, float] = {}
    for fk, rv in rates.items():
        try:
            val = float(rv)
        except Exception:
            continue
        if val > 0.0:
            out[str(fk)] = val
    return out

def _gid(g):
    c,n,gg = g
    return f"{c}-{n}-{gg}"

def _base_tid(s: str) -> str:
    s = str(s)
    return s[:-2] if s.endswith("_U") or s.endswith("_D") else s

def _f(x, d=0.0):
    try:
        return float(x)
    except Exception:
        return float(d)

# --- KEY FIX: normalize cluster ids, incl. "R0-RC1" -> "C1"
_R0RC_RE = re.compile(r"^R0[-_]?RC\s*(\d+)$", re.IGNORECASE)

def _norm_cid(x) -> str:
    """
    Normalize anything ('C1','1','R0-RC1','Cluster1','(C1,N1,G1)') -> 'C1'.
    """
    s = str(x).strip()
    if s.startswith("(") and "," in s:
        s = s[1:s.find(",")].strip().strip("'\"")
    if s.lower().startswith("cluster"):
        s = s[len("cluster"):]
    m = _R0RC_RE.match(s)
    if m:
        return f"C{int(m.group(1))}"
    if s and (s[0] in ("C", "c")) and s[1:].isdigit():
        return f"C{int(s[1:])}"
    if s.isdigit():
        return f"C{int(s)}"
    return s if s else "C1"

def _link_for_cluster(R0_RC_Links: dict, cid_like) -> dict:
    """Fetch per-cluster link record with tolerant keys and fallbacks."""
    cid = _norm_cid(cid_like)
    if cid in R0_RC_Links:
        return R0_RC_Links[cid]
    # try numeric alternates
    n = cid[1:] if cid.startswith("C") else cid
    if str(n) in R0_RC_Links: return R0_RC_Links[str(n)]
    if isinstance(n, str) and n.isdigit() and int(n) in R0_RC_Links: return R0_RC_Links[int(n)]
    # last resort: first record (keeps solver alive but logs)
    # Use tiny but nonzero rates to avoid division-by-zero in PWL
    print(f"[WARN] no link entry for {cid_like}; using first/degenerate link.")
    return next(iter(R0_RC_Links.values()), {"ul_rate_kBps":1.0,"dl_rate_kBps":1.0,"ul_prop_s":0.0,"dl_prop_s":0.0})

def _build_pwl_points(R: float, rho_max: float = 0.9, K: int = 2):
    """Breakpoints for phi(x)=1/(R-x), denser near rho_max."""
    R = max(1e-6, float(R))
    K = max(2, int(K))
    xs, ys = [], []
    for k in range(K):
        u = rho_max * (k / (K - 1))**2  # denser near rho_max
        x = u * R
        y = 1.0 / max(1e-9, (R - x))
        xs.append(x); ys.append(y)
    return xs, ys

def sanitize_pwl(xs, ys, rtol=1e-6, atol=1e-12, need_monotone_x=True):
    assert len(xs) == len(ys) and len(xs) >= 2
    pairs = sorted(zip(xs, ys), key=lambda p: p[0])
    clean = [pairs[0]]
    for x, y in pairs[1:]:
        x0, y0 = clean[-1]
        if abs(x - x0) <= max(atol, rtol * max(abs(x0), abs(x), 1.0)):
            clean[-1] = (x, y)
        else:
            clean.append((x, y))
    xs_c, ys_c = zip(*clean)
    span = xs_c[-1] - xs_c[0]
    if len(xs_c) < 2 or abs(span) <= max(atol, rtol * max(abs(xs_c[-1]), abs(xs_c[0]), 1.0)):
        delta = max(atol, rtol)
        xs_c = (xs_c[0], xs_c[0] + delta)
        ys_c = (ys_c[0], ys_c[-1])
    if need_monotone_x:
        xs_m, ys_m = [xs_c[0]], [ys_c[0]]
        for i in range(1, len(xs_c)):
            if xs_c[i] > xs_m[-1] + max(atol, rtol * max(abs(xs_m[-1]), abs(xs_c[i]), 1.0)):
                xs_m.append(xs_c[i]); ys_m.append(ys_c[i])
            else:
                xs_m.append(xs_m[-1] + max(atol, rtol)); ys_m.append(ys_c[i])
        xs_c, ys_c = xs_m, ys_m
    return list(xs_c), list(ys_c)

def dbg_pwl(xs, tag):
    xs_sorted = sorted(xs)
    if len(xs_sorted) < 2: 
        print(f"[DBG2] {tag}: insufficient points"); return
    gaps = [xs_sorted[i+1] - xs_sorted[i] for i in range(len(xs_sorted)-1)]
    print(f"[DBG2] {tag}: min_gap={min(gaps):.3e}, max_gap={max(gaps):.3e}, span={xs_sorted[-1]-xs_sorted[0]:.3e}, n={len(xs_sorted)}")

def _prune_freqs(F, RATE, *, tol=1e-9):
    """
    Keep at most one frequency per distinct effective RATE[g][f].
    Two rates are equal if |r1-r2| <= tol*max(1,|r1|,|r2|).
    Among equal-rate freqs, keep the numerically highest freq label.
    """
    def _fnum(s, default=float("-inf")):
        try: return float(s)
        except: return default

    F2 = {}
    for g, flist in F.items():
        pairs = []
        for f in flist:
            try:
                r = float(RATE[g][f])
            except Exception:
                continue
            if r > 0.0:
                pairs.append((r, str(f)))

        pairs.sort(key=lambda p: (p[0], _fnum(p[1])))

        kept = []
        for r, f in pairs:
            if not kept:
                kept.append((r, f))
                continue
            r0, f0 = kept[-1]
            if abs(r - r0) <= tol * max(1.0, abs(r), abs(r0)):
                if _fnum(f) > _fnum(f0):
                    kept[-1] = (r0, f)
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


def _static_signature(G, F, RATE, R0_RC_Links, rho, Kpts, *, use_dl=False):
    g_key = tuple(sorted(
        (_gid(g),
         tuple(sorted(F[g])),
         tuple(sorted((str(f), _q(RATE[g][f])) for f in F[g])))
        for g in G))

    clusters = sorted({g[0] for g in G})
    link_key = tuple(sorted(
        (c,
         _q(_f(R0_RC_Links[c].get("ul_rate_kBps"), 0.0)),
         _q(_f(R0_RC_Links[c].get("dl_rate_kBps"), 0.0)))
        for c in clusters))

    return ("v2", g_key, link_key, _q(rho), int(Kpts), bool(use_dl))


def _ensure_static(snapshot, G, F, RATE, R0_RC_Links, rho, Kpts):
    """
    Build/reuse static model:
      a_active[g], y[g,f], Q_gpu[g], Lambda_gpu[g],
      link PWLs (UL/DL), GPU PWLs (+ indicators), and soft caps.
    """
    use_dl = bool(snapshot.get("use_downlink", False))
    tt = _get_tictok(snapshot, default_enabled=False)
    sig = _static_signature(G, F, RATE, R0_RC_Links, rho, Kpts, use_dl=use_dl)
    if _MODEL_CACHE.get("sig") == sig and _MODEL_CACHE.get("m") is not None:
        return _MODEL_CACHE["m"], _MODEL_CACHE["H"]


    m = gp.Model("minlatency")
    m.Params.OutputFlag = int(snapshot.get("solver_log", 0))
    _apply_solver_params(m, snapshot)

    clusters = sorted({g[0] for g in G})

    # --- static vars ---
    a_active   = {g: m.addVar(vtype=GRB.BINARY, name=f"a_active[{_gid(g)}]") for g in G}
    y          = {g: {f: m.addVar(vtype=GRB.BINARY, name=f"y[{_gid(g)},{f}]") for f in F[g]} for g in G}
    Q_gpu      = {g: m.addVar(lb=0.0, name=f"Q_gpu[{_gid(g)}]") for g in G}
    Lambda_gpu = {g: m.addVar(lb=0.0, name=f"Lambda_gpu[{_gid(g)}]") for g in G}

    lam_ul_c = {c: m.addVar(lb=0.0, name=f"lam_ul[{c}]") for c in clusters}
    U_c      = {c: m.addVar(lb=0.0, name=f"U_ul[{c}]")  for c in clusters}
    s_ul_cap = {c: m.addVar(lb=0.0, name=f"s_ul_cap[{c}]") for c in clusters}

    lam_dl_c, D_c, s_dl_cap = {}, {}, {}
    if use_dl:
        lam_dl_c = {c: m.addVar(lb=0.0, name=f"lam_dl[{c}]") for c in clusters}
        D_c      = {c: m.addVar(lb=0.0, name=f"D_dl[{c}]")  for c in clusters}
        s_dl_cap = {c: m.addVar(lb=0.0, name=f"s_dl_cap[{c}]") for c in clusters}

    s_rho = {g: m.addVar(lb=0.0, name=f"s_rho[{_gid(g)}]") for g in G}

    # Global max arrival rate across GPUs – used as a linear
    # load-balancing proxy in the objective.
    Lambda_max = m.addVar(lb=0.0, name="Lambda_max")
    for g in G:
        m.addConstr(
            Lambda_max >= Lambda_gpu[g],
            name=f"C_Lambda_max[{_gid(g)}]",
        )

    # --- GPU PWLs & one-freq-if-active ---
    for g in G:
        for f in F[g]:
            R = max(1e-6, float(RATE[g][f]))
            q = m.addVar(lb=0.0, name=f"Q_gpu_pwl[{_gid(g)},{f}]")
            xs, ys = sanitize_pwl(*_build_pwl_points(R, rho_max=rho, K=Kpts))
            m.addGenConstrPWL(Lambda_gpu[g], q, xs, ys, name=f"PWL_gpu[{_gid(g)},{f}]")
            # NOTE: this indicator is named 'pin_freq[...]' – we will not reuse this name elsewhere
            m.addGenConstrIndicator(y[g][f], True, Q_gpu[g] == q,
                                    name=f"pin_freq[{_gid(g)},{f}]")

        m.addConstr(gp.quicksum(y[g][f] for f in F[g]) == a_active[g],
                    name=f"C_onefreq_if_active[{_gid(g)}]")

    # --- Link PWLs + soft caps ---
    for c in clusters:
        Lrec = _link_for_cluster(R0_RC_Links, c)
        r_ul = max(1e-6, _f(Lrec.get("ul_rate_kBps"), 1.0))
        xsU, ysU = sanitize_pwl(*_build_pwl_points(r_ul, rho_max=rho, K=Kpts))
        m.addGenConstrPWL(lam_ul_c[c], U_c[c], xsU, ysU, name=f"PWL_ul[{c}]")
        m.addConstr(lam_ul_c[c] <= rho * r_ul + s_ul_cap[c], name=f"C_ul_cap[{c}]")

        if use_dl:
            r_dl = max(1e-6, _f(Lrec.get("dl_rate_kBps"), 1.0))
            xsD, ysD = sanitize_pwl(*_build_pwl_points(r_dl, rho_max=rho, K=Kpts))
            m.addGenConstrPWL(lam_dl_c[c], D_c[c], xsD, ysD, name=f"PWL_dl[{c}]")
            m.addConstr(lam_dl_c[c] <= rho * r_dl + s_dl_cap[c], name=f"C_dl_cap[{c}]")

    # --- pre-computed upper bounds used in Big-M ---
    QMAX = {}
    for g in G:
        ymax = 0.0
        for f in F[g]:
            xs, ys = sanitize_pwl(*_build_pwl_points(max(1e-6, float(RATE[g][f])), rho_max=rho, K=Kpts))
            if ys:
                ymax = max(ymax, max(ys))
        QMAX[g] = ymax

    UDMAX = {}
    for c in clusters:
        Lrec = _link_for_cluster(R0_RC_Links, c)
        r_ul = max(1e-6, _f(Lrec.get("ul_rate_kBps"), 1.0))
        xsU, ysU = sanitize_pwl(*_build_pwl_points(r_ul, rho_max=rho, K=Kpts))
        if use_dl:
            r_dl = max(1e-6, _f(Lrec.get("dl_rate_kBps"), 1.0))
            xsD, ysD = sanitize_pwl(*_build_pwl_points(r_dl, rho_max=rho, K=Kpts))
            UDMAX[c] = (max(ysU) if ysU else 0.0, max(ysD) if ysD else 0.0)
        else:
            UDMAX[c] = (max(ysU) if ysU else 0.0, 0.0)

    H = {
        "clusters": clusters,
        "a_active": a_active,
        "y": y,
        "Q_gpu": Q_gpu,
        "Lambda_gpu": Lambda_gpu,   # original
        "Lambda_g": Lambda_gpu,     # alias used by latency “boost” term
        "lam_ul_c": lam_ul_c, "U_c": U_c, "s_ul_cap": s_ul_cap,
        "lam_dl_c": lam_dl_c, "D_c": D_c, "s_dl_cap": s_dl_cap,
        "s_rho": s_rho,
        "Lambda_max": Lambda_max,
        "QMAX": QMAX, "UDMAX": UDMAX,
        "use_dl": use_dl,
    }



    _MODEL_CACHE.update({"sig": sig, "m": m, "H": H})
    return m, H

def _add_task_block(
    m, H, t0, G, RATE, FLOPS, ULKB, DLKB,
    LAMBDA_t0, TAIL0_map, tau_sum, rho_cap,
    util_guess, bg_util_weight,
    lambda_bg_norm,
):
    """
    Add vars/rows for the single 'first task' and return handles.
    Lambda_gpu[g] includes both background load lambda_bg[g] and the
    new task stream x[g] * LAMBDA_t0 * FLOPS[t0].
    """
    x     = {g: m.addVar(vtype=GRB.BINARY, name=f"x[{t0},{_gid(g)}]") for g in G}
    Dcomp = m.addVar(lb=0.0, name=f"Dcomp[{t0}]")
    Dsys  = m.addVar(lb=0.0, name=f"Dsys[{t0}]")
    Dprop = m.addVar(lb=0.0, name=f"Dprop[{t0}]")
    Dall  = m.addVar(lb=0.0, name=f"Doverall[{t0}]")
    s_dead= m.addVar(lb=0.0, name=f"ddl_slack[{t0}]")

    rows = []

    # a_active >= x (link) and assign one GPU
    for g in G:
        rows.append(
            m.addConstr(
                H["a_active"][g] - x[g] >= 0,
                name=f"link_a_ge_x[{t0},{_gid(g)}]",
            )
        )
    rows.append(
        m.addConstr(
            gp.quicksum(x[g] for g in G) == 1,
            name=f"C_assign[{t0}]",
        )
    )

    # --- Lambda definition = background + new stream ---
    util_guess     = util_guess or {}
    lambda_bg_norm = lambda_bg_norm or {}

    for g in G:
        gid    = _gid(g)
        lam_bg = float(lambda_bg_norm.get(gid, 0.0))  # FLOPs/s already on g
        rows.append(
            m.addConstr(
                H["Lambda_gpu"][g]
                == lam_bg + x[g] * float(LAMBDA_t0) * float(FLOPS[t0]),
                name=f"def_Lambda[{gid}]",
            )
        )

    # --- rho soft cap with background utilization U_bg(g) ---
    for g in G:
        gid = _gid(g)

        # Background utilization (0..1), scaled if desired
        U_bg_raw = float(util_guess.get(gid, 0.0) or 0.0)
        U_bg     = max(0.0, min(1.0, bg_util_weight * U_bg_raw))

        # Capacity of GPU g at its chosen frequency
        cap_g = gp.quicksum(H["y"][g][f] * float(RATE[g][f]) for f in H["y"][g])

        # Fraction of capacity still available for the new stream under rho_cap
        avail_frac = rho_cap - U_bg
        if avail_frac < 0.0:
            avail_frac = 0.0  # let slack s_rho absorb violation

        rows.append(
            m.addConstr(
                H["Lambda_gpu"][g] <= avail_frac * cap_g + H["s_rho"][g],
                name=f"C_rho_soft[{gid}]",
            )
        )

    # --- UL/DL aggregation by cluster (DL only if used) ---
    use_dl = bool(H.get("use_dl", False))
    for c in H["clusters"]:
        rows.append(
            m.addConstr(
                H["lam_ul_c"][c]
                == gp.quicksum(
                    x[g] * float(LAMBDA_t0) * float(ULKB[t0])
                    for g in G if g[0] == c
                ),
                name=f"def_lam_ul[{c}]",
            )
        )
        if use_dl and H["lam_dl_c"]:
            rows.append(
                m.addConstr(
                    H["lam_dl_c"][c]
                    == gp.quicksum(
                        x[g] * float(LAMBDA_t0) * float(DLKB[t0])
                        for g in G if g[0] == c
                    ),
                    name=f"def_lam_dl[{c}]",
                )
            )

    # --- Big-M linearizations for Dcomp/Dsys/Dprop ---
    UL = float(ULKB[t0])
    DL = float(DLKB[t0])
    Fl = float(FLOPS[t0])

    for g in G:
        c = g[0]
        Umax, Dmax = H["UDMAX"][c]
        M_comp = Fl * H["QMAX"][g] + float(TAIL0_map[g])
        M_sys  = Umax * UL + ((Dmax * DL) if use_dl else 0.0)
        M_prop = abs(float(tau_sum[c]))

        rows.append(
            m.addConstr(
                Dcomp >= Fl * H["Q_gpu"][g] + float(TAIL0_map[g])
                        - M_comp * (1 - x[g]),
                name=f"C_comp_lin[{t0},{_gid(g)}]",
            )
        )

        if use_dl:
            rows.append(
                m.addConstr(
                    Dsys >= H["U_c"][c] * UL + H["D_c"][c] * DL
                           - M_sys * (1 - x[g]),
                    name=f"C_sys_lin[{t0},{_gid(g)}]",
                )
            )
        else:
            rows.append(
                m.addConstr(
                    Dsys >= H["U_c"][c] * UL - M_sys * (1 - x[g]),
                    name=f"C_sys_lin[{t0},{_gid(g)}]",
                )
            )

        rows.append(
            m.addConstr(
                Dprop >= tau_sum[c] - M_prop * (1 - x[g]),
                name=f"C_prop_lin[{t0},{_gid(g)}]",
            )
        )

    rows.append(
        m.addConstr(Dall == Dcomp + Dsys + Dprop, name=f"C_all[{t0}]")
    )

    return {
        "vars": [*x.values(), Dcomp, Dsys, Dprop, Dall, s_dead],
        "constrs": rows,
        "x": x,
        "Dcomp": Dcomp,
        "Dsys": Dsys,
        "Dprop": Dprop,
        "Dall": Dall,
        "s_dead": s_dead,
    }


def _remove_task_block(m, handles):
    m.remove(handles.get("temp_rows", []))     # remove deadline row(s)
    m.remove(handles["constrs"])               # rows created inside _add_task_block(...)
    m.remove(handles["vars"])                  # vars created for the task block
    m.update()


def _collect(snapshot: Dict[str, Any], new_task: List[Dict[str, Any]]):
    """
    Normalize snapshot + tasks for the optimizer (STREAMING-ONLY).
    Returns:
      T, G, F, RATE, TAIL, FLOPS, DEADL, ULKB, DLKB, GPUINFO, LAMBDA, R0_RC_Links

    Conventions / units:
      - Compute side is *scaled to GFLOP/s* to keep numerics well-conditioned:
          FLOPS[t] /= 1e9,  RATE[g][f] /= 1e9
      - Link side stays in kB and kB/s as in config/logs.
    """
    # ---- scaling (compute domain only) ----

    # ---- Tasks (IDs + attributes) ----
    T = [str(t["Task_ID"]) for t in new_task]
    # FLOPs scaled to GFLOPs
    FLOPS = {str(t["Task_ID"]): _f(t.get("Workload_FLOPs"), 0.0) for t in new_task}
    DEADL = {str(t["Task_ID"]): _f(t.get("Task_Deadline"), float("inf")) for t in new_task}
    # Link volumes (already kB in pipeline)
    ULKB  = {str(t["Task_ID"]): _f(t.get("UL_Total_kB"), 0.0) for t in new_task}
    DLKB  = {str(t["Task_ID"]): _f(t.get("DL_Total_kB"), 0.0) for t in new_task}

    # ---- Arrival rates (tasks/s) ----
    LAMBDA: Dict[str, float] = {}
    for t in new_task:
        tid = str(t["Task_ID"])
        lam_raw = (t.get("Lambda") or t.get("lambda") or
                   t.get("arrival_rate") or t.get("Task_Arrival_Rate"))
        lam = _f(lam_raw, 0.0)
        if lam <= 0.0:
            raise RuntimeError(f"Streaming-only: Lambda missing/zero for task {tid}.")
        LAMBDA[tid] = lam

    # ---- GPUs (normalize keys; scale rates to GFLOP/s) ----
    G, F, RATE, TAIL, GPUINFO = [], {}, {}, {}, {}
    dropped = []

    for gk, info in (snapshot.get("gpus") or {}).items():
        try:
            g = _norm_gpu_key(gk)  # ('C#','N#','G#') tuple
        except Exception as e:
            dropped.append(f"skip GPU key {gk!r}: {e}")
            continue

        rate_map_raw = _extract_rates(info)  # accepts 'rates'/'service_rates'/'freqs'/...
        if not rate_map_raw:
            dropped.append(f"skip {_gid(g)}: no usable rate map")
            continue

        # scale each service rate to GFLOP/s and sort frequencies numerically when possible
        def _as_float(x):
            try: return float(x)
            except: return float("inf")

        F[g] = sorted((str(k) for k in rate_map_raw.keys()), key=_as_float, reverse=False)
        RATE[g] = {str(k): float(v) for k, v in rate_map_raw.items()}  # scaled
        # TAIL[g] = _f(info.get("tail"), 0.0)  # seconds of residual tail
        TAIL[g] = _tail_seconds(info)

        GPUINFO[g] = dict(info)
        G.append(g)

    if dropped:
        print("[WARN] Optimizer dropped some GPUs:\n  - " + "\n  - ".join(dropped))

    R0_RC_Links: Dict[str, Dict[str, float]] = {}
    links_src = (snapshot.get("R0_RC_Links") or snapshot.get("links") or {})
    if links_src:
        for k, d in links_src.items():
            c = _norm_cid(k)

            def _pick(dct, *keys, default=None):
                for kk in keys:
                    if kk in dct: return dct[kk]
                return default

            # accept ul_rate_Kbps/ul_rate_kBps/ul_rate_kbps (dl analogs too)
            ul_rate = _pick(d, "ul_rate_Kbps", "ul_rate_kBps", "ul_rate_kbps", default=None)
            dl_rate = _pick(d, "dl_rate_Kbps", "dl_rate_kBps", "dl_rate_kbps", default=None)
            ul_prop = _pick(d, "ul_prop_s", "uplink_prop_delay", default=0.0)
            dl_prop = _pick(d, "dl_prop_s", "downlink_prop_delay", default=0.0)

            R0_RC_Links[c] = {
                "ul_rate_kBps": _f(ul_rate, 1e9),   # permissive default avoids infeasibility
                "dl_rate_kBps": _f(dl_rate, 1e9),
                "ul_prop_s":    _f(ul_prop, 0.0),
                "dl_prop_s":    _f(dl_prop, 0.0),
            }
    else:
        # permissive defaults for all clusters found in GPUs
        for c in sorted({g[0] for g in G}):
            R0_RC_Links[c] = {
                "ul_rate_kBps": 1000000, "dl_rate_kBps": 1000000,
                "ul_prop_s": 0.001,   "dl_prop_s": 0.001,
            }
    
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
def _solve_latency(snapshot: Dict[str, Any],
                   new_task: List[Dict[str, Any]], objective: str = "latency",
                   pinned_assignments: Optional[Dict[str, Dict[str, str]]] = None,
                   pinned_frequencies: Optional[Dict[str, str]] = None) -> Dict[str, Any]:

    # Collect
    T, G, F, RATE, TAIL, FLOPS, DEADL, ULKB, DLKB, GPUINFO, LAMBDA, R0_RC_Links = _collect(snapshot, new_task)
    if not T or not G:
        raise RuntimeError("min-latency optimizer: empty task or GPU set")
    t0 = sorted(T)[0]
    lam_t0 = float(_f(LAMBDA.get(t0, 0.0), 0.0))
    if lam_t0 <= 0.0:
        raise RuntimeError(f"Streaming-only: Lambda missing/zero for task {t0}.")
    now = float(_f(snapshot.get("now"), 0.0))

    # # knobs
    # OD = snapshot.get("Optimizer_Defaults", {}) or {}

    # rho  = float(snapshot.get("pwl_rho_max", OD.get("pwl_rho_max", 0.8)))
    # Kpts = int(snapshot.get("pwl_points",  OD.get("pwl_points", 2)))

    # link_cap_w = float(snapshot.get(
    #     "link_cap_penalty_weight",
    #     OD.get("link_cap_penalty_weight", OD.get("link_penalty_weight", 0.1)),
    # ))

    # deadline_w = float(snapshot.get(
    #     "deadline_penalty_weight",
    #     OD.get("deadline_penalty_weight", 1e6),
    # ))

    # freq_eps = float(snapshot.get(
    #     "freq_tiebreak_eps",
    #     OD.get("freq_tiebreak_eps", 0.0),
    # ))
    
    # load_balance_w = float(snapshot.get(
    #     "load_balance_weight",
    #     OD.get("load_balance_weight", 0.0),
    # ))


    # rho_cap = float(snapshot.get("rho_cap", OD.get("rho_cap", 0.6)))

    # # Background utilization map (0..1) — use explicit guess only
    # util_guess = {}
    # raw_util = snapshot.get("utilization_guess") or {}
    # if isinstance(raw_util, dict):
    #     util_guess = {
    #         str(k): float(v) for k, v in raw_util.items()
    #         if v not in (None, "")
    #     }

    # bg_util_weight = float(snapshot.get("bg_util_weight", 1.0))

    # # Background compute load lambda_bg[g] in FLOPs/s
    # lambda_bg_raw = snapshot.get("lambda_bg", {}) or {}
    # if hasattr(lambda_bg_raw, "items"):
    #     lambda_bg_norm = {
    #         str(k): float(v)
    #         for k, v in lambda_bg_raw.items()
    #         if v not in (None, "")
    #     }
    # else:
    #     lambda_bg_norm = {}

    
    # # Optional: latency-mode overrides – now QUEUE-AWARE
    # if str(objective).startswith("latency"):
    #     snapshot = dict(snapshot)  # copy to avoid mutating caller

    #     # Keep queues comfortably below saturation (tunable).
    #     snapshot["pwl_rho_max"] = snapshot.get("pwl_rho_max", 0.80)
    #     snapshot["rho_cap"]     = snapshot.get("rho_cap",     0.75)

    #     # Turn ON queue / capacity penalties instead of disabling them.
    #     snapshot["bg_util_weight"]          = snapshot.get("bg_util_weight",          1.0)
    #     snapshot["link_cap_penalty_weight"] = snapshot.get("link_cap_penalty_weight", 0.1)
    #     snapshot["gpu_cap_penalty_weight"]  = snapshot.get("gpu_cap_penalty_weight",  10.0)

    #     # Small frequency penalty so the solver can prefer slightly slower
    #     # but less loaded GPUs instead of always hammering the fastest.
    #     snapshot["freq_penalty_weight"]     = snapshot.get("freq_penalty_weight", 1e-4)

    #     # Refresh knobs from the updated snapshot
    #     rho            = float(snapshot["pwl_rho_max"])
    #     rho_cap        = float(snapshot["rho_cap"])
    #     bg_util_weight = float(snapshot["bg_util_weight"])

    
    # # # Optional: latency-mode overrides (only if you want them)
    # # if str(objective).startswith("latency"):
    # #     snapshot = dict(snapshot)  # copy to avoid mutating caller

    # #     # Let queues run hot; don't punish rho or rate in latency mode
    # #     snapshot["pwl_rho_max"]            = snapshot.get("pwl_rho_max", 0.90)
    # #     snapshot["rho_cap"]                = snapshot.get("rho_cap", 0.6)
    # #     snapshot["bg_util_weight"]         = 0.0
    # #     snapshot["link_cap_penalty_weight"]= 0.0
    # #     snapshot["gpu_cap_penalty_weight"] = 0.0
    # #     snapshot["freq_penalty_weight"]    = 0.0   # keep 0 for fixed; maybe 1e-5 for adaptive

    # #     # refresh knobs from snapshot
    # #     rho      = 0.6
    # #     rho_cap  = 0.80
    # #     link_cap_w = 0.0
    # #     gpu_cap_w  = 0.0
    # #     freq_penalty_w = 0.0

    # #     # ignore background utilization soft cap in latency mode
    # #     bg_util_weight = 0.0
    # #     snapshot["bg_util_weight"] = 0.0

    # static_every = int(snapshot.get("static_rebuild_every",
    #                                 OD.get("static_rebuild_every", 0)))

    # # Tell static builder whether DL is used for this solve
    # snapshot = dict(snapshot)
    # snapshot["use_downlink"] = any(_f(DLKB[tid], 0.0) > 0.0 for tid in T)

    # _REBUILDS["solves"] += 1
    # if _REBUILDS["solves"] % REBUILD_EVERY == 0:
    #     _clear_static_cache()

    # # Static model
    # m, H = _ensure_static(snapshot, G, F, RATE, R0_RC_Links, rho, Kpts)

    # # Cache max frequency per GPU for tie-breaking / freq penalty
    # H["fmax_g"] = {g: max(float(f) for f in H["y"][g]) for g in G}

    # # Pin or seed frequencies (NO new constraints named 'pin_freq[...]')
    # if pinned_frequencies:
    #     for g in G:
    #         gid = _gid(g)
    #         if gid in pinned_frequencies:
    #             fpin = str(pinned_frequencies[gid])
    #             for f in F[g]:
    #                 if str(f) == fpin:
    #                     pass  # keep default bounds; y[fpin] follows a_active
    #                 else:
    #                     H["y"][g][f].ub = 0.0
    # else:
    #     # warm-start
    #     for g in G:
    #         if F[g]:
    #             f0 = max(F[g], key=lambda ff: float(RATE[g][ff]))
    #             for f in F[g]:
    #                 H["y"][g][f].start = 1.0 if f == f0 else 0.0

    # # propagation delays for linearization
    # clusters = H["clusters"]
    # tau_sum = {
    #     c: _f(_link_for_cluster(R0_RC_Links, c).get("ul_prop_s"), 0.0)
    #      + (_f(_link_for_cluster(R0_RC_Links, c).get("dl_prop_s"), 0.0)
    #         if H["use_dl"] else 0.0)
    #     for c in clusters
    # }
    # TAIL0 = {g: _f(TAIL.get(g), 0.0) for g in G}

    # # per-task block (pass lambda_bg_norm as new arg)
    # handles = _add_task_block(
    #     m, H, t0, G, RATE, FLOPS, ULKB, DLKB,
    #     lam_t0, TAIL0, tau_sum, rho_cap,
    #     util_guess, bg_util_weight,
    #     lambda_bg_norm,
    # )

    # # deadline (only if finite and positive)
    # temp_rows = []                               
    # Dt = float(_f(DEADL.get(t0), float("inf")) - now)
    # if math.isfinite(Dt) and Dt > 1e-9:
    #     r = m.addConstr(handles["Dall"] <= Dt + handles["s_dead"],
    #                     name=f"C_deadline[{t0}]")
    #     temp_rows.append(r)                        

    # handles["temp_rows"] = temp_rows

    # # objective
    # # Tiny, increasing penalty with rate so lower rates are preferred
    # freq_pen = gp.LinExpr(0.0)
    # if freq_eps > 0.0:
    #     for g in G:
    #         for f in F[g]:
    #             freq_pen += float(RATE[g][f]) * H["y"][g][f]

    # obj = handles["Dall"]
    # # Optional: penalize the max per-GPU arrival rate to encourage
    # # spreading the stream instead of hot-spotting one GPU.
    # if load_balance_w > 0.0 and H.get("Lambda_max") is not None:
    #     obj += load_balance_w * H["Lambda_max"]
    # if link_cap_w > 0.0:
    #     obj += link_cap_w * gp.quicksum(H["s_ul_cap"][c] for c in clusters)
    #     if H["use_dl"] and H["s_dl_cap"]:
    #         obj += link_cap_w * gp.quicksum(H["s_dl_cap"][c] for c in clusters)

    # # penalize GPU rho slack so violating rho has a cost
    # gpu_cap_w = float(snapshot.get(
    #     "gpu_cap_penalty_weight",
    #     OD.get("gpu_cap_penalty_weight", 0.0),
    # ))
    # obj += gpu_cap_w * gp.quicksum(H["s_rho"][g] for g in G)

    # if deadline_w > 0.0:
    #     obj += deadline_w * handles["s_dead"]

    # if freq_eps > 0.0:
    #     freq_pen = gp.quicksum(float(RATE[g][f]) * H["y"][g][f] for g in G for f in F[g])
    #     obj += freq_eps * freq_pen

    # freq_penalty_w = float(snapshot.get(
    #     "freq_penalty_weight",
    #     OD.get("freq_penalty_weight", 0.0),
    # ))

    # if freq_penalty_w > 0.0:
    #     obj += freq_penalty_w * gp.quicksum(
    #         (float(f) / max(1e-9, float(H["fmax_g"][g]))) * H["y"][g][f]
    #         for g in G for f in H["y"][g]
    #     )


    # m.setObjective(obj, GRB.MINIMIZE)

    # ------------------------------------------------------------------
    # knobs
    # ------------------------------------------------------------------
    OD = snapshot.get("Optimizer_Defaults", {}) or {}

    rho  = float(snapshot.get("pwl_rho_max", OD.get("pwl_rho_max", 0.8)))
    Kpts = int(snapshot.get("pwl_points",  OD.get("pwl_points", 2)))

    link_cap_w = float(snapshot.get(
        "link_cap_penalty_weight",
        OD.get("link_cap_penalty_weight", OD.get("link_penalty_weight", 0.0)),
    ))

    gpu_cap_w = float(snapshot.get(
        "gpu_cap_penalty_weight",
        OD.get("gpu_cap_penalty_weight", 0.0),
    ))

    # Additional penalty on rho violations (s_rho)
    rho_penalty = float(snapshot.get(
        "rho_penalty",
        OD.get("rho_penalty", 0.0),
    ))

    deadline_w = float(snapshot.get(
        "deadline_penalty_weight",
        OD.get("deadline_penalty_weight", 0.0),
    ))

    freq_eps = float(snapshot.get(
        "freq_tiebreak_eps",
        OD.get("freq_tiebreak_eps", 0.0),
    ))

    load_balance_w = float(snapshot.get(
        "load_balance_weight",
        OD.get("load_balance_weight", 0.0),
    ))

    rho_cap = float(snapshot.get("rho_cap", OD.get("rho_cap", 0.0)))

    bg_util_weight = float(snapshot.get("bg_util_weight", OD.get("bg_util_weight", 0.0),))


    # Background utilization (0..1) and bg load (FLOPs/s)
    util_guess = {}
    raw_util = snapshot.get("utilization_guess") or {}
    if isinstance(raw_util, dict):
        util_guess = {
            str(k): float(v) for k, v in raw_util.items()
            if v not in (None, "")
        }


    lambda_bg_raw = snapshot.get("lambda_bg", {}) or {}
    if hasattr(lambda_bg_raw, "items"):
        lambda_bg_norm = {
            str(k): float(v)
            for k, v in lambda_bg_raw.items()
            if v not in (None, "")
        }
    else:
        lambda_bg_norm = {}

    # ------------------------------------------------------------------
    # Latency-mode overrides: queue-aware + load-balanced
    # ------------------------------------------------------------------
    if str(objective).startswith("latency"):
        # don't clamp, just cap if OD forgot
        rho     = float(snapshot.get("pwl_rho_max", rho))
        rho_cap = float(snapshot.get("rho_cap", rho_cap))

        # let OD decide whether penalties exist
        if "link_cap_penalty_weight" in snapshot or "link_cap_penalty_weight" in OD:
            link_cap_w = float(snapshot.get("link_cap_penalty_weight",
                                OD.get("link_cap_penalty_weight", link_cap_w)))

        if "gpu_cap_penalty_weight" in snapshot or "gpu_cap_penalty_weight" in OD:
            gpu_cap_w = float(snapshot.get("gpu_cap_penalty_weight",
                                OD.get("gpu_cap_penalty_weight", gpu_cap_w)))

        bg_util_weight = float(snapshot.get("bg_util_weight",
                                OD.get("bg_util_weight", bg_util_weight)))

        load_balance_w = float(snapshot.get("load_balance_weight",
                                OD.get("load_balance_weight", load_balance_w)))
    # if str(objective).startswith("latency"):
    #     # keep queues comfortably below saturation
    #     rho     = min(rho,     0.80)
    #     rho_cap = min(rho_cap, 0.75)

    #     # ensure non-zero penalties for capacity violations
    #     if link_cap_w == 0.0:
    #         link_cap_w = OD.get("link_cap_penalty_weight",
    #                             OD.get("link_penalty_weight", 0.3))
    #     if gpu_cap_w == 0.0:
    #         gpu_cap_w = OD.get("gpu_cap_penalty_weight", 0.1)

    #     # keep bg_util_weight active (gives queue-awareness via util_guess)
    #     if "bg_util_weight" in snapshot:
    #         bg_util_weight = float(snapshot["bg_util_weight"])
    #     else:
    #         bg_util_weight = 1.0

    #     # stronger load-balancing in latency mode unless user overrides
    #     if load_balance_w == 0.0:
    #         load_balance_w = OD.get("load_balance_weight", 0.0) or 1.0

    # ------------------------------------------------------------------
    # Static model
    # ------------------------------------------------------------------
    snapshot = dict(snapshot)
    snapshot["use_downlink"] = any(_f(DLKB[tid], 0.0) > 0.0 for tid in T)

    _REBUILDS["solves"] += 1
    if _REBUILDS["solves"] % REBUILD_EVERY == 0:
        _clear_static_cache()

    m, H = _ensure_static(snapshot, G, F, RATE, R0_RC_Links, rho, Kpts)

    # Cache max frequency per GPU for tie-breaking / freq penalty
    H["fmax_g"] = {g: max(float(f) for f in H["y"][g]) for g in G}

    # Pin or seed frequencies
    if pinned_frequencies:
        for g in G:
            gid = _gid(g)
            if gid in pinned_frequencies:
                fpin = str(pinned_frequencies[gid])
                for f in F[g]:
                    if str(f) != fpin:
                        H["y"][g][f].ub = 0.0
    else:
        for g in G:
            if not F[g]:
                continue

            # F[g] is already sorted low → high
            freqs_sorted = list(F[g])

            if len(freqs_sorted) <= 2:
                # With only 1–2 bins, just warm-start at fastest
                f0 = max(freqs_sorted, key=lambda ff: float(RATE[g][ff]))
            else:
                # Use the middle bin as neutral warm-start
                mid_idx = len(freqs_sorted) // 2
                f0 = freqs_sorted[mid_idx]

            for f in F[g]:
                H["y"][g][f].start = 1.0 if f == f0 else 0.0


    # propagation delays for linearization
    clusters = H["clusters"]
    tau_sum = {
        c: _f(_link_for_cluster(R0_RC_Links, c).get("ul_prop_s"), 0.0)
         + (_f(_link_for_cluster(R0_RC_Links, c).get("dl_prop_s"), 0.0)
            if H["use_dl"] else 0.0)
        for c in clusters
    }
    TAIL0 = {g: _f(TAIL.get(g), 0.0) for g in G}

    # per-task block
    handles = _add_task_block(
        m, H, t0, G, RATE, FLOPS, ULKB, DLKB,
        lam_t0, TAIL0, tau_sum, rho_cap,
        util_guess, bg_util_weight,
        lambda_bg_norm,
    )

    # deadline (only if finite and positive)
    temp_rows = []
    Dt = float(_f(DEADL.get(t0), float("inf")) - now)
    if math.isfinite(Dt) and Dt > 1e-9:
        r = m.addConstr(handles["Dall"] <= Dt + handles["s_dead"],
                        name=f"C_deadline[{t0}]")
        temp_rows.append(r)
    handles["temp_rows"] = temp_rows

    # ------------------------------------------------------------------
    # Objective: end-to-end delay + queue/soft-cap penalties
    # ------------------------------------------------------------------
    obj = gp.LinExpr(0.0)

    # 1) end-to-end latency for this (first) task stream
    obj += handles["Dall"]

    # 2) load-balancing: penalize hottest GPU arrival rate
    if load_balance_w > 0.0 and H.get("Lambda_max") is not None:
        obj += load_balance_w * H["Lambda_max"]

    # 3) link capacity slacks (UL and optionally DL)
    if link_cap_w > 0.0:
        obj += link_cap_w * gp.quicksum(H["s_ul_cap"][c] for c in clusters)
        if H["use_dl"] and H["s_dl_cap"]:
            obj += link_cap_w * gp.quicksum(H["s_dl_cap"][c] for c in clusters)

    # # 4) GPU rho slacks – keep GPUs away from saturation
    # if gpu_cap_w > 0.0:
    #     obj += gpu_cap_w * gp.quicksum(H["s_rho"][g] for g in G)

    # 4) GPU rho slacks – keep GPUs away from saturation
    if gpu_cap_w > 0.0 or rho_penalty > 0.0:
        obj += (gpu_cap_w + rho_penalty) * gp.quicksum(H["s_rho"][g] for g in G)


    # 5) deadline slack (if a finite deadline exists)
    if deadline_w > 0.0:
        obj += deadline_w * handles["s_dead"]

    # 6) tiny preference for lower rates (tiebreak) if freq_eps > 0
    if freq_eps > 0.0:
        freq_pen = gp.quicksum(
            float(RATE[g][f]) * H["y"][g][f]
            for g in G for f in F[g]
        )
        obj += freq_eps * freq_pen

    # 7) optional explicit freq penalty (e.g., prefer lower normalized freq)
    freq_penalty_w = float(snapshot.get(
        "freq_penalty_weight",
        OD.get("freq_penalty_weight", 0.0),
    ))
    if freq_penalty_w > 0.0:
        obj += freq_penalty_w * gp.quicksum(
            (float(f) / max(1e-9, float(H["fmax_g"][g]))) * H["y"][g][f]
            for g in G for f in H["y"][g]
        )
    
    # 8) (optional) reward high freq when GPU is busy:
    #    subtract term so MINIMIZE prefers higher freq on high-Λ_g GPUs
    extra_lat_boost_w = float(snapshot.get(
        "extra_lat_boost_w",
        OD.get("extra_lat_boost_w", 0.0),
    ))

    if extra_lat_boost_w > 0.0:
        obj -= extra_lat_boost_w * gp.quicksum(
            H["Lambda_gpu"][g] * gp.quicksum(
                (float(f) / max(1e-9, float(H["fmax_g"][g]))) * H["y"][g][f]
                for f in F[g]
            )
            for g in G
        )

    
    m.setObjective(obj, GRB.MINIMIZE)


    # Optional: pin GPU for t0 by base TID
    if pinned_assignments:
        tb = _base_tid(t0)
        if tb in pinned_assignments:
            tgt = pinned_assignments[tb]
            gpin = (str(tgt["Cluster"]), str(tgt["Node"]), str(tgt["GPU"]))
            for g in G:
                if g == gpin: handles["x"][g].lb = 1.0
                else:         handles["x"][g].ub = 0.0

    # Solve
    m.optimize()
    # _dump_model(m, "minlatency_model")
    # for g in G:
    #     ys = {f: H["y"][g][f].X for f in F[g]}
    #     print(_gid(g), ys)


    if m.SolCount == 0:
        _remove_task_block(m, handles)
        raise RuntimeError("MILP returned no incumbent.")

    # Read result
    g_star = max(G, key=lambda gg: handles["x"][gg].X)
    c, n, gg = g_star
    assign = {
        t0: {"Cluster": c, "Node": n, "GPU": gg},
        _base_tid(t0): {"Cluster": c, "Node": n, "GPU": gg}
    }

    freqs = {}
    for g in G:
        if handles["x"][g].X > 0.5:                 # GPU actually chosen for this task
            chosen = next((str(f) for f in F[g] if H["y"][g][f].X > 0.5), None)
            if chosen is None and F[g]:
                chosen = max(F[g], key=lambda f: float(RATE[g][f]))
            freqs[_gid(g)] = chosen

    out = {
        "assignments": assign,
        "frequencies": freqs,
        "status": "optimal" if m.Status == GRB.OPTIMAL else f"status_{m.Status}",
        "objective": float(m.ObjVal),
        "use_rate_form": True,
        "per_job_delays": {t0: {"Doverall": float(handles["Dall"].X)}},
    }
    # _dump_json(out, "minlatency_solution_assignments")

    # Clean up the task block so the static model stays constant-size
    _remove_task_block(m, handles)

    return out


# ----------------- FastPolicy entrypoints -----------------

def optimize_latency(snapshot, new_task, catalog=None, store=None):
    """
    FastPolicy-compatible latency entrypoint.
    Delegates to the existing incremental MILP solver.
    """
    return solve_incremental(snapshot, new_task)

def optimize(snapshot, new_task, objective="latency", catalog=None, store=None):
    """
    Generic FastPolicy fallback entrypoint.
    """
    if objective == "latency":
        return solve_incremental(snapshot, new_task)
    else:
        raise RuntimeError(f"Objective '{objective}' not supported in this optimizer.")


def solve_incremental(snapshot: Dict[str, Any],
                      new_task: List[Dict[str, Any]],
                      objective: str = "latency",
                      pinned_assignments: Optional[Dict[str, Dict[str, str]]] = None,
                      pinned_frequencies: Optional[Dict[str, str]] = None
                      ) -> Dict[str, Any]:
    # IMPORTANT: pass objective through to the core solver
    return _solve_latency(
        snapshot,
        new_task,
        objective=objective,
        pinned_assignments=pinned_assignments,
        pinned_frequencies=pinned_frequencies,
    )


def solve(snapshot: Dict[str, Any],
          new_task: List[Dict[str, Any]],
          objective: str = "latency") -> Dict[str, Any]:
    # Thin wrapper; already passes objective correctly
    return solve_incremental(snapshot, new_task, objective=objective)


# legacy alias
def solve_latency(snapshot: Dict[str, Any], new_task: List[Dict[str, Any]]):
    return solve_incremental(snapshot, new_task, objective="latency")
