# power_shared.py 

from typing import Dict, Any, Tuple

def _gid(g):
    # turn (C,N,G) -> "C-N-G"
    return f"{g[0]}-{g[1]}-{g[2]}"

def build_power_params(snapshot: Dict[str, Any],
                       G: Tuple[Tuple[str, str, str], ...],
                       F,
                       RATE) -> Tuple[Dict, Dict, Dict, Dict]:
    """
    Returns:
      P_static_W[g]
      phi_power[g]
      exp_const[g]     # FORCED to 1.0 for this min-power formulation
      freq_value[f]    # numeric MHz for each frequency key
    """

    # Where meta/specs live in snapshots
    meta = (snapshot.get("GPU_Specs_Meta")
            or snapshot.get("meta")
            or {}) or {}

    # ---- uses exponent 1 (linear), not cube ----
    FORCE_LINEAR_POWER_EXP = True
    forced_exp = 2.0

    # FastPolicy snapshots usually carry GPU specs under snapshot["gpus"]
    gpuinfo = snapshot.get("gpus") or snapshot.get("GPU_Specs") or {}

    def _p_static_of(info: Dict[str, Any]) -> float:
        for k in ("P_static_W", "P_static", "P_idle_W",
                  "p_static_w", "p_idle_w", "P_st", "P_st_W"):
            if k in info and info[k] not in (None, ""):
                try:
                    v = float(info[k])
                    if v > 0.0:
                        return v
                except Exception:
                    pass
        return 60.0

    # ---- Build freq_value ONCE (do not overwrite per GPU) ----
    freq_value: Dict[str, float] = {}
    for g in G:
        for f in (F.get(g) or []):
            fs = str(f)
            if fs not in freq_value:
                try:
                    freq_value[fs] = float(fs)
                except Exception:
                    pass

    P_static_W: Dict[Any, float] = {}
    exp_const:  Dict[Any, float] = {}
    phi_power:  Dict[Any, float] = {}

    for g in G:
        gid = _gid(g)

        spec = gpuinfo.get(g, None)
        if spec is None:
            spec = gpuinfo.get(gid, {}) or {}

        Pst = _p_static_of(spec)

        # Prefer explicit P_max_W if provided; else fallback to Pst (no dynamic range)
        try:
            Pmax = float(spec.get("P_max_W", spec.get("P_max", Pst)))
        except Exception:
            Pmax = Pst

        # ---- EXPONENT: force linear if requested ----
        if FORCE_LINEAR_POWER_EXP:
            e = forced_exp
        else:
            pexp_def = float(meta.get("power_exp", 1.0))
            try:
                e = float(spec.get("power_exp", pexp_def))
            except Exception:
                e = pexp_def

        P_static_W[g] = float(Pst)
        exp_const[g]  = float(e)

        # numeric fmax for this GPU
        fnums = []
        for f in (F.get(g) or []):
            fs = str(f)
            if fs in freq_value:
                fnums.append(freq_value[fs])
        fmax = max(fnums) if fnums else 1.0

        # ---- phi calibration consistent with exponent ----
        # Pdyn_max = Pmax - Pst = phi * fmax^e  => phi = (Pmax - Pst)/fmax^e
        denom = max(fmax, 1e-9) ** float(e)
        phi = (float(Pmax) - float(Pst)) / max(denom, 1e-12)

        # keep strictly positive to avoid numerical issues in PWL
        phi_power[g] = max(1e-12, float(phi))

    return P_static_W, phi_power, exp_const, freq_value
