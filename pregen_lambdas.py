import sys
import subprocess
from pathlib import Path

# Which lambdas / seeds to pre-generate
LAMBDAS = [4.0, 5.0, 6.0, 7.0, 8.0]
SEEDS   = [41, 42, 43, 44, 45]


# Root folder = directory containing this script
ROOT = Path(__file__).resolve().parent

# Output folder: ROOT/runs
OUT_DIR = ROOT / "runs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

python_bin = sys.executable  # use current venv python
sim_script = ROOT / "queuing_system_simulation_incremental.py"

for L in LAMBDAS:
    for run_idx, seed in enumerate(SEEDS, start=1):
        prefix = f"exp1_L{L}__run{run_idx}"
        pregen_path = OUT_DIR / f"{prefix}_pregen.csv"

        cmd = [
            python_bin,
            str(sim_script),
            "--config", str(ROOT / "config.json"),
            "--lambda", str(L),
            "--seed", str(seed),
            "--strategy", "least-load_adaptive",  # ignored in pregen mode
            "--pregen-out", str(pregen_path),
        ]

        print(">", " ".join(cmd))
        subprocess.run(cmd, check=True, cwd=ROOT)


# python3 pregen_lambdas.py
# source venv/bin/activate
