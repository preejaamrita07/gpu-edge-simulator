# GPU Edge Simulator

A high-fidelity, DVFS-aware, network-aware GPU scheduling simulator for edge clusters.

## Features

- Multi-cluster GPU hierarchy  
- Uplink/downlink queuing (R0, Rc routers)  
- GPU DVFS with per-frequency power/energy logging  
- Optimizer-based scheduling (min-power, min-latency, max-efficiency)  
- Least-load, random, round-robin heuristics  
- SimPy-driven packet/task/job pipeline  
- Config-driven simulation (`config.json`)  
- Automatic CSV output for:
  - packet summary
  - DVFS energy windows
  - per-frequency stats
  - transition matrix

## Structure

See `src/` for the main simulation code.

## Running the Simulator

```bash
python src/main.py --config configs/config.json --strategy opt_power_adaptive
