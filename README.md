
# GPU-Aware Edge Computing Simulator

This repository contains a **GPU-aware edge computing simulator** for studying
latency, power, and efficiency trade-offs using **incremental (online) optimizers**.

The simulator models:
- Packetized uplink and downlink communication
  - Jobs → tasks → packets
  - Uplink/downlink queues (R0, Rc)
- GPU task aggregation and DVFS-based processing
  - Power/energy accounting (static + dynamic)
- Multi-cluster, multi-node, multi-GPU topologies
- - Multiple scheduling strategies:
  - Optimizer-based (min-power / min-latency / max-efficiency)
  - Heuristic (least-load)
- Online optimization under dynamic load

---

## Key Features

- Incremental (online) optimization
- Min-latency, min-power, and max-efficiency objectives
- DVFS-aware GPU service rates and power models
- Full queuing pipeline: R0 → GPU → Rc
- Comparable multi-strategy experiments

---
# Architecture and Queuing Model

The system follows a fixed processing pipeline:

Uplink Phase:
R0 ingress → R0 egress → propagation → GPU aggregation

Processing Phase:
GPU queue → DVFS-based service → task completion

Downlink Phase:
Rc ingress →  propagation → R0 ingress

Key properties:
- All packets of a task are aggregated before GPU service
- Tasks are non-preemptive once GPU service begins
- Queues are explicitly modeled with arrival, waiting, and service delays

File:
  - queuing_system_simulation_incremental.py

    Core simulator engine.

    Responsibilities:

    Job, task, and packet lifecycle management

    Uplink and downlink queuing (R0 and Rc routers)

    GPU task aggregation and processing

    DVFS-aware service and power tracking

    Deadline checks and job drops

    Runtime logging and metrics collection

    This file orchestrates the entire event-driven simulation.

  File:
    - fast_decider.py

      Lightweight runtime decision layer used by incremental optimizers.
      This module is designed to keep online decisions fast and stable.

-----

# Incremental Optimizers

All optimizers operate **online**, making decisions per task arrival.

## Min-Latency Optimizer
- Objective: minimize overall delay

File:
- live_min_latency_optimizer_incremental.py

## Min-Power Optimizer
- Objective: minimize power consumption

File:
- live_min_power_optimizer_incremental.py
- power_shared.py (GPU power model)

## Max-Efficiency Optimizer
- Objective: maximize performance per watt
- Balances throughput and power

File:
- live_max_efficiency_optimizer_incremental.py

--------

# Full strategy set

least-load-fixed
least-load-adaptive

opt_latency_fixed
opt_power_fixed
opt_efficiency_fixed

opt_latency_adaptive
opt_power_adaptive
opt_efficiency_adaptive

All strategies share identical workload traces, network models, and queuing dynamics, ensuring fair and controlled comparison across heuristic, fixed, and adaptive policies.
---------

# Configuration Guide

All system parameters are defined in `config.json`.

## Key Sections

- Cluster and GPU topology
- GPU architectures and DVFS service rates
- Network links (R0 ↔ Rc)
- Job, task, and packet parameters
- Optimizer defaults and constraints

-------
## Output Artifacts:

task_packets_summary_*.csv – detailed packet/task logs

dvfs_energy_windows.csv – dvfs log

Plots for latency, power, and violation comparisons

-------

##  Repository Structure
.
├── config.json
├── queuing_system_simulation_incremental.py
├── fast_decider.py
├── live_max_efficiency_optimizer_incremental.py
├── live_min_latency_optimizer_incremental.py
├── live_min_power_optimizer_incremental.py
├── power_shared.py
├── pregen_lambdas.py
├── run_all_strategies.py
├── run_all_strategies.sh
├── plot_from_task_packets.py

------

# 1. Running Pre-generated logs for simulation

File:
- pregen_lambdas.py 

  Generates arrival rate (λ) profiles for experiments.

```bash
python pregen_lambdas.py

--------

# 2. Run all strategies script

File: 
- run_all_strategies.sh

```bash
chmod +x run_all_strategies.sh
python run_all_strategies.sh

------
## Run All Strategies (.py file)

File:
-  run_all_strategies.py

  Experiment driver script.

  Runs the simulator across:

  -  Multiple objectives (latency, power, efficiency)

  -  Multiple λ values

  -  Multiple seeds

```bash
python run_all_strategies.py

or

python queuing_system_simulation_incremental.py \
  --config config.json \
  --multi "L:4.0, 5.0, 6.0, 7.0, 8.0;runs:5;seed0:41;outdir:runs;prefix:exp1" \
  --enable-adaptive-freq \
  --fast-decider \
  --opt-module-latency    live_min_latency_optimizer_incremental \
  --opt-module-power      live_min_power_optimizer_incremental \
  --opt-module-efficiency live_max_efficiency_optimizer_incremental \
  --admission soft

This executes:

 - Min-latency optimization

 - Min-power optimization

 - Max-efficiency optimization
------

# 3. Plotting

```bash
python plot_from_task_packets.py

