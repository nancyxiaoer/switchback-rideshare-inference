# Finite-Sample Randomization Inference for Switchback Rideshare Experiments

This project studies how the switchback interval affects causal-effect estimation
and randomization-based inference in a dynamic ridesharing environment with
carryover.

> **Research question:** When treatment changes can persist through the platform
> state, how should the switchback interval be chosen to balance carryover bias,
> estimator variance, statistical power, and the need to update policies frequently?

## Main findings

The final experiment contains 100 independent simulator trajectories for each
switchback interval, with 500,000 events per trajectory and 2,000 Monte Carlo
resamples per randomization test.

| Switchback interval \(L\) | Naive bias | Naive RMSE | Pure-history share | Total-effect CRT rejection rate |
| ---: | ---: | ---: | ---: | ---: |
| 1,000 | -0.473 | 0.475 | 3.2% | 0% (0/95 valid tests) |
| 5,000 | -0.032 | 0.107 | 50.0% | 43% |
| 10,000 | 0.027 | 0.114 | 75.2% | 62% |

The long-run all-treatment versus all-control effect is **1.2784**.

- Frequent switching at \(L=1000\) produces severe downward bias because outcomes
  remain contaminated by earlier assignments.
- Increasing \(L\) to 5,000 or 10,000 nearly eliminates the naive estimator's bias.
- Pure-history Horvitz--Thompson estimation reduces exposure contamination but has
  substantially larger finite-sample variance.
- The total-effect CRT has almost no usable randomized sections at \(L=1000\), but
  its empirical power rises to 43% and 62% at \(L=5000\) and \(L=10000\).
- Across all 300 anticipation tests, the rejection rate is exactly 5%, consistent
  with nominal size in a simulator without anticipatory behavior.

![Estimator performance across switchback intervals](results/final/estimation_performance.png)

![Randomization-inference results](results/final/randomization_inference.png)

## Experimental design

| Component | Specification |
| --- | --- |
| Simulator | XP-Gym rideshare pricing environment |
| Events per trajectory | 500,000 |
| Switchback intervals | 1,000; 5,000; 10,000 events |
| Independent replications | 100 seeds per interval (1000--1099) |
| Treatment probability | 0.5 at the block level |
| Assumed carryover memory | 5,000 events |
| Carryover horizons tested | 0; 1,000; 2,500; 5,000; 10,000 events |
| Randomization resamples | 2,000 per test |
| Primary outcomes | Bias, RMSE, rejection rate, usable randomized units |

The same seed index is used across the three interval settings to support paired
comparisons. Randomization seeds inside the tests are deterministically derived
from the experiment specification.

## Statistical methods

### Effect estimation

- **Naive difference in means:** compares average rewards under treatment and
  control without adjusting for history.
- **Pure-history Horvitz--Thompson estimator:** uses events whose most recent
  \(m+1\) assignments are entirely treatment or entirely control and weights them
  by their exact switchback exposure probabilities.

### Randomization tests

- **Total-effect CRT:** tests a partially sharp null of no positive total policy
  effect after conditioning on constant-assignment sections.
- **Carryover CRT:** tests whether treatment carryover lasts beyond a candidate
  horizon \(m\).
- **Anticipation PIRT:** checks whether present outcomes appear to depend on future
  assignments; in this application it functions as a design-validity check.

Rejection of a carryover null provides evidence against that candidate horizon.
Failure to reject does not prove that carryover is absent. The horizon grid is
therefore treated as a diagnostic profile rather than a point estimator.

## Engineering contributions

- Replaced the event-level Python rollout with `jax.lax.scan` and verified it
  against the reference Python loop.
- Built a deterministic, resumable pipeline for 300 trajectories and 2,100
  randomization tests.
- Reduced pure-history detection from \(O(Tm)\) to \(O(T)\) using a sliding-window
  update.
- Computed exact pure-history exposure probabilities under equal-length
  switchback blocks.
- Added synthetic null/alternative calibration, unit tests, unavailable-test
  reporting, Wilson intervals, and reproducible random seeds.

## Repository structure

```text
.
|-- configs/                 # Prespecified experiment settings
|-- data/generated/          # Local NPZ trajectories (not committed)
|-- results/
|   |-- final/               # R=100 tables, public manifest, and figures
|   |-- validation/          # Synthetic size/power calibration
|   `-- benchmarks/          # O(Tm) versus O(T) runtime benchmark
|-- scripts/                 # Generation, validation, inference, and benchmarks
|-- src/switchback_lab/      # Reusable inference and rollout implementation
|-- tests/                   # Mathematical and implementation checks
|-- pyproject.toml
`-- README.md
```

## Reproduce the core analysis

Python 3.10 or 3.11 is recommended.

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[dev]"
pytest -q
```

Run the controlled size/power validation:

```bash
python scripts/run_synthetic_validation.py --replications 200 --resamples 2000
```

Run the algorithm benchmark:

```bash
python scripts/benchmark_pure_history.py
```

## Reproduce the XP-Gym experiment

The simulator is an optional upstream dependency and is not vendored here.
Install [XP-Gym](https://github.com/atzheng/xp_gym) in the same environment,
then validate the compiled rollout before generating full trajectories.

```bash
python scripts/check_xp_gym_rollout.py --n-steps 1000 --switch-every 1000 --seed 2026 --compare-python-loop
python scripts/check_xp_gym_rollout.py --n-steps 500000 --switch-every 1000 --seed 2026
```

Generate the complete experiment. The command is resumable and skips existing
trajectory files.

```bash
python scripts/generate_xp_gym_multiseed.py --n-steps 500000 --block-lengths 1000 5000 10000 --seed-start 1000 --seed-count 100 --memory 5000 --output-dir data/generated
```

Run the prespecified inference analysis:

```bash
python scripts/run_multiseed_inference.py --input-dir data/generated --output-dir results/final --resamples 2000 --total-memory 5000 --carryover-memories 0 1000 2500 5000 10000 --true-ate 1.278396906
```

Full simulator generation took approximately 11.4 CPU hours on the reference
Windows machine. Once trajectories were available, all 2,100 randomization tests
completed in under one minute.

## Results and data availability

The repository includes:

- all final summary tables and figures;
- all 2,100 row-level randomization-test results;
- a sanitized trajectory manifest without local machine paths;
- the final synthetic calibration and runtime benchmark.

The 500,000-event NPZ trajectories are omitted because they are reproducible and
unnecessary for reviewing the analysis. They can be regenerated using the command
above.

## Limitations

- Results describe one simulated rideshare environment and parameter setting;
  they are not causal claims about a real platform.
- Only three switchback intervals are evaluated.
- A carryover rejection profile does not identify an exact structural memory
  length.
- Total-effect CRT availability depends on realized constant-assignment sections;
  five \(L=1000\) trajectories had no usable section and are reported as
  unavailable rather than discarded.
- The carryover horizon grid is exploratory and is not adjusted for family-wise
  multiple testing.

## Attribution

The ridesharing simulator is adapted from the MIT-licensed
[XP-Gym](https://github.com/atzheng/xp_gym) project associated with:

- Farias, Li, Peng, and Zheng, *Markovian Interference in Experiments*,
  [arXiv:2206.02371](https://arxiv.org/abs/2206.02371).

The pairwise randomization-test ideas are based on:

- Zhong, *Unconditional Randomization Tests for Interference*,
  [arXiv:2409.09243](https://arxiv.org/abs/2409.09243).

Third-party source files and paper PDFs are intentionally not redistributed in
this repository.
