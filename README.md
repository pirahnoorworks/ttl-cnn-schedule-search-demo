# TTL CNN Schedule Search Demo

Standalone portfolio demo from the code dump that showcases **tensor template language CNN workloads** and compares multiple search techniques for schedule selection.

This project is inspired by the dump's tensor expression CNN flow (`Conv2D`) and runtime schedule-selection logic in pipeline stages.

## What this demo shows

- Tensor template language style data model (`Tensor<4, float>`) for CNN tensors.
- CNN convolution workload execution with tunable schedule parameters.
- Side-by-side comparison of schedule search techniques:
  - Grid search
  - Random search
  - Hill-climbing local search
- Quantitative output:
  - best latency per technique
  - evaluated candidate count
  - speedup vs naive baseline schedule

## Schedule Parameters

Each candidate schedule searches over:

- `tileH`, `tileW` : output tile size
- `channelBlock` : channel blocking factor
- `numThreads` : worker threads used by the evaluator
- `kernelOuter` : loop-order strategy (`kernel-outer` vs `channel-outer`)

## Workloads

- `small` : quick sanity run
- `resnet-stage` : typical mid-size stage
- `vgg-stage` : larger workload

## Build and Run

### Ubuntu / Linux

```bash
cd code/portfolio/ttl-cnn-schedule-search-demo
bash scripts/run_demo.sh
```

### Windows PowerShell

```powershell
cd code/portfolio/ttl-cnn-schedule-search-demo
./scripts/run_demo.ps1
```

## CLI Options

```text
ttl_schedule_search_demo [options]
  --workload <name>       small|resnet-stage|vgg-stage (default small)
  --warmup <int>          Warmup iterations per candidate (default 1)
  --iters <int>           Timed iterations per candidate (default 2)
  --random-budget <int>   Number of random candidates (default 40)
  --hill-steps <int>      Max hill-climb improvement rounds (default 6)
  --max-threads <int>     Max thread count in candidate pool (default 8)
  --grid-max-evals <int>  Max candidates for grid search (default 150)
```

## Example Runs

Faster exploratory run:

```bash
bash scripts/run_demo.sh --workload small --iters 1 --grid-max-evals 60
```

More realistic stage search:

```bash
bash scripts/run_demo.sh --workload resnet-stage --warmup 1 --iters 2 --random-budget 60 --hill-steps 8 --grid-max-evals 180
```

