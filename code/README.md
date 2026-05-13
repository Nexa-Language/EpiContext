# EpiContext Code

This directory contains the implementation of EpiContext framework.

## Structure

- `epicontext/`: Core framework implementation
  - `core.py`: Core classes (ContextGraph, EpigeneticOperators, FitnessFunction, EpiContextRouter)
  - `agent.py`: Agent implementation
  - `benchmarks/`: Benchmark datasets and evaluation
  - `utils/`: Utility functions
- `experiments/`: Experiment scripts
- `results/`: Experiment results

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```python
from epicontext.core import EpiContextRouter

router = EpiContextRouter()
payload = router.process_turn(...)
```

## Experiments

Run experiments:
```bash
python experiments/run_main.py
```
