# Chain of Thought Experiments

Implementation of Wang, Xuezhi, et al. "Self-consistency improves chain of thought reasoning in language models." arXiv preprint arXiv:2203.11171 (2022).

## Features

- Support for multiple language models (OPT, BLOOM, GPT-2, T5)
- Chain of Thought (CoT) reasoning implementation
- **Self-consistency evaluation**

## Usage

Basic usage:

```python
from experiments.src.experiment import ChainOfThoughtExperiment

# Initialize experiment
experiment = ChainOfThoughtExperiment(model_name="facebook/opt-1.3b")

# Run evaluation
results = experiment.evaluate_question(
    question="Your math question here",
    correct_answer="42",
    num_samples=5
)
```

Run full experiment:

```bash
python -m experiments.main
```

## Configuration

Modify in `main.py`:
- your Model
- Number of samples
- Temperature
- Problem sets or benchmarks (gsm8, reclor...)

## Testing

Some tests:

```bash
python -m pytest tests/
```