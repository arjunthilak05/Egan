# Experiment Guide

This document explains how to run all experiments for the Entanglement-Regularized QGAN paper.

---

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run Simple Test

```bash
python run_experiments.py --experiment toy --epochs 50
```

This runs a quick toy distribution experiment with 50 epochs (should take ~5-10 minutes).

---

## Main Experiments

### Experiment 1: Toy Distribution (Mode Coverage)

**Goal**: Demonstrate that entropy regularization improves mode coverage.

**Command**:
```bash
python run_experiments.py --experiment toy_comparison --epochs 200 --n-modes 8
```

**What it does**:
- Tests 5 different models:
  - Classical GAN (baseline)
  - Standard QGAN (no entropy regularization)
  - Entropy QGAN with α=0.5
  - Entropy QGAN with α=1.0
  - Entropy QGAN with α=2.0
- Generates 2D Gaussian mixture with 8 modes
- Measures mode coverage for each model
- Creates comparison plots

**Expected output**:
- `results/comparison/mode_coverage_comparison.png` - Bar chart comparing coverage
- Individual result folders for each model

**Expected results**:
- Classical GAN: 50-60% coverage
- Standard QGAN: 60-70% coverage
- Entropy QGAN (α=1.0): 85-95% coverage ✅

**Time**: ~30-40 minutes (200 epochs × 5 models)

---

### Experiment 2: Ablation Study (Effect of α)

**Goal**: Show how different entropy weights affect performance.

**Command**:
```bash
python run_experiments.py --experiment ablation --epochs 200
```

**What it does**:
- Tests α ∈ {0.0, 0.5, 1.0, 2.0, 5.0}
- Measures mode coverage, diversity, and entropy for each
- Creates ablation plots

**Expected output**:
- Summary table in terminal
- Individual result folders

**Expected pattern**:
- α=0: Low coverage (~60%)
- α=1: High coverage (~90%)
- α=5: Slightly lower quality but still high coverage

**Time**: ~30-40 minutes

---

### Experiment 3: Single Model Test

**Goal**: Test a specific configuration.

**Commands**:

Test entropy-regularized QGAN:
```bash
python run_experiments.py --experiment toy --model entropy_qgan --alpha 1.0 --epochs 200
```

Test standard QGAN (baseline):
```bash
python run_experiments.py --experiment toy --model standard_qgan --epochs 200
```

Test classical GAN (baseline):
```bash
python run_experiments.py --experiment toy --model classical_gan --epochs 200
```

**Time**: ~6-8 minutes per model

---

### Experiment 4: Run Everything

**Goal**: Generate all results for the paper.

**Command**:
```bash
python run_experiments.py --experiment all --epochs 200
```

**Warning**: This will take 1-2 hours!

---

## Understanding the Results

### Directory Structure

After running experiments, you'll find:

```
results/
├── comparison/
│   └── mode_coverage_comparison.png  # Main comparison figure
├── toy_entropy_qgan_modes8/
│   ├── training_curves.png           # Loss & entropy over time
│   ├── distribution.png              # Generated vs real samples
│   └── entropy_dist.png              # Entropy distribution
├── toy_standard_qgan_modes8/
│   └── ...
└── toy_classical_gan_modes8/
    └── ...
```

### Key Metrics

**Mode Coverage**: Percentage of true modes covered by generated samples
- **Goal**: Maximize (100% = perfect)
- **Interpretation**: Higher = better diversity, less mode collapse

**Sample Diversity**: Average distance to k-nearest neighbors
- **Goal**: Maximize
- **Interpretation**: Higher = samples more spread out

**Entanglement Entropy**: von Neumann entropy S(ρ_A)
- **Goal**: Maximize (up to log(d) where d=2^{n_A})
- **Interpretation**: Higher = more quantum correlations

---

## Paper Figures

### Figure 1: Mode Coverage Comparison
**File**: `results/comparison/mode_coverage_comparison.png`

**Shows**: Bar chart of mode coverage for different methods

**Expected**: Entropy QGAN clearly outperforms baselines

---

### Figure 2: Training Curves
**File**: `results/toy_entropy_qgan_modes8/training_curves.png`

**Shows**:
- Generator/Discriminator loss over time
- Entanglement entropy evolution
- Alpha (regularization weight) schedule

**Expected**: Entropy increases during training, stabilizing at high value

---

### Figure 3: Generated Distributions
**File**: `results/toy_entropy_qgan_modes8/distribution.png`

**Shows**: Side-by-side comparison of real vs generated 2D distributions

**Expected**: Generated samples cover all 8 modes

---

### Figure 4: Ablation Study
**Run ablation experiment**, then manually create plot from results

**Shows**: Mode coverage vs. alpha

**Expected**: Coverage peaks around α=1.0, then plateaus

---

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'pennylane'"

**Solution**:
```bash
pip install pennylane pennylane-lightning
```

### Issue: "RuntimeError: Attempting to deserialize object on a CUDA device"

**Solution**: Use `--device cpu` if you don't have CUDA:
```bash
python run_experiments.py --experiment toy --device cpu
```

### Issue: Experiments are too slow

**Solutions**:
1. Reduce epochs: `--epochs 50`
2. Reduce qubit count: Edit `src/utils/config.py`, set `n_qubits = 6`
3. Use smaller circuit: Edit config, set `n_layers = 1`

### Issue: Mode coverage is low even with entropy regularization

**Possible causes**:
1. Alpha too small: Try `--alpha 2.0`
2. Not enough training: Try `--epochs 300`
3. Learning rate too high: Edit config, set `lr_g = 0.0005`

---

## Advanced Usage

### Custom Configuration

Edit `src/utils/config.py` to customize:
- Quantum circuit architecture
- Network dimensions
- Training hyperparameters

### Adding New Experiments

Create a new file in `experiments/`:

```python
import sys
sys.path.append('..')

from src.models import create_qgan_model
# ... your experiment code
```

### Visualizing Custom Metrics

```python
from src.evaluation import plot_entropy_distribution

# Your code to compute entropies
plot_entropy_distribution(entropies, save_path="my_plot.png")
```

---

## Expected Runtime (on CPU)

| Experiment | Epochs | Time |
|-----------|--------|------|
| Single toy | 50 | 5-10 min |
| Single toy | 200 | 20-30 min |
| Comparison | 200 | 1-1.5 hours |
| Ablation | 200 | 1-1.5 hours |
| All | 200 | 2-3 hours |

**Note**: With GPU, these times can be 2-3x faster for large circuits.

---

## Citation

If you use this code, please cite:

```bibtex
@article{entropy_qgan_2025,
  title={Entanglement Entropy Regularization for Diversity-Enforced Quantum Generative Adversarial Networks},
  author={},
  journal={arXiv preprint},
  year={2025}
}
```

---

## Questions?

See README.md for general project info.

For issues, check GitHub issues or contact the authors.
