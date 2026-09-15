# Reconstruction Tutorial

This tutorial will guide you through using LORAKS reconstruction in PyMRITools with both AC-LORAKS and P-LORAKS methods.

## Prerequisites

Before starting, make sure you have:

- PyMRITools installed
- A k-space dataset (NIfTI or PyTorch tensor)
- Basic understanding of MRI reconstruction concepts

- CUDA-capable GPU (optional but recommended for speed)

## Quick Start: AC-LORAKS

### Step 1: Prepare Your Data

```python
import torch
import numpy as np

# Load your undersampled k-space data
# Shape should be: [batch, num_coils, n_phase, n_read]
k_space = torch.load("undersampled_k_space.pt")

print(f"Input shape: {k_space.shape}")
# Expected: [batch_size, num_coils, n_phase, n_read]
```

### Step 2: Configure AC-LORAKS

```python
from pymritools.recon.loraks.loraks import Loraks, LoraksOptions
from pymritools.recon.loraks.ac_loraks import AcLoraksOptions
from pymritools.recon.loraks.loraks import (
    OperatorType, RankReduction, RankReductionMethod
)

# Create AC-LORAKS configuration
options = AcLoraksOptions(
    loraks_neighborhood_size=5,      # 5×5 neighborhood
    loraks_matrix_type=OperatorType.S,   # Use S-matrix
    rank=RankReduction(
        method=RankReductionMethod.HARD_CUTOFF,
        value=120                        # Keep 120 singular values
    ),
    regularization_lambda=0.0,          # Strict data consistency
    max_num_iter=20,                   # 20 iterations
    solver_type=SolverType.LEASTSQUARES,   # Fast CGD solver
    nullspace_algorithm=NullspaceAlgorithm.EIGH  # Eigendecomposition
)
```

### Step 3: Run Reconstruction

```python
# Create AC-LORAKS instance
loraks = Loraks.create(options)

# Run reconstruction
k_recon = loraks.reconstruct(k_space)

print(f"Output shape: {k_recon.shape}")
# Same shape as input: [batch_size, num_coils, n_phase, n_read]
```

### Step 4: Analyze Results

```python
import matplotlib.pyplot as plt

# Convert to image domain
from pymritools.utils import ifft_to_k

# Take first batch and combine coils
img_recon = ifft_to_k(k_recon[0])  # Shape: [n_phase, n_read]
img_combined = torch.sqrt(torch.sum(torch.abs(img_recon)**2, dim=0))

# Display
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(img_combined.cpu().numpy(), cmap='gray', origin='lower')
plt.title('AC-LORAKS Reconstruction')
plt.colorbar()

# Compare with undersampled data (if available)
img_undersampled = ifft_to_k(k_space[0])
img_combined_us = torch.sqrt(torch.sum(torch.abs(img_undersampled)**2, dim=0))

plt.subplot(1, 2, 2)
plt.imshow(img_combined_us.cpu().numpy(), cmap='gray', origin='lower')
plt.title('Undersampled (GRAPPA 2x)')
plt.colorbar()

plt.tight_layout()
plt.show()
```

## Advanced: P-LORAKS

### When to Use P-LORAKS

- Highly undersampled data (R > 8)
- Limited or no auto-calibration region
- Complex undersampling patterns

### Step 1: Configure P-LORAKS

```python
from pymritools.recon.loraks.p_loraks import PLoraks

# Create P-LORAKS with builder pattern
p_loraks = (PLoraks()
    .with_patch_shape((-1, -1, -1, 7, 7))    # 7×7 phase/read neighborhood
    .with_s_matrix()                         # Use S-matrix
    .with_sv_auto_soft_cutoff(percentage=0.85)  # Automatic rank selection
    .with_regularization_lambda(0.5)         # Penalized data consistency
    .with_max_iterations(60)                  # More iterations
    .with_constant_learning_rate(1e-3)        # Fixed learning rate
    .with_device('cuda:0'))                  # GPU acceleration
)

# Or using configuration class
p_options = PLoraksOptions(
    patch_shape=(0, 0, 0, 7, 7),
    lowrank_algorithm=LowRankAlgorithmType.TORCH_LOWRANK_SVD
)

p_loraks.configure(p_options)
```

### Step 2: Run Reconstruction

```python
# Reconstruct using P-LORAKS
k_recon_p = p_loraks.reconstruct(k_space)

print(f"P-LORAKS output shape: {k_recon_p.shape}")
```

## Parameter Tuning Guide

### Neighborhood Size

**Rule of thumb:**
- Small (3-5): Good for smooth images, fast computation
- Medium (5-7): Good balance, most cases
- Large (7-11): Better for complex textures, slower

```python
# Try different neighborhood sizes
for nb_size in [3, 5, 7, 9]:
    options = AcLoraksOptions(loraks_neighborhood_size=nb_size)
    loraks = Loraks.create(options)
    result = loraks.reconstruct(k_space)
    # Save and compare results
```

### Rank Selection

**Methods:**
```python
# Hard cutoff (fast)
rank1 = RankReduction(method=HARD_CUTOFF, value=150)

# ReLU shift (soft)
rank2 = RankReduction(method=RELU_SHIFT, value=0.1)

# Automatic (adaptive)
rank3 = RankReduction(method=RELU_SHIFT_AUTOMATIC, value=0.9)
```

### Data Consistency Weighting

```python
# Strict data consistency ( fastest convergence)
options = AcLoraksOptions(regularization_lambda=0.0)

# Penalized approach ( better for noisy data)
options = AcLoraksOptions(regularization_lambda=0.5)

# Heavy data consistency (for very noisy data)
options = AcLoraksOptions(regularization_lambda=2.0)
```

## Common Workflows

### Scenario 1: Standard Parallel Imaging (2x GRAPPA)

```python
# 2x acceleration with 24-line ACS region
options = AcLoraksOptions(
    loraks_neighborhood_size=5,
    rank=RankReduction(method=HARD_CUTOFF, value=100),
    solver_type=SolverType.LEASTSQUARES,
    nullspace_algorithm=NullspaceAlgorithm.EIGH
)

loraks = Loraks.create(options)
k_recon = loraks.reconstruct(k_space_2x)
```

### Scenario 2: Highly Undersampled (4x R)

```python
# 4x acceleration, limited ACS
p_loraks = (PLoraks()
    .with_patch_shape((-1, -1, -1, 9, 9))    # Larger neighborhood
    .with_s_matrix()
    .with_sv_auto_soft_cutoff(percentage=0.80)
    .with_regularization_lambda(1.0)
    .with_device('cuda:0'))

k_recon_4x = p_loraks.reconstruct(k_space_4x)
```

### Scenario 3: Multi-Echo Data

```python
# Process 4 echoes
echo_data = torch.stack([echo1, echo2, echo3, echo4], dim=0)
# Shape: [4, n_coils, n_phase, n_read]

loraks = Loraks.create(AcLoraksOptions(loraks_neighborhood_size=5))
k_recon_multi = loraks.reconstruct(echo_data)
# Returns: [4, n_coils, n_phase, n_read]

# Process each echo separately
for i, echo in enumerate(echo_data):
    k_recon_echo = loraks.reconstruct(echo.unsqueeze(0))
    # Save or analyze each echo
```

## GPU Performance Optimization

### Memory Management

```python
# Reduce batch size for large datasets
batched_k_space = k_space[:4]  # Process first 4 batches first
k_recon_batch1 = loraks.reconstruct(batched_k_space)

# Process remaining batches
for i in range(4, len(k_space), 4):
    batch = k_space[i:i+4]
    k_recon_batch = loraks.reconstruct(batch)
```

### Algorithm Selection

```python
# For speed (AC-LORAKS)
options = AcLoraksOptions(
    solver_type=SolverType.LEASTSQUARES,
    nullspace_algorithm=NullspaceAlgorithm.TORCH_LR
)

# For accuracy (AC-LORAKS)
options = AcLoraksOptions(
    solver_type=SolverType.AUTOGRAD,
    nullspace_algorithm=NullspaceAlgorithm.SOR_SVD
)

# For large problems
options = AcLoraksOptions(
    nullspace_algorithm=NullspaceAlgorithm.RSVD  # Memory efficient
)
```

## Troubleshooting

### Common Issues

**Issue 1: Out of Memory**
```python
# Solution: Reduce neighborhood size
options = AcLoraksOptions(loraks_neighborhood_size=3)

# Solution: Reduce batch size
k_space_subset = k_space[:2]  # Process 2 batches
```

**Issue 2: Poor Reconstruction Quality**
```python
# Solution: Increase neighborhood size
options = AcLoraksOptions(loraks_neighborhood_size=7)

# Solution: Use S-matrix instead of C-matrix
options = AcLoraksOptions(loraks_matrix_type=OperatorType.S)

# Solution: Try regularization
options = AcLoraksOptions(regularization_lambda=0.5)
```

**Issue 3: Slow Convergence**
```python
# Solution: Use least squares solver (AC-LORAKS)
options = AcLoraksOptions(solver_type=SolverType.LEASTSQUARES)

# Solution: Reduce iterations
options = AcLoraksOptions(max_num_iter=15)

# Solution: Use faster SVD
options = AcLoraksOptions(nullspace_algorithm=NullspaceAlgorithm.TORCH_LR)
```

## Next Steps

Now that you've learned basic LORAKS reconstruction:

- Try different acceleration factors and compare results
- Experiment with parameter tuning for your specific data
- Explore the integration with other processing modules
- Read the [Reconstruction documentation](recon.md) for detailed theory
- Check the [API Reference](api.md#loraks-reconstruction) for complete class reference

## CLI Usage

```bash
# AC-LORAKS reconstruction
python -m pymritools.recon.loraks.recon \
    -ik undersampled_k_space.pt \
    -p 5 \
    -rank 120 \
    -alg AC-LORAKS-ls \
    -t S \
    -mni 20

# P-LORAKS reconstruction (via configuration file)
python -m pymritools.recon.loraks.recon \
    -c ploraks_config.json
```

This tutorial provides a practical introduction to LORAKS reconstruction. For more advanced usage and theoretical background, see the full documentation.