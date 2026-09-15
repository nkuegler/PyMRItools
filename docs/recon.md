# MRI Reconstruction Methods

## Overview

PyMRITools provides advanced reconstruction algorithms based on LORAKS (Low-Rank Modeling of Local k-Space Neighborhoods), exploiting structured correlations in MRI data to enable reconstruction from undersampled acquisitions.

## LORAKS Reconstruction

### Theoretical Background

**LORAKS (Low-Rank Modeling of Local k-Space Neighborhoods)** exploits the fact that MRI data exhibits strong correlations in local k-space neighborhoods due to physical constraints and limited support.

#### Key Mathematical Concepts

**1. Local Neighborhood Structure:**
MRI data shows correlations in local k-space regions due to smoothness and support constraints.

**2. C-Matrix Operator:**
The C-matrix builds correlation matrices from local neighborhoods:

```
C[i,j] = k[i] * conj(k[j])  for local neighborhood pairs
```

**3. S-Matrix Operator:**
The S-matrix enforces conjugate symmetry, often preferred for magnitude images:

```
S P = C - P_mirrored     (P: forward half, P_mirrored: conjugate symmetric)
S M = C + P_mirrored
```

**4. Low-Rank Property:**
The C/S matrices have low numerical rank due to data structure constraints.

**5. Optimization Problem:**
```
minimize ||M(k)||_* + λ * ||P⊙(k - k_sampled)||_2²
```

Where:
- `||·||_*` is the nuclear norm (low-rank proxy)
- `M` is the C or S operator
- `P` is the sampling mask
- `λ` is the regularization parameter

### AC-LORAKS vs P-LORAKS

| Aspect | AC-LORAKS | P-LORAKS |
|--------|-----------|----------|
| **Data Requirement** | Auto-calibration region available | No calibration region needed |
| **Method** | Extract nullspace from AC region | Direct optimization on full matrix |
| **Computation** | Faster (nullspace computed once) | Slower (matrix re-built each iteration) |
| **Memory** | Lower (works with nullspace subspace) | Higher (stores full matrices) |
| **Best For** | Parallel imaging with ACS | Highly undersampled data |

## AC-LORAKS (Auto-Calibration LORAKS)

### Overview

AC-LORAKS is designed for sequences with auto-calibration regions, providing fast reconstruction by leveraging the fully sampled central k-space.

### Algorithm Steps

1. **Extract AC Region:** Get fully-sampled auto-calibration region from k-space center
2. **Compute Matrix:** Build C/S matrix from AC region
3. **Calculate Nullspace:** Compute trailing right singular vectors (nullspace basis)
4. **Linear System:** Express reconstruction as linear system in nullspace
5. **Solve:** Optimize Mf = b via conjugate gradient or autograd

### Configuration

```python
from pymritools.recon.loraks.ac_loraks import AcLoraksOptions
from pymritools.recon.loraks.loraks import LoraksImplementation, OperatorType, RankReduction, RankReductionMethod

options = AcLoraksOptions(
    loraks_type=LoraksImplementation.AC_LORAKS,
    loraks_neighborhood_size=5,           # 5×5 neighborhood
    loraks_matrix_type=OperatorType.S,   # S-matrix (symmetry)
    rank=RankReduction(
        method=RankReductionMethod.HARD_CUTOFF,
        value=150                        # Keep first 150 singular values
    ),
    regularization_lambda=0.0,           # 0 = strict data consistency
    max_num_iter=30,
    computation_type=ComputationType.FFT,  # Fast FFT computation
    solver_type=SolverType.LEASTSQUARES,    # Conjugate gradient solver
    nullspace_algorithm=NullspaceAlgorithm.EIGH  # Eigendecomposition
)
```

### Key Parameters

| Parameter | Effect | Typical Values |
|-----------|--------|----------------|
| `loraks_neighborhood_size` | Spatial extent of correlation | 3-11 (odd) |
| `loraks_matrix_type` | C: correlation only, S: symmetry enforcement | S preferred for magnitude |
| `rank` | Expected numerical rank | 10-200 (depends on matrix size) |
| `regularization_lambda` | Data consistency weighting | 0 (strict), 0.1-10 (penalized) |
| `max_num_iter` | Optimization iterations | 20-100 |
| `solver_type` | CGD (faster) or autograd (flexible) | CGD for speed |

### Python API Usage

```python
from pymritools.recon.loraks.loraks import Loraks, LoraksOptions
from pymritools.recon.loraks.ac_loraks import AcLoraksOptions
import torch

# Create options
options = AcLoraksOptions(
    loraks_neighborhood_size=5,
    rank=RankReduction(method=RankReductionMethod.HARD_CUTOFF, value=150),
    solver_type=SolverType.LEASTSQUARES,
    nullspace_algorithm=NullspaceAlgorithm.EIGH
)

# Create AC-LORAKS instance
ac_loraks = Loraks.create(options)

# Reconstruct (input shape: [batch, nct, np, nr])
k_recon = ac_loraks.reconstruct(k_space_input)

# Alternative: Direct instantiation
from pymritools.recon.loraks.ac_loraks import AcLoraks

ac_loraks_direct = AcLoraks()
ac_loraks_direct.configure(options)
k_recon = ac_loraks_direct.reconstruct(k_space_input)
```

### CLI Usage

```bash
python -m pymritools.recon.loraks.recon \
    -ik input_k_space.pt \
    -ia affine.pt \
    -p 5 \
    -rank 180 \
    -l 0.0 \
    -alg AC-LORAKS-ls \
    -t S \
    -mni 25 \
    -b 4 \
    -mdt eigh
```

### Advanced Features

**Multiple Nullspace Algorithms:**

```python
# Eigendecomposition (most stable)
nullspace_algorithm=NullspaceAlgorithm.EIGH

# Torch low-rank SVD (fast)
nullspace_algorithm=NullspaceAlgorithm.TORCH_LR

# Randomized SVD (memory efficient)
nullspace_algorithm=NullspaceAlgorithm.RSVD

# Randomized nullspace (for large problems)
nullspace_algorithm=NullspaceAlgorithm.RANDOM_NS

# Subspace Orbit Randomized SVD (most accurate)
nullspace_algorithm=NullspaceAlgorithm.SOR_SVD
```

**Different Solver Types:**

```python
# Conjugate gradient descent (fastest)
solver_type=SolverType.LEASTSQUARES

# Autograd optimization (more flexible)
solver_type=SolverType.AUTOGRAD
```

## P-LORAKS (Penalized LORAKS)

### Overview

P-LORAKS is designed for highly undersampled data without auto-calibration regions, using direct optimization with rank penalties.

### Algorithm Steps

1. **Build Matrix:** Construct C/S matrix from all sampled points
2. **Formulate Problem:** Low-rank optimization problem
3. **Direct Optimization:** Use autograd to optimize with rank penalty
4. **Adaptive Learning Rate:** Apply Barzilai-Borwein or other adaptive methods

### Configuration (Builder Pattern)

```python
from pymritools.recon.loraks.p_loraks import PLoraks
from pymritools.recon.loraks.loraks import OperatorType

# Using builder pattern for fluent interface
p_loraks = (PLoraks()
    .with_patch_shape((-1, -1, -1, 5, 5))    # Full spatial, 5×5 in phase/read
    .with_sample_directions((0, 0, 0, 1, 1))  # Sample in phase/read dimensions
    .with_s_matrix()                         # Use S-matrix operator
    .with_sv_hard_cutoff(rank=150)           # Hard rank cutoff
    .with_regularization_lambda(0.5)         # Penalized reconstruction
    .with_max_iterations(50)                 # Maximum iterations
    .with_constant_learning_rate(1e-3)        # Fixed learning rate
    .with_device('cuda:0'))                  # GPU acceleration

# Or using configuration dataclass
from pymritools.recon.loraks.p_loraks import PLoraksOptions
from pymritools.recon.loraks.loraks import LowRankAlgorithmType

p_options = PLoraksOptions(
    lowrank_algorithm=LowRankAlgorithmType.TORCH_LOWRANK_SVD,
    patch_shape=(0, 0, 0, 5, 5),
    sample_directions=(0, 0, 0, 1, 1)
)

p_loraks = PLoraks()
p_loraks.configure(p_options)
```

### Advanced Builder Options

**Rank Reduction Methods:**

```python
# Hard rank cutoff
.with_sv_hard_cutoff(rank=150)

# ReLU shift (soft cutoff)
.with_sv_soft_cutoff(tau=0.1)

# Automatic ReLU percentage
.with_sv_auto_soft_cutoff(percentage=0.9)  # Keep 90% of cumulative SV

# Custom rank reduction
.with_rank_reduction(RankReduction(method=..., value=...))
```

**Learning Rate Strategies:**

```python
# Fixed learning rate
.with_constant_learning_rate(1e-3)

# Linearly decaying learning rate
.with_linear_learning_rate(min_lr=1e-5, max_lr=1e-2)
```

**Matrix Operators:**

```python
# C-matrix (correlation only)
.with_c_matrix()

# S-matrix (symmetry enforcement)
.with_s_matrix()
```

### Python API Usage

```python
from pymritools.recon.loraks.p_loraks import PLoraks

# Create P-LORAKS with builder pattern
p_loraks = (PLoraks()
    .with_patch_shape((-1, -1, -1, 5, 5))
    .with_s_matrix()
    .with_sv_hard_cutoff(rank=150)
    .with_regularization_lambda(0.5)
    .with_max_iterations(50)
    .with_device('cuda:0'))

# Reconstruct
k_recon = p_loraks.reconstruct(k_space_input)
```

## Configuration Classes

### Base Class: LoraksOptions

```python
from pymritools.recon.loraks.loraks import LoraksOptions

LoraksOptions(
    loraks_type=LoraksImplementation.AC_LORAKS,
    loraks_neighborhood_size=5,
    loraks_matrix_type=OperatorType.S,
    rank=RankReduction(method=RankReductionMethod.HARD_CUTOFF, value=150),
    regularization_lambda=0.0,
    max_num_iter=30,
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
)
```

### AC-LORAKS: AcLoraksOptions

```python
from pymritools.recon.loraks.ac_loraks import AcLoraksOptions

AcLoraksOptions(
    # Inherits all LoraksOptions parameters
    computation_type=ComputationType.FFT,      # FFT or regular computation
    solver_type=SolverType.LEASTSQUARES,        # Least squares or autograd
    nullspace_algorithm=NullspaceAlgorithm.EIGH  # Nullspace computation method
)
```

### P-LORAKS: PLoraksOptions

```python
from pymritools.recon.loraks.p_loraks import PLoraksOptions

PLoraksOptions(
    # Inherits all LoraksOptions parameters
    patch_shape=(0, 0, 0, 5, 5),              # Spatial patch dimensions
    sample_directions=(0, 0, 0, 1, 1),        # Sampling directions
    lowrank_algorithm=LowRankAlgorithmType.TORCH_LOWRANK_SVD  # SVD algorithm
)
```

## Enums and Types

### Main Enums

```python
# LORAKS implementation type
LoraksImplementation.AC_LORAKS  # Auto-calibration LORAKS
LoraksImplementation.P_LORAKS   # Penalized LORAKS

# Operator type
OperatorType.C  # C-matrix (correlation only)
OperatorType.S  # S-matrix (symmetry enforcement)

# Rank reduction method
RankReductionMethod.HARD_CUTOFF            # Hard rank cutoff (int)
RankReductionMethod.RELU_SHIFT              # ReLU shift threshold (float)
RankReductionMethod.RELU_SHIFT_AUTOMATIC    # Automatic percentage (0-1)

# AC-LORAKS solver type
SolverType.LEASTSQUARES  # Conjugate gradient descent
SolverType.AUTOGRAD      # Autograd optimization

# AC-LORAKS nullspace algorithm
NullspaceAlgorithm.EIGH         # Eigendecomposition
NullspaceAlgorithm.TORCH_LR     # Torch low-rank SVD
NullspaceAlgorithm.RSVD         # Randomized SVD
NullspaceAlgorithm.SOR_SVD      # Subspace orbit randomized SVD
NullspaceAlgorithm.RANDOM_NS    # Randomized nullspace

# P-LORAKS low-rank algorithm
LowRankAlgorithmType.TORCH_LOWRANK_SVD  # Torch built-in low-rank SVD
LowRankAlgorithmType.RANDOM_SVD          # Randomized SVD
LowRankAlgorithmType.SOR_SVD             # Subspace orbit randomized SVD
```

## Data Preparation

### Input Format

LORAKS expects k-space data in the following format:

```python
# Input tensor shape: [batch, nct, np, nr]
# Where:
# - batch: number of batches / echoes
# - nct: number of compressed channels (or total channels)
# - np: number of phase encoding points
# - nr: number of readout points

k_space_input = torch.randn(4, 12, 256, 256, dtype=torch.complex64)
```

### Batching Utilities

```python
from pymritools.recon.loraks.utils import prepare_k_space_to_batches, unprepare_batches_to_k_space
from pymritools.recon.loraks.utils import extract_channel_batch_indices

# Transform to batched format
k_batched, batch_channel_indices = prepare_k_space_to_batches(
    k_space_input,
    batch_channel_indices  # Optional K-means clustering
)

# Transform back to original format
k_original = unprepare_batches_to_k_space(
    k_batched,
    batch_channel_indices,
    original_shape=k_space_input.shape
)

# K-means clustering for channel batching
batch_indices = extract_channel_batch_indices(
    k_data=k_space_input,
    batch_size_channels=8  # Cluster channels into groups of 8
)
```

## Performance Optimization

### Memory Requirements

LORAKS memory requirements depend on:
- Neighborhood size: Larger neighborhoods require more memory
- Matrix type: S-matrix ~2x larger than C-matrix
- Batch size: Larger batches increase memory usage

**Approximate memory calculation:**
```python
# For AC-LORAKS
memory = batch_entries * neighborhood_size^2 * complex_bytes * over_sampling

# For P-LORAKS
memory = sampled_points * neighborhood_size^2 * complex_bytes
```

### GPU Acceleration

```python
# Automatic GPU detection
settings.use_gpu = True  # Will use CUDA if available

# Manual device selection
settings.gpu_device = 0  # Use first GPU
device = torch.device('cuda:0')
```

### Optimization Strategies

**1. Choose Algorithm Based on Data:**
- **AC available**: Use AC-LORAKS (faster, less memory)
- **Highly undersampled**: Use P-LORAKS (more flexible)
- **Limited AC region**: Use P-LORAKS with small regularization

**2. Tune Parameters:**
- **Neighborhood size**: Start with 5, adjust based on data
- **Rank**: Start with 80-90% of matrix size, reduce if memory is limited
- **Lambda**: Start with 0.0, increase if data inconsistency issues

**3. Algorithm Selection:**
- **Nullspace (AC)**: Use EIGH for stability, RSVD for large problems
- **Solver (AC)**: Use LEASTSQUARES for speed, AUTOGRAD for flexibility
- **SVD (P)**: Use TORCH_LOWRANK_SVD for speed, SOR_SVD for accuracy

## Common Use Cases

### Scenario 1: Standard Parallel Imaging

```python
from pymritools.recon.loraks.ac_loraks import AcLoraksOptions

# 2x acceleration with 24-line ACS region
options = AcLoraksOptions(
    loraks_neighborhood_size=5,
    rank=RankReduction(method=HARD_CUTOFF, value=120),
    solver_type=SolverType.LEASTSQUARES,
    nullspace_algorithm=NullspaceAlgorithm.EIGH
)

loraks = Loraks.create(options)
k_recon = loraks.reconstruct(k_space_2x_accelerated)
```

### Scenario 2: Highly Undersampled Data

```python
from pymritools.recon.loraks.p_loraks import PLoraks

# 4x acceleration, no AC region
p_loraks = (PLoraks()
    .with_patch_shape((-1, -1, -1, 7, 7))    # Larger neighborhood
    .with_s_matrix()
    .with_sv_auto_soft_cutoff(percentage=0.85)
    .with_regularization_lambda(0.8)         # Higher penalty
    .with_max_iterations(80)                  # More iterations
    .with_device('cuda:0'))

k_recon = p_loraks.reconstruct(k_space_4x_accelerated)
```

### Scenario 3: Multi-Echo Data

```python
# Batch process multiple echoes
echo_data = torch.stack([echo1, echo2, echo3, echo4], dim=0)  # [4, nct, np, nr]

loraks = Loraks.create(AcLoraksOptions(loraks_neighborhood_size=5))
k_recon = loraks.reconstruct(echo_data)  # Returns [4, nct, np, nr]
```

## Error Handling and Troubleshooting

### Common Issues

**1. Out of Memory**
- Reduce neighborhood size
- Reduce rank parameter
- Reduce batch size
- Use more aggressive rank reduction

**2. Poor Reconstruction Quality**
- Increase neighborhood size
- Use S-matrix instead of C-matrix
- Tune regularization lambda
- Try different nullspace algorithms

**3. Slow Convergence**
- Use LEASTSQUARES solver (AC-LORAKS)
- Reduce max iterations with higher tolerance
- Use TORCH_LOWRANK_SVD for faster SVD

**4. Data Consistency Issues**
- Check input data format and dimensions
- Verify sampling mask is correct
- Use regularization (lambda > 0) for noisy data

## See Also

- [Mathematical Foundations](#theoretical-background) - Complete LORAKS theory
- [API Reference](api.md#loraks-reconstruction) - Complete class and method documentation
- [Configuration System](../contribute/writing_code.md#configuration) - How to use BaseClass configuration
- [GPU Acceleration](#performance-optimization) - Performance considerations
- [Simulation](simulation.md) - Generate test data for reconstruction