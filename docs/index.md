# PyMRITools - Python Tools for MRI

Welcome to the PyMRITools documentation. PyMRITools is a comprehensive Python toolbox for MRI sequence programming, reconstruction, simulation, processing, and analysis.

## Overview

PyMRITools provides state-of-the-art algorithms and utilities for:

- **Simulation**: Bloch equation simulations using the EMC method
- **Sequence Programming**: Event-based MRI pulse sequence design
- **Reconstruction**: Advanced LORAKS (Low-Rank Modeling of Local k-Space Neighborhoods) reconstruction
- **Processing**: Data preprocessing and post-processing including denoising and artifact correction
- **Modeling**: Parameter estimation including B1+ mapping and dictionary fitting

## Key Features

!!! note "GPU Acceleration"
    Most computationally intensive operations support GPU acceleration via PyTorch for significant performance improvements.

!!! tip "CLI & Python API"
    All major tools provide both command-line interfaces and Python APIs for maximum flexibility in automated pipelines and interactive use.

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/schmidt-jo/PyMRItools.git
cd PyMRItools

# Using uv (recommended)
uv sync

# Or using conda
mamba env create -f environment.yml
mamba activate mri_tools_env
pip install --no-deps -e .
```

### Basic Usage

**Command Line:**

```bash
# LORAKS reconstruction
python -m pymritools.recon.loraks.recon -ik k_space.pt -p 5 -rank 180

# B1+ AFI mapping
python -m pymritools.modeling.b1_afi.modeling -i afi.nii -fa 60 -n 5

# LCPCA denoising
python -m pymritools.processing.denoising.lcpca.denoise -i noisy.nii -p 0
```

**Python API:**

```python
from pymritools.recon.loraks.loraks import Loraks, LoraksOptions
from pymritools.modeling.b1_afi.modeling import Settings, processing
from pymritools.processing.denoising.lcpca import denoise_lcpca

# LORAKS reconstruction
options = LoraksOptions(loraks_neighborhood_size=5)
loraks = Loraks.create(options)
k_recon = loraks.reconstruct(k_space)

# B1+ mapping
settings = Settings(input_file="afi.nii", flip_angle=60.0, ratio_tr2_tr1=5)
processing(settings)

# LCPCA denoising
denoised = denoise_lcpca(input_data=noisy_data, p=0, device='cuda')
```

## Module Overview

### Simulation (`pymritools.simulation`)

**EMC (echo modulation curve) simulation** using Bloch equations for accurate modeling of multi-echo MRI sequences:

- Multi-parametric simulation (T₁, T₂, B₁⁺, B₀)
- Slice profile modeling with isochromats
- GPU acceleration for large parameter spaces
- Database generation for dictionary matching

### Sequence Programming (`pymritools.seqprog`)

**Event-based sequence programming** paradigm for flexible MRI pulse sequence design:

- Atomic events: RF, GRAD, ADC, DELAY
- Sequence kernels and composition
- ISMRMRD integration for raw data handling
- Support for MESE, MEGESSE sequences

### Reconstruction (`pymritools.recon.loraks`)

**LORAKS (Low-Rank Modeling of Local k-Space Neighborhoods)** - advanced reconstruction exploiting structured correlations:

- **AC-LORAKS**: Fast reconstruction with auto-calibration regions
- **P-LORAKS**: Penalized reconstruction for highly undersampled data
- Multiple matrix operators (C and S matrices)
- Flexible rank reduction strategies
- GPU-accelerated optimization

### Data Processing (`pymritools.processing`)

Comprehensive data preprocessing and post-processing toolbox:

- **Coil Compression**: PCA-based channel reduction
- **Coil Sensitivity**: rSoS-based sensitivity map estimation
- **LCPCA Denoising**: Local Clinical PCA with Marchenko-Pastur inequality
- **Gibbs Unringing**: Ringing artifact removal using sub-voxel optimization

### Modeling (`pymritools.modeling`)

Parameter estimation and model fitting algorithms:

- **B1+ AFI Mapping**: Actual Flip-angle Imaging
- **Dictionary Fitting**: Grid search for T₁/T₂/B₁⁺ estimation
- **Mono-exponential Fitting**: Decay model fitting
- **ESPIRiT**: Eigenvalue-based coil sensitivity estimation

## Mathematical Foundations

### LORAKS Theory

LORAKS exploits low-rank structure in local k-space neighborhoods:

```
C-Matrix: C[i,j] = k[i] * conj(k[j])
S-Matrix: S P = C - P_mirrored

Optimization: minimize ||M(k)||_* + λ * ||P⊙(k - k_sampled)||_2²
```

### Bloch Equations (EMC)

EMC simulation solves the Bloch equations:

```
dMz/dt = (M0 - Mz)/T1
dMxy/dt = -Mxy/T2 + γ * M × B(t)

Matrix form: M(t + dt) = P(dt) × M(t)
```

### AFI B1+ Mapping

The AFI method uses two acquisitions with different TR values:

```
r = S₁/S₂
α = arccos((r·n - 1)/(n - r))
B1⁺ = (α/α_set) × 100
```

## Tutorials

- [Reconstruction Tutorial](tutorials/recon_tutorial.md) - LORAKS reconstruction with AC-LORAKS and P-LORAKS
- [Modeling Tutorial](tutorials/modeling_tutorial.md) - B1+ mapping, dictionary fitting, and mono-exponential fitting
- [Simulation Tutorial](tutorials/simulation_tutorial.md) - EMC simulation for database generation
- [Complete Workflow Tutorial](tutorials/complete_workflow_tutorial.md) - End-to-end pipeline from raw data to final parameters

## Common Patterns

### Configuration Management

All tools use a consistent configuration system:

```python
@dataclass
class Settings(BaseClass):
    config_file: str = ""          # JSON configuration file
    out_path: str = ""             # Output directory
    use_gpu: bool = True           # GPU acceleration
    gpu_device: int = 0            # GPU device ID
    visualize: bool = True         # Visualization toggle
    debug: bool = False            # Debug mode
    slurm: bool = False           # Slurm mode
```

### GPU Acceleration

```python
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
settings.use_gpu = True
```

### Batch Processing

```python
settings.batch_size = 3000  # Process large datasets in batches
settings.process_slice = False  # Process full volume or slice-by-slice
```

## Performance Considerations

- **GPU vs CPU**: GPU acceleration provides 10-100x speedup for most operations
- **Memory requirements**: LORAKS requires significant GPU memory for large k-space datasets
- **Batch processing**: Use batched processing for large datasets to manage memory
- **Algorithm selection**: Choose AC-LORAKS for speed, P-LORAKS for highly undersampled data

## Citation

If you use PyMRITools in your research, please cite the appropriate references:

- EMC Simulation and Grid Search Fitting: [Ben-Eliezer et al. 2015](https://doi.org/10.1002/mrm.25156)
- LORAKS Reconstruction: [Haldar et al. 2014](https://pubmed.ncbi.nlm.nih.gov/24595341/)
- AC-LORAKS: [Tamir et al. 2017](https://ieeexplore.ieee.org/document/7164018/)
- P-LORAKS: [Liang et al. 2014](https://pubmed.ncbi.nlm.nih.gov/25952136/)
- LCPCA Denoising: [Does et al. 2019](https://doi.org/10.1002/mrm.27658)
- ESPIRiT: [Uecker et al. 2014](https://doi.org/10.1002/mrm.25195)
- Gibbs Unringing: [Kellner et al. 2016](https://doi.org/10.1002/mrm.26054)
- B1+ AFI: [Yarnykh 2007](https://doi.org/10.1002/mrm.21194)

## Getting Started

- [Installation Guide](installation.md) - Detailed installation instructions
- [Contributing](contribute/contributing.md) - How to contribute to PyMRITools
- [Tutorials](#tutorials) - Module-specific tutorials
- [API Reference](api.md) - Complete Python API reference

## Reference Documentation

- [Reconstruction](recon.md) - Complete LORAKS theory and algorithms
- [Modeling](modeling.md) - All modeling approaches with mathematical foundations
- [Processing](processing.md) - All processing algorithms with error analysis
- [Simulation](simulation.md) - EMC simulation with Bloch equation details
- [Sequence Programming](sequence_programming.md) - Event-based sequence building tutorial
- [API Reference](api.md) - Complete class and method documentation