# Theoretical Modeling of MR Data

## Overview

PyMRITools provides comprehensive modeling tools for parameter estimation from MRI data, including B1+ mapping, dictionary fitting, mono-exponential decay fitting, and coil sensitivity estimation.

## B1+ AFI Mapping

### Theoretical Background

**AFI (Actual Flip-angle Imaging)** is a dual-acquisition method that uses two acquisitions with different TR values to measure the actual flip angle and thus estimate B1+ field inhomogeneity.

#### Mathematical Principles

**1. Signal Ratio:**
```
r = S₁/S₂
```

**2. Flip Angle Calculation:**
```
α = arccos((r·n - 1)/(n - r))
```
where `n = TR₂/TR₁` is the ratio of repetition times.

**3. B1+ Scaling:**
```
B1⁺ (%) = (α/α_set) × 100
```

**4. Error Propagation:**
```
Variance in ratio: σᵣ² ≈ (σ₁S₂ + σ₂S₁)²/S₁⁴
Variance in flip angle: σₐ² = |∂α/∂r|²·σᵣ²
Variance in B1⁺: σ_B1² = (18000/π·α_set)²·σₐ²
```

### Configuration

```python
from pymritools.modeling.b1_afi.modeling import Settings

settings = Settings(
    input_file="afi_echoes.nii",           # Input AFI echo images
    input_affine="",                       # Empty for NIfTI files
    flip_angle=60.0,                      # Set flip angle [degrees]
    ratio_tr2_tr1=5,                     # TR₂/TR₁ ratio
    smoothing_kernel=3.0,                 # Gaussian smoothing size
    noise_mask="noise_mask.nii",          # Noise mask for error estimation
    input_in_image_domain=True,            # Input is in image domain
    out_path="./output"                   # Output directory
)
```

### Key Parameters

| Parameter | Purpose | Typical Values |
|-----------|---------|----------------|
| `input_file` | Path to AFI echo images | Two-echo NIfTI file |
| `flip_angle` | Set flip angle in degrees | 50-70° (common: 60°) |
| `ratio_tr2_tr1` | TR₂/TR₁ ratio | 4-8 (common: 5-6) |
| `smoothing_kernel` | Gaussian smoothing size | 2-5 pixels |
| `noise_mask` | Noise regions for error | Background voxels |

### Python API Usage

```python
from pymritools.modeling.b1_afi.modeling import Settings, processing

# Full processing pipeline
settings = Settings(
    input_file="afi_echoes.nii",
    flip_angle=60.0,
    ratio_tr2_tr1=5,
    smoothing_kernel=3.0,
    noise_mask="noise_mask.nii",
    out_path="./b1_output"
)

processing(settings)

# Step-by-step processing
import torch
from pymritools.modeling.b1_afi.modeling import calculate_b1

# Load AFI data (shape: [..., 2] for two echoes)
b1_data = torch_load("afi_echoes.pt")

# Calculate B1+ map
b1_map = calculate_b1(
    b1_data=b1_data,
    r_tr21=5.0,
    smoothing_kernel=3.0,
    flip_angle_set_deg=60.0
)

# Calculate error map
from pymritools.modeling.b1_afi.modeling import calculate_error_map
import numpy as np

noise_mask = np.load("noise_mask.npy")
b1_error = calculate_error_map(
    b1_data=b1_data,
    mask=noise_mask,
    flip_angle_set_deg=60.0,
    ratio_tr_n=5.0
)
```

### Core Functions

#### `calculate_b1()`
```python
def calculate_b1(
    b1_data: torch.Tensor,
    r_tr21: float,
    smoothing_kernel: float = 3,
    flip_angle_set_deg: float = 60.0
) -> torch.Tensor
    """
    Core B1+ mapping calculation from AFI signals.
    
    Parameters:
        b1_data: Multi-echo B1 data (shape: [..., 2] for two echoes)
        r_tr21: TR₂/TR₁ ratio
        smoothing_kernel: Gaussian smoothing kernel size
        flip_angle_set_deg: Set flip angle in degrees
    
    Returns:
        B1+ percentage map
    """
```

#### `calculate_error_map()`
```python
def calculate_error_map(
    b1_data: torch.Tensor,
    mask: torch.Tensor,
    flip_angle_set_deg: float,
    ratio_tr_n: float,
    path_visuals: pathlib.Path = None
) -> torch.Tensor
    """
    Compute comprehensive error map for B1+ estimation.
    
    Parameters:
        b1_data: B1+ data
        mask: Noise mask
        flip_angle_set_deg: Set flip angle [degrees]
        ratio_tr_n: TR ratio
        path_visuals: Optional path for visualization
    
    Returns:
        B1+ error map
    """
```

#### `variance_b1()`
```python
def variance_b1(
    value_sigma_alpha: torch.Tensor,
    value_set_alpha: float = 60.0
) -> torch.Tensor
    """
    Calculate B1+ variance from flip angle variance.
    
    Parameters:
        value_sigma_alpha: Alpha variance map
        value_set_alpha: Set flip angle [degrees]
    
    Returns:
        B1+ variance map
    """
```

### CLI Usage

```bash
python -m pymritools.modeling.b1_afi.modeling \
    -i afi_echoes.nii \
    -fa 60.0 \
    -n 5 \
    -s 3.0 \
    -m noise_mask.nii \
    -o ./output
```

### Error Analysis

**Variance Propagation:**

The B1+ estimation includes full error propagation accounting for:
- Ratio variance: From noise in both echoes
- Flip angle variance: Derived from ratio variance
- B1+ variance: Scaled from flip angle variance

**Noise Mask:**
- Provides background voxels for noise estimation
- Required for error calculation
- Should include regions with no signal

### Common Use Cases

**Scenario 1: Standard AFI Acquisition**

```python
settings = Settings(
    input_file="afi_60degree_TR1_TR5.nii",
    flip_angle=60.0,
    ratio_tr2_tr1=5,
    smoothing_kernel=3.0
)
processing(settings)
```

**Scenario 2: Custom Error Analysis**

```python
# Calculate B1+ with error analysis
b1_map = calculate_b1(b1_data, r_tr21=5.0, smoothing_kernel=3.0, flip_angle_set_deg=60.0)
b1_error = calculate_error_map(b1_data, noise_mask, flip_angle_set_deg=60.0, ratio_tr_n=5.0)

# Plot with error bands
plt.figure()
plt.imshow(b1_map.T, cmap='viridis', origin='lower')
cbar = plt.colorbar()
cbar.set_label('B1+ (%)')
plt.title(f'B1+ Map (±{np.mean(b1_error):.2f}% error)')
```

## Dictionary Fitting

### Theoretical Background

**Dictionary matching** uses pre-computed signal patterns (from EMC simulation) to estimate tissue parameters via brute-force search.

#### Algorithm Principles

**1. Database Construction:**
```
Size: [N_T1, N_T2, N_B1, ETL]
Content: Magnitude signals for parameter combinations
```

**2. Matching Process:**
```
For each voxel:
    min L₂ = ||S_voxel - S_DB[i,j,k]||²
    Find best matching: (T₁*, T₂*, B₁⁺*) = arg min L₂
```

**3. B1+ Regularization:**
```
Penalty term: P_B1 = |B₁_voxel - B₁_DB[k]|
Result: Reduces search space to nearest B1+ values
```

**4. MEGESSE R2* Extraction:**
```
ln(S_GRE(t)/S_DB_GRE(t)) = ln(S₀) - R₂*·t
Solve: y = a₀ + a₁·t where R₂* = -a₁
```

### Configuration

```python
from pymritools.config.emc import EmcFitSettings

settings = EmcFitSettings(
    input_data="emc_data.nii",           # Multi-echo data
    input_database="database.pkl",        # Pre-computed EMC database
    input_b1="b1_map.nii",              # B1+ regularization (optional)
    input_b1_err="",                    # B1+ error (optional)
    batch_size=3000,                    # Batch size for memory
    rsos_channel_combine=False,         # Channel combination method
    low_rank_regularisation=None,      # Optional low-rank preprocessing
    b1_error_cutoff=0.15,              # B1+ error threshold
    tx_factor=1.0,                     # TX scaling factor
    out_name="",                       # Output name
    save_name_prefix="",               # File prefix
    out_path="./output"                # Output directory
)
```

### Python API Usage

```python
from pymritools.modeling.dictionary.grid_search import fit, fit_r2_fn
from pymritools.config.emc import EmcFitSettings
import torch

# Method 1: Using settings class
settings = EmcFitSettings(
    input_data="emc_data.nii",
    input_database="database.pkl",
    input_b1="b1_map.nii",
    batch_size=3000,
    out_path="./fitting_output"
)

fit(settings)

# Method 2: Direct function call
input_data = torch.randn(100, 100, 16)  # [spatial, temporal]
database = torch.randn(10, 20, 5, 16)   # [T1, T2, B1, temporal]
t2_vals = torch.linspace(25, 100, 20)   # [ms]
b1_vals = torch.linspace(0.6, 1.6, 5)

t2_map, b1_map, l2_err = fit_r2_fn(
    input_data=input_data,
    db_mag=database,
    t2_vals=t2_vals,
    b1_vals=b1_vals,
    batch_size=1000
)
```

### Core Functions

#### `fit_r2_fn()`
```python
def fit_r2_fn(
    input_data: torch.Tensor,
    db_mag: torch.Tensor,
    t2_vals: torch.Tensor,
    b1_vals: torch.Tensor,
    t1_vals: torch.Tensor = torch.tensor([1.5]),
    b1_data: torch.Tensor = None,
    batch_size: int = 1000,
    device: torch.device = torch.get_default_device()
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Comprehensive R2 fitting using dictionary search.
    
    Parameters:
        input_data: Input data with temporal dimension
        db_mag: Database magnitudes [T1, T2, B1, ETL]
        t2_vals: T2 search range [seconds]
        b1_vals: B1 search range
        t1_vals: T1 values (optional, default: [1.5s])
        b1_data: B1+ regularization data (optional)
        batch_size: Process voxels in batches
        device: Computation device
    
    Returns:
        (T2 map, B1 map, L2 error map)
    """
```

#### `fit()`
```python
def fit(settings: EmcFitSettings) -> None:
    """
    Main fitting pipeline for R2 parameter estimation.
    
    Process:
        1. Load data → setup database → optional B1+ regularization
        2. Batched fitting → save results
    """
```

#### `megesse()`
```python
def megesse(settings: EmcFitSettings) -> None:
    """
    MEGESSE-specific fitting for combined R2/R2* estimation.
    
    Additional Process:
        Standard R2 fitting + gradient echo R2* extraction
        - Linear regression on GRE time differences
        - R2* = -slope of ln(S) vs time
    """
```

### CLI Usage

```bash
# Standard R2 dictionary fitting
python -m pymritools.modeling.dictionary.grid_search \
    -i emc_data.nii \
    -db database.pkl \
    -b1 b1_map.nii \
    -bs 3000 \
    -o ./output

# MEGESSE R2/R2* fitting
python -m pymritools.modeling.dictionary.grid_search --grid_combined_r2_r2s_search \
    -i megesse_data.nii \
    -db database.pkl \
    -b0 b0_map.nii \
    -o ./megesse_output
```

### Database Structure

**Dimensions:**
```
database magnitude: [N_T1, N_T2, N_B1, ETL]
database phase: [N_T1, N_T2, N_B1, ETL]
```

**Parameter Ranges:**
```python
# Typical parameter ranges
t1_list = [0.8, 1.0, 1.2, 1.5, 1.8, 2.0, 2.5, 3.0]  # [seconds]
t2_list = [[20, 80, 2.0], [30, 70, 2.0], [40, 60, 2.0]]  # [[start, end, step] ms]
b1_list = [0.6, 0.8, 1.0, 1.2, 1.4]  # Relative B1+ scaling
```

### Memory Optimization

**Batch Processing:**
```python
# Adjust batch size based on available GPU memory
settings.batch_size = 3000  # Default, adjust as needed

# For large datasets
settings.batch_size = 1000  # Smaller batches to fit GPU memory
```

### Common Use Cases

**Scenario 1: Standard R2 Fitting**

```python
settings = EmcFitSettings(
    input_data="emc_16echo.nii",
    input_database="emc_database.pkl",
    batch_size=3000,
    b1_error_cutoff=0.15
)
fit(settings)
```

**Scenario 2: B1-Regularized Fitting**

```python
settings = EmcFitSettings(
    input_data="emc_data.nii",
    input_database="database.pkl",
    input_b1="b1_map.nii",           # Use B1+ map
    input_b1_err="b1_error.nii",     # Use B1+ error map
    b1_error_cutoff=0.15             # Filter unreliable B1+ values
)
fit(settings)
```

## Mono-exponential Decay Fitting

### Theoretical Background

**Mono-exponential decay model** fits the standard T2 decay: S(t) = S₀·exp(-R·t).

#### Mathematical Derivation

**1. Signal Model:**
```
S(t) = S₀ · exp(-R · t)
```
where S₀ is baseline signal and R is decay rate.

**2. Linearization:**
```
ln(S) = ln(S₀) - R·t
```
Reduces to linear form: y = b₀ + b₁·t

**3. Least Squares Solution:**
```
minimize: Σᵢ [ln(Sᵢ) - (ln(S₀) - R·tᵢ)]²
Solve: (AᵀA)b = AᵀY where A = [1, -t]
```

**4. Parameter Extraction:**
```
S₀ = exp(b₀)
R = -b₁
T₂ = 1/R  [seconds]
```

### Configuration

```python
from pymritools.modeling.decay.mexp import MEXPSettings

settings = MEXPSettings(
    input_data="multi_echo.nii",       # Multi-echo data
    input_affine="",                   # Empty for NIfTI files
    input_in_img_space=True,            # Input is in image domain
    echo_times=[0.01, 0.02, 0.03, 0.04, 0.05],  # Echo times [seconds]
    out_path="./output"                # Output directory
)
```

### Python API Usage

```python
from pymritools.modeling.decay.mexp import fit, process, MEXPSettings
import torch

# Method 1: Using settings
settings = MEXPSettings(
    input_data="multi_echo.nii.gz",
    echo_times=[0.01, 0.02, 0.03, 0.04, 0.05],  # [seconds]
    out_path="./mexp_output"
)
process(settings)

# Method 2: Direct fitting
# Simulate data with known parameters
te = torch.tensor([0.01, 0.02, 0.03, 0.04, 0.05])  # [seconds]
s0_true = 100.0
r_true = 20.0  # R = 1/T2
data = s0_true * torch.exp(-r_true * te)

# Add spatial dimensions
data_3d = data.view(1, 1, 1, -1).expand(128, 128, 1, -1)

# Fit
s0_est, r_est = fit(data_3d, te)

# Calculate T2
t2_est = 1.0 / r_est
print(f"Estimated T2: {t2_est.mean():.4f} seconds")

# Plot results
import matplotlib.pyplot as plt
fig, (ax1, ax2) = plt.subplots(1, 2)

ax1.imshow(s0_est.squeeze().cpu().numpy(), cmap='gray')
ax1.set_title(f'S₀ (Ground truth: {s0_true})')
ax1.axis('off')

ax2.imshow(t2_est.squeeze().cpu().numpy(), cmap='jet')
ax2.set_title(f'T₂ (Ground truth: {1/r_true:.4f}s)')
ax2.axis('off')
plt.colorbar(ax2.images[0], ax=ax2)
```

### Core Functions

#### `fit()`
```python
def fit(
    data: torch.Tensor,
    te: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Fits mono-exponential decay model S(t) = S₀·exp(-R·t).
    
    Parameters:
        data: Input tensor with time as last dimension
        te: Echo times [seconds] corresponding to time points
    
    Returns:
        (S₀ map, R map) where T₂ = 1/R
    
    Method:
        Least squares fitting using log transform and linear regression
        
    Process:
        1. Log transform: ln(S) = ln(S₀) - R·t
        2. Linear solve: minimize ||Y - Ab||² where Y = ln(S)
        3. Extract S₀ = exp(b₀), R = -b₁
    """
```

#### `process()`
```python
def process(settings: MEXPSettings) -> None:
    """
    Main processing pipeline for mono-exponential fitting.
    
    Process:
        Load data → Set device → Fit model → Save results
        
    Saves:
        - s0_map.nii: Baseline signal
        - r_map.nii: Decay rate
        - t2_map.nii: T₂ values (T₂ = 1/R)
    """
```

### CLI Usage

```bash
# Standard mono-exponential fitting
python -m pymritools.modeling.decay.mexp \
    -i multi_echo.nii \
    -te 0.01 0.02 0.03 0.04 0.05 \
    -o ./mexp_output

# k-space input
python -m pymritools.modeling.decay.mexp \
    -i multi_echo_k.pt \
    -ia affine.pt \
    -te 0.01 0.02 0.03 \
    -ii false \
    -o ./mexp_output
```

### Error Analysis

**Linear Regression Accuracy:**
The mono-exponential fitting uses ordinary least squares on log-transformed data, which is optimal for Gaussian noise.

**Data Requirements:**
- Minimum 3 echo times recommended
- Echo times should span the expected T2 range
- Longer echo times provide better R estimation

### Common Use Cases

**Scenario 1: T2 Mapping**

```python
import numpy as np

# Generate echo times spanning expected T2 range
t2_range = 0.08  # Expected T2 in seconds
echo_times = np.linspace(0.01, 0.3, 16)  # 16 echoes, up to ~4× T2

settings = MEXPSettings(
    input_data="multi_echo_16echo.nii",
    echo_times=echo_times.tolist()
)
process(settings)
```

**Scenario 2: R1 Mapping (T1 inversion recovery)**

```python
# Use same mono-exponential fitting for T1 (inversion recovery)
# Model: S(t) = |S₀ - 2S₀·exp(-t/T1)| approximated as exponential
settings = MEXPSettings(
    input_data="inversion_recovery.nii",
    echo_times=[0.1, 0.2, 0.4, 0.8, 1.6],  # Multiple inversion times
)
process(settings)
```

## ESPIRiT Coil Sensitivity Estimation

### Theoretical Background

**ESPIRiT (Eigenvalue Approach to Autocalibrating Parallel MRI)** uses eigenvalue decomposition to estimate multiple coil sensitivity maps.

#### Algorithm Principles

**1. Calibration Matrix Construction:**
```
Extract AC region: k_AC = k[x_center±n, y_center±n, z_center±n, :]
Build Hankel: H from sliding windows over k_AC
```

**2. Singular Value Analysis:**
```
H = U Σ Vᴴ
Keep: V[:, :n_components] where σᵢ ≥ threshold·σ_max
```

**3. Kernel Extraction:**
```
Reshape V to k-space kernels: K[x,y,z,nc,n_comp]
Flip and conjugate: K̃ = K*.flip()
```

**4. Image Space Transformation:**
```
S_kernel = FFT(K̃, dims=(0,1,2))
Scale: S_kernel × √(product(spatial_dims))/√(kernel_size^d)
```

**5. Point-wise SVD:**
```
For each spatial voxel:
    Eigenvalue decomposition: C = U Λ Uᴴ
    Select: eigenvalues λ > threshold
    Sensitivity maps: eigenvectors corresponding to λ > 1
```

### Python API Usage

```python
from pymritools.modeling.espirit.functions import map_estimation
import torch

# Simulate multi-channel k-space data
n_read, n_phase, n_slice, n_channels = 256, 256, 1, 8
k_data = torch.randn(n_read, n_phase, n_slice, n_channels, dtype=torch.complex64)

# Estimate sensitivity maps
sensitivity_maps = map_estimation(
    k_rpsc=k_data,
    kernel_size=6,                                    # Even kernel size recommended
    num_ac_lines=24,                                 # Calibration region size
    rank_fraction_ac_matrix=0.01,                    # AC matrix rank threshold
    eigenvalue_cutoff=0.99,                         # Eigenvalue threshold for selection
    device=torch.device("cuda")
)

# Access results
print(f"Number of sensitivity maps: {sensitivity_maps.shape[0]}")
print(f"Maps shape: {sensitivity_maps.shape}")  # [n_maps, n_read, n_phase, n_slice, n_channels]

# First map (typically SENSE-like)
sense_map = sensitivity_maps[0]  # Shape: [read, phase, slice, channels]
map_magnitude = torch.abs(sense_map)

# Coil combination using maps
img_k = torch.randn(n_read, n_phase, n_slice, n_channels, dtype=torch.complex64)

# For each map, apply coil combination
combined_images = []
for i in range(sensitivity_maps.shape[0]):
    coil_weights = sensitivity_maps[i].conj()
    combined = torch.sum(img_k * coil_weights, dim=-1)
    combined_images.append(combined)
```

### Core Functions

#### `map_estimation()`
```python
def map_estimation(
    k_rpsc: torch.Tensor,
    kernel_size: int,
    num_ac_lines: int,
    rank_fraction_ac_matrix: float = 0.01,
    eigenvalue_cutoff: float = 0.99,
    device: torch.device = torch.get_default_device()
) -> torch.Tensor:
    """
    Main ESPIRiT sensitivity map estimation algorithm.
    
    Parameters:
        k_rpsc: Multi-channel k-space data (n_read, n_phase, n_slice, n_channels)
        kernel_size: K-space kernel size (even number recommended)
        num_ac_lines: Calibration region size
        rank_fraction_ac_matrix: Auto-calibration matrix rank threshold
        eigenvalue_cutoff: Eigenvalue threshold for map selection
        device: Computation device
    
    Returns:
        ESPIRiT sensitivity maps (n_maps, n_read, n_phase, n_slice, n_channels)
    
    Algorithm Steps:
        1. Extract calibration region from k-space center
        2. Build Hankel matrix from calibration data
        3. Perform SVD and select significant singular values
        4. Extract GRAPPA-like kernels
        5. Transform kernels to image space
        6. Point-wise SVD for eigenvalue decomposition
        7. Select eigenvalues > threshold for sensitivity maps
    """
```

#### Other Utility Functions

```python
def get_hankel_indices(nx: int, ny: int, nz: int, kernel_size: int) -> torch.Tensor:
    """Generate indexing for Hankel matrix construction"""

def built_hankel(k_ac: torch.Tensor, kernel_size: int) -> torch.Tensor:
    """Construct Hankel matrix from calibration region"""

def get_ker_imgs(kernels: torch.Tensor, kernel_size: int) -> torch.Tensor:
    """Convert k-space kernels to image space"""
```

### Sensitivity Map Properties

- **First map**: Typically has eigenvalue ≈ 1 (SENSE-like)
- **Additional maps**: Capture coil geometry variations
- **Number of maps**: Equals number of significant singular values
- **Use cases**: Parallel imaging, coil compression, artifact correction

### Common Use Cases

**Scenario 1: SENSE Reconstruction**

```python
# Get first map for SENSE-like reconstruction
sense_map = sensitivity_maps[0].conj()  # Shape: [read, phase, slice, channels]

# Apply coil combination
coil_combined = torch.sum(img_k * sense_map, dim=-1)  # SENSE image
```

**Scenario 2: Coil Compression**

```python
# Use first N sensitivity maps for compression
n_compressed = 4
sens_weights = sensitivity_maps[:n_compressed]

# Compress data
compressed_data = torch.einsum('rcxy,mxy->mxyc', img_k, sens_weights.conj())
```

**Scenario 3: Artifact Analysis**

```python
# Analyze different sensitivity maps
for i in range(sensitivity_maps.shape[0]):
    map_i = sensitivity_maps[i].abs()
    print(f"Map {i}: Mean magnitude = {map_i.mean():.4f}")
    
    # Visualize individual maps
    import matplotlib.pyplot as plt
    plt.imshow(map_i[..., 0, 0].cpu().numpy(), cmap='jet')
    plt.colorbar()
    plt.title(f'ESPIRiT Map {i}')
    plt.show()
```

## Configuration System

All modeling modules use the `BaseClass` configuration system with CLI integration:

```python
@dataclass
class BaseClass(Serializable):
    config_file: str = ""          # Config file path
    out_path: str = ""             # Output directory
    use_gpu: bool = True           # GPU acceleration
    gpu_device: int = 0            # GPU device ID
    visualize: bool = True         # Visualization toggle
    debug: bool = False            # Debug mode
    slurm: bool = False           # Slurm mode
```

### Common Configuration Pattern

```python
# Load from JSON file
settings = MyConfig.from_json("config.json")

# Load from CLI
settings = MyConfig.from_cli(args, parser)

# Manual construction
settings = MyConfig(param1=value1, param2=value2)
```

## Mathematical Summary

| Algorithm | Model | Keys | Output |
|-----------|-------|------|--------|
| AFI B1+ | r = S₁/S₂, α = arccos((r·n-1)/(n-r)) | TR ratio, flip angle | B1+ map (%) |
| Dictionary | min‖S_voxel - S_DB‖² | Database, search ranges | T₁, T₂, B1⁺ |
| Mono-exp | S(t) = S₀·exp(-R·t) | Echo times | S₀, T₂ = 1/R |
| ESPIRiT | H = UΣVᴴ, S_maps = FFT(V*) | Kernel size, AC region | Sensitivity maps |

## See Also

- [API Reference](api.md#modeling) - Complete class and method documentation
- [Simulation](simulation.md) - Generate EMC databases for dictionary fitting
- [Configuration System](../contribute/writing_code.md#configuration) - How to use BaseClass configuration
- [GPU Acceleration](#performance-optimization) - Memory and performance considerations