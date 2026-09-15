# Modeling Tutorial

This tutorial will guide you through using PyMRITools modeling tools for parameter estimation from MRI data, including B1+ mapping, dictionary fitting, and mono-exponential decay fitting.

## Prerequisites

- PyMRITools installed
- Multi-echo MRI data (for fitting)
- AFI echo images (for B1+ mapping)
- Basic understanding of MRI parameter estimation

## B1+ AFI Mapping Tutorial

### Understanding AFI Method

AFI (Actual Flip-angle Imaging) uses two acquisitions with different TR values to measure B1+ field inhomogeneity:
- **First acquisition**: Short TR₁
- **Second acquisition**: Long TR₂ (typically 5× TR₁)

The flip angle is calculated from the signal ratio, then converted to B1+ scaling factor.

### Step 1: Load AFI Data

```python
import torch
import numpy as np

# Load AFI echo images (shape: [..., 2] for two echoes)
afi_data = torch.load("afi_echoes.pt")

print(f"AFI data shape: {afi_data.shape}")
# Should be [..., spatial_dims..., 2]
```

### Step 2: Configure B1+ Mapping

```python
from pymritools.modeling.b1_afi.modeling import Settings

settings = Settings(
    input_file="afi_echoes.nii",
    flip_angle=60.0,                      # Set flip angle in degrees
    ratio_tr2_tr1=5,                       # TR₂/TR₁ ratio (common: 5-6)
    smoothing_kernel=3.0,                 # Gaussian smoothing kernel size
    noise_mask="background.nii.gz",       # Noise regions for error estimation
    out_path="./b1_output"
)
```

### Step 3: Run B1+ Estimation

```python
from pymritools.modeling.b1_afi.modeling import processing

# Run full processing pipeline
processing(settings)

# Output files saved to ./b1_output/:
# - b1_map.nii.gz: B1+ percentage map
# - b1_error.nii.gz: B1+ error map
# - alpha_map.nii.gz: Flip angle map
```

### Step 4: Analyze Results and Visualize

```python
import matplotlib.pyplot as plt
from pymritools.utils import nifti_load

# Load results
b1_map, img = nifti_load("b1_output/b1_map.nii.gz")
b1_error, _ = nifti_load("b1_output/b1_error.nii.gz")

# Create figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# B1+ map
im1 = ax1.imshow(b1_map.T, cmap='RdBu_r', origin='lower', vmin=80, vmax=120)
ax1.set_title('B1+ Map (%)')
ax1.axis('off')
cbar1 = plt.colorbar(im1, ax=ax1)
cbar1.set_label('B1+ (%)')

# Error map
im2 = ax2.imshow(b1_error.T, cmap='hot', origin='lower')
ax2.set_title('B1+ Error Map')
ax2.axis('off')
cbar2 = plt.colorbar(im2, ax=ax2)
cbar2.set_label('Error (%)')

plt.tight_layout()
plt.show()
```

## Dictionary Fitting Tutorial

### Understanding Dictionary Matching

Dictionary matching uses pre-computed signal patterns (from EMC simulation) to estimate tissue parameters by finding the best match in a database.

### Step 1: Load Data and Database

```python
import torch

# Load multi-echo data (shape: [spatial..., temporal])
eme_data = torch.load("emc_16echo.nii.gz")
print(f"EMC data shape: {eme_data.shape}")

# Load pre-computed EMC database
database = torch.load("emc_database.pkl")
print(f"Database shape: {database.shape}")
# Expected: [N_T1, N_T2, N_B1, ETL]
```

### Step 2: Configure Dictionary Fitting

```python
from pymritools.config.emc import EmcFitSettings

settings = EmcFitSettings(
    input_data="emc_data.nii",
    input_database="database.pkl",
    input_b1="b1_map.nii",           # Optional B1+ regularization
    batch_size=3000,                   # Process 3000 voxels per batch
    b1_error_cutoff=0.15,            # Filter unreliable B1+ values
    rsos_channel_combine=False,       # Channel combination method
    out_name="r2_fit",
    out_path="./fitting_output"
)
```

### Step 3: Run Dictionary Fitting

```python
from pymritools.modeling.dictionary.grid_search import fit

# Run fitting pipeline
fit(settings)

# Output files:
# - r2_fit_t2.nii.gz: T2 map [seconds]
# - r2_fit_b1.nii.gz: B1+ map [percentage]
# - r2_fit_l2.nii.gz: L2 error map
```

### Step 4: Access Fitting Parameters

```python
import matplotlib.pyplot as plt
from pymritools.utils import nifti_load

# Load results
t2_map, _ = nifti_load("fitting_output/r2_fit_t2.nii.gz")
b1_map, _ = nifti_load("fitting_output/r2_fit_b1.nii.gz")

# Convert T2 to ms
t2_ms = t2_map * 1000

# Create figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# T2 map
im1 = ax1.imshow(t2_ms.T, cmap='jet', origin='lower', vmin=30, vmax=150)
ax1.set_title('T₂ Map [ms]')
ax1.axis('off')
cbar1 = plt.colorbar(im1, ax=ax1)
cbar1.set_label('T₂ [ms]')

# B1+ map
im2 = ax2.imshow(b1_map.T, cmap='RdBu_r', origin='lower', vmin=80, vmax=120)
ax2.set_title('B1+ Map [%]')
ax2.axis('off')
cbar2 = plt.colorbar(im2, ax=ax2)
cbar2.set_label('B1+ [%]')

plt.tight_layout()
plt.show()
```

### Alternative: Direct Function Call

```python
from pymritools.modeling.dictionary.grid_search import fit_r2_fn
import torch

# Prepare data
input_data = torch.randn(128, 128, 16)  # [spatial, temporal]
database = torch.randn(8, 12, 5, 16)   # [T1, T2, B1, temporal]
t2_vals = torch.linspace(30, 120, 20)  # [ms]
b1_vals = torch.linspace(0.6, 1.4, 5)

# Direct fitting
t2_map, b1_map, l2_err = fit_r2_fn(
    input_data=input_data.unsqueeze(0).unsqueeze(0),  # Add batch and coil dims
    db_mag=database,
    t2_vals=t2_vals,
    b1_vals=b1_vals,
    batch_size=1000
)

print(f"T2 map shape: {t2_map.shape}")
print(f"B1 map shape: {b1_map.shape}")
print(f"L2 error shape: {l2_err.shape}")
```

## Mono-exponential Decay Fitting Tutorial

### Understanding Mono-exponential Model

The mono-exponential decay model: S(t) = S₀·exp(-R·t) where:
- S₀ is baseline signal
- R is decay rate (1/T₂)
- T₂ = 1/R [seconds]

### Step 1: Load Multi-echo Data

```python
import torch

# Load multi-echo data (shape: [spatial..., temporal])
echo_data = torch.load("multi_echo_data.pt")
print(f"Echo data shape: {echo_data.shape}")

# Define echo times in seconds
echo_times = torch.tensor([0.01, 0.02, 0.03, 0.04, 0.05])
print(f"Echo times: {echo_times}")
```

### Step 2: Configure Mono-exponential Fitting

```python
from pymritools.modeling.decay.mexp import MEXPSettings

settings = MEXPSettings(
    input_data="multi_echo.nii",
    echo_times=[0.01, 0.02, 0.03, 0.04, 0.05],  # [seconds]
    out_path="./mexp_output"
)
```

### Step 3: Run Fitting

```python
from pymritools.modeling.decay.mexp import process

# Run fitting pipeline
process(settings)

# Output files:
# - s0_map.nii.gz: Baseline signal map
# - r_map.nii.gz: Decay rate map
# - t2_map.nii.gz: T₂ values (T₂ = 1/R)
```

### Step 4: Analyze Results

```python
import matplotlib.pyplot as plt
from pymritools.utils import nifti_load

# Load results
s0_map, _ = nifti_load("mexp_output/s0_map.nii.gz")
r_map, _ = nifti_load("mexp_output/r_map.nii.gz")

# Calculate T₂
t2_map = 1.0 / r_map

# Create figure
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

# S₀ map
im1 = ax1.imshow(s0_map.T, cmap='gray', origin='lower')
ax1.set_title('S₀ Map')
ax1.axis('off')
cbar1 = plt.colorbar(im1, ax=ax1)

# R map
im2 = ax2.imshow(r_map.T, cmap='jet', origin='lower')
ax2.set_title('R Map [1/s]')
ax2.axis('off')
cbar2 = plt.colorbar(im2, ax=ax2)

# T₂ map
im3 = ax3.imshow(t2_map.T, cmap='jet', origin='lower', vmin=20, vmax=150)
ax3.set_title('T₂ Map [ms]')
ax3.axis('off')
cbar3 = plt.colorbar(im3, ax=ax3)

plt.tight_layout()
plt.show()
```

### Alternative: Direct Fitting with Custom Echo Times

```python
from pymritools.modeling.decay.mexp import fit

# Simulate test data
t = torch.tensor([0.01, 0.02, 0.03, 0.04, 0.05])  # Echo times [seconds]
s0_true = 100.0
r_true = 20.0  # R = 1/T₂
data = s0_true * torch.exp(-r_true * t)

# Add spatial dimensions
data_3d = data.view(1, 1, 1, -1).expand(64, 64, 1, -1)

# Fit
s0_est, r_est = fit(data_3d, t)

# Calculate T₂
t2_est = 1.0 / r_est

print(f"True S₀: {s0_true:.2f}, Estimated: {s0_est.mean():.2f}")
print(f"True R: {r_true:.2f}, Estimated: {r_est.mean():.2f}")
print(f"True T₂: {1/r_true:.4f}s, Estimated: {t2_est.mean():.4f}s")
```

## Common Workflows

### Workflow 1: Complete Parameter Estimation

```python
from pymritools.modeling.b1_afi.modeling import processing as b1_process
from pymritools.modeling.dictionary.grid_search import fit
from pymritools.modeling.decay.mexp import process as mexp_process

# Step 1: B1+ mapping
b1_settings = Settings(input_file="afi.nii", flip_angle=60.0, ratio_tr2_tr1=5)
b1_process(b1_settings)

# Step 2: Dictionary fitting (using B1+ map)
fit_settings = EmcFitSettings(
    input_data="emc_data.nii",
    input_database="database.pkl",
    input_b1="b1_output/b1_map.nii.gz",
    b1_error_cutoff=0.15
)
fit(fit_settings)

# Step 3: Optional: Mono-exponential T₂ fitting
mexp_settings = MEXPSettings(input_data="multi_echo.nii", 
                              echo_times=[0.01, 0.02, 0.03, 0.04, 0.05])
mexp_process(mexp_settings)
```

### Workflow 2: Simulation + Dictionary Fitting

```python
# Step 1: Run EMC simulation to create database
from pymritools.config.emc import EmcParameters, EmcSimSettings
from pymritools.simulation.emc.sequence import MESE

# Configure simulation
params = EmcParameters(etl=16, esp=9.0)
sim_settings = EmcSimSettings(
    t1_list=[1.0, 1.5, 2.0],
    t2_list=[[30, 60, 5.0]],
    b1_list=[0.8, 1.0, 1.2],
    database_name="custom_database.pkl"
)

# Run simulation
mese_sim = MESE(params=params, settings=sim_settings)
mese_sim.simulate()
db = mese_sim.get_db()

# Step 2: Use database for fitting
fit_settings = EmcFitSettings(
    input_data="test_data.nii",
    input_database="custom_database.pkl",
    batch_size=2000
)
fit(fit_settings)
```

### Workflow 3: Parameter Error Analysis

```python
# Access fitting quality metrics
l2_err, _ = nifti_load("fitting_output/r2_fit_l2.nii.gz")

# Calculate statistics
import numpy as np
print(f"Mean L² error: {l2_err.mean():.4f}")
print(f"Median L² error: {np.median(l2_err):.4f}")
print(f"Standard deviation: {l2_err.std():.4f}")

# Visualize error distribution
plt.figure()
plt.hist(l2_err.flatten(), bins=50, edgecolor='black')
plt.xlabel('L² Error')
plt.ylabel('Frequency')
plt.title('Fitting Error Distribution')
plt.show()

# Mask out high-error voxels
error_mask = l2_err < np.percentile(l2_err, 95)
print(f"High-error voxels: {(error_mask == 0).sum()} ({(error_mask == 0).sum()/error_mask.numel()*100:.1f}%)")
```

## Parameter Optimization

### Optimizing Echo times for T₂ Estimation

```python
import numpy as np

# Rule of thumb: echo times should span up to 3× expected T₂
expected_t2 = 0.08  # 80 ms
max_echo_time = 3 * expected_t2

# Generate different echo time schemes
echo_schemes = [
    np.linspace(0.01, 0.06, 5),         # 5 echoes, up to 60 ms
    np.linspace(0.01, 0.12, 8),         # 8 echoes, up to 120 ms
    np.linspace(0.01, 0.18, 10),        # 10 echoes, up to 180 ms
    np.linspace(0.01, 0.24, 12),        # 12 echoes, up to 240 ms
]

for i, echo_times in enumerate(echo_schemes):
    settings = MEXPSettings(
        input_data="multi_echo.nii",
        echo_times=echo_times.tolist(),
        out_path=f"./mexp_scheme_{i}"
    )
    process(settings)
    # Compare T₂ maps from different schemes
```

### Database Parameter Ranges

```python
from pymritools.config.emc import EmcSimSettings

# Wide parameter range for general fitting:
settings = EmcSimSettings(
    t1_list=[0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.7, 1.9, 2.1, 2.3, 2.5, 2.7, 2.9, 3.1, 3.3, 3.5],
    t2_list=[[20, 150, 5.0]],  # Single wide T2 range
    b1_list=[0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0],
)

# Focused parameter range for specific application:
settings_focused = EmcSimSettings(
    t1_list=[0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0],
    t2_list=[
        [30, 50, 2.0],   # Short T₂: brain tissue
        [60, 80, 2.0],   # Medium T₂: gray matter
        [90, 110, 2.0],  # Long T₂: CSF
    ],
    b1_list=[0.8, 1.0, 1.2],  # Narrow B1+ range
)
```

## Troubleshooting

### Common Issues

**Issue 1: B1+ Map Shows Unrealistic Values**
```python
# Check TR ratio is correct
print(f"Using TR ratio: {settings.ratio_tr2_tr1}")
# Suggested: 4-8 for 60° flip angle

# Check flip angle is correctly specified
print(f"Set flip angle: {settings.flip_angle}°")
# Suggested: 50-70° for most AFI protocols
```

**Issue 2: Dictionary Fitting Produces Errors**
```python
# Check database dimensions match expected format
print(f"Database shape: {database.shape}")
# Should be: [N_T1, N_T2, N_B1, ETL]

# Check input data dimensions
print(f"Input data shape: {input_data.shape}")
# Should have: [spatial..., temporal] where temporal == ETL

# Reduce batch size for GPU memory
settings.batch_size = 1000  # Try smaller batch size
```

**Issue 3: Mono-exponential Fitting Gives Negative T₂**
```python
# This can happen if data includes noise or negative values
# Apply magnitude operation
data_mag = torch.abs(data)

# Or filter low signal areas
mask = reduce(data, 'sum(-1)') > threshold
data_masked = data * mask
```

## CLI Usage Examples

### B1+ AFI Mapping
```bash
python -m pymritools.modeling.b1_afi.modeling \
    -i afi_echoes.nii \
    -fa 60.0 \
    -n 5 \
    -s 3.0 \
    -m noise_mask.nii \
    -o ./b1_output
```

### Dictionary Fitting
```bash
python -m pymritools.modeling.dictionary.grid_search \
    -i emc_data.nii \
    -db database.pkl \
    -b1 b1_map.nii \
    -bs 3000 \
    -o ./fitting_output
```

### MEGESSE R2/R2* Fitting
```bash
python -m pymritools.modeling.dictionary.grid_search --grid_combined_r2_r2s_search \
    -i megesse_data.nii \
    -db database.pkl \
    -b0 b0_map.nii \
    -o ./megesse_output
```

### Mono-exponential Fitting
```bash
python -m pymritools.modeling.decay.mexp \
    -i multi_echo.nii \
    -te 0.01 0.02 0.03 0.04 0.05 \
    -o ./mexp_output
```

## Next Steps

- Try different fitting parameters and compare results
- Use simulation to create custom databases for your specific applications
- Combine B1+ regularization with dictionary fitting for improved accuracy
- Explore ESPIRiT for coil sensitivity estimation
- Read the [Modeling documentation](modeling.md) for detailed theory
- Check the [API Reference](api.md#modeling) for complete class reference

This tutorial provides practical guidance for using PyMRITools modeling tools to estimate MRI parameters from your data.