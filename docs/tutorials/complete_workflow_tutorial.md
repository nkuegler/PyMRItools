# Complete Workflow Tutorial

This tutorial demonstrates how to combine different PyMRITools modules in a complete MRI analysis pipeline from raw data to final parameter maps.

## Overview

The complete pipeline includes:

1. **Data Preprocessing**: Coil compression and sensitivity estimation
2. **Denoising**: LCPCA denoising with noise bias correction
3. **Reconstruction**: LORAKS reconstruction from undersampled data
4. **Parameter Estimation**: Dictionary fitting for T₂ mapping
5. **Quality Control**: Error analysis and validation

## Prerequisites

- PyMRITools installed
- Multi-channel k-space data with appropriate sampling
- Understanding of each module's role in the pipeline

## Complete Pipeline Example

### Step 1: Data Loading and Preprocessing

```python
import torch
import numpy as np
from pymritools.utils import nifti_load, nifti_save

# Load raw k-space data (e.g., from scanner)
# Shape: [n_read, n_phase, n_slice, n_coils, n_echoes]
k_space_raw, img_info = nifti_load("multi_echo_raw_data.nii.gz")

print(f"Raw data shape: {k_space_raw.shape}")

# Step 1a: Check data format
if len(k_space_raw.shape) == 5:
    print("Data format: [n_read, n_phase, n_slice, n_coils, n_echoes]")
elif len(k_space_raw.shape) == 4:
    print("Data format: [n_read, n_phase, n_slice, n_coils]")
    # Add echo dimension if needed
    k_space_raw = k_space_raw.unsqueeze(-1)  # [..., n_coils, 1]
else:
    raise ValueError(f"Unexpected data shape: {k_space_raw.shape}")

# Step 1b: Check data type (should be complex)
if not torch.is_complex(k_space_raw):
    print("Warning: Data is real-valued, converting to complex")
    # Assume real part is magnitude
    k_space_raw = k_space_raw.astype(torch.complex64)
```

### Step 2: Channel Compression

```python
from pymritools.processing.coil_sens import compress_channels_2d
from pymritools.utils import torch_save

# Create sampling pattern (assuming GRAPPA 2x acceleration)
n_phase = k_space_raw.shape[1]
n_coils = k_space_raw.shape[3]

sampling_mask = torch.ones(k_space_raw.shape, dtype=torch.bool)
# Skip every other phase line for 2x acceleration
sampling_mask[:, 1::2, :, :, :] = False

# Compress from 32 coils to 12 coils
print(f"Compressing from {n_coils} coils to 12 coils...")

k_space_compressed = compress_channels_2d(
    input_k_space=k_space_raw,
    sampling_pattern=sampling_mask,
    num_compressed_channels=12,
    use_ac_data=True,              # Use ACS region
    visualize=True,
    device='cuda:0'
)

print(f"Compressed shape: {k_space_compressed.shape}")

# Save compressed data
torch_save(k_space_compressed, "k_compressed_12ch.pt")
```

### Step 3: Coil Sensitivity Estimation

```python
from pymritools.config.processing.coil_sens import CoilSensConfig
from pymritools.processing.coil_sens import smap

# Estimate sensitivity maps for first echo
echo_data = k_space_compressed[:, :, :, :, 0]  # First echo only
echo_np = echo_data.cpu().numpy()

# Save temporary NIfTI for sensitivity estimation
nifti_save(
    echo_np,
    img_info.header.get_best_affine(),
    path_to_dir="./temp",
    file_name="echo1_for_smap"
)

# Configure sensitivity estimation
sens_settings = CoilSensConfig(
    input_data="./temp/echo1_for_smap.nii.gz",
    input_in_img_space=False,  # Input is k-space
    coil_dimension=3,          # Channel is 4th dimension
    smoothing_kernel=12,
    out_path="./sensitivity_output"
)

# Run sensitivity estimation
smap(sens_settings)

# Load sensitivity maps
sensitivity_maps = []
for i in range(12):
    try:
        smap_i, _ = nifti_load(f"./sensitivity_output/smap_{i:02d}.nii.gz")
        sensitivity_maps.append(torch.from_numpy(smap_i))
    except:
        print(f"Warning: Could not load sensitivity map for coil {i}")
        # Create uniform map as fallback
        sensitivity_maps.append(torch.ones_like(echo_np, dtype=torch.complex64))

sensitivity_maps = torch.stack(sensitivity_maps, dim=-1)  # [n_read, n_phase, n_slice, n_coils]
print(f"Sensitivity maps loaded: {sensitivity_maps.shape}")
```

### Step 4: LCPCA Denoising

```python
from pymritools.processing.denoising.lcpca import denoise_lcpca

# Denoise each echo separately
denoised_echoes = []

for echo_idx in range(k_space_compressed.shape[-1]):
    echo_data = k_space_compressed[:, :, :, :, echo_idx]
    
    # Convert to image space for denoising
    from pymritools.utils import fft_to_img
    echo_img = fft_to_img(echo_data)
    
    # Denoise with automatic component selection
    denoised_img, _, _ = denoise_lcpca(
        input_data=echo_img,
        p=0,  # Use MP inequality automatic selection
        noise_bias_correction=True,
        visualize=echo_idx == 0  # Show progress for first echo only
    )
    
    # Convert back to k-space
    from pymritools.utils import ifft_to_k
    denoised_k = ifft_to_k(denoised_img)
    
    denoised_echoes.append(denoised_k)

# Stack denoised echoes
k_space_denoised = torch.stack(denoised_echoes, dim=-1)
print(f"Denoised k-space shape: {k_space_denoised.shape}")

# Save denoised data
torch_save(k_space_denoised, "k_denoised_12ch.pt")
```

### Step 5: LORAKS Reconstruction

```python
from pymritools.recon.loraks.loraks import Loraks, LoraksOptions
from pymritools.recon.loraks.ac_loraks import AcLoraksOptions
from pymritools.recon.loraks.loraks import (
    OperatorType, RankReduction, RankReductionMethod, SolverType, NullspaceAlgorithm
)
from pymritools.utils import ifft_to_k

# Reconstruct each echo with AC-LORAKS
reconstructed_echoes = []

for echo_idx in range(k_space_denoised.shape[-1]):
    echo_k = k_space_denoised[:, :, :, :, echo_idx]
    
    # Configure for 2x acceleration
    options = AcLoraksOptions(
        loraks_neighborhood_size=5,
        loraks_matrix_type=OperatorType.S,
        rank=RankReduction(
            method=RankReductionMethod.HARD_CUTOFF,
            value=100
        ),
        regularization_lambda=0.0,
        max_num_iter=20,
        solver_type=SolverType.LEASTSQUARES,
        nullspace_algorithm=NullspaceAlgorithm.EIGH,
        device='cuda:0'
    )
    
    # Reconstruct
    loraks = Loraks.create(options)
    k_reconstructed = loraks.reconstruct(echo_k)
    
    reconstructed_echoes.append(k_reconstructed)

# Stack reconstructed echoes
k_reconstructed = torch.stack(reconstructed_echoes, dim=-1)
print(f"Reconstructed k-space shape: {k_reconstructed.shape}")

# Save reconstructed data
torch_save(k_reconstructed, "k_reconstructed_12ch.pt")

# Convert first echo to image for quality check
from pymritools.utils import fft_to_img, root_sum_of_squares

recon_img = fft_to_img(k_reconstructed[:, :, :, :, 0])
recon_combined = root_sum_of_squares(recon_img, coil_dim=-1)

# Save first reconstructed image
nifti_save(
    recon_combined.cpu().numpy(),
    img_info.header.get_best_affine(),
    path_to_dir="./output",
    file_name="first_echo_loraks"
)
```

### Step 6: Coil Combination

```python
# Combine coils across all echoes
combined_echoes = []

for echo_idx in range(k_reconstructed.shape[-1]):
    echo_k = k_reconstructed[:, :, :, :, echo_idx]
    
    # Option 1: Use sensitivity maps
    if sensitivity_maps is not None and sensitivity_maps.shape[-1] == echo_k.shape.pop(-1):
        # Apply sensitivity maps
        weights = sensitivity_maps.conj()
        combined = torch.sum(echo_k * weights, dim=-1)
    else:
        # Fallback: Root sum-of-squares
        combined = root_sum_of_squares(echo_k, coil_dim=-1)
    
    combined_echoes.append(combined)

# Stack combined echoes
k_combined = torch.stack(combined_echoes, dim=-1)
print(f"Combined k-space shape: {k_combined.shape}")

# Save combined data
torch_save(k_combined, "k_combined_singlech.pt")
```

### Step 7: Dictionary Fitting for T₂ Mapping

```python
from pymritools.modeling.dictionary.grid_search import fit, fit_r2_fn
from pymritools.config.emc import EmcFitSettings
from pymritools.utils import nifti_load

# Convert combined k-space to image domain
echo_images = []
for echo_idx in range(k_combined.shape[-1]):
    echo_img_ext = ifft_to_k(k_combined[:, :, :, :, echo_idx])
    echo_images.append(echo_img_ext.abs().cpu().numpy())

# Stack echoes as [n_read, n_phase, n_slice, n_echoes]
data_4d = np.stack(echo_images, axis=-1)

# Save for fitting
nifti_save(
    data_4d,
    img_info.header.get_best_affine(),
    path_to_dir="./fitting_input",
    file_name="multi_echo_loraks"
)

# Configure dictionary fitting
fit_settings = EmcFitSettings(
    input_data="./fitting_input/multi_echo_loraks.nii.gz",
    input_database="emc_database_16echo.pkl",
    batch_size=2000,
    out_name="loraks_fit",
    out_path="./fitting_output"
)

# Run dictionary fitting
print("Running dictionary fitting...")
fit(fit_settings)
print("Dictionary fitting complete!")

# Load results
t2_map, _ = nifti_load("fitting_output/loraks_fit_t2.nii.gz")
b1_map, _ = nifti_load("fitting_output/loraks_fit_b1.nii.gz")

print(f"T2 map shape: {t2_map.shape}")
print(f"T2 range: {t2_map.min()*1000:.1f} to {t2_map.max()*1000:.1f} ms")
```

### Step 8: Quality Control and Visualization

```python
import matplotlib.pyplot as plt

# Visualize results
fig = plt.figure(figsize=(16, 12))

# Reconstructed images
for i in range(min(4, k_reconstructed.shape[-1])):
    recon_img = fft_to_img(k_reconstructed[:, :, :, :, i])
    recon_combined = root_sum_of_squares(recon_img, coil_dim=-1)
    
    ax = fig.add_subplot(3, 4, i+1)
    ax.imshow(recon_combined.cpu().numpy(), cmap='gray', origin='lower')
    ax.set_title(f'Echo {i+1} (LORAKS 2x)')
    ax.axis('off')

# T₂ map
ax = fig.add_subplot(3, 4, 5)
im1 = ax.imshow(t2_map.T * 1000, cmap='jet', origin='lower', vmin=20, vmax=150)
ax.set_title(f'T₂ Map [ms]\nRange: {t2_map.min()*1000:.1f}-{t2_map.max()*1000:.1f} ms')
ax.axis('off')
plt.colorbar(im1, ax=ax)

# B1+ map (if available)
if b1_map is not None:
    ax = fig.add_subplot(3, 4, 6)
    im2 = ax.imshow(b1_map.T, cmap='RdBu_r', origin='lower', vmin=80, vmax=120)
    ax.set_title('B1+ Map [%]')
    ax.axis('off')
    plt.colorbar(im2, ax=ax)

plt.tight_layout()
plt.savefig('./output/complete_workflow.png', dpi=150, bbox_inches='tight')
plt.show()

# Quality metrics
print("\nQuality Metrics:")
print(f"T₂ map statistics:")
print(f"  Mean: {t2_map.mean()*1000:.2f} ms")
print(f"  Std:  {t2_map.std()*1000:.2f} ms")
print(f"  Median: {np.median(t2_map)*1000:.2f} ms")

# Check for outliers
t2_outliers = (t2_map < 0.02) | (t2_map > 0.15)
print(f"  Out-of-range voxels: {t2_outliers.sum()} ({t2_outliers.sum()/t2_outliers.numel()*100:.1f}%)")
```

## Optimization Strategies

### Memory Management

```python
# Process data in chunks if memory is limited
chunk_size = 2  # Process 2 echoes at a time

for start_echo in range(0, k_denoised.shape[-1], chunk_size):
    end_echo = min(start_echo + chunk_size, k_denoised.shape[-1])
    
    # Process chunk
    chunk_data = k_denoised[:, :, :, :, start_echo:end_echo]
    
    # Reconstruct chunk
    options = AcLoraksOptions(loraks_neighborhood_size=5)
    loraks = Loraks.create(options)
    k_recon_chunk = loraks.reconstruct(chunk_data)
    
    # Save intermediate results
    torch_save(k_recon_chunk, f"recon_echoes_{start_echo}_{end_echo}.pt")
```

### Parallel Processing

```python
import concurrent.futures

def process_single_echo(echo_idx, k_space_denoised, options):
    """Process single echo for parallel execution"""
    echo_k = k_space_denoised[:, :, :, :, echo_idx]
    loraks = Loraks.create(options)
    return loraks.reconstruct(echo_k)

# Process echoes in parallel
options = AcLoraksOptions(loraks_neighborhood_size=5)
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
    future_to_echo = {
        executor.submit(process_single_echo, idx, k_denoised, options): idx
        for idx in range(k_denoised.shape[-1])
    }
    
    reconstructed_echoes = []
    for future in concurrent.futures.as_completed(future_to_echo):
        echo_idx = future_to_echo[future]
        k_recon = future.result()
        reconstructed_echoes.append((echo_idx, k_recon))

# Sort by echo index
reconstructed_echoes.sort(key=lambda x: x[0])
k_reconstructed = torch.stack([x[1] for x in reconstructed_echoes], dim=-1)
```

## Pipeline Validation

### Comparing Reconstruction Methods

```python
# Compare AC-LORAKS vs P-LORAKS
from pymritools.recon.loraks.p_loraks import PLoraks

# AC-LORAKS reconstruction
ac_options = AcLoraksOptions(
    loraks_neighborhood_size=5,
    rank=RankReduction(method=HARD_CUTOFF, value=100)
)
ac_loraks = Loraks.create(ac_options)
k_ac = ac_loraks.reconstruct(k_denoised[:, :, :, :, 0])

# P-LORAKS reconstruction
p_loraks = (PLoraks()
    .with_patch_shape((-1, -1, -1, 7, 7))
    .with_s_matrix()
    .with_sv_hard_cutoff(rank=100)
    .with_device('cuda:0'))
k_p = p_loraks.reconstruct(k_denoised[:, :, :, :, 0])

# Compare
ac_img = root_sum_of_squares(fft_to_img(k_ac), coil_dim=-1)
p_img = root_sum_of_squares(fft_to_k(k_p), coil_dim=-1)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
ax1.imshow(ac_img.cpu().numpy(), cmap='gray', origin='lower')
ax1.set_title('AC-LORAKS Reconstruction')
ax1.axis('off')

ax2.imshow(p_img.cpu().numpy(), cmap='gray', origin='lower')
ax2.set_title('P-LORAKS Reconstruction')
ax2.axis('off')

plt.tight_layout()
plt.show()

# Calculate differences
diff = torch.abs(ac_img - p_img)
print(f"Mean absolute difference: {diff.mean():.4e}")
print(f"Max absolute difference: {diff.max():.4e}")
```

### Parameter Sensitivity Analysis

```python
# Analyze how reconstruction parameters affect results
neighborhood_sizes = [3, 5, 7, 9]

t2_maps = []
for nb_size in neighborhood_sizes:
    options = AcLoraksOptions(
        loraks_neighborhood_size=nb_size,
        rank=RankReduction(method=HARD_CUTOFF, value=80 + nb_size*10)
    )
    loraks = Loraks.create(options)
    k_recon = loraks.reconstruct(k_denoised[:, :, :, :, 0])
    
    # Convert to image space
    recon_img = root_sum_of_squares(fft_to_k(k_recon), coil_dim=-1)
    
    # Save for comparison (skip fitting for speed)
    torch_save(recon_img, f"recon_nb{nb_size}.pt")

# Compare reconstruction quality
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
for i, nb_size in enumerate(neighborhood_sizes):
    img, _ = nifti_load(f"recon_nb{nb_size}.pt", return_img=True)
    ax = axes[i//2, i%2]
    ax.imshow(img, cmap='gray', origin='lower')
    ax.set_title(f'Neighborhood Size: {nb_size}')
    ax.axis('off')

plt.tight_layout()
plt.show()
```

## Common Error Scenarios

### Handling Missing Files

```python
import os

# Check for database, create if missing
if not os.path.exists("emc_database_16echo.pkl"):
    print("Database not found, creating minimal database...")
    
    # Create simple database using first echo data
    from pymritools.modeling.dictionary.grid_search import create_simple_database
    
    # Use first echo to create T2 database
    first_echo = fft_to_k(k_combined[:, :, :, :, 0])
    simple_db = create_simple_database(
        echo_data=first_echo,
        echo_times=[0.01, 0.02, 0.03, 0.04, 0.05],
        t2_range=[30, 120],  # [ms]
        b1_range=[0.8, 1.2],
        database_name="simple_database.pkl"
    )
    
    fit_settings.input_database = "simple_database.pkl"
else:
    fit_settings.input_database = "emc_database_16echo.pkl"
```

### Data Shape Adjustments

```python
# Helper function to ensure correct data format
def ensure_k_space_format(data, expected_dims):
    """Ensure k-space data has expected dimensions"""
    
    # Remove extra dimensions
    while len(data.shape) > expected_dims:
        data = data.squeeze()
    
    # Add missing dimensions if needed
    while len(data.shape) < expected_dims:
        data = data.unsqueeze(0)
    
    # Ensure complex type
    if not torch.is_complex(data):
        data = data.astype(torch.complex64)
    
    return data

# Use in pipeline
k_space_formatted = ensure_k_space_format(k_space_raw, expected_dims=5)
print(f"Formatted data shape: {k_space_formatted.shape}")
```

## CLI Pipeline Orchestration

### Create Shell Script

```bash
#!/bin/bash
# complete_pipeline.sh - Complete MRI analysis pipeline

set -e  # Exit on error

echo "=== Starting Complete MRI Pipeline ==="

# Step 1: Coil compression
echo "Step 1: Coil compression"
python -c "
import torch
from pymritools.processing.coil_sens import compress_channels_2d
from pymritools.utils import torch_load, torch_save

k_space = torch_load('multi_echo_raw_data.pt')
sampling_mask = torch.ones(k_space.shape, dtype=torch.bool)
sampling_mask[:, 1::2, :, :, :] = False  # 2x acceleration

k_compressed = compress_channels_2d(
    k_space, sampling_mask, 12, use_ac_data=True, visualize=True, device='cuda:0'
)
torch_save(k_compressed, 'k_compressed.pt')
"

# Step 2: Sensitivity estimation  
echo "Step 2: Sensitivity estimation"
python -m pymritools.processing.coil_sens.smap \
    -i ./temp/echo1_for_smap.nii.gz \
    -o ./sensitivity_output \
    -dim 3 \
    -k 12

# Step 3: LCPCA denoising
echo "Step 3: LCPCA denoising"
python -m pymritools.processing.denoising.lcpca.denoise \
    -i multi_echo_concatenated.nii.gz \
    -o ./denoised_output \
    -p 0 \
    -nbc

# Step 4: LORAKS reconstruction
echo "Step 4: LORAKS reconstruction"
python -m pymritools.recon.loraks.recon \
    -ik k_denoised.pt \
    -p 5 \
    -rank 100 \
    -alg AC-LORAKS-ls \
    -t S \
    -mni 20

# Step 5: Dictionary fitting
echo "Step 5: Dictionary fitting"
python -m pymritools.modeling.dictionary.grid_search \
    -i multi_echo_loraks.nii.gz \
    -db emc_database_16echo.pkl \
    -bs 2000 \
    -o ./fitting_output

echo "=== Pipeline Complete ==="
echo "Results saved in ./output/"
```

## Summary

This complete pipeline demonstrates how to:

1. **Preprocess** raw multi-echo k-space data
2. **Compress** channels while preserving information
3. **Estimate** coil sensitivities for parallel imaging
4. **Denoise** using LCPCA with noise correction
5. **Reconstruct** undersampled data with LORAKS
6. **Estimate** T₂ parameters using dictionary matching
7. **Validate** results with quality metrics

The pipeline can be adapted for different acceleration factors, parameter ranges, and quality requirements.

## Next Steps

- Try different reconstruction algorithms and parameters
- Add additional processing steps (e.g., Gibbs unringing)
- Extend the pipeline for MEGESSE data
- Implement quality control and automated parameter validation
- Explore [Individual Module Tutorials](./) for detailed module-specific guidance
- Read the [API Reference](../api.md) for complete function documentation

This tutorial provides a complete, production-ready pipeline for quantitative MRI analysis using PyMRITools.