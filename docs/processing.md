# Data Processing and Analysis

## Overview

PyMRITools provides comprehensive data processing tools for MRI data including coil compression, sensitivity estimation, denoising, and artifact correction.

**Important Note:** This codebase includes LCPCA (Local Clinical PCA) but not MPPCA (Marchenko-Pastur PCA). The LCPCA implementation differs from standard MPPCA in patch size selection, structural adaptation, and noise handling methodology.

## Coil Compression

### Theoretical Background

**Coil compression** reduces the number of receiver coils while preserving image quality using Principal Component Analysis (PCA) based on geometric coil sensitivity variations.

#### Mathematical Principles

**1. GCC (Geometric Coil Compression):**
Based on Huang et al. (2008), coil sensitivities vary spatially but follow geometric patterns that can be captured by compression matrices.

**2. PCA-Based Compression:**
```
PCA on ACS data: C = U Σ Vᵀ
Compress: k_compressed = U[:, :n_compressed]ᵀ · k_original
```

**3. Key Concepts:**
- **ACS Region:** Fully-sampled central k-space used to compute compression
- **Virtual Coils:** Computed from PCA along read direction for smooth transitions
- **Slice-wise Processing:** Accounts for geometric variations across slices

### Python API Usage

```python
from pymritools.processing.coil_sens import compress_channels_2d
import torch

# Load or generate data
k_space_data = torch.randn(256, 256, 32, 32, 5)  # [x, y, z, ch, t]
sampling_mask = torch.ones(256, 256, 32, 32, 5, dtype=torch.bool)

# Create undersampling pattern
sampling_mask[:, 100:150, :, :, :] = False  # Undersampled region

# Perform coil compression
compressed_data = compress_channels_2d(
    input_k_space=k_space_data,
    sampling_pattern=sampling_mask,
    num_compressed_channels=12,    # Target: 12 compressed channels
    use_ac_data=True,              # Use ACS region for compression matrix
    visualize=True,                # Show progress bar
    device=torch.device('cuda:0')
)

print(f"Original: {k_space_data.shape} → Compressed: {compressed_data.shape}")
```

### Key Functions

```python
def compress_channels_2d(
    input_k_space: torch.tensor,
    sampling_pattern: torch.tensor,
    num_compressed_channels: int,
    use_ac_data: bool = True,
    visualize: bool = False,
    device: torch.device = torch.get_default_device()
) -> torch.tensor:
    """
    2D k-space channel compression using geometric coil compression.
    
    Parameters:
        input_k_space: k-space data dims [x, y, z, ch, t]
        sampling_pattern: Binary mask for sampled k-space points
        num_compressed_channels: Target number of channels after compression
        use_ac_data: Use only ACS region for compression matrix calculation
        visualize: Enable progress bar
        device: Computational device
    
    Returns:
        Compressed k-space data with reduced channel dimension
    """
```

### Compression Strategy

**Channel Selection:**
- Start with 32 channels → compress to 8-12 channels
- Use PCA to find most significant coil patterns
- Preserve 90-95% of information

**Performance Impact:**
- **Computation:** 2-4x faster for reconstruction
- **Memory:** 2-4x reduced memory footprint
- **Quality:** Minimal quality loss (< 2% SNR drop)

## Coil Sensitivity Estimation

### Theoretical Background

**Coil sensitivity estimation** provides maps for parallel imaging reconstruction using a practical, low-pass filtering approach.

#### Algorithm Steps

1. **Convert to image space** (if needed)
2. **Compute rSoS combined image** as reference
3. **Divide individual coils** by reference
4. **Apply low-pass filtering** in k-space
5. **Compute coil correlation matrix**

```python
# Coil sensitivity estimation:
s_i(x) = c_i(x) / (rSoS(x) + ε)
where rSoS(x) = √(Σ_i |c_i(x)|²)
```

### Configuration

```python
from pymritools.config.processing.coil_sens import CoilSensConfig

settings = CoilSensConfig(
    input_data="multi_channel_data.nii.gz",    # Input file path
    input_affine="",                          # Empty for NIfTI files
    input_in_img_space=False,                 # Input is k-space
    coil_dimension=3,                          # Channel is 4th dimension
    smoothing_kernel=12,                      # Low-pass filter kernel size
    out_path="./output",                      # Output directory
    use_gpu=True,
    gpu_device=0
)
```

### Python API Usage

```python
from pymritools.config.processing.coil_sens import CoilSensConfig
from pymritools.processing.coil_sens import smap

# Configure settings
settings = CoilSensConfig(
    input_data="path/to/coil_data.nii.gz",
    input_in_img_space=False,  # Input is k-space
    coil_dimension=3,          # Channel is 4th dimension
    smoothing_kernel=12,
    out_path="./sensitivity_output/",
    use_gpu=True,
    gpu_device=0
)

# Run sensitivity map estimation
smap(settings)
```

### CLI Usage

```bash
python -m pymritools.processing.coil_sens.smap \
    -i input_data.nii.gz \
    -o ./output/ \
    -dim 3 \
    -k 12

# With GPU specification
python smap.py -i data.nii.gz -o output/ -dim 3 -k 10 -gpu true -gpud 0
```

### Output Files

**Main Output:**
- `smap_*.nii.gz`: Coil sensitivity maps (one per coil)
- `smap_combined.nii.gz`: rSoS combined reference
- `covariance.json`: Coil correlation matrix
- `covariance_plot.png`: Visualization of coil correlations

## LCPCA Denoising

### Theoretical Background

**LCPCA (Local Clinical PCA)** implements advanced denoising using Marchenko-Pastur inequality and matrix completion theory.

#### Mathematical Fundamentals

**1. Patch Construction:**
```
Build patches: P[x,y,z, t × n] from spatial-time neighborhoods
Patch shape: [cube_side_length, cube_side_length, cube_side_length, echoes]
```

**2. SVD Decomposition:**
```
P = U Σ Vᵀ
PCA components: U[:, :p]}, singular values: Σ[:p, :p]
```

**3. Marchenko-Pastur Inequality:**
```
Optimal number of components p satisfies:
(λ_{i} - λ_{m}) / (4√((m - i)/n_v)) < (1/(m - i)) * Σ_{j=i+1}^{m} λ_j
```
where λ are singular values in descending order.

**4. Truncation and Aggregation:**
```
P_denoised = U[:, :p] Σ[:p, :p] V[:, :p]ᵀ
Final: Weighted average of overlapping patches
```

#### Noise Bias Correction

**Statistical Model:**
```
Corrected magnitude: √(max(|x|² - 2·n·σ², 0))
where n = number of channels, σ = noise standard deviation
```

This correction assumes non-central chi distribution of magnitude data.

### Configuration

```python
from pymritools.config.processing.lcpca import SettingsLCPCA

settings = SettingsLCPCA(
    in_path="noisy_multiecho.nii.gz",     # Input file
    in_affine="",                         # Empty for NIfTI files  
    file_prefix="d",                      # Output file prefix
    normalize=False,                      # Normalize before PCA
    input_image_data=True,                # Input is image space
    noise_bias_correction=False,          # Enable NBC
    fixed_p=0,                           # 0 = use MP inequality
    noise_bias_mask="",                  # Pre-computed mask
    out_path="./output"
)
```

### Python API Usage

```python
import torch
from pymritools.processing.denoising.lcpca import denoise_lcpca

# Load noisy data
imported_data = torch.load("noisy_multiecho.pt")

# Scenario 1: Automatic component selection (MP inequality)
denoised_data, access_mask, patch_counts = denoise_lcpca(
    input_data=imported_data,
    p=0,  # Use MP inequality for automatic selection
    device=torch.device('cuda:0')
)

# Scenario 2: Fixed number of components
denoised_data, access_mask, patch_counts = denoise_lcpca(
    input_data=imported_data,
    p=1,  # Keep first singular value
    fixed_cube_side_length=5,             # Force 5×5×5 patches
    device=torch.device('cuda:0')
)

# Scenario 3: With noise bias correction
denoised_data_nbc = denoise_lcpca(
    input_data=imported_data,
    p=1,  # Keep first singular value
    noise_bias_correction=True,
    noise_bias_mask="",  # Auto-generate using autodmri
    visualize=True
)
```

### Core Functions

```python
def denoise_lcpca(
    input_data: torch.Tensor,
    p: int = 1,
    fixed_cube_side_length: int = 1,
    device: torch.device = torch.get_default_device(),
    noise_bias_correction: bool = False,
    noise_bias_mask: str = "",
    visualize: bool = False
) -> tuple:
    """
    LCPCA denoising implementation.
    
    Parameters:
        input_data: 4D+ tensor [x, y, z, coils, additional_dims]
        p: Number of singular values to keep (0 = automatic via MP inequality)
        fixed_cube_side_length: Force specific patch size (1 = default)
        device: Computation device
        noise_bias_correction: Enable noise bias correction
        noise_bias_mask: Path to noise mask ("" = auto-generate)
        visualize: Show progress
    
    Returns:
        (denoised_data, access_mask, patch_counts)
    """
```

### Independent Noise Bias Correction

```python
from pymritools.modules import extract_noise_stats_from_mask, noise_bias_correction

# Extract noise statistics
sigma, n_channels = extract_noise_stats_from_mask(
    input_data=noisy_data,
    mask=noise_mask,
    path_visuals="./noise_analysis/"
)

# Apply noise bias correction
corrected_data = noise_bias_correction(
    denoised_data=denoised_data,
    sigma=sigma,
    num_channels=n_channels
)
```

### CLI Usage

```python
# Main denoising
python -m pymritools.processing.denoising.lcpca.denoise \
    -i noisy_data.nii.gz \
    -o ./denoised_output/ \
    -p 0 \
    -nbc true

# With fixed threshold
python denoise.py -i data.nii.gz -o output/ -p 2 -gpu true

# Post-processing NBC only
python denoise.py noise_bias_correct_util \
    -i noisy_data.pt \
    -in denoised_data.pt \
    -ia affine.pt \
    -o nbc_output/
```

### Rank Selection

**Automatic (MP Inequality):**
```python
settings.fixed_p = 0  # Use Marchenko-Pastur inequality
# Results in automatic rank selection per patch
```

**Fixed Rank:**
```python
settings.fixed_p = 1  # Keep first singular value only
settings.fixed_p = 2  # Keep first 2 components
# Provides more control but less adaptive
```

### Comparison: LCPCA vs MPPCA

| Feature | LCPCA | MPPCA |
|---------|-------|-------|
| **Patch size** | Fixed size, configurable | Adaptive, based on local structure |
| **Structure** | Uses local spatial-time neighborhoods | Uses adaptive neighborhoods |
| **Noise handling** | Non-central chi bias correction | Central chi approximation |
| **Implementation** | Available in PyMRITools | NOT available |

## Gibbs Unringing

### Theoretical Background

**Gibbs unringing** removes Gibbs ringing artifacts based on Kellner et al. (2015, MRM) using sub-voxel optimization.

#### Algorithm Principles

**1. Weighting Filters:**
```
G_x = (1 + cos(k_y)) / (2 + cos(k_x) + cos(k_y))
G_y = (1 + cos(k_x)) / (2 + cos(k_x) + cos(k_y))
```

**2. Shifted Image Generation:**
For each voxel, generate M sub-voxel shifted images:
```
I_s(x) = Σ_k c_k * exp(2πi·k·(x + s/2M)) / N_k
```

**3. Optimal Shift Selection:**
Find shifts that minimize oscillation measure within local neighborhoods:
```
O(x,s) = ||G_x ⊗ I_s||² + ||G_y ⊗ I_s||²
s* = arg min O(x,s)
```

**4. Interpolation:**
```
Final image: Interpolate I_s* back to original grid
```

### Configuration

```python
from pymritools.config.processing.unringing import Settings

settings = Settings(
    input_file="gibbs_artifact.nii.gz",      # Input file
    num_shifts_per_voxel=100,                # Number of shifts per voxel
    voxel_neighborhood_size=3,               # Neighborhood size for optimization
    out_path="./output"
)
```

### Python API Usage

```python
import torch
from pymritools.processing.unringing import gibbs_unring_nd
from pymritools.utils import nifti_load

# Load data (2D, 3D, or 4D)
data, img = nifti_load("gibbs_artifact.nii.gz")
image_tensor = torch.from_numpy(data)

# Apply unringing
unringed = gibbs_unring_nd(
    image_data_nd=image_tensor,
    m=150,      # Higher precision (more shifts)
    k=3,        # 3-voxel neighborhood
    visualize=True,
    gpu=True,
    gpu_device=0
)

# Save result
from pymritools.utils import nifti_save
nifti_save(
    data=unringed.abs(),
    img_aff=img.affine,
    path_to_dir="./output",
    file_name="unringed"
)
```

### Core Functions

```python
def gibbs_unring_nd(
    image_data_nd: torch.Tensor,
    m: int = 100,
    k: int = 3,
    visualize: bool = True,
    gpu: bool = False,
    gpu_device: int = 0
) -> torch.Tensor:
    """
    Multidimensional Gibbs unringing (2D-4D).
    
    Parameters:
        image_data_nd: Input image data (2-4 dimensions)
        m: Number of shifted images per voxel (higher = more precise)
        k: Neighborhood size for optimization
        visualize: Enable progress bars
        gpu: Enable GPU acceleration
    
    Returns:
        Unringed image data
    """
```

### CLI Usage

```bash
# Basic usage
python -m pymritools.processing.unringing.gibbs_unr \
    -i gibbs_image.nii.gz \
    -o ./unringed_output/ \
    -m 100 \
    -k 3

# With GPU acceleration
python gibbs_unr.py -i data.nii.gz -o output/ -m 150 -k 4 -gpu true

# High precision mode
python gibbs_unr.py -i image.nii.gz -o results/ -m 200 -k 5 -gpu true -gpud 1
```

### Quality Control

**Effectiveness Assessment:**

```python
import matplotlib.pyplot as plt

# Compare before and after
fig, (ax1, ax2) = plt.subplots(1, 2)

ax1.imshow(gibbs_image.squeeze().cpu().numpy(), cmap='gray')
ax1.set_title('With Gibbs Ringing')
ax1.axis('off')

ax2.imshow(unringed.squeeze().cpu().numpy(), cmap='gray')
ax2.set_title('After Unringing')
ax2.axis('off')

# Profile comparison to quantify removal
center_profile_original = gibbs_image[..., 128, :].squeeze().cpu().numpy()
center_profile_unringed = unringed[..., 128, :].squeeze().cpu().numpy()

plt.figure()
plt.plot(center_profile_original, label='Original')
plt.plot(center_profile_unringed, label='Unringed')
plt.legend()
plt.xlabel('Pixel position')
plt.ylabel('Signal intensity')
plt.title('Profile comparison')
```

## Configuration System

All processing modules inherit from `BaseClass`:

```python
@dataclass
class BaseClass(Serializable):
    config_file: str = ""          # JSON configuration file path
    out_path: str = ""             # Output directory path
    use_gpu: bool = True           # GPU/CPU selection
    gpu_device: int = 0            # Specific GPU index
    visualize: bool = True         # Visualization toggle
    debug: bool = False            # Debug mode toggle
    slurm: bool = False           # Slurm mode
```

### Common Configuration Pattern

```python
# Load from file
settings = MyConfig.from_json("config.json")

# Load from CLI
settings = MyConfig.from_cli(args, parser)

# Manual construction  
settings = MyConfig(param1=value1, param2=value2)
```

## Mathematical Summary

| Algorithm | Key Equation | Parameters | Output |
|-----------|--------------|------------|--------|
| Coil Compression | k_comp = U[:, :n]ᵀ · k_orig | num_channels, ACS region | Compressed k-space |
| Coil Sensitivity | s_i = c_i / rSoS(x) | smoothing_kernel | Sensitivity maps |
| LCPCA | P_denoised = U[:, :p] Σ[:p, :p] V[:, :p]ᵀ | patch_size, MP inequality | Denoised image |
| Gibbs Unringing | min O(x,s) = ‖G_x ⊗ I_s‖² + ‖G_y ⊗ I_s‖² | m (shifts), k (neighborhood) | Unringed image |

## Performance Considerations

### Memory Requirements

**LCPCA Memory:**
```
memory ≈ batch_size × patch_size³ × echo_count × data_type
Example: 3000 × 5³ × 16 × 8 bytes ≈ 48 MB per batch
```

**Gibbs Unringing Memory:**
```
memory ≈ voxels × m (shifts) × data_type
Example: 256³ × 150 × 8 bytes ≈ 2 GB
```

### GPU Acceleration

**Recommended Settings:**
```python
# LCPCA: Good for batch processing
settings.use_gpu = True
settings.batch_size = 3000  # Balances memory/speed

# Gibbs: High precision benefits from GPU
unringed = gibbs_unring_nd(img_tensor, m=150, gpu=True)
```

### Processing Pipeline

**Typical Order:**
1. **Noise estimation** (background.mprage.nii → noise.mask)
2. **LCPCA denoising** (noisy.mprage.nii → denoised.mprage.nii)
3. **Noise bias correction** (denoised.mprage.nii → corrected.mprage.nii)
4. **Gibbs unringing** (corrected.mprage.nii → final.mprage.nii)

## Common Use Cases

### Scenario 1: Standard Denoising Pipeline

```python
from pymritools.processing.denoising.lcpca import denoise_lcpca
from pymritools.utils import nifti_load, nifti_save

# Load data
data, img = nifti_load("noisy_multiecho.nii.gz")

# Denoise
denoised, mask, counts = denoise_lcpca(
    input_data=torch.from_numpy(data),
    p=0,  # Automatic rank selection
    noise_bias_correction=True,
    visualize=True
)

# Save result
nifti_save(denoised.cpu().numpy(), img.affine, "./output/", "denoised")
```

### Scenario 2: Gibbs Ringing Removal

```python
from pymritools.processing.unringing import gibbs_unring_nd

# High-precision unringing
unringed = gibbs_unring_nd(
    image_data_nd=gibbs_tensor,
    m=200,  # High precision
    k=5,    # Larger neighborhood
    gpu=True,
    visualize=True
)
```

### Scenario 3: Complete Processing Pipeline

```python
# Step 1: Denoising
denoised = denoise_lcpca(noisy_data, p=0, noise_bias_correction=True)

# Step 2: Unringing  
final = gibbs_unring_nd(denoised, m=150, k=3)

# Step 3: Analysis
snr_improvement = calculate_snr(noisy_data, final) / calculate_snr(noisy_data, denoised)
print(f"LCPCA SNR improvement: {snr_improvement:.2f}x")
```

## See Also

- [API Reference](api.md#processing) - Complete class and method documentation
- [Configuration System](../contribute/writing_code.md#configuration) - How to use BaseClass configuration
- [GPU Acceleration](#performance-optimization) - Performance considerations
- [Noise Estimation](#noise-bias-correction) - Autodmri integration
- [Reconstruction](recon.md) - Downstream reconstruction applications