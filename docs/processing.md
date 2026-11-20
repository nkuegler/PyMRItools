# Processing

`pymritools.processing` contains image and k-space processing steps that are typically applied between
reconstruction and modeling:

| Tool | Module | Command |
|---|---|---|
| [PCA denoising](#pca-denoising) | `processing.denoising.lcpca` | `processing_denoise_lcpca` |
| [Noise bias correction](#noise-bias-correction) | `processing.denoising.lcpca` | part of the denoising command |
| [Gibbs unringing](#gibbs-unringing) | `processing.unringing` | `processing_unring` |
| [Coil compression](#coil-compression) | `processing.coil_compression` | Python only |
| [Coil sensitivity estimates](#coil-sensitivity-estimates) | `processing.coil_sens.smap` | `python -m pymritools.processing.coil_sens.smap` |

## PCA denoising

The denoising follows the local PCA approach for multi-echo relaxometry data by Does et al. (2019):
the redundancy of the echo dimension in small spatial neighborhoods is used to separate signal and noise
components, and the number of signal components is chosen with the Marchenko–Pastur (MP) criterion of
random matrix theory (Veraart et al. 2016).

### Algorithm

For every slice and channel independently:

1. For every in-plane voxel position, take a square patch of \(n\times n\) voxels with \(n = \lceil\sqrt{m}\rceil\),
   where \(m\) is the number of echoes.
   The patch forms a Casorati matrix \(\mathbf{X}\in\mathbb{C}^{n_v\times m}\) with \(n_v = n^2 \approx m\) voxels.
2. Subtract the mean over the echo dimension and compute the SVD \(\mathbf{X} = \mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^H\)
   with eigenvalues \(\lambda_i = \sigma_i^2/n_v\).
3. Determine which components are noise.
    With \(M = \min(m, n_v)\), a component \(p\) is classified as noise when

    \[
    \frac{\lambda_{p+1} - \lambda_{M}}{4\sqrt{(M-p)/n_v}} < \frac{1}{M-p}\sum_{i>p} \lambda_i ,
    \]

    i.e. when the spread of the remaining eigenvalues is compatible with the MP distribution of pure noise.
    Alternatively, a fixed number of components `fixed_p` is kept.

4. Reconstruct the patch from the kept components and add the mean back.

5. Patches overlap. Every voxel is the weighted average of all patch estimates containing it,
   with weights \(\theta = 1/(1+p)\) that favour patches with fewer signal components (Manjón et al.).

Real and complex input is supported.

### Command line usage

```shell
uv run processing_denoise_lcpca -i mese_echoes.nii.gz -o ./denoised
```

| Option | Alias | Default | Description |
|---|---|---|---|
| `--in_path` | `-i` | example file | Input `.nii` or `.pt` file, `[x, y, z, (channel), echo]`. |
| `--in_affine` | `-ia` | `""` | Affine `.pt` file (for `.pt` input). |
| `--input_image_data` | `-iimg` | `True` | Set to `False` for k-space input; data is then transformed to image space along the first two dimensions. |
| `--fixed_p` | `-p` | `0` | Number of components to keep; `0` determines them per patch with the MP criterion. |
| `--noise_bias_correction` | `-nbc` | `False` | Apply the [noise bias correction](#noise-bias-correction) after denoising. |
| `--noise_bias_mask` | `-nbm` | `""` | NIfTI mask of background (noise-only) voxels; estimated automatically if empty. |
| `--normalize` | `-n` | `False` | Not used in the current implementation. |
| `--file_prefix` | `-fp` | `d` | Not used in the current implementation. |

Output files:

| File | Content |
|---|---|
| `denoised_data.pt` | Denoised data (same dimensions as the input, singleton dimensions removed) |
| `denoised_data.nii` | Denoised data; for multi-channel input combined by root-sum-of-squares |
| `avg_p.nii` | Number of kept components, accumulated over all patches containing the voxel |
| `denoised_data_nbc.pt` / `.nii`, `noise_mask.nii` | With `--noise_bias_correction` |

### Python usage

```python
import torch
from pymritools.processing.denoising.lcpca import denoise_lcpca

# data: [x, y, z, channel, echo] or [x, y, z, echo]
denoised, weights, num_p = denoise_lcpca(input_data=data, p=0, device=torch.device("cuda"))
```

`denoise_lcpca` (alias of `core_fn`) returns the denoised data, the accumulated patch weights and the accumulated number of kept components.
Note that the Python default is `p=1` (keep one component); pass `p=0` for the MP criterion.

## Noise bias correction

Magnitude data has a positive noise floor (Rician or, for root-sum-of-squares combined multi-channel data,
non-central χ distribution).
The correction removes the first-order bias of the squared signal:

\[
S_\mathrm{corr} = \sqrt{\max\left(|S|^2 - 2N\sigma^2,\, 0\right)}
\]

with the noise level \(\sigma\) and the effective number of channels \(N\).
Both are estimated from background voxels with the moment estimators of the non-central χ distribution
(`pymritools.processing.denoising.stats.non_central_chi`).
If no mask is given, background voxels are detected with [autodmri](https://github.com/samuelstjean/autodmri)
(St-Jean et al. 2020) along all three axes and eroded.

```python
from pymritools.processing.denoising.lcpca import (
    extract_noise_mask, extract_noise_stats_from_mask, noise_bias_correction
)

mask = extract_noise_mask(input_data=data, erode_iter=2)
sigma, n_channels = extract_noise_stats_from_mask(input_data=data, mask=mask)
data_corrected = noise_bias_correction(denoised_data=denoised, sigma=sigma, num_channels=n_channels)
```

The correction assumes stationary noise, i.e. one \(\sigma\) for the whole volume.

## Gibbs unringing

Removes Gibbs ringing with the local sub-voxel shift method of Kellner et al. (2016).
For each image line, the image is shifted by sub-voxel amounts \(s/2M\) (\(s = -M,\dots,M-1\)).
For every voxel the shift that minimises the total variation in a small neighborhood to the left or right
(size \(k\)) is selected, and the result is interpolated back onto the original grid.
For 2D images, the method is applied along x and y on images filtered with the weighting filters
\(G_x = (1+\cos k_y)/(2+\cos k_x+\cos k_y)\) and \(G_y = (1+\cos k_x)/(2+\cos k_x+\cos k_y)\), and the two results are summed.
The data is processed slice by slice (in-plane, first two dimensions).

```shell
uv run processing_unring -i mese_echoes.nii.gz -o ./unring -m 100 -k 3
```

| Option | Alias | Default | Description |
|---|---|---|---|
| `--input_file` | `-i` | `""` | Input NIfTI file, 2D to 4D (`x y (z) (t)`). |
| `--num_shifts_per_voxel` | `-m` | `100` | Number of sub-voxel shifts \(M\). |
| `--voxel_neighborhood_size` | `-k` | `3` | Neighborhood size \(k\) for the oscillation measure. |

The output is written as `ur_<input name>.nii` (magnitude).

```python
import torch
from pymritools.processing.unringing import gibbs_unring_nd

unringed = gibbs_unring_nd(image_data_nd=torch.from_numpy(data), m=100, k=3, gpu=True, visualize=False)
```

!!! warning "Known limitation"
    `gibbs_unring_nd` logs the device name with `torch.cuda.get_device_name()`, which fails without a CUDA device.

## Coil compression

`compress_channels_2d` reduces the number of receive channels with geometric-decomposition coil compression
(GCC; Zhang et al. 2013, based on the SVD compression of Huang et al. 2008), for example to speed up the LORAKS reconstruction.

- The read direction is detected from the sampling pattern (the fully sampled direction).
- The data is transformed to image space along the read direction.
- For every read position, compression matrices are computed by an SVD of the AC data (or all sampled data) over phase encodes and echoes,
  and aligned along the read direction.
- The compression is computed per slice, because the slices of 2D multi-slice data are already in image space.

```python
from pymritools.processing.coil_compression import compress_channels_2d

# k_space: [x, y, z, channel, echo], sampling_mask: [x, y, echo]
k_compressed = compress_channels_2d(
    input_k_space=k_space, sampling_pattern=sampling_mask,
    num_compressed_channels=16, use_ac_data=True
)
```

If the requested number of channels is larger than the number of available AC lines, it is reduced accordingly.

## Coil sensitivity estimates

`pymritools.processing.coil_sens.smap` computes a very approximate coil sensitivity estimate:
every channel image is divided by the root-sum-of-squares combination, averaged over echoes, and low-pass filtered
by keeping only a central `smoothing_kernel × smoothing_kernel` k-space region.
It also writes the correlation matrix of the channel sensitivities (`smap_cov.json`, `smap_cov_plot.html`)
and the maps as `smap.nii`.

```shell
python -m pymritools.processing.coil_sens.smap -i channel_images.pt -ia affine.pt -o ./smap -dim 3 -k 10
```

| Option | Alias | Default | Description |
|---|---|---|---|
| `--input_data` | `-i` | `""` | `.pt` or `.nii` input with at least 4 dimensions. |
| `--input_affine` | `-ia` | `""` | Affine `.pt` file (required for `.pt` input). |
| `--input_in_img_space` | `-ii` | `True` | Set to `False` for k-space input. |
| `--coil_dimension` | `-dim` | `3` | Channel dimension (0-based). |
| `--smoothing_kernel` | `-k` | `10` | Size of the central k-space region kept for smoothing. |

!!! warning "Known limitation"
    For NIfTI input, the return values of `nifti_load` are unpacked in the wrong order in `smap()`; use `.pt` input.

For proper sensitivity maps, use [ESPIRiT](modeling.md#espirit-sensitivity-maps).
