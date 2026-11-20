# Reconstruction (LORAKS)

`pymritools.recon.loraks` reconstructs undersampled multi-channel, multi-echo k-space with
LORAKS (low-rank modeling of local k-space neighborhoods).
The implementation runs on the GPU and reconstructs all channels and echoes of a slice jointly,
which makes the complementary sampling across echoes of the [PyMRItools sequences](sequence_programming.md#k-space-sampling) usable.

## Background

### LORAKS matrices

LORAKS (Haldar 2014) uses the fact that MR images have limited spatial support and slowly varying phase.
Both properties imply linear dependencies between neighboring k-space samples.
If the k-space values in a local neighborhood (here: a square patch of side length \(n_b\)) are collected for every
patch position, the resulting structured matrix has low rank.

Let \(\mathcal{P}_C(k)\) be the operator that builds the matrix of all neighborhoods of the k-space \(k\)
(for multiple channels and echoes, the neighborhoods of all channels and echoes are stacked, which gives the joint
multi-channel / multi-contrast formulation of P-LORAKS and Joint-LORAKS).

- **C matrix:** \( \mathbf{C} = \mathcal{P}_C(k) \).
  Low rank follows from the limited image support.
- **S matrix:** additionally uses the neighborhoods at the mirrored positions \(-\mathbf{k}\),
  which exploits smooth image phase (conjugate symmetry).
  With \(\mathbf{S}_{+} = \mathcal{P}_C(k)\) and \(\mathbf{S}_{-}\) built from the point-reflected neighborhoods,
  the real-valued S matrix is

    \[
    \mathbf{S} =
    \begin{bmatrix}
    \mathrm{Re}(\mathbf{S}_{+} - \mathbf{S}_{-}) & -\mathrm{Im}(\mathbf{S}_{+} - \mathbf{S}_{-}) \\
    \mathrm{Im}(\mathbf{S}_{+} + \mathbf{S}_{-}) & \mathrm{Re}(\mathbf{S}_{+} + \mathbf{S}_{-})
    \end{bmatrix}.
    \]

The S matrix is usually the better choice and is the default.
The operators and their adjoints are implemented with linear index tensors and `index_add`
(`pymritools.recon.loraks.operators`), without loops.

### AC-LORAKS

AC-LORAKS (Haldar 2015) estimates the approximate nullspace of the LORAKS matrix from the fully sampled
auto-calibration (AC) region once, instead of computing a singular value decomposition in every iteration:

1. Build the LORAKS matrix from the sampled data and keep only the neighborhoods that are completely sampled.
2. Compute its nullspace basis \(\mathbf{N}\): all singular vectors beyond the chosen rank \(r\).
3. Reconstruct the k-space by solving

    \[
    \hat{k} = \arg\min_k \; \lVert \mathcal{A}k - d \rVert_2^2 + \lambda \, \lVert \mathcal{P}(k)\,\mathbf{N} \rVert_F^2,
    \]

    where \(\mathcal{A}\) selects the sampled points and \(d\) is the measured data.
    For \(\lambda = 0\) (default), the measured samples are kept exactly and only the missing samples are estimated
    by minimizing \( \lVert \mathcal{P}(k)\,\mathbf{N} \rVert_F^2 \).

The product with the nullspace is evaluated as a convolution with nullspace filters in the Fourier domain
(as in the reference AC-LORAKS implementation), which is fast and memory efficient.
The resulting least squares problem is solved with conjugate gradients.
Alternatively, it can be minimized directly with gradient descent (PyTorch autograd, Barzilai–Borwein step size).

### P-LORAKS (experimental)

The P-LORAKS implementation in `p_loraks.py` does not need an AC region.
In each iteration it builds the LORAKS matrix of the current estimate, computes a low-rank approximation
\(\mathbf{M}_r\) with a (randomized) truncated SVD, and takes a gradient step on

\[
\mathcal{L}(k) = \lVert \mathcal{A}k - d \rVert_2 + \lambda \, \lVert \mathcal{P}(k) - \mathbf{M}_r \rVert_F
\]

(or only on the low-rank term with strict data consistency for \(\lambda = 0\)).
It is slower and needs much more memory than AC-LORAKS.

## Command line usage

```shell
uv run recon_loraks -ik raw/k_space.pt -ia raw/affine.pt -o ./recon \
    -alg AC-LORAKS-ls -t S -p 5 -rank 150 -mni 25 -b 8
```

| Option | Alias | Default | Description |
|---|---|---|---|
| `--in_k_space` | `-ik` | `./examples/raw_data/results/k_space.pt` | Input k-space `.pt` file `[read, phase, slice, channel, echo]` (missing trailing dimensions are added). |
| `--in_affine` | `-ia` | `""` | Affine `.pt` file for the NIfTI output (identity if not given). |
| `--loraks_algorithm` | `-alg` | `AC-LORAKS-ls` | `AC-LORAKS-ls` (conjugate gradients), `AC-LORAKS` (autograd gradient descent), `P-LORAKS`. |
| `--matrix_type` | `-t` | `S` | LORAKS matrix, `S` or `C`. |
| `--patch_size` | `-p` | `5` | Side length of the square k-space neighborhood. |
| `--rank` | `-rank` | `180` | Rank \(r\) of the LORAKS matrix. |
| `--reg_lambda` | `-l` | `0.0` | Regularization \(\lambda\); `0` enforces strict data consistency. |
| `--max_num_iter` | `-mni` | `25` | Maximum number of iterations. |
| `--batch_size` | `-b` | `-1` | Channels per batch; `-1` reconstructs all channels jointly. Must divide the number of channels. |
| `--matrix_decomposition_type` | `-mdt` | `eigh` | Nullspace computation for AC-LORAKS: `eigh`, `lr_svd`, `r_svd`, `sor_svd`, `rand_ns`. |
| `--process_slice` | `-ps` | `False` | Reconstruct only the central slice (for testing parameters). |
| `--conv_tol` | `-tol` | `1e-3` | Convergence tolerance (currently not passed on; AC-LORAKS uses a fixed tolerance of `1e-5`). |

The rank depends on the matrix size, i.e. on the patch size and the number of jointly reconstructed channels and echoes.
Start with the `--process_slice` option to tune it.

### Processing steps

1. Load the k-space and add missing dimensions up to `[read, phase, slice, channel, echo]`.
2. Group channels into batches (`--batch_size`): channels are clustered by their correlation (k-means)
   so that strongly correlated channels are reconstructed together.
3. Reshape to batches of `[slice × channel batch, channels × echoes, phase, read]` and pad even spatial dimensions by one sample.
4. Reconstruct each batch (one slice and channel batch, all echoes jointly).
5. Undo padding and batching, transform to image space, and save.

### Output files

| File | Content |
|---|---|
| `input_img.nii` | Magnitude of the zero-filled input (central slice) |
| `recon_k.pt` | Reconstructed k-space `[read, phase, slice, channel, echo]`; note that the slice dimension is additionally transformed to k-space (inverse FFT along slices) |
| `recon_img.pt` | Reconstructed complex images |
| `recon_img_rsos.nii` | Root-sum-of-squares channel combination |
| `affine.pt` | Affine used for the NIfTI files |

## Python usage

The end-user interface is the factory `Loraks.create(options)`, which returns an `AcLoraks` or `PLoraks` object
depending on `options.loraks_type`.
`reconstruct()` expects batched k-space with dimensions `[batch, channels × echoes, phase, read]`.
The helpers in `pymritools.recon.loraks.utils` convert from and to the toolbox k-space layout:

```python
import torch
from pymritools.utils import torch_load, fft_to_img, root_sum_of_squares
from pymritools.recon.loraks.loraks import Loraks, OperatorType
from pymritools.recon.loraks.ac_loraks import AcLoraksOptions, SolverType, NullspaceAlgorithm
from pymritools.recon.loraks.utils import (
    check_channel_batch_size_and_batch_channels, prepare_k_space_to_batches,
    unprepare_batches_to_k_space, pad_input, unpad_output
)

k = torch_load("raw/k_space.pt")                      # [read, phase, slice, channel, echo]

batch_idx = check_channel_batch_size_and_batch_channels(k_space_rpsct=k, batch_size_channels=-1)
k_batched, in_shape = prepare_k_space_to_batches(k_space_rpsct=k, batch_channel_indices=batch_idx)
k_batched, padding = pad_input(k_batched)

opts = AcLoraksOptions(
    loraks_neighborhood_size=5,
    loraks_matrix_type=OperatorType.S,
    regularization_lambda=0.0,
    max_num_iter=25,
    solver_type=SolverType.LEASTSQUARES,
    nullspace_algorithm=NullspaceAlgorithm.EIGH,
    device=torch.device("cuda:0"),
)
opts.rank.value = 150

loraks = Loraks.create(opts)
k_recon = loraks.reconstruct(k_batched)               # returned on the CPU

k_recon = unpad_output(k_space=k_recon, padding=padding)
k_recon = unprepare_batches_to_k_space(
    k_batched=k_recon, batch_channel_indices=batch_idx, original_shape=in_shape
)
img = fft_to_img(k_recon, dims=(0, 1))
rsos = root_sum_of_squares(img, dim_channel=-2)
```

### Options

`LoraksOptions` (base class) and `AcLoraksOptions`:

| Field | Default | Description |
|---|---|---|
| `loraks_type` | `AC_LORAKS` | `LoraksImplementation.AC_LORAKS` or `P_LORAKS` |
| `loraks_neighborhood_size` | `5` | Patch side length |
| `loraks_matrix_type` | `OperatorType.S` | `S` or `C` |
| `rank` | `RankReduction(HARD_CUTOFF, 150)` | AC-LORAKS uses `rank.value` as the rank |
| `regularization_lambda` | `0.0` | \(\lambda\), `0` = strict data consistency |
| `max_num_iter` | `30` | Iterations of the solver |
| `device` | CUDA if available | Torch device |
| `solver_type` | `SolverType.LEASTSQUARES` | Conjugate gradients, or `AUTOGRAD` for gradient descent |
| `nullspace_algorithm` | `NullspaceAlgorithm.EIGH` | Eigen-decomposition of \(\mathbf{M}\mathbf{M}^H\), or randomized alternatives (`TORCH_LR`, `RSVD`, `SOR_SVD`, `RANDOM_NS`) |

For P-LORAKS, use `PLoraksOptions` from `pymritools.recon.loraks.p_loraks`.
It adds `patch_shape`, `sample_directions` and `lowrank_algorithm`, and `rank` can be one of three
`RankReductionMethod`s: `HARD_CUTOFF` (keep the first `value` singular values), `RELU_SHIFT`
(subtract `value` from all singular values and clip at zero), or `RELU_SHIFT_AUTOMATIC`
(`value` in (0, 1): keep the singular values that make up this fraction of their cumulative sum).
`PLoraks` also offers a builder-style interface (`with_patch_shape`, `with_s_matrix`, `with_sv_hard_cutoff`, `with_device`, …).

!!! warning "Known limitations"
    - P-LORAKS from the command line passes the base `LoraksOptions`, so the patch size option is not used, and
      the rank is replaced by an automatically chosen hard cutoff. Use the Python interface for P-LORAKS.
    - `--conv_tol` is not passed to the solver.
    - The zero-filled sampling mask is deduced from the data (`|k| > 1e-10`); exact zeros in measured data are treated as missing.
