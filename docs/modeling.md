# Modeling

`pymritools.modeling` estimates quantitative parameters from (processed) images:

| Method | Module | Usage |
|---|---|---|
| [Dictionary matching](#dictionary-matching) (T2 / R2, B1+, B0) | `modeling.dictionary.grid_search_channels` | `python -m pymritools.modeling.dictionary.grid_search_channels` |
| [Voxel-wise pattern matching function](#pattern-matching-function) | `modeling.dictionary` | Python |
| [Mono-exponential fitting](#mono-exponential-fitting) | `modeling.decay.mexp` | Python (see note) |
| [AFI B1+ mapping](#afi-b1-mapping) | `modeling.b1_afi.modeling` | `modeling_afi_b1` |
| [ESPIRiT sensitivity maps](#espirit-sensitivity-maps) | `modeling.espirit.functions` | Python |

## Dictionary matching

Dictionary matching compares the measured echo train of each voxel with simulated
[EMC curves](simulation.md) and assigns the parameters of the best matching curve (Ben-Eliezer et al. 2015).
Both the measured curves \(\mathbf{s}\) and the dictionary curves \(\mathbf{d}_j\) are normalized to unit L2 norm along
the echoes, and the best match minimises the Euclidean distance

\[
\hat{j} = \arg\min_j \left\lVert \frac{|\mathbf{s}|}{\lVert \mathbf{s}\rVert_2} - \frac{|\mathbf{d}_j|}{\lVert \mathbf{d}_j\rVert_2} \right\rVert_2 ,
\]

which is equivalent to maximising the normalized inner product.
Because T2 and B1+ both change the shape of a MESE echo train, the matching benefits from a B1+ map from an
independent measurement (e.g. [AFI](#afi-b1-mapping)).

### Algorithm (`grid_search_channels`)

The command line tool `pymritools.modeling.dictionary.grid_search_channels` supports channel-combined and
channel-wise (not combined) data:

1. **Load data and dictionary.**
   NIfTI input `[x, y, z, echo]` gets a channel dimension (and a slice dimension for 2D input);
   `.pt` input is expected as `[x, y, z, channel, echo]`.
   k-space input (`--input_in_image_space false`) is transformed to image space along the first two dimensions.
   Magnitudes of data and dictionary are normalized along the echoes.
2. **Background fields (B1+, B0).**
    - Without B1+ and B0 input, both are estimated by matching the root-sum-of-squares combined data against the full dictionary.
    - If only B1+ is given, B0 is estimated from the sub-dictionary closest to the given B1+ value per voxel
      (and vice versa).
    - If a B1+ error map is given (`--input_b1_err`), the input B1+ and the dictionary estimate are combined:
      \(B_1 = w\,B_{1,\mathrm{EMC}} + (1-w)\,B_{1,\mathrm{input}}\) with \(w = \min(|e|, e_\mathrm{cut})/e_\mathrm{cut}\),
      the relative error \(e\) and the cutoff \(e_\mathrm{cut}\) = `--b1_error_cutoff`.
    - Estimated maps are smoothed with a Gaussian filter (σ = 10 voxels).
    - B1+ input given in percent (maximum > 10) is divided by 100 and then by `--tx_factor`.
3. **Regularised T2 matching.**
   For every slice and channel, each voxel is matched against the sub-dictionary at its B1+ and B0 value,
   leaving T1 and T2 as free parameters.
   With `--low_rank_regularisation r`, the curves of 3×3 voxel patches are first replaced by a rank-\(r\) approximation;
   the results of overlapping patches are averaged with weights inversely proportional to the residual.
4. **Channel combination.**
   Channel-wise R2 maps are combined with weights derived from the squared matching residual \(\rho\):
   \(w = (\rho_\mathrm{max} - \min(\rho, \rho_\mathrm{max}))/\rho_\mathrm{max}\) with \(\rho_\mathrm{max}\) = median(ρ)/3,
   smoothed in k-space, and \(R_{2} = \sum_c w_c R_{2,c} / \sum_c w_c\).
   For combined data (one channel) this is the identity.

### Command line usage

```shell
python -m pymritools.modeling.dictionary.grid_search_channels \
    -i denoised/denoised_data.nii -db emc/db_mese.pkl -b1 b1/b1_afi.nii -o ./fit
```

| Option | Alias | Default | Description |
|---|---|---|---|
| `--input_data` | `-i` | `""` | Input `.nii` or `.pt` data. |
| `--input_affine` | `-ia` | `""` | Affine `.pt` for `.pt` input (identity otherwise). |
| `--input_database` | `-db` | `""` | Dictionary database `.pkl` from the [EMC simulation](simulation.md). |
| `--input_b1` | `-b1` | `""` | B1+ map (`.nii`), relative or in percent. Estimated from the data if not given. |
| `--input_b1_err` | `-b1e` | `""` | Relative B1+ error map (`.nii`), e.g. `b1_rel_err.nii` from `modeling_afi_b1`. |
| `--b1_error_cutoff` | `-b1ec` | `0.15` | Relative error above which only the dictionary estimate of B1+ is used. |
| `--tx_factor` | `-txf` | `1.0` | Transmit voltage reduction factor of the scan (e.g. for SAR reasons), to rescale the B1+ input. |
| `--input_b0` | `-b0` | `""` | B0 map (`.nii`) [Hz]. Estimated from the data if not given. |
| `--input_in_image_space` | `-iimg` | `False` | Set to `True` for image input! Otherwise the data is Fourier transformed first. |
| `--rsos_channel_combine` | `-rsos` | `False` | Combine channels by root-sum-of-squares before matching. |
| `--low_rank_regularisation` | `-lr` | `None` | Rank for the patch-based low-rank regularisation. |
| `--process_slice` | `-ps` | `False` | Process a single slice only. |

!!! warning
    `--input_in_image_space` defaults to `False`, i.e. the input is treated as k-space.
    Set it to `True` for reconstructed or denoised images.

Output files:

| File | Content |
|---|---|
| `t2.nii` | T2 [ms] per channel |
| `r2.nii` | R2 [1/s] per channel |
| `r2_combined.nii` | Channel-combined R2 [1/s] |
| `optimize_residual.nii` | Squared L2 residual of the match |
| `weights.nii` | Channel combination weights |
| `b1_estimate.nii`, `b1_estimate_smoothed.nii` | B1+ used for the matching (input or estimate) |
| `b0_estimate.nii`, `b0_estimate_smoothed.nii` | B0 used for the matching (input or estimate) |

### Pattern matching function

For custom pipelines, `pymritools.modeling.dictionary.r2_pattern_matching` (`fit_r2_fn` in `grid_search.py`)
matches magnitude curves voxel-wise, in batches, optionally restricted to a given B1+ map:

```python
import torch
from pymritools.config.database import DB
from pymritools.modeling.dictionary import r2_pattern_matching
from pymritools.utils import normalize_data

db = DB.load("emc/db_mese.pkl")
db_mag, _ = db.get_torch_tensors_t1t2b1b0e()
t1_vals, t2_vals, b1_vals, _ = db.get_t1_t2_b1_b0_values()
# use a single B0 value and normalize the curves
db_mag, _ = normalize_data(db_mag[:, :, :, 0].to(torch.float32))
data, _ = normalize_data(data.abs().to(torch.float32))       # data: [..., echoes]

t2, b1, l2 = r2_pattern_matching(
    input_data=data, db_mag=db_mag,
    t2_vals=t2_vals.float(), b1_vals=b1_vals.float(), t1_vals=t1_vals.float(),
    b1_data=None, batch_size=2000, device=torch.device("cuda"),
)
```

The function does not normalize its inputs; T2 is returned in the units of the database (s).

### Legacy grid search

`pymritools.modeling.dictionary.grid_search` contains the earlier voxel-wise command line tool (`fit`) and a
combined R2 / R2' fit for MEGESSE data (`megesse`).

!!! warning "Known limitations"
    - `grid_search.fit` unpacks seven return values from `setup_db`, which returns eight; the command line tool
      currently fails. Use `grid_search_channels`.
    - `grid_search.megesse` reads the echo times from a hard-coded path.
    - In `grid_search_channels`, `--process_slice` selects slice `n_z // 3` of the data but the central slice of the B1+ / B0 maps.

## Mono-exponential fitting

`pymritools.modeling.decay.mexp.fit` fits \(S(t) = S_0\,e^{-R t}\) for every voxel by linear least squares on the
logarithm of the signal:

\[
\ln S(t_i) = \ln S_0 - R\,t_i .
\]

```python
import torch
from pymritools.modeling.decay.mexp import fit

te = torch.tensor([9.5, 19.0, 28.5, 38.0, 47.5, 57.0, 66.5]) * 1e-3    # [s]
s0, r = fit(data=magnitude_data.to(torch.float32), te=te)             # echoes in the last dimension
```

- The last dimension of `data` must match the number of echo times.
- Echo times are expected in seconds; if all are larger than 1, they are assumed to be in ms and converted.
- The computation runs on the device of `data`.
- Zero or negative signal values produce invalid logarithms, which are set to zero.
  The log-linear fit weights late (low SNR) echoes as much as early ones; denoise first or restrict the echoes for noisy data.

The module also contains a command line interface (`process(settings)`, `main()`) that reads a NIfTI file and writes
`S0.nii` and `R.nii`.

!!! warning "Known limitation"
    The default value of `MEXPSettings.echo_times` is not a valid default factory, so the settings class cannot be created
    without echo times and the command line tool currently fails at start-up. Use the Python function.

## AFI B1+ mapping

Actual flip-angle imaging (Yarnykh 2007) acquires two interleaved spoiled gradient echo signals \(S_1\), \(S_2\)
after the repetition times \(TR_1\) and \(TR_2 = n\,TR_1\).
With \(r = S_2/S_1\), the actual flip angle is

\[
\alpha = \arccos\left(\frac{r\,n - 1}{n - r}\right),
\]

and the relative B1+ is \(100\cdot\alpha/\alpha_\mathrm{nominal}\) in percent.

```shell
uv run modeling_afi_b1 -i afi.nii.gz -o ./b1 -fa 60 -n 5 -s 3
```

| Option | Alias | Default | Description |
|---|---|---|---|
| `--input_file` | `-i` | `""` | AFI images; the last dimension holds \(S_1\) and \(S_2\) (`[x, y, z, 2]` or `[x, y, z, channel, 2]` for `.pt`). |
| `--input_affine` | `-ia` | `""` | Affine `.pt` for `.pt` input. |
| `--flip_angle` | `-fa` | `60.0` | Nominal flip angle [°]. |
| `--ratio_tr2_tr1` | `-n` | `5` | \(n = TR_2/TR_1\). |
| `--smoothing_kernel` | `-s` | `3` | σ of the Gaussian filter applied to the flip angle map [voxels]; `0` disables smoothing. |
| `--noise_mask` | `-m` | `""` | Background mask (`.nii` or `.pt`); estimated with autodmri if not given. |
| `--input_in_image_domain` | `-iimgd` | `True` | Set to `False` for k-space input. |

The noise level is estimated from background voxels, and the error of the B1+ map is calculated by error propagation
of the noise through the AFI equation.

Output files:

| File | Content |
|---|---|
| `b1_afi.nii` | B1+ [%] |
| `b1_rel_err.nii` | Relative error of B1+ (clamped to ±2) |
| `b1_afi_ref.nii` | Reference image (\(S_1\)) |
| `b1_afi_corr-poly.nii` | B1+ after an empirical cubic polynomial correction with fixed coefficients |
| `noise_mask.nii` | Estimated background mask (if not given) |
| `coil_sensitivities.nii`, `b1_map_wavg.nii` | Only for multi-channel `.pt` input: ESPIRiT sensitivities and the sensitivity-weighted combination of channel-wise B1+ maps |

The relevant functions are also available from Python:

```python
from pymritools.modeling.b1_afi.modeling import calculate_b1, calculate_error_map

b1 = calculate_b1(b1_data=afi_data, r_tr21=5, smoothing_kernel=3, flip_angle_set_deg=60.0)
```

## ESPIRiT sensitivity maps

`pymritools.modeling.espirit.functions.map_estimation` computes ESPIRiT coil sensitivity maps (Uecker et al. 2014):

1. Extract the central `num_ac_lines` of k-space in every dimension with more than one sample.
2. Build the calibration (block-Hankel) matrix of all `kernel_size` neighborhoods and compute its SVD.
3. Keep the singular vectors with \(\sigma_i \ge\) `rank_fraction_ac_matrix` \(\cdot\,\sigma_1\) (at least as many as channels).
4. Transform the kernels to image space and compute a point-wise SVD; keep eigenvectors with eigenvalue \(> \) `eigenvalue_cutoff`.

```python
import torch
from pymritools.modeling.espirit.functions import map_estimation

# k-space [read, phase, slice, channel] (a fifth dimension is averaged)
maps = map_estimation(
    k_rpsc=k_space, kernel_size=6, num_ac_lines=24,
    rank_fraction_ac_matrix=0.01, eigenvalue_cutoff=0.9,
    device=torch.device("cuda"),
)
sens = maps[0]   # first set of maps, [read, phase, slice, channel]
```

Odd kernel sizes are increased by one.
