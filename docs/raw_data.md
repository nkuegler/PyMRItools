# Raw Data

`pymritools.seqprog.rawdata` reads Siemens twix raw data (`.dat`) with
[twixtools](https://github.com/pehses/twixtools) and sorts it into k-space tensors in the
[toolbox convention](concepts.md#dimension-conventions) `[read, phase, slice, channel, echo]`.

Two sources are supported:

- **Pulseq sequences built with PyMRItools** (`--seq_type pulseq`, default):
  the ADCs of a Pulseq sequence carry no meaningful loop counters, so the sorting uses the sampling configuration
  written by the [sequence build](sequence_programming.md#output-files) (`*_sampling_config.pkl`) together with the
  sequence parameters (`*_pulseq_config.json`).
- **Vendor (Siemens) sequences** (`--seq_type siemens`): sorting via the twixtools high-level `map_twix` interface.

## Command line usage

```shell
uv run seqprog_raw_data \
    -i meas_MID00042_FID07553_pulseq_mese.dat \
    -seq mese_pulseq_config.json \
    -samp mese_sampling_config.pkl \
    -o ./raw
```

| Option | Alias | Default | Description |
|---|---|---|---|
| `--input_file` | `-i` | `""` | Raw data file (`.dat`). |
| `--input_sequence_config` | `-seq` | `""` | `*_pulseq_config.json` of the sequence build (Pulseq only). |
| `--input_sample_config` | `-samp` | `""` | `*_sampling_config.pkl` of the sequence build (Pulseq only). |
| `--remove_os` | `-rmos` | `True` | Remove the readout oversampling. |
| `--split_read_polarity` | `-spr` | `True` | Separate readouts of opposite polarity (identifiers containing `bu` / `bd`). |
| `--seq_type` | `-st` | `pulseq` | `pulseq` or `siemens`. |

plus the [common options](concepts.md#configuration-command-line-and-json).

## Processing steps (Pulseq)

1. Read the twix file, remove sync data, and get the number of coils.
2. For each acquisition type of the sampling pattern (e.g. `fs_acq`, `acq_bu`, `acq_bd`, `noise_scan`):
    - noise scans are stored separately,
    - navigator acquisitions are skipped (not implemented yet),
    - imaging ADCs are placed on the Cartesian grid at their phase encode, slice and echo number.
      The read positions follow from the registered k-space trajectory of the acquisition (nearest grid point, no gridding).
3. If noise scans exist, compute a noise pre-whitening matrix from the noise covariance and apply it to all channels.
4. Optionally remove the readout oversampling (crop in image space along the read direction) and re-apply the sampling mask.
5. Swap read and phase dimensions for `acq_phase_dir = "RL"`, and compute the affine from the twix geometry and the sequence parameters.

## Output files

| File | Content |
|---|---|
| `k_space.pt` | Complex k-space `[read, phase, slice, channel, echo]`, slices in image space |
| `k_sampling_mask.pt` | Boolean sampling mask `[read, phase, echo]` |
| `k_space_bu.pt`, `k_space_bd.pt` | Instead of `k_space.pt` when readout polarities are separated (MEGESSE) |
| `k_sampling_mask_bu.pt`, `k_sampling_mask_bd.pt`, `echo_nums_bu.pt`, `echo_nums_bd.pt` | Masks and echo indices of the two polarities |
| `affine.pt` | 4×4 affine matrix for NIfTI output |
| `k_noise_scans.pt` | Pre-whitened noise scans |
| `naive_rsos_recon.nii`, `sampling_pattern.nii` | Zero-filled FFT + root-sum-of-squares image and sampling mask (with `--visualize`) |
| `nii_settings.json` | Header information (Siemens data only) |

The k-space and affine files are the inputs of the [LORAKS reconstruction](recon.md).

## Helpers

`pymritools.seqprog.rawdata.utils` contains:

- `remove_oversampling(data, data_in_k_space=True, read_dir=0, os_factor=2)`: crop the oversampled field of view along `read_dir`.
- `rmos()` and `recon_naive_rsos()`: small command line wrappers (no registered command) for removing oversampling
  from a `.pt` file and for a zero-filled FFT + root-sum-of-squares reconstruction.

!!! warning "Known limitations"
    - `load_pulseq_rd` logs the device name with `torch.cuda.get_device_name()`, which fails on machines without CUDA.
    - `--split_read_polarity` is not passed on to the sorting function; polarities are always split when the
      acquisition identifiers contain `bu` / `bd`.
