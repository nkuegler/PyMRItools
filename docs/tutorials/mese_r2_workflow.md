# Tutorial: MESE R2 Mapping

This tutorial walks through the complete pipeline for R2 / T2 mapping with an accelerated multi-echo spin-echo
(MESE) sequence: sequence build, raw data sorting, LORAKS reconstruction, denoising, EMC dictionary simulation,
B1+ mapping and dictionary matching.
It needs your own scanner data; the commands show the interfaces between the tools.

All commands are run from the repository root inside the environment (`uv run …`, or activate `.venv`).
Each tool prints its full configuration at start-up and accepts `--help`.

```text
work/
├── seq/      sequence files (step 1)
├── raw/      sorted k-space (step 2)
├── recon/    LORAKS reconstruction (step 3)
├── denoised/ denoised images (step 4)
├── emc/      dictionary (step 5)
├── b1/       B1+ map (step 6)
└── fit/      parameter maps (step 7)
```

## 1. Build the sequence

Create a system specification file for your scanner, e.g. `work/specs.json` (see
[Sequence Programming](../sequence_programming.md#system-specifications-pulseqsystemspecs)), and a parameter file,
e.g. starting from `examples/seqprog/mese/params.json`.

```shell
uv run pulseq_mese -s work/specs.json -p work/params.json -o work/seq
```

This writes, among others:

- `work/seq/mese_sequence.seq` – run this on the scanner with the Pulseq interpreter,
- `work/seq/mese_pulseq_config.json` and `work/seq/mese_sampling_config.pkl` – needed in step 2,
- `work/seq/mese_kernels.pkl` and `work/seq/mese_te.json` – needed in step 5.

Check the sampling pattern and kernel plots in `work/seq/figs/` and the simulated slice profiles in
`work/seq/pulse_profiles.html`.

## 2. Sort the raw data

```shell
uv run seqprog_raw_data \
    -i work/meas_MID00042_FID07553_pulseq_mese.dat \
    -seq work/seq/mese_pulseq_config.json \
    -samp work/seq/mese_sampling_config.pkl \
    -o work/raw
```

Result: `work/raw/k_space.pt` (`[read, phase, slice, channel, echo]`, noise pre-whitened, oversampling removed),
`work/raw/affine.pt`, and a zero-filled preview `work/raw/naive_rsos_recon.nii`.

## 3. Reconstruct with AC-LORAKS

Test the parameters on the central slice first:

```shell
uv run recon_loraks -ik work/raw/k_space.pt -ia work/raw/affine.pt -o work/recon_test \
    -ps true -p 5 -rank 150
```

Then reconstruct the whole volume.
With many receive channels, use channel batches (`-b`, must divide the number of channels) to reduce memory:

```shell
uv run recon_loraks -ik work/raw/k_space.pt -ia work/raw/affine.pt -o work/recon -p 5 -rank 150 -b 8
```

Result: `work/recon/recon_img.pt` (complex, channel-wise) and the root-sum-of-squares image `work/recon/recon_img_rsos.nii`.

## 4. Denoise

Denoise the channel-combined magnitude images and correct the noise floor of the magnitude data:

```shell
uv run processing_denoise_lcpca -i work/recon/recon_img_rsos.nii -o work/denoised -nbc true
```

Result: `work/denoised/denoised_data.nii` and the bias-corrected `work/denoised/denoised_data_nbc.nii`.
Gibbs unringing (`processing_unring`) can be applied afterwards if needed.

## 5. Simulate the dictionary

Create an EMC parameter file `work/emc_params.json` that matches the sequence (echo train length and flip angles;
all timings are taken from the kernel file):

```json
{
  "etl": 6,
  "excitation_angle": 90.0,
  "excitation_phase": 90.0,
  "refocus_angle": [140.0],
  "refocus_phase": [0.0]
}
```

Simulate T2 from 10 to 200 ms and B1+ from 0.3 to 1.7 for a fixed T1 and no B0 offset.
The value lists are set in a settings file `work/emc_settings.json`:

```json
{
  "out_path": "work/emc",
  "emc_params_file": "work/emc_params.json",
  "kernel_file": "work/seq/mese_kernels.pkl",
  "te_file": "work/seq/mese_te.json",
  "database_name": "db_mese.pkl",
  "t1_list": [1.5],
  "t2_list": [[10, 200, 1]],
  "b1_list": [[0.3, 1.7, 0.02]],
  "b0_list": [0.0]
}
```

```shell
uv run simulation_emc_mese -c work/emc_settings.json
```

!!! tip
    Set value lists in the JSON file rather than on the command line, and check the parsed values in the
    configuration printed at start-up (see [Simulation](../simulation.md#settings-emcsimsettings)).

Result: `work/emc/db_mese.pkl`.

## 6. B1+ map

From an AFI acquisition with a nominal flip angle of 60° and \(TR_2/TR_1 = 5\), stored as `[x, y, z, 2]`:

```shell
uv run modeling_afi_b1 -i work/afi.nii.gz -o work/b1 -fa 60 -n 5
```

Result: `work/b1/b1_afi.nii` (percent) and `work/b1/b1_rel_err.nii`.
The B1+ map must be registered to the MESE images (same grid and orientation) before the next step.

## 7. Dictionary matching

```shell
python -m pymritools.modeling.dictionary.grid_search_channels \
    -i work/denoised/denoised_data_nbc.nii -iimg true \
    -db work/emc/db_mese.pkl \
    -b1 work/b1/b1_afi_registered.nii -b1e work/b1/b1_rel_err_registered.nii \
    -o work/fit
```

Result: `work/fit/t2.nii` [ms], `work/fit/r2.nii` / `r2_combined.nii` [1/s], the residual map and the B1+ / B0 maps used.

Without a B1+ map, B1+ is estimated jointly from the data, which works but is less robust against noise.

## Python

All steps are also available as Python functions, see the respective tool pages and the
[API reference](../api/index.md).
