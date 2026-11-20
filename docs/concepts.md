# Concepts & Conventions

This page describes the conventions shared by all tools: how settings are passed, which dimension orders are expected,
how Fourier transforms are defined, and which file formats are read and written.

## Configuration: command line and JSON

All command line tools are built on [simple-parsing](https://github.com/lebrice/SimpleParsing).
Each tool has a settings dataclass derived from `pymritools.config.BaseClass`.
Every field of that dataclass is a command line option (long name `--field_name` plus an optional short alias),
and the whole dataclass can be stored as or loaded from a JSON file.

The base class provides the options shared by all tools:

| Option | Alias | Default | Description |
|---|---|---|---|
| `--config_file` | `-c` | `""` | JSON file with entries of the settings dataclass. |
| `--out_path` | `-o` | `""` | Output directory. **Required** – the tools stop with an error if it is empty. |
| `--use_gpu` | `-gpu` | `True` | Use a CUDA device if available. |
| `--gpu_device` | `-gpud` | `0` | Index of the CUDA device. |
| `--visualize` | `-v` | `True` | Write additional figures (mostly interactive plotly `.html` files). |
| `--debug` | `-d` | `False` | Debug mode (only used by few tools). |
| `--slurm` | | `False` | Intended for cluster runs: reduce visual output and progress bars. |

`BaseClass.from_cli()` combines both sources in the following order:

1. Parse the command line (missing options take their default values).
2. If `--config_file` points to an existing file, load all settings from it.
3. Re-apply every command line option whose value **differs from the default**.

!!! note
    Because of step 3, a command line option that is explicitly set to its *default* value does not override a different value from the configuration file.
    Edit the JSON file in that case.

Boolean options take an explicit value on the command line, e.g. `--use_gpu false`.
List options (e.g. value ranges of the simulation) are best set in the JSON configuration file,
because list entries given on the command line are not always converted to numbers.

A typical JSON configuration file just mirrors the dataclass fields:

```json
{
  "out_path": "./results/denoising",
  "use_gpu": true,
  "in_path": "./data/mese_echoes.nii.gz",
  "fixed_p": 0
}
```

In Python, the dataclasses can be used directly, they do not need the command line:

```python
from pymritools.config.processing import DenoiseSettingsLCPCA

settings = DenoiseSettingsLCPCA(in_path="mese.nii.gz", out_path="./out")
settings.save_json("denoise_config.json", indent=2)
settings = DenoiseSettingsLCPCA.load("denoise_config.json")
```

Example configuration files are provided in the `examples/` directory of the repository.
Some of them are older than the current settings classes and may contain outdated keys; check them against `--help`.

## Dimension conventions

Most tools work on PyTorch tensors with a fixed order of dimensions.
The abbreviation in variable names tells the order, e.g. `k_space_rpsct`.

| Symbol | Dimension |
|---|---|
| `r` / `x` | read direction (frequency encoding) |
| `p` / `y` | phase encoding direction |
| `s` / `z` | slice |
| `c` | receive channel (coil) |
| `t` / `e` | echo / contrast / time |

- **k-space** (output of the raw data tools, input of LORAKS): `[n_read, n_phase, n_slice, n_channels, n_echoes]`.
  For 2D multi-slice sequences the slice dimension is already in image space ("hybrid" k-space), so only the first two
  dimensions are Fourier encoded.
- **Images** (NIfTI in and output): `[x, y, z, echoes]` for channel-combined data or
  `[x, y, z, channels, echoes]` for channel-wise data.
  Several tools add singleton dimensions if channel or echo dimensions are missing.

## Fourier transforms

`pymritools.utils` provides two centered n-dimensional FFT helpers:

```python
from pymritools.utils import fft_to_img, ifft_to_k

img = fft_to_img(k_space, dims=(0, 1))   # fftshift(fftn(ifftshift(k)))
k = ifft_to_k(img, dims=(0, 1))          # fftshift(ifftn(ifftshift(img)))
```

Both accept numpy arrays and torch tensors and use `norm="backward"` by default.
Note that, contrary to the more common convention, the transform *into* image space is the forward FFT.
The pair is consistent within PyMRItools, but images can appear flipped when mixing these helpers with other software.

## File formats

| Format | Used for | Helpers |
|---|---|---|
| `.pt` | complex k-space and image tensors, affine matrices | `torch_save(data, path_to_file, file_name)`, `torch_load(path)` |
| `.nii` / `.nii.gz` | images and parameter maps | `nifti_save(data, img_aff, path_to_dir, file_name)`, `nifti_load(path)` |
| `.npy` | numpy arrays | `numpy_save`, `numpy_load` |
| `.pkl` | EMC dictionary databases (`DB`), sequence kernels, sampling configuration | `DB.save` / `DB.load`, `Kernel.load`, `Sampling.load` |
| `.json` | settings, sequence parameters, echo times, RF pulses | `save_json` / `load` of the dataclasses |
| `.seq` | Pulseq sequence files | written by the sequence tools |

Details of the I/O helpers:

- `torch_load` maps tensors to the CPU and casts to `float32` / `complex64` (integer and boolean tensors are kept).
- `nifti_load` returns `(data, nibabel_image)`; `data` is a numpy array.
- `nifti_save` accepts a nibabel image, a numpy array or a torch tensor as affine.
  Complex data is written as two files `<name>_magnitude.nii` and `<name>_phase.nii`.
- When a tool reads `.pt` data and writes NIfTI output, it needs the 4×4 affine matrix as `.pt` file
  (e.g. `affine.pt` from the raw data tools). Without it, the identity matrix is used, and the images may be misaligned in NIfTI viewers.

## Devices

Tools select `cuda:<gpu_device>` if `use_gpu` is set and CUDA is available, and fall back to the CPU otherwise.
Python functions usually take a `device` argument or run on the device of their input tensor.
Large problems are processed in batches (slices, channels, voxels); reduce the respective batch size options when running out of GPU memory.

## Logging

Command line tools set up [rich](https://github.com/Textualize/rich) logging with `setup_program_logging()` and print the full configuration at start-up.
When using the Python API, configure logging yourself, e.g.:

```python
import logging
from pymritools.config import setup_program_logging

setup_program_logging(name="My pipeline", level=logging.INFO)
```
