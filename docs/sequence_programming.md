# Sequence Programming

`pymritools.seqprog` builds 2D multi-slice [Pulseq](https://pulseq.github.io/) sequences on top of
[PyPulseq](https://github.com/imr-framework/pypulseq).
Besides the `.seq` file, every build writes the information needed later in the pipeline:
the sampling pattern for sorting the raw data, the sequence parameters, the echo times, and the RF / gradient
kernels for the [EMC simulation](simulation.md).

## Building blocks

The code is organised in three layers.

### Events (`pymritools.seqprog.core.events`)

Thin wrappers around the PyPulseq event types that keep track of durations, delays and raster times:

| Class | Content | Factory methods |
|---|---|---|
| `RF` | RF pulse shape, flip angle, phase, frequency offset, bandwidth, timing | `make_gauss_pulse`, `make_sinc_pulse`, `load_from_pypsi_pulse` (from an `RFPulse` file) |
| `GRAD` | Gradient on one channel (`x`, `y`, `z`) | `make_trapezoid`, `make_slice_selective`, `sym_grad` |
| `ADC` | Readout | `make_adc` |
| `DELAY` | Delay block | `make_delay` |

Each event provides `get_duration()` and `to_simple_ns()`, which returns the `SimpleNamespace` object PyPulseq expects.
`RFPulse` (`pymritools.config.rf`) is a serializable container for externally designed pulse shapes.

### Kernels (`pymritools.seqprog.core.Kernel`)

A `Kernel` is one sequence block: an RF event, read / phase / slice gradients, an ADC and a delay.
Class methods create the blocks used by the sequences, e.g.

- `Kernel.excitation_slice_sel(...)` – slice-selective excitation with pre- and rephasing gradients,
- `Kernel.refocus_slice_sel_spoil(...)` – slice-selective refocusing with spoiling (crusher) gradients,
- `Kernel.acquisition_fs(...)`, `acquisition_pf_undersampled(...)`, `acquisition_sym_undersampled(...)`,
  `acquisition_fid_nav(...)` – readouts,
- `Kernel.spoil_all_grads(...)` – end-of-train spoiling.

Kernels can be plotted (`Kernel.plot(path, name)`, interactive `.html`), copied, and pickled (`save` / `load`).

### Sequences (`pymritools.seqprog.sequences`)

`Sequence2D` is the abstract base class of all 2D sequences.
It handles everything that is shared between sequences:

- system limits (`PulseqSystemSpecs` → `pypulseq.Opts`),
- slice positions, gaps, and interleaved or sequential slice order,
- the phase encoding sampling pattern (fully sampled central lines plus undersampled outer lines, see below),
- noise pre-scans (`number_noise_scans`) for coil noise decorrelation,
- optional EPI-style navigators outside of the slice slab (`use_navs`),
- the TR / slice delay calculation,
- writing all output files and plots.

The concrete sequences are

| Class | Module | Description |
|---|---|---|
| `MESE` | `pymritools.seqprog.sequences.mese` | Multi-echo spin echo with one readout per refocusing pulse |
| `MEGESSE` | `pymritools.seqprog.sequences.megesse_sym` | Multi-echo gradient and spin echo: per refocusing pulse three gradient echoes before and three after the spin echo (7 readouts) |
| `MEGESSE` | `pymritools.seqprog.sequences.megesse_asym` | Multi-echo gradient and spin echo: per refocusing pulse one spin echo and four gradient echoes (5 readouts) |

The MEGESSE variants alternate the readout polarity ("blip up" / "blip down", kernels `acq_bu` and `acq_bd`).

New sequences derive from `Sequence2D` and implement `_build_variant()` (timing calculations)
and `_loop_slices()` (adding the blocks per phase encode and slice).

## Command line usage

```shell
uv run pulseq_mese -s system_specs.json -p params.json -o ./mese_seq
python -m pymritools.seqprog.sequences.megesse_asym -s system_specs.json -p params.json -o ./megesse_seq
```

Both the system specifications file (`-s`) and the parameter file (`-p`) are required.

### Configuration (`PulseqConfig`)

In addition to the [common options](concepts.md#configuration-command-line-and-json):

| Option | Alias | Default | Description |
|---|---|---|---|
| `--system_specs_file` | `-s` | `""` | JSON file with the scanner limits (`PulseqSystemSpecs`). |
| `--parameter_file` | `-p` | `""` | JSON file with the sequence parameters (`PulseqParameters2D`). |
| `--pulse_file_excitation` | `-rfe` | `""` | Optional external pulse shape file for the excitation pulses. |
| `--pulse_file_refocusing` | `-rfr` | `""` | Optional external pulse shape file for the refocusing pulses. |
| `--report` | `-r` | `False` | Write a PyPulseq test report and timing check (`report.txt`). Slow. |
| `--simulate_pulse_profiles` | `-spp` | `True` | Simulate the slice profiles of one echo train with the EMC code after the build. |
| `--version` | | `"1.0"` | Sequence version string. |

### System specifications (`PulseqSystemSpecs`)

```json
{
  "b_0": 6.98,
  "max_grad": 100.0,
  "grad_unit": "mT/m",
  "max_slew": 140.0,
  "slew_unit": "T/m/s",
  "rise_time": 0,
  "grad_raster_time": 1e-05,
  "rf_dead_time": 0.0001,
  "rf_raster_time": 1e-06,
  "rf_ringdown_time": 3e-05,
  "adc_dead_time": 2e-05,
  "gamma": 42577478.518
}
```

The values above are the defaults of the class; adapt them to your scanner (and to your local safety limits).

### Sequence parameters (`PulseqParameters2D`)

The most important fields (see `pymritools/config/seqprog/settings.py` for all of them):

| Group | Field | Default | Unit / meaning |
|---|---|---|---|
| Resolution | `resolution_fov_read` | 212.0 | FoV in read direction [mm] |
| | `resolution_fov_phase` | 100.0 | FoV in phase direction [% of read FoV] |
| | `resolution_base` | 212 | Number of read encodes (rounded up to a multiple of 4) |
| | `resolution_slice_thickness` | 1.0 | [mm] |
| | `resolution_slice_num` | 30 | Number of slices |
| | `resolution_slice_gap` | 200.0 | Gap between slices [% of slice thickness] |
| Acceleration | `number_central_lines` | 36 | Fully sampled central phase encodes (AC region) |
| | `acceleration_factor` | 4.0 | Undersampling factor of the outer phase encodes |
| | `sampling_pattern` | `"optimized"` | `optimized`, `weighted_random`, `random`, `interleaved_lines`, `grappa` |
| Excitation | `excitation_rf_fa`, `excitation_rf_phase` | 90.0, 90.0 | [°] |
| | `excitation_rf_time_bw_prod`, `excitation_duration` | 2.0, 2000 | time-bandwidth product, [µs] |
| Refocusing | `refocusing_rf_fa`, `refocusing_rf_phase` | 140.0, 0.0 | [°], single value or one value per echo |
| | `refocusing_rf_time_bw_prod`, `refocusing_duration` | 2.0, 2500 | time-bandwidth product, [µs] |
| | `refocusing_grad_slice_scale` | 1.5 | Widens the refocused slice relative to the excited slice |
| Spoiling | `grad_moment_slice_spoiling`, `grad_moment_slice_spoiling_end` | 0.0, 2500.0 | [Hz] (gradient moment in units of γ·G·t) |
| Readout | `readout_dwell_us`, `oversampling` | 4, 2 | [µs], readout oversampling factor |
| | `acq_phase_dir` | `"RL"` | Phase encoding direction, `RL` or `PA` |
| Timing | `esp` | 8.92 | Echo spacing [ms] |
| | `etl` | 6 | Echo train length |
| | `tr` | 4500.0 | Repetition time [ms] |
| | `tr_et_delay` | 20.0 | Delay between echo trains of successive slices [ms] |
| Other | `number_noise_scans` | 10 | Noise-only readouts before the sequence |
| | `use_navs` | `False` | EPI navigators |
| | `rf_adapt_z` | `False` | Slice-dependent RF scaling |

Some values are adapted during the build, and the build log reports each change:

- The readout bandwidth follows from dwell time, oversampling and base resolution:
  \( \mathrm{BW} = 1 / (n_\mathrm{read} \cdot \mathrm{os} \cdot t_\mathrm{dwell}) \).
- The MESE implementation calculates the minimum echo spacing from the kernel durations and uses it as ESP.
- If `tr_et_delay` is larger than 1 ms, TR is calculated from the echo train duration plus this delay for all slices,
  and the larger of this value and the given `tr` is used.

### k-space sampling

Phase encodes are sampled per echo:
the central `number_central_lines` are acquired for every echo, and the outer lines are undersampled by `acceleration_factor`.
The outer lines can differ between echoes (complementary sampling), which is what the joint
multi-echo [LORAKS reconstruction](recon.md) uses:

- `optimized`: density weighted towards the AC region with a repulsion radius in the phase encode and echo directions,
  so that neighbouring echoes do not sample the same or adjacent lines,
- `weighted_random` / `random`: random outer lines with / without a weighting towards the center,
- `interleaved_lines`: regular undersampling shifted by one line per echo,
- `grappa`: the same regular undersampling for all echoes.

The random number generator is seeded, so the patterns are reproducible.

### Output files

With the base name `<name>` (`mese` or `megesse` for the command line tools), the output directory contains:

| File | Content |
|---|---|
| `<name>_sequence.seq` | Pulseq sequence file |
| `<name>_calibration_sequence.seq` | Optional calibration sequence (if the variant uses one) |
| `<name>_pulseq_config.json` | Final sequence parameters (after all adaptions) – needed for the [raw data](raw_data.md) sorting |
| `<name>_sampling_config.pkl` | `Sampling` object: sampling pattern of every ADC and the k-space trajectories – needed for the raw data sorting |
| `<name>_kernels.pkl` | Dictionary of the RF / gradient kernels (`excitation`, `refocus_1`, `refocus_2`, …, `acq`) – input for the [EMC simulation](simulation.md) |
| `<name>_te.json` | Echo times [ms] – input for the EMC simulation |
| `<name>_pulseq_report.json` | Timing errors reported by PyPulseq when writing (if any) |
| `report.txt` | PyPulseq test report (only with `--report`) |
| `figs/` | Kernel, sequence and sampling pattern plots (with `--visualize`) |
| `pulse_profiles.html` / `.pkl` | Simulated slice profiles (with `--simulate_pulse_profiles`) |

## Python usage

```python
from pymritools.config.seqprog import PulseqConfig, PulseqSystemSpecs, PulseqParameters2D
from pymritools.seqprog.sequences import build
from pymritools.seqprog.sequences.mese import MESE

config = PulseqConfig(out_path="./mese_seq", visualize=False, simulate_pulse_profiles=False)
specs = PulseqSystemSpecs.load("system_specs.json")
params = PulseqParameters2D.load("params.json")

mese = MESE(config=config, specs=specs, params=params)
build(config=config, sequence=mese, name="mese")
```

Individual events and kernels can be used to compose new sequences:

```python
import pypulseq as pp
from pymritools.seqprog.core import events

system = pp.Opts(max_grad=40, grad_unit="mT/m", max_slew=150, slew_unit="T/m/s")
rf = events.RF.make_gauss_pulse(flip_angle_rad=0.5 * 3.1416, system=system, duration_s=2e-3)
print(rf.get_duration())
```

!!! note
    The event and kernel factories take many keyword arguments that depend on each other.
    Check their signatures in the [API reference](api/seqprog.md) and the existing sequences for working examples.
