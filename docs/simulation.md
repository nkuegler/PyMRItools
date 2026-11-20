# Simulation (EMC)

`pymritools.simulation.emc` simulates echo modulation curves (EMC), i.e. the echo amplitudes of a multi-echo
sequence as a function of T1, T2, B1+ and B0.
The approach follows Ben-Eliezer et al. (2015): the actual RF pulse shapes and slice-selective gradients are
simulated across the slice profile, so stimulated and indirect echoes from imperfect refocusing are part of the
dictionary.
The resulting dictionary (database) is used for [dictionary matching](modeling.md#dictionary-matching).

## Simulation model

**Spatial sample.**
The slice direction \(z\) is sampled by `sample_number` isochromats in \([-L_z, L_z]\) with \(L_z\) = `length_z`.
Their equilibrium magnetization follows a flat-top profile (generalized normal distribution).
Each isochromat carries the vector \((M_x, M_y, M_z, M_0)\); the fourth (constant) component allows writing the
T1 recovery as a matrix product.

**RF pulses and gradients.**
Pulses and gradients are sampled on a time raster \(\Delta t\) (hard-pulse approximation).
Per time step, an isochromat at position \(z\) is rotated by

- \(\phi_x = 2\pi\gamma\,\Delta t\,B_{1,x}\) and \(\phi_y = 2\pi\gamma\,\Delta t\,B_{1,y}\) about x and y
  (the RF amplitude is scaled by the B1+ value of the dictionary entry),
- \(\phi_z = 2\pi\gamma\,\Delta t\,G_z z\) about z (slice gradient),

with \(\gamma\) in Hz/T.

**Relaxation and off-resonance.**
Free precession over a duration \(\Delta t\) is applied as

\[
\begin{pmatrix} M_x \\ M_y \\ M_z \\ M_0 \end{pmatrix}
\leftarrow
\mathbf{R}_z(2\pi\,\Delta B_0\,\Delta t)
\begin{pmatrix}
E_2 & 0 & 0 & 0\\
0 & E_2 & 0 & 0\\
0 & 0 & E_1 & 1-E_1\\
0 & 0 & 0 & 1
\end{pmatrix}
\begin{pmatrix} M_x \\ M_y \\ M_z \\ M_0 \end{pmatrix},
\quad E_{1,2} = e^{-\Delta t/T_{1,2}},
\]

with the B0 offset \(\Delta B_0\) in Hz.

**Signal.**
For each echo, the signal is the sum of \(M_x + iM_y\) over all isochromats at the echo time.
With `signal_fourier_sampling`, the readout is instead simulated with the readout gradient applied along \(z\)
and the slice profile is resolved in `acquisition_number` bins.

All dictionary entries are simulated in parallel as one tensor with dimensions
`[T1, T2, B1, B0, samples, 4]`, on the GPU if available.

## Sequences

| Class | Module | Command |
|---|---|---|
| `MESE` | `pymritools.simulation.emc.sequence.mese` | `simulation_emc_mese` |
| `MEGESSE` (symmetric) | `pymritools.simulation.emc.sequence.megesse_sym` | `python -m pymritools.simulation.emc.sequence.megesse_sym` |
| `MEGESSE` (asymmetric) | `pymritools.simulation.emc.sequence.megesse_asym` | `python -m pymritools.simulation.emc.sequence.megesse_asym` |

All derive from `pymritools.simulation.emc.sequence.base_sequence.Simulation`,
which sets up the device, the `SimulationData` tensors and the RF pulse, and provides `simulate()`, `get_db()` and `save()`.

The RF and gradient waveforms can come from two sources:

1. **Kernel file (recommended).** The [sequence build](sequence_programming.md#output-files) writes
   `<name>_kernels.pkl` and `<name>_te.json`.
   With `--kernel_file` and `--te_file` set, the exact pulse and gradient shapes of the sequence are simulated,
   and durations and echo spacing are taken from the kernels.
   The echo train length and the flip angles (`etl`, `excitation_angle`, `refocus_angle`) are still taken from the
   EMC parameters, so they must match the sequence.
2. **Parameter file.** Without a kernel file, trapezoidal gradients are built from the amplitudes and durations in the
   EMC parameter file, and the pulse shape is read from `--pulse_file` (an `RFPulse` object as JSON or pickle).

## Command line usage

```shell
uv run simulation_emc_mese \
    -emc emc_params.json -kernels mese_kernels.pkl -te mese_te.json \
    -o ./emc -db db_mese.pkl
```

or with a configuration file, as in the repository example:

```shell
uv run simulation_emc_mese -c examples/simulation/emc/emc_settings.json
```

### Settings (`EmcSimSettings`)

| Option | Alias | Default | Description |
|---|---|---|---|
| `--emc_params_file` | `-emc` | `./examples/simulation/emc/emc_params.json` | EMC parameter file (see below). For the MESE command it is always loaded if set. |
| `--kernel_file` | `-kernels` | `./examples/simulation/kernels/kernels` | Sequence kernels `.pkl`; if the file does not exist, the parameter file is used. |
| `--te_file` | `-te` | `""` | Echo times `.json` [ms]; required with a kernel file. |
| `--pulse_file` | `-pul` | example file | RF pulse shape, used without a kernel file. |
| `--database_name` | `-db` | `database_test.pkl` | File name of the resulting database. |
| `--t1_list` | `-var_t1` | `[1.5]` | T1 values [s]. |
| `--t2_list` | `-var_t2` | `[[25, 30, 0.5], [30, 35, 1]]` | T2 values [ms]. |
| `--b1_list` | `-var_b1` | `[0.6, 1.6, 0.1]` | B1+ values (relative to nominal). |
| `--b0_list` | `-var_b0` | `[[0, 1, 2]]` | B0 offsets [Hz]. |
| `--signal_fourier_sampling` | `-sfs` | `False` | Simulate the readout instead of summing over the slice profile. |
| `--resample_pulse_to_dt_us` | `-rptdt` | `5.0` | Resample pulses to this raster [µs] to speed up the simulation. |
| `--b0_batch_size` | `-bs` | `1` | Number of B0 values simulated at once. |
| `--b1_batch_size` | `-bsb1` | `-1` | Number of B1 values simulated at once (`-1`: all). |

**Value lists.**
Each entry of `t2_list`, `b1_list` and `b0_list` is either a single value or a range `[start, stop, step]`
(stop excluded, as in `torch.arange`).
A flat list of numbers is interpreted as single values, so `[0.6, 1.6, 0.1]` means the three values 0.6, 1.6 and 0.1.
Write `[[0.6, 1.6, 0.05]]` for a range.
`t1_list` only takes single values.

**Batching.**
The simulation loops over batches of B0 and B1 values to limit the memory usage.
Intermediate databases of each batch are written to `<out_path>/tmp/` and combined at the end.

### Parameters (`EmcParameters`)

Without a kernel file, the parameter file defines the sequence:

| Field | Default | Unit / meaning |
|---|---|---|
| `etl` | 16 | Echo train length |
| `esp` | 9.0 | Echo spacing [ms] |
| `bw` | 349 | Readout bandwidth [Hz/px] |
| `excitation_angle`, `excitation_phase` | 90.0, 90.0 | [°] |
| `gradient_excitation`, `duration_excitation` | -18.5, 2560.0 | Slice-select gradient [mT/m], pulse duration [µs] |
| `gradient_excitation_rephase`, `duration_excitation_rephase` | -10.51, 1080.0 | [mT/m], [µs] |
| `refocus_angle`, `refocus_phase` | [140.0], [0.0] | [°], one value per echo (the last value is repeated up to `etl`) |
| `gradient_refocus`, `duration_refocus` | -36.2, 3584.0 | [mT/m], [µs] |
| `gradient_crush`, `duration_crush` | -38.7, 1000.0 | Crusher [mT/m], [µs] |
| `*_verse_lobes` | 0.0 | Gradient and duration of VERSE lobes |
| `tes` | [0.0] | Echo times |
| `sample_number` | 1000 | Number of isochromats along z |
| `length_z` | 0.005 | Half extent of the simulated sample [m] |
| `acquisition_number` | 50 | Number of bins for `signal_fourier_sampling` |

### Output files

| File | Content |
|---|---|
| `<database_name>.pkl` | The dictionary as `DB` object |
| `emc_settings.json`, `emc_params.json` | Settings and (updated) parameters used |
| `tmp/*.pkl` | Databases of the individual B0 / B1 batches |
| `figs/` | Pulse / gradient shapes, kernels and database plots (with `--visualize`) |

## The dictionary database (`DB`)

`pymritools.config.database.DB` stores the simulated curves in a [polars](https://pola.rs/) DataFrame with the
columns `index, t1, t2, b1, b0, echo_num, magnitude, phase` (`t1`, `t2` in s), together with the `EmcParameters` used.

```python
from pymritools.config.database import DB

db = DB.load("emc/db_mese.pkl")
mag, phase = db.get_torch_tensors_t1t2b1b0e()        # each [n_t1, n_t2, n_b1, n_b0, n_echoes]
t1, t2, b1, b0 = db.get_t1_t2_b1_b0_values()          # unique values (torch tensors)
db.plot(out_path="emc/figs", t2_range_ms=(20, 80), b1_range=(0.5, 1.4), format="html")

combined = DB.add_dbs(db_1, db_2)                     # merge databases simulated for different value ranges
```

## Python usage

```python
from pymritools.config.emc import EmcSimSettings, EmcParameters
from pymritools.simulation.emc.sequence import MESE

settings = EmcSimSettings(
    out_path="./emc", kernel_file="seq/mese_kernels.pkl", te_file="seq/mese_te.json",
    t1_list=[1.5], t2_list=[[20, 100, 1]], b1_list=[[0.5, 1.5, 0.05]], b0_list=[0.0],
    visualize=False,
)
# with a kernel file, timings come from the kernels; echo train length and flip angles from the parameters
params = EmcParameters(etl=6, excitation_angle=90.0, refocus_angle=[140.0], refocus_phase=[0.0])

sim = MESE(params=params, settings=settings)
sim.simulate()
db = sim.get_db()
db.save("./emc/db_mese.pkl")
```

`pymritools.simulation.emc.sequence.mese.simulate(settings, params)` runs the batched simulation of the command line tool.

## Single pulse profiles

`pymritools.simulation.emc.sequence.pulse` simulates the excitation and first refocusing pulse of a kernel file
and plots the magnetization profiles across the slice (`mag_profiles.html`):

```shell
python -m pymritools.simulation.emc.sequence.pulse -kf seq/mese_kernels.pkl -o ./pulse_profiles
```

When building sequences, `--simulate_pulse_profiles` performs a similar check for a whole echo train
(`simulate_pulse_profiles` in `mese.py`).
