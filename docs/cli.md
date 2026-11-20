# Command Line Tools

All tools accept `--help`, a JSON configuration file via `-c`, and require an output directory via `-o`
(see [Concepts](concepts.md#configuration-command-line-and-json)).

## Registered commands

These commands are defined in `[project.scripts]` of `pyproject.toml` and are available in the activated environment
(or via `uv run <command>`):

| Command | Module | Purpose | Docs |
|---|---|---|---|
| `pulseq_mese` | `pymritools.seqprog.sequences.mese` | Build a Pulseq MESE sequence | [Sequence Programming](sequence_programming.md) |
| `seqprog_raw_data` | `pymritools.seqprog.rawdata.rd_read` | Sort twix raw data into k-space tensors | [Raw Data](raw_data.md) |
| `recon_loraks` | `pymritools.recon.loraks.recon` | AC-LORAKS / P-LORAKS reconstruction | [Reconstruction](recon.md) |
| `processing_denoise_lcpca` | `pymritools.processing.denoising.lcpca.denoise` | PCA denoising with optional noise bias correction | [Processing](processing.md#pca-denoising) |
| `processing_unring` | `pymritools.processing.unringing.gibbs_unr` | Gibbs unringing | [Processing](processing.md#gibbs-unringing) |
| `simulation_emc_mese` | `pymritools.simulation.emc.sequence.mese` | EMC simulation of a MESE sequence | [Simulation](simulation.md) |
| `modeling_afi_b1` | `pymritools.modeling.b1_afi.modeling` | B1+ map from AFI data | [Modeling](modeling.md#afi-b1-mapping) |

## Module entry points

Further tools have a `main()` function but no registered command.
Run them as modules:

```shell
python -m <module> --help
```

| Module | Purpose | Docs |
|---|---|---|
| `pymritools.seqprog.sequences.megesse_sym` | Build a Pulseq MEGESSE sequence with symmetric echoes | [Sequence Programming](sequence_programming.md) |
| `pymritools.seqprog.sequences.megesse_asym` | Build a Pulseq MEGESSE sequence with asymmetric echoes | [Sequence Programming](sequence_programming.md) |
| `pymritools.simulation.emc.sequence.megesse_sym` | EMC simulation of the symmetric MEGESSE sequence | [Simulation](simulation.md) |
| `pymritools.simulation.emc.sequence.megesse_asym` | EMC simulation of the asymmetric MEGESSE sequence | [Simulation](simulation.md) |
| `pymritools.simulation.emc.sequence.pulse` | Slice profile simulation of a single RF pulse from a kernel file | [Simulation](simulation.md#single-pulse-profiles) |
| `pymritools.modeling.dictionary.grid_search_channels` | Dictionary matching (R2, B1+, B0) of combined or channel-wise MESE data | [Modeling](modeling.md#dictionary-matching) |
| `pymritools.modeling.dictionary.grid_search` | Older voxel-wise dictionary grid search | [Modeling](modeling.md#legacy-grid-search) |
| `pymritools.modeling.decay.mexp` | Mono-exponential decay fit | [Modeling](modeling.md#mono-exponential-fitting) |
| `pymritools.processing.coil_sens.smap` | Rough coil sensitivity estimates (coil / RSoS, low-pass filtered) | [Processing](processing.md#coil-sensitivity-estimates) |

!!! note
    The `pymritools.optimization` scripts (variable flip angle and SMS pulse optimization) and the
    `cvae` / `differential_evolution` modules in `pymritools.modeling.dictionary` are experiments with hard-coded settings.
    They are not documented as tools.

## Example

```shell
# show all options
uv run processing_denoise_lcpca --help

# run with a configuration file, overriding the output path on the command line
uv run processing_denoise_lcpca -c examples/processing/denoising/config.json -o ./results/denoising
```
