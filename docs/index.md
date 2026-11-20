# PyMRItools

PyMRItools is a Python toolbox for MRI sequence programming, raw data handling, reconstruction, simulation, processing and quantitative modeling.
Most numerically heavy parts are written in [PyTorch](https://pytorch.org/) and can run on a GPU.

The toolbox grew out of a research workflow for quantitative R2 / T2 mapping with multi-echo spin-echo (MESE) and
multi-echo gradient and spin-echo (MEGESSE) sequences built with [Pulseq](https://pulseq.github.io/).
The individual tools can be used on their own, but they are designed to work as one pipeline:

```text
Sequence programming (seqprog) ── .seq, sampling_config.pkl, pulseq_config.json
        │                    └─── kernels.pkl, te.json ──┐
        ▼                                                ▼
Scanner measurement                              EMC Bloch simulation (simulation.emc)
        │                                                │
        ▼                                                │
Raw data sorting (seqprog.rawdata) ── k_space.pt         │
        │                                                │
        ▼                                                │
Reconstruction (recon.loraks): AC-LORAKS / P-LORAKS      │
        │                                                │
        ▼                                                │
Processing: denoising, unringing, coil compression       │
        │                                                │ dictionary (DB .pkl)
        ▼                                                ▼
Modeling: dictionary matching (T2/R2, B1+), AFI B1+, mono-exponential fit
```

## Modules

| Module | Content | Documentation |
|---|---|---|
| `pymritools.seqprog` | Event / kernel based Pulseq sequence programming (MESE, MEGESSE), sampling patterns | [Sequence Programming](sequence_programming.md) |
| `pymritools.seqprog.rawdata` | Sorting Siemens twix raw data of Pulseq or vendor sequences into k-space tensors | [Raw Data](raw_data.md) |
| `pymritools.recon.loraks` | GPU implementation of AC-LORAKS and P-LORAKS for undersampled multi-channel, multi-echo data | [Reconstruction](recon.md) |
| `pymritools.processing` | PCA denoising, noise bias correction, Gibbs unringing, coil compression, coil sensitivity estimates | [Processing](processing.md) |
| `pymritools.simulation.emc` | Echo modulation curve (EMC) Bloch simulations of slice-selective MESE / MEGESSE sequences | [Simulation](simulation.md) |
| `pymritools.modeling` | Dictionary matching (T2/R2, B1+), mono-exponential fitting, AFI B1+ mapping, ESPIRiT maps | [Modeling](modeling.md) |
| `pymritools.config` | Configuration dataclasses (command line + JSON), the dictionary database `DB`, RF pulse objects | [Concepts](concepts.md) |
| `pymritools.utils` | FFT helpers, data I/O (NIfTI, torch, numpy), indexing helpers, randomized SVDs, phantoms | [API Reference](api/utils.md) |

## Quick start

Install the package (see [Installation](installation.md) for details):

```shell
git clone https://github.com/schmidt-jo/PyMRItools.git
cd PyMRItools
uv sync
```

Every tool reads its settings from command line arguments and/or a JSON configuration file.
An output path (`-o`) is always required.
Run any tool with `--help` to see all options:

```shell
# AC-LORAKS reconstruction of a k-space tensor
uv run recon_loraks -ik k_space.pt -ia affine.pt -o ./recon -p 5 -rank 150

# PCA denoising of a multi-echo NIfTI image
uv run processing_denoise_lcpca -i mese_echoes.nii.gz -o ./denoised

# B1+ map from a two-TR AFI acquisition
uv run modeling_afi_b1 -i afi.nii.gz -o ./b1 -fa 60 -n 5
```

The same functionality is available from Python, e.g.:

```python
import torch
from pymritools.recon.loraks.loraks import Loraks
from pymritools.recon.loraks.ac_loraks import AcLoraksOptions

opts = AcLoraksOptions(loraks_neighborhood_size=5, max_num_iter=20)
opts.rank.value = 150
loraks = Loraks.create(opts)
# k-space batches with dims [batch, channels * echoes, phase, read]
k_recon = loraks.reconstruct(k_batched)
```

See [Command Line Tools](cli.md) for a list of all entry points and the
[MESE R2 mapping tutorial](tutorials/mese_r2_workflow.md) for a complete pipeline.

!!! warning "Research code"
    PyMRItools is research software under active development.
    Interfaces change, not every combination of options is tested, and some modules are experimental.
    Known limitations are marked in the respective pages.
    Please check results carefully and get in touch via
    [GitHub issues](https://github.com/schmidt-jo/PyMRItools/issues) if something does not work.

## References

Parts of the implemented code are based on the work of peers.
If you use the respective tools, please consider citing their contributions:

- **EMC simulation and dictionary fitting:**
  Ben-Eliezer N, Sodickson DK, Block KT.
  Rapid and accurate T2 mapping from multi-spin-echo data using Bloch-simulation-based reconstruction.
  *Magn Reson Med* 2015;73:809–817. [doi:10.1002/mrm.25156](https://doi.org/10.1002/mrm.25156)
- **LORAKS:**
  Haldar JP. Low-rank modeling of local k-space neighborhoods (LORAKS) for constrained MRI.
  *IEEE Trans Med Imaging* 2014;33:668–681. [doi:10.1109/TMI.2013.2293974](https://doi.org/10.1109/TMI.2013.2293974)
- **AC-LORAKS:**
  Haldar JP. Autocalibrated LORAKS for fast constrained MRI reconstruction.
  *Proc IEEE ISBI* 2015:910–913. [doi:10.1109/ISBI.2015.7164018](https://doi.org/10.1109/ISBI.2015.7164018)
- **P-LORAKS:**
  Haldar JP, Zhuo J. P-LORAKS: Low-rank modeling of local k-space neighborhoods with parallel imaging data.
  *Magn Reson Med* 2016;75:1499–1514. [doi:10.1002/mrm.25717](https://doi.org/10.1002/mrm.25717)
- **Joint multi-contrast LORAKS:**
  Bilgic B, Kim TH, Liao C, et al. Improving parallel imaging by jointly reconstructing multi-contrast data.
  *Magn Reson Med* 2018;80:619–632. [doi:10.1002/mrm.27076](https://doi.org/10.1002/mrm.27076)
- **PCA denoising:**
  Does MD, Olesen JL, Harkins KD, et al. Evaluation of principal component analysis image denoising on multi-exponential MRI relaxometry.
  *Magn Reson Med* 2019;81:3503–3514. [doi:10.1002/mrm.27658](https://doi.org/10.1002/mrm.27658);
  Veraart J, Fieremans E, Novikov DS. Diffusion MRI noise mapping using random matrix theory.
  *Magn Reson Med* 2016;76:1582–1593. [doi:10.1002/mrm.26059](https://doi.org/10.1002/mrm.26059)
- **Noise bias correction and noise estimation:**
  Manjón JV, Coupé P, Buades A. MRI noise estimation and denoising using non-local PCA.
  *Med Image Anal* 2015;22:35–47. [doi:10.1016/j.media.2015.01.004](https://doi.org/10.1016/j.media.2015.01.004);
  St-Jean S, De Luca A, Tax CMW, Viergever MA, Leemans A. Automated characterization of noise distributions in diffusion MRI data.
  *Med Image Anal* 2020;65:101758. [doi:10.1016/j.media.2020.101758](https://doi.org/10.1016/j.media.2020.101758)
  ([autodmri](https://github.com/samuelstjean/autodmri))
- **Gibbs unringing:**
  Kellner E, Dhital B, Kiselev VG, Reisert M. Gibbs-ringing artifact removal based on local subvoxel-shifts.
  *Magn Reson Med* 2016;76:1574–1581. [doi:10.1002/mrm.26054](https://doi.org/10.1002/mrm.26054)
- **Coil compression:**
  Huang F, Vijayakumar S, Li Y, Hertel S, Duensing GR. A software channel compression technique for faster reconstruction with many channels.
  *Magn Reson Imaging* 2008;26:133–141. [doi:10.1016/j.mri.2007.04.010](https://doi.org/10.1016/j.mri.2007.04.010);
  Zhang T, Pauly JM, Vasanawala SS, Lustig M. Coil compression for accelerated imaging with Cartesian sampling.
  *Magn Reson Med* 2013;69:571–582. [doi:10.1002/mrm.24267](https://doi.org/10.1002/mrm.24267)
- **ESPIRiT:**
  Uecker M, Lai P, Murphy MJ, et al. ESPIRiT — an eigenvalue approach to autocalibrating parallel MRI: where SENSE meets GRAPPA.
  *Magn Reson Med* 2014;71:990–1001. [doi:10.1002/mrm.24751](https://doi.org/10.1002/mrm.24751)
- **AFI B1+ mapping:**
  Yarnykh VL. Actual flip-angle imaging in the pulsed steady state: a method for rapid three-dimensional mapping of the transmitted radiofrequency field.
  *Magn Reson Med* 2007;57:192–200. [doi:10.1002/mrm.21120](https://doi.org/10.1002/mrm.21120)
- **Pulseq / PyPulseq:**
  Layton KJ, Kroboth S, Jia F, et al. Pulseq: a rapid and hardware-independent pulse sequence prototyping framework.
  *Magn Reson Med* 2017;77:1544–1552. [doi:10.1002/mrm.26235](https://doi.org/10.1002/mrm.26235);
  Ravi KS, Geethanath S, Vaughan JT. PyPulseq: a Python package for MRI pulse sequence design.
  *J Open Source Softw* 2019;4:1725. [doi:10.21105/joss.01725](https://doi.org/10.21105/joss.01725)

The software itself can be cited via its Zenodo record: [doi:10.5281/zenodo.13992707](https://doi.org/10.5281/zenodo.13992707).
