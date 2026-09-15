# Sequence Programming

## Overview

PyMRITools provides an **event-based sequence programming** paradigm for flexible MRI pulse sequence design, enabling creation of complex sequences from atomic building blocks.

## Theoretical Background

### Event-Based Paradigm

The sequence programming module uses an event-based approach where each MRI sequence is composed of atomic events:

- **RF (Radio Frequency)**: RF pulses for excitation and refocusing
- **GRAD (Gradient)**: Magnetic field gradients for spatial encoding
- **ADC (Analog-to-Digital Converter)**: Signal sampling events
- **DELAY**: Timing delays for sequence control

### Architecture

**Event Hierarchy:**
```
Events → Kernels (sequence blocks) → Sequences
```

**Event Types:**
```
Core Events:
- RF: RF pulses (excitation, refocusing)
- GRAD: Gradients (prephase, rephase, spoiling)
- ADC: Analog-to-digital conversion
- DELAY: Timing delays
```

### Timing Constraints

Different timing rasters for precision control:

| Event Type | Raster Time | Purpose |
|------------|-------------|---------|
| RF raster | 1 μs | RF pulse timing precision |
| Gradient raster | 10 μs | Gradient waveform timing |
| Block raster | 10 μs | Event block alignment |
| ADC raster | 100 ns | ADC sampling precision |

## Core Events

### RF (Radio Frequency Pulse)

**Overview:** RF pulses are the fundamental excitation events in MRI sequences.

#### Pulse Generation

**Sinc Pulse:**
```python
from pymritools.seqprog.core import events
from pypulseq import Opts

system = Opts()

# Create 90° sinc pulse
rf = events.RF.make_sinc_pulse(
    flip_angle_rad=np.pi/2,  # 90° in radians
    system=system,
    duration_s=2e-3,  # 2 ms pulse
    time_bw_prod=4.0,  # Time-bandwidth product
    apodization=0.0,  # No apodization
    pulse_type='excitation',
    phase_rad=0.0
)

print(f"Bandwidth: {rf.bandwidth_hz} Hz")
print(f"Duration: {rf.get_duration()} seconds")
```

**Gaussian Pulse:**
```python
# Create Gaussian pulse
rf = events.RF.make_gauss_pulse(
    flip_angle_rad=np.pi/2,
    system=system,
    duration_s=3e-3,
    time_bw_prod=2.0
)
```

**Load from File:**
```python
# Load pre-customized pulse (e.g., SLR pulse)
rf = events.RF.load_from_pypsi_pulse(
    fname="pulse_file.pypsi",
    flip_angle_rad=np.pi/2,
    phase_rad=0.0,
    system=system,
    duration_s=2e-3,
    pulse_type='excitation'
)
```

#### RF Key Properties

```python
# RF pulse properties
rf.flip_angle_deg    # Flip angle in degrees
rf.flip_angle_rad    # Flip angle in radians
rf.phase_deg         # Phase in degrees
rf.phase_rad         # Phase in radians
rf.bandwidth_hz      # Bandwidth in Hz
rf.time_bandwidth    # Time-bandwidth product
rf.t_delay_s         # Dead time before pulse
rf.t_duration_s      # Pulse duration
rf.t_dead_time_s     # Dead time after pulse
```

#### RF Methods

```python
# Resample pulse to different duration
rf.resample_to_new_duration(duration_s=1.5e-3)

# Get pulse duration
duration = rf.get_duration()

# Place on RF raster
new_duration = rf.set_on_raster(1.500001e-3)

# Plot pulse
rf.plot()

# Convert to pypulseq namespace
namespace = rf.to_simple_ns()
```

### GRAD (Gradient Event)

**Overview:** Gradients provide spatial encoding and are used for slice selection, phase encoding, frequency encoding, and spoiling.

#### Trapezoidal Gradients

```python
from pymritools.seqprog.core import events
from pypulseq import Opts

system = Opts()

# Create readout gradient (area-based)
grad_read = events.GRAD.make_trapezoid(
    channel='x',          # 'x', 'y', or 'z'
    system=system,
    area=1e-3,            # Area [T·s/m]
    amplitude=None,       # Let amplitude be calculated
    delay_s=0.0,          # Delay before gradient
    duration_s=None,      # Let duration be calculated
    flat_area=0,          # Additional flat area
    flat_time=None,       # Let flat time be calculated
    ramp_time=None        # Let ramp time be calculated
)

# Create frequency encode gradient (amplitude-based)
grad_freq = events.GRAD.make_trapezoid(
    channel='x',
    system=system,
    amplitude=0.025,      # 25 mT/m
    area=None,
    duration_s=0.003     # 3 ms readout
)
```

#### Slice-Selective Gradients

```python
# Create slice-selective gradient with rephasing
grad_slice, delay_to_rf, spoil_duration = events.GRAD.make_slice_selective(
    pulse_bandwidth_hz=-rf.bandwidth_hz,  # Negative for slice selection
    slice_thickness_m=0.003,              # 3 mm slice
    duration_s=2e-3,                      # 2 ms gradient
    system=system,
    pre_moment=0.0,                       # No additional moment
    re_spoil_moment=0.0,                  # No rephase moment
    rephase=1.0,                          # Add rephasing
    t_minimum_re_grad=0.0,                # Minimum rephase gradient
    adjust_ramp_area=0.0,                 # No area adjustment
    excitation=True                       # Excitation gradient
)

print(f"Delay to RF start: {delay_to_rf} seconds")
print(f"Spoiling duration: {spoil_duration} seconds")
```

#### Symmetric Gradients

```python
# Create symmetric gradient pair for spoiling
grad_spoil = events.GRAD.sym_grad(
    system=system,
    channel='z',           # Slice direction spoiling
    pre_delay=0.0,
    area_lobe=0.002,       # Area of each lobe
    amplitude_lobe=None,   # Calculated from area
    duration_lobe=0.001,   # 1 ms per lobe
    duration_between=0.0,  # No delay
    reverse_second_lobe=False  # Same polarity both lobes
)
```

#### Gradient Calculation

```python
# Calculate gradient areas
asymmetric_area = grad_slice.calculate_asym_area(forward=True)

# Place on gradient raster
new_duration = grad_slice.set_on_raster(3.00001e-3, return_delay=False)

# Get gradient properties
print(f"Amplitude: {grad_slice.amplitude} T/m")
print(f"Duration: {grad_slice.get_duration()} seconds")
print(f"Area: {grad_slice.area} T·s/m")
print(f"Flat time: {grad_slice.t_flat_time_s} seconds")
```

### ADC (Analog-to-Digital Converter)

**Overview:** ADC events define when and how the MR signal is sampled.

```python
from pymritools.seqprog.core import events
from pypulseq import Opts

system = Opts()

# Create ADC for readout
adc = events.ADC.make_adc(
    system=system,
    num_samples=256,      # Number of samples (including oversampling)
    delay_s=0.0,          # Dead time before sampling
    duration_s=3.2e-3,    # Total sampling duration = dwell × num_samples
    dwell=0.0,            # Calculated from duration_s and num_samples
    freq_offset_hz=0.0,   # Frequency offset (for frequency-encoded readouts)
    phase_offset_rad=0.0  # Phase offset (for phase-encoded readouts)
)

print(f"Dwell time: {adc.t_dwell_s} seconds")
print(f"Sampling bandwidth: {1/adc.t_dwell_s/1e6} MHz")
```

### DELAY (Timing Delay)

**Overview:** DELAY events provide precise timing control between sequence elements.

```python
from pymritools.seqprog.core import events
from pypulseq import Opts

system = Opts()

# Create delay for TE collection
delay_te = events.DELAY.make_delay(
    delay_s=0.01,  # 10 ms delay
    system=system
)

# Check compatibility with gradient raster
is_compatible = delay_te.check_on_block_raster()

# Force alignment to gradient raster if needed
delay_te.set_on_block_raster()

print(f Delay duration: {delay_te.get_duration()} seconds")
```

## Sequence Kernels

### Overview

Kernels are reusable sequence blocks that combine multiple events. They can be saved, loaded, and reused across different sequences.

### Creating Custom Kernels

```python
from pymritools.seqprog.core import Kernel, events

# Create kernel with multiple events
kernel = Kernel(rf=rf, grad_read=grad_read, grad_phase=grad_phase, adc=adc)

# Add events
kernel.rf = rf
kernel.grad_slice = grad_slice  
kernel.grad_read = grad_read
kernel.adc = adc
kernel.delay = delay_te

# Save kernel for reuse
kernel.save("my_excitation_kernel.pkl")
```

### Loading Reusable Kernels

```python
# Load previously saved kernel
loaded_kernel = Kernel.load("my_excitation_kernel.pkl")

# Copy kernel
kernel_copy = loaded_kernel.copy()

# Get kernel duration
duration = loaded_kernel.get_duration()

# List non-empty events
events_list = loaded_kernel.list_events()

# Convert all to pypulseq namespaces
ns_list = loaded_kernel.list_events_to_ns()
```

### Built-in Kernels

**Excitation with Slice Selection:**
```python
from pymritools.seqprog.core import Kernel
from pymritools.config.seqprog import PulseqParameters2D, PulseqSystemSpecs

params = PulseqParameters2D(
    resolution_voxel_size_read=0.7,
    resolution_slice_thickness=3.0,
    bandwidth=730.0
)

specs = PulseqSystemSpecs()

# Create excitation kernel
exc_kernel = Kernel.excitation_slice_sel(
    params=params,
    system=specs,
    pulse_file="rf_sinc.pkl",
    read_grad_pre_area=1e-3,
    spoiling_moment=6.0
)
```

**Refocusing with Spoiling:**
```python
# Create refocusing kernel
refocus_kernel, pe_time = Kernel.refocus_slice_sel_spoil(
    params=params,
    system=specs,
    pulse_file="rf_sinc.pkl",
    pulse_num=0,  # First refocusing pulse
    duration_spoiler=0.002,  # 2 ms spoiling
    return_pe_time=True  # Return phase encode time
)
```

**Acquisition:**
```python
# Create full-sampling acquisition
acq_kernel = Kernel.acquisition_fs(
    params=params,
    system=specs,
    invert_grad_read_dir=False,
    relax_grad_stress=False
)

# Create undersampled acquisition
und_acq_kernel = Kernel.acquisition_sym_undersampled(
    params=params,
    system=specs,
    invert_grad_dir=False,
    asym_accelerated=False
)
```

**FID Navigator:**
```python
# Create FID navigator
nav_kernel = Kernel.acquisition_fid_nav(
    params=params,
    system=specs,
    line_num=32,           # Line number
    reso_degrading=1.0/6   # 1/6 resolution
)
```

## Sequence Building

### MESE Sequence

**Overview:** MESE (Multi-Echo Spin Echo) is a classic quantitative MRI sequence with multiple spin echoes.

```python
from pymritools.seqprog.sequences.mese import MESE, build
from pymritools.config.seqprog import PulseqConfig, PulseqSystemSpecs, PulseqParameters2D

# Configure system
specs = PulseqSystemSpecs()
specs.max_grad = 40.0  # mT/m
specs.max_slew = 180.0  # T/m/s

# Configure parameters
params = PulseqParameters2D()
params.resolution_n_read = 320
params.resolution_n_phase = 320
params.resolution_slice_num = 15
params.resolution_voxel_size_read = 0.7  # mm
params.resolution_voxel_size_phase = 0.7  # mm
params.resolution_slice_thickness = 3.0  # mm
params.bandwidth = 730.0  # Hz/px
params.etl = 16  # Echo train length
params.esp = 20.0  # ms - Echo spacing
params.dwell = 2.0  # μs
params.excitation_rf_rad_fa = np.pi / 2
params.excitation_duration = 2000  # μs
params.refocusing_rf_rad_fa = [np.pi] * 16  # 180° refocusing
params.refocusing_duration = 2500  # μs

# Configure output
config = PulseqConfig()
config.out_path = "./sequence_output"
config.pulse_file = "rf_sinc.npy"
config.visualize = True

# Build sequence
mese = MESE(config=config, specs=specs, params=params)
build(config=config, sequence=mese, name="mese_3mm")
```

### Accelerated MESE

```python
# Configure for GRAPPA acceleration (2x)
params.acceleration_factor = 2.0
params.sampling_pattern = "grappa"
params.number_central_lines = 32  # ACS region
params.number_outer_lines = 128   # Undersampled lines
params.resolution_n_phase = 320

mese = MESE(config=config, specs=specs, params=params)
build(config=config, sequence=mese, name="mese_grappa_2x")
```

### MEGESSE Sequence

**Symmetric MEGESSE:**
```python
from pymritools.seqprog.sequences.megesse_sym import MEGESSE

params = PulseqParameters2D()
params.etl = 12  # Number of SE echoes
params.resolution_n_read = 256
params.resolution_n_phase = 256

megesse = MEGESSE(config=config, specs=specs, params=params)
build(config=config, sequence=megesse, name="megesse_sym_3mm")
# Produces 61 total echoes: 3 initial GRE + 12×(GRE GRE SE GRE GRE)
```

**Asymmetric MEGESSE:**
```python
from pymritools.seqprog.sequences.megesse_asym import MEGESSE

params = PulseqParameters2D()
params.etl = 12  # SE echoes
params.resolution_n_read = 256
params.resolution_n_phase = 256

megesse = MEGESSE(config=config, specs=specs, params=params)
build(config=config, sequence=megesse, name="megesse_asym_3mm")
```

## Configuration Classes

### System Specifications

```python
@dataclass
class PulseqSystemSpecs(BaseClass):
    # System limits
    max_grad: float = 45.0              # mT/m
    max_slew: float = 200.0             # T/m/s
    grad_unit: str = "mT/m"
    slew_unit: str = "T/m/s"
    
    # Raster times
    grad_raster_time: float = 10e-6     # 10 μs
    rf_raster_time: float = 1e-6       # 1 μs
    adc_raster_time: float = 100e-9    # 100 ns
    
    # Dead/ringdown times
    rf_dead_time: float = 100e-6       # 100 μs
    rf_ringdown_time: float = 30e-6    # 30 μs
    adc_dead_time: float = 10e-6       # 10 μs
    rise_time: float = 100e-6
    
    # Physics
    gamma: float = 42.577              # MHz/T
    b_0: float = 3.0                   # Tesla
```

### Sequence Parameters

```python
@dataclass
class PulseqParameters2D(BaseClass):
    # Dimensions
    resolution_n_read: int = 256
    resolution_n_phase: int = 256
    resolution_slice_num: int = 10
    resolution_voxel_size_read: float = 0.7      # mm
    resolution_voxel_size_phase: float = 0.7     # mm
    resolution_slice_thickness: float = 3.0      # mm
    resolution_slice_gap: float = 10.0          # %
    
    # Acquisition
    bandwidth: float = 730.0                     # Hz/px
    oversampling: float = 2.0
    etl: int = 16                               # Echo train length
    esp: float = 20.0                           # Echo spacing (ms)
    dwell: float = 2.0                          # Sampling dwell time (μs)
    
    # Sampling
    acceleration_factor: float = 1.0
    sampling_pattern: str = "full"
    number_central_lines: int = 0
    number_outer_lines: int = 0
    
    # RF
    excitation_rf_rad_fa: float = np.pi / 2
    excitation_rf_rad_phase: float = 0.0
    excitation_duration: int = 2000              # μs
    excitation_rf_time_bw_prod: float = 4.0
    excitation_pulse_type: str = "sinc"
    
    refocusing_rf_rad_fa: list[float] = [np.pi]
    refocusing_rf_rad_phase: list[float] = [0.0]
    refocusing_duration: int = 2000               # μs
    
    # Gradients
    grad_moment_slice_spoiling: float = 6000.0   # 1/m
    read_grad_spoiling_factor: float = -0.5
    read_grad_fid_spoil_samples: int = 8
    
    # Directions
    read_dir: str = "RL"  # Right-Left
    phase_dir: str = "AP"  # Anterior-Posterior
    slice_dir: str = "FH"  # Foot-Head
    
    # Utility methods
    def get_fov(self) -> tuple[float, float, float]:
        """Return FOV in mm"""
        
    def get_voxel_size(self) -> tuple[float, float, float]:
        """Return voxel sizes in mm"""
        
    def get_acquisition_time(self) -> float:
        """Return total acquisition time in seconds"""
```

## Raw Data Handling

### Loading Pulseq Data

```python
from pymritools.seqprog.rawdata.rd_read import pulseq_rd_to_torch
from pymritools.config.seqprog import RD

rd_config = RD()
rd_config.input_file = "scan_data.dat"
rd_config.input_sequence_config = "sequence_sampling_config.pkl"
rd_config.input_sequence_config = "sequence_pulseq_config.json"
rd_config.out_path = "./processed_data"
rd_config.remove_os = True  # Remove oversampling
rd_config.visualize = True

pulseq_rd_to_torch(config=rd_config)
# Outputs:
# - k_space.pt: Complex k-space data [n_read, n_phase, n_slice, n_coils, n_echoes]
# - k_sampling_mask.pt: Binary sampling mask
# - k_noise_scans.pt: Noise scan data
# - affine.pt: Affine transformation matrix
```

### Loading Siemens Data

```python
rd_config.seq_type = "siemens"
rd_config.input_file = "siemens_raw.dat"
rd_config.input_sequence_config = "siemens_config.json"

siemens_rd_to_torch(config=rd_config)
# Handles Siemens specific data format and coordinate transforms
```

### Data Processing

```python
from pymritools.seqprog.rawdata.rd_read import (
    remove_oversampling,
    get_whitening_matrix,
    noise_whitening
)

# Remove oversampling
data_no_os = remove_oversampling(
    data=data,
    data_in_k_space=True,
    read_dir=0,
    os_factor=2
)

# Calculate whitening matrix from noise scans
whitening_matrix = get_whitening_matrix(noise_scans)

# Apply noise whitening
whitened_data, sigma = noise_whitening(noise_data, device='cuda')
```

## Utility Functions

### Timing Calculation

```python
# Calculate k-space trajectory
trajectory = kernel.get_k_space_trajectory(
    pre_read_area=1e-3,      # Pre-phase encode area
    fs_grad_area=2e-3        # Frequency encode area
)

# Plot k-space trajectory
plt.figure()
plt.plot(trajectory[:, 0], trajectory[:, 1], 'bo-')
plt.xlabel('k_x [1/m]')
plt.ylabel('k_y [1/m]')
plt.title('K-Space Trajectory')
plt.grid(True)
plt.axis('equal')
```

### Affine Calculation

```python
from pymritools.seqprog.rawdata.rd_read import get_affine

affine = get_affine(
    geom=geometry,                    # Scanner geometry
    voxel_sizes_mm=np.array([0.7, 0.7, 3.0]),
    fov_mm=np.array([224, 224, 30]),
    vox_offset=-0.5,
    slice_gap_mm=3.0
)
```

## Common Use Cases

### Scenario 1: Basic Sequence Generation

```python
specs = PulseqSystemSpecs()
params = PulseqParameters2D(
    resolution_n_read=256,
    etl=8,
    esp=15.0
)
config = PulseqConfig(out_path="./basic_seq")

mese = MESE(config=config, specs=specs, params=params)
build(config=config, sequence=mese, name="basic_mese")
```

### Scenario 2: Custom RF Pulse

```python
# Use custom SLR pulse instead of sinc
config = PulseqConfig()
config.pulse_file = "custom_slr_pulse.pkl"
config.pulse_file_refocusing = "custom_slr_refocus.pkl"

mese = MESE(config=config, specs=specs, params=params)
build(config=config, sequence=mese, name="custom_rf_mese")
```

### Scenario 3: Variable Flip Angle Sequence

```python
# Variable flip angle refocusing (VFA)
params.refocusing_rf_rad_fa = [
    np.pi * 0.9,    # 162°
    np.pi * 0.95,   # 171°
    np.pi,          # 180°
    np.pi * 1.05,   # 189°
    np.pi * 1.1     # 198°
] + [np.pi] * 11   # Rest 180°

mese = MESE(config=config, specs=specs, params=params)
build(config=config, sequence=mese, name="vfa_mese")
```

### Scenario 4: High-Resolution Sequence

```python
params = PulseqParameters2D(
    resolution_n_read=512,
    resolution_n_phase=512,
    resolution_slice_num=20,
    resolution_voxel_size_read=0.35,
    resolution_slice_thickness=2.0,
    bandwidth=1460.0,  # Higher BW for resolution
    dwell=2.0         # Maintain SNR
)

mese = MESE(config=config, specs=specs, params=params)
build(config=config, sequence=mese, name="highres_mese")
```

##参见

- [API Reference](api.md#sequence-programming) - Complete class and method documentation
- [Simulation](simulation.md) - Simulation of generated sequences
- [Configuration System](../contribute/writing_code.md#configuration) - How to use BaseClass configuration
- [RF Pulse Design](config/emc/) - EMC pulse parameter specification
- [Event Composition](#sequence-kernels) - Building complex sequences from simple events