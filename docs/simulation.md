# Simulation

## Overview

PyMRITools provides Bloch equation simulation capabilities using the **EMC (echo modulation curve)** method, enabling accurate modeling of multi-echo MRI sequences for parameter estimation and validation.

## EMC (echo modulation curve) Simulation

### Theoretical Background

**EMC simulation** solves the Bloch equations to model magnetization evolution through complex pulse sequences, accounting for slice profiles and spatial variation.

#### Bloch Equations

**Fundamental Equations:**
```
dMz/dt = (M0 - Mz)/T1
dMxy/dt = -Mxy/T2 + γ * M × B(t)
```

Where:
- M = magnetization vector [Mx, My, Mz]
- T1, T2 = relaxation times
- γ = gyromagnetic ratio (42.577 MHz/T for protons)
- B(t) = effective magnetic field (gradients + RF pulses)

#### Matrix Propagation Form

**Discrete Time Steps:**
```
M(t + dt) = P(dt) × M(t)
```

**Propagation Matrix:**
```
P = R_RF(α, φ) × R_grad(θ) × R_relax(dt)
```

Where P combines:
- **R_RF(α, φ)**: RF rotation matrices for flip angle α and phase φ
- **R_grad(θ)**: Gradient precession with angle θ
- **R_relax(dt)**: Relaxation matrices for T1 and T2

#### Multi-Component Modeling

**Parameter Space:**
```
- T1: Longitudinal relaxation times [seconds]
- T2: Transverse relaxation times [milliseconds]
- B1⁺: RF field inhomogeneity factors
- B0: Static field inhomogeneity factors [Hz]
```

**Slice Profile Modeling:**
```
- Multi-isochromat approach along z-axis
- Sample resolution: typically 1000-2000 points
- Spatial resolution: determined by slice gradient
```

### Configuration

### Simulation Parameters

```python
from pymritools.config.emc import EmcParameters

params = EmcParameters(
    # Timing parameters
    etl=16,                           # Echo train length
    esp=9.0,                          # Echo spacing [ms]
    bw=349,                           # Bandwidth [Hz/px]
    
    # Excitation parameters
    excitation_angle=90.0,             # Flip angle [°]
    excitation_phase=90.0,             # Phase [°]
    gradient_excitation=-18.5,         # Gradient [mT/m]
    duration_excitation=2560.0,        # Duration [µs]
    
    # Refocusing parameters
    refocus_angle=[140.0],             # Flip angles [°]
    refocus_phase=[0.0],               # Phases [°]
    gradient_refocus=-36.2,            # Gradient [mT/m]
    duration_refocus=3584.0,           # Duration [µs]
    
    # Spatial parameters
    tes=[0.0],                        # Echo times [s]
    sample_number=1000,                # Sample points along slice
    length_z=0.005,                    # Slice length [m]
    acquisition_number=50              # Number of acquisition bins
)
```

### Simulation Settings

```python
from pymritools.config.emc import EmcSimSettings

settings = EmcSimSettings(
    # Input files
    pulse_file="./examples/simulation/emc/pulse_default.pkl",
    kernel_file="./examples/simulation/kernels/kernels.pkl",
    te_file="echo_times.json",
    
    # Configuration
    out_path="./simulation_output",
    database_name="my_database.pkl",
    use_gpu=True,
    gpu_device="0",
    visualize=True,
    
    # Simulation options
    signal_fourier_sampling=False,   # Use k-space sampling or sum acquisition
    sim_type="mese",                  # Simulation type
    resample_pulse_to_dt_us=5.0,      # Pulse time resolution
    
    # Parameter ranges to simulate
    t1_list=[0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0],  # [seconds]
    t2_list=[[30, 40, 2.0], [50, 60, 2.0], [70, 80, 2.0]],  # [[start, stop, step] ms]
    b1_list=[0.8, 1.0, 1.2],                         # B1+ scaling factors
    b0_list=[[0, 10, 2]],                             # B0 offsets [start, stop, step] Hz
    
    # Batch processing
    b0_batch_size=3,
    b1_batch_size=4
)
```

### SimulationData Class

**Key Properties:**
```python
class SimulationData:
    def __init__(self, params: EmcParameters, settings: EmcSimSettings, device):
        self.device: torch.device = device
        
        # Parameter arrays
        self.t1_vals: torch.Tensor          # T1 values to simulate
        self.t2_vals: torch.Tensor          # T2 values to simulate
        self.b1_vals: torch.Tensor          # B1+ values to simulate
        self.b0_vals: torch.Tensor          # B0 values to simulate
        
        # Spatial sampling
        self.sample_axis: torch.Tensor      # Spatial positions along z-axis
        self.sample: torch.Tensor           # Bulk magnetization profile
        
        # Magnetization evolution
        self.m_init: torch.Tensor           # Initial magnetization state
        self.magnetization_propagation: torch.Tensor  # [t1, t2, b1, b0, samples, 4]
        
        # Signal output
        self.signal_mag: torch.Tensor       # Magnitude [t1, t2, b1, b0, etl]
        self.signal_phase: torch.Tensor     # Phase [t1, t2, b1, b0, etl]
        self.signal_tensor: torch.Tensor    # Complex k-space sampling
```

**Dimensions:**
```
magnetization_propagation: [num_t1s, num_t2s, num_b1s, num_b0s, num_samples, 4]
signal_mag: [num_t1s, num_t2s, num_b1s, num_b0s, etl]
signal_phase: [num_t1s, num_t2s, num_b1s, num_b0s, etl]
```

## MESE Simulation

### Overview

**MESE (Multi-Echo Spin Echo)** simulation models the classic multi-echo spin echo sequence widely used for quantitative MRI.

### Python API Usage

```python
from pymritools.config.emc import EmcParameters, EmcSimSettings
from pymritools.simulation.emc.sequence.mese import MESE
import torch

# Configure parameters
params = EmcParameters()
params.etl = 16
params.esp = 9.0  # ms
params.excitation_angle = 90.0
params.refocus_angle = [180.0] * 16
params.length_z = 0.005  # 5 mm slice thickness
params.sample_number = 1200

# Configure settings
settings = EmcSimSettings()
settings.pulse_file = "rf_pulse.pkl"
settings.out_path = "./mese_sim_results"
settings.t1_list = [0.8, 1.0, 1.2, 1.5, 1.8, 2.0, 2.5, 3.0]  # seconds
settings.t2_list = [[30, 40, 2.5], [45, 55, 2.5], [60, 70, 2.5], [75, 85, 2.5]]  # ms
settings.b1_list = [0.8, 1.0, 1.2]
settings.visualize = True
settings.use_gpu = True

# Create and run simulation
mese_sim = MESE(params=params, settings=settings)
mese_sim.simulate()

# Get results
db = mese_sim.get_db()
mese_sim.save(save_db=True, save_config=True)
```

### Key Methods

```python
class MESE(Simulation):
    def _prep(self) -> None:
        """Prepare gradient pulses and excitations for MESE sequence"""
    
    def _prep_from_kernels(self, kernel_file: pathlib.Path) -> None:
        """Load pulse/gradient data from serialized kernel file"""
    
    def _prep_from_params(self) -> None:
        """Generate gradient pulses from configuration parameters"""
    
    def _set_device(self) -> None:
        """Move all gradient pulses to device"""
    
    def _register_sequence_timings(self) -> None:
        """Calculate timing for excitation-to-refocus and refocus-to-acquisition"""
    
    def _simulate(self) -> None:
        """Execute MESE simulation with matrix propagation"""
    
    def simulate_pulse_profiles(
        self, settings: EmcSimSettings, params: EmcParameters,
        force_sim: bool = False, path: pathlib.Path = None,
        slice_thickness: float = 0.0007, gap: float = 65.0
    ) -> None:
        """Simulate and visualize RF pulse profiles for slice characterization"""
```

## MEGESSE Simulation

### Overview

**MEGESSE (Multi-Echo Gradient and Spin Echo)** simulation provides combined gradient echo and spin echo signals for advanced parameter estimation.

### Symmetric MEGESSE

```python
from pymritools.simulation.emc.sequence import MEGESSE_SYM

# Symmetric MEGESSE: 5 echoes per RF (GRE, GRE, SE, GRE, GRE)
megesse_sym = MEGESSE_SYM(params=params, settings=settings)
megesse_sym.simulate()
megesse_sym.save()
```

### Asymmetric MEGESSE

```python
from pymritools.simulation.emc.sequence import MEGESSE_ASYM

# Asymmetric MEGESSE: 4 GRE + 1 SE pattern
megesse_asym = MEGESSE_ASYM(params=params, settings=settings)
megesse_asym.simulate()
megesse_asym.save()
```

### Configuration Differences

**Symmetric MEGESSE:**
```
Echo pattern: GRE, GRE, SE, GRE, GRE
Periodic: Same pattern for each RF pulse
Timing: Symmetric spacing around SE echo
```

**Asymmetric MEGESSE:**
```
Echo pattern: GRE, GRE, GRE, GRE, SE
Aperiodic: Different interleaving strategy
Timing: Optimized for phase encoding
```

## Core Simulation Functions

### Matrix Propagation

```python
def matrix_propagation_grad_pulse_multidim(
    sim_data: SimulationData,
    pulse_x: torch.Tensor,
    pulse_y: torch.Tensor,
    grad: torch.Tensor,
    dt_s: torch.Tensor
) -> torch.Tensor:
    """
    Calculate propagation matrix per time step.
    
    Combines:
    - RF rotations (x, y axes)
    - Gradient precession (z axis)
    - Relaxation (T1, T2)
    
    Returns:
        Matrix [num_b1s, num_samples, 4, 4]
    """
```

### Magnetization Propagation

```python
def propagate_matrix_mag_vector(
    propagation_matrix: torch.tensor,
    sim_data: SimulationData
) -> SimulationData:
    """
    Propagate magnetization through matrix.
    
    Dimensions:
        propagation_matrix: [t1s, t2s, b1s, b0s, samples, 4, 4]
        sim_data.magnetization: [t1s, t2s, b1s, b0s, samples, 4]
    
    Operation: M' = P × M
    """
```

### Signal Acquisition

```python
def sample_acquisition(
    etl_idx: int,
    params: EmcParameters,
    sim_data: SimulationData,
    acquisition_grad: torch.tensor,
    dt_s: torch.tensor
) -> SimulationData:
    """
    Sample acquisition with k-space readout gradient.
    
    Returns Fourier sampling results for reconstruction.
    """
```

```python
def sum_sample_acquisition(
    etl_idx: int,
    params: EmcParameters,
    sim_data: SimulationData,
    acquisition_duration_s: torch.tensor,
    partial_fourier: float = 1.0
) -> SimulationData:
    """
    Sum magnetization at echo center.
    
    Stores magnitude and phase in signal tensors.
    """
```

## Working with Kernels

### Loading Pre-computed Kernels

```python
params = EmcParameters()
params.etl = 16
params.tes = [20.0, 40.0, 60.0, 80.0, 100.0, 120.0, 140.0, 160.0,
             180.0, 200.0, 220.0, 240.0, 260.0, 280.0, 300.0, 320.0]

settings = EmcSimSettings()
settings.kernel_file = "kernels.pkl"  # Pre-calculated RF/grad blocks
settings.te_file = "echo_times.json"
settings.out_path = "./kernel_sim"

mese = MESE(params=params, settings=settings)
mese._prep_from_kernels(pathlib.Path("kernels.pkl"))
mese.simulate()
mese.save()
```

### Kernel Format

**Serialized Structure:**
```python
# kernels.pkl contains:
{
    'excitation': GradPulse object,
    'refocus': [GradPulse, GradPulse, ...],  # One per refocus pulse
    'acquisition': GradPulse object,
    'te_array': [20.0, 40.0, ...]  # Echo times
}
```

## Pulse Profile Simulation

### Individual Pulse Analysis

```python
from pymritools.config.emc import PulseSimulationSettings
from pymritools.simulation.emc.sequence.pulse import Pulse

pulse_settings = PulseSimulationSettings(
    pulse_name="test_pulse",
    kernel_file="pulse_kernel.pkl",
    sample_length=0.004,              # Sample length in meters
    num_samples=1000,                  # Number of isochromats
    pulse_duration=2000.0,             # Pulse duration in ms
    pulse_flip_angle=90.0,             # Flip angle in degrees
)

pulse_sim = Pulse(settings=pulse_settings)
pulse_sim.simulate()

# Plot magnetization profiles
mag_list = pulse_sim.data.magnetization_propagation[0, 0, 0, 0].cpu().numpy()

from pymritools.simulation.emc.sequence.pulse import plot
sample_axis = pulse_sim.data.sample_axis
plot(mag_list, sample_axis, "./pulse_profiles/")
```

## GPU Acceleration

### Batch Processing

```python
settings = EmcSimSettings()
settings.use_gpu = True
settings.gpu_device = 0

# Configure batch sizes for memory efficiency
settings.b0_batch_size = 3   # Process 3 B0 values at once
settings.b1_batch_size = 4   # Process 4 B1 values at once
```

### Memory Estimation

**Approximate Memory Calculation:**
```
memory = num_t1 × num_t2 × num_b1 × num_b0 × num_samples × num_params bytes
```

**Example:**
```
8 T1 × 9 T2 × 3 B1 × 2 B0 × 200 samples × 4 params × 8 bytes ≈ 9 MB
```

## Database Management

### Database Structure

```python
class DB:
    """Database for storing simulation results"""
    
    def __init__(self):
        self.t1: list = []       # T1 values [seconds]
        self.t2: list = []       # T2 values [milliseconds]
        self.b1: list = []       # B1 values
        self.b0: list = []       # B0 values [Hz]
        self.etl: list = []      # Echo train length
        self.m_h: list = []      # Signal magnitude [N_total, etl]
        self.p_h: list = []      # Signal phase [N_total, etl]
        
    def add_entry(self, t1, t2, b1, b0, signal_mag, signal_phase):
        """Add simulation result to database"""
        
    def save(self, file_name: str):
        """Save database to pickle file"""
        
    @classmethod
    def load(cls, file_name: str):
        """Load database from pickle file"""
```

### Saving Results

```python
from pymritools.config.database import DB

# Save simulation results
mese_sim.save(save_db=True, save_config=True)

# Output files:
# - my_database.pkl: Complete database with all signals
# - emc_config.json: Simulation parameters and settings
# - emc_pulse.pkl: RF pulse waveform
```

## Common Use Cases

### Scenario 1: Standard MESE Dictionary

```python
params = EmcParameters(etl=16, esp=9.0)
settings = EmcSimSettings(
    t1_list=[0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0],
    t2_list=[[30, 60, 2.5], [70, 100, 2.5]],
    b1_list=[0.8, 1.0, 1.2],
    database_name="standard_mese_db.pkl"
)

mese_sim = MESE(params=params, settings=settings)
mese_sim.simulate()
db = mese_sim.get_db()
```

### Scenario 2: Extended Parameter Space

```python
# Large parameter space exploration
settings = EmcSimSettings(
    t1_list=[0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.7, 1.9, 2.1, 2.3, 2.5, 2.7, 2.9, 3.1, 3.3],
    t2_list=[[20, 120, 5.0]],  # Single wide T2 range
    b1_list=[0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5],
    b1_batch_size=4,  # Process B1 in batches
    use_gpu=True
)

# Total simulations: 15 T1 × 21 T2 × 11 B1 = 3465
```

### Scenario 3: B0 Inhomogeneity Study

```python
# Study effect of B0 frequency offsets
settings = EmcSimSettings(
    t1_list=[1.0, 1.5, 2.0],
    t2_list=[[30, 60, 5.0]],
    b1_list=[1.0],
    b0_list=[[-100, 100, 10]],  # -100 to 100 Hz in 10 Hz steps
    b0_batch_size=5,
    database_name="b0_effects_db.pkl"
)

# Examine B0 effects on signal phase
b0_vals = range(-100, 101, 10)  # -100 to 100 Hz
for i, b0_val in enumerate(b0_vals):
    phase_response = db.p_h[i * 3]  # 每3个T1值对应一个B0值
```

### Scenario 4: Pulse Profile Characterization

```python
# Analyze pulse excitation profile
settings = PulseSimulationSettings(
    pulse_name="90degree_excitation",
    sample_length=0.007,    # 7 mm slice
    num_samples=1500,       # Fine spatial resolution
    pulse_duration=2560,    # 2560 μs pulse
)

pulse_sim = Pulse(settings=settings)
pulse_sim.simulate()
pulse_sim.plot()

# Analyze flip angle profile
mag_profiles = pulse_sim.data.magnetization_propagation.squeeze()
final_flip_angles = torch.sqrt(mag_profiles[:, 0]**2 + mag_profiles[:, 1]**2) * 180/π
print(f"Target flip angle: 90°")
print(f"Actual profile range: {final_flip_angles.min():.1f}° to {final_flip_angles.max():.1f}°")
```

## Mathematical Foundations

### Complete Bloch Equation Solution

**Discrete Time Integration:**
```
For each time step dt:
    1. Calculate RF rotation matrix R_RF(α, φ)
    2. Calculate gradient precession R_grad(θ = γ·G·dt)
    3. Calculate relaxation matrix R_relax(dt)
    4. Combine: P = R_RF × R_grad × R_relax
    5. Apply: M(t + dt) = P × M(t)
```

**Relaxation Matrix:**
```
R_relax(dt) = [
    [exp(-dt/T2),    0,            0,           1 - exp(-dt/T1)],
    [0,             exp(-dt/T2),  0,           0              ],
    [0,             0,            exp(-dt/T2),  0              ],
    [0,             0,            0,           1              ]
]
```

**Gradient Precession:**
```
θ_z = γ * G_z * dt
R_grad(θ_z) = [
    [cos(θ_z), -sin(θ_z), 0, 0],
    [sin(θ_z),  cos(θ_z), 0, 0],
    [0,          0,         1, 0],
    [0,          0,         0, 1]
]
```

**RF Rotation:**
```
R_RF(α, φ) = rotation around:
    M_x: flip angle α, phase φ = 0°
    M_y: flip angle α, phase φ = 90°
ST = rotation_matrix_x(α) * rotation_matrix_y(α)
```

### Signal Equations

**Magnitude Signal:**
```
S_mag = sqrt(M_x² + M_y²)
```

**Phase Signal:**
```
S_phase = atan2(M_y, M_x)
```

**K-Space Sampling:**
```
k_x = γ * ∫G_x(t)dt  (readout gradient)
k_y = γ * ∫G_y(t)dt  (phase encode gradient)
```

## Configuration Examples

### Basic MESE Configuration

```json
{
  "pulse_file": "rf_sinc.pkl",
  "out_path": "./sim_output",
  "database_name": "mese_db.pkl",
  "use_gpu": true,
  "visualize": true,
  "signal_fourier_sampling": false,
  "t1_list": [0.8, 1.0, 1.2, 1.5, 2.0],
  "t2_list": [[30, 60, 2.5], [70, 100, 2.5]],
  "b1_list": [0.8, 1.0, 1.2],
  "b0_list": [[0, 20, 5]]
}
```

### High-Resolution MEGESSE

```json
{
  "pulse_file": "rf_slr_high_res.pkl",
  "kernel_file": "kernels_high_res.pkl",
  "sim_type": "megesse_siemens",
  "sample_number": 2000,
  "resample_pulse_to_dt_us": 2.0,
  "acquisition_number": 100,
  "t1_list": [0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
  "t2_list": [[20, 120, 5.0]],
  "b1_list": [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3],
  "b1_batch_size": 4
}
```

## See Also

- [API Reference](api.md#simulation) - Complete class and method documentation
- [Dictionary Fitting](modeling.md#dictionary-fitting) - Using EMC database for parameter estimation
- [Configuration System](../contribute/writing_code.md#configuration) - How to use BaseClass configuration
- [GPU Acceleration](#gpu-acceleration) - Memory and performance considerations
- [Sequence Programming](sequence_programming.md) - Generate sequences for simulation