# Simulation Tutorial

This tutorial will guide you through using PyMRITools EMC (echo modulation curve) simulation to generate Bloch equation-based MRI signal databases for parameter estimation.

## Prerequisites

- PyMRITools installed
- Basic understanding of MRI physics and Bloch equations
- Parameters: RF pulse file, gradient specifications

## Understanding EMC Simulation

**EMC (echo modulation curve)** simulates magnetization evolution through multi-echo sequences by solving the Bloch equations:

```
dMz/dt = (M0 - Mz)/T1
dMxy/dt = -Mxy/T2 + γ * M × B(t)
```

The simulation accounts for:
- **Slice profiles**: Multi-isochromat modeling along z-axis
- **Multi-parameter space**: T₁, T₂, B₁⁺, B₀ variations
- **Complex pulse shapes**: Real RF and gradient waveforms
- **GPU acceleration**: Efficient batch processing

## Quick Start: MESE Simulation

### Step 1: Configure Simulation Parameters

```python
from pymritools.config.emc import EmcParameters, EmcSimSettings
import torch

# Sequence parameters
params = EmcParameters()
params.etl = 16                          # Echo train length
params.esp = 9.0                         # Echo spacing [ms]
params.duration_excitation = 2560.0      # Excitation pulse [µs]
params.duration_refocus = 3584.0         # Refocusing pulse [µs]
params.excitation_angle = 90.0           # [degrees]
params.refocus_angle = [180.0] * 16      # [degrees]
params.sample_number = 1200               # Sample points along slice
params.length_z = 0.003                    # Slice thickness [m]
```

### Step 2: Parameter Ranges to Simulate

```python
# Configure settings with parameter ranges
settings = EmcSimSettings()
settings.pulse_file = "rf_sinc.pkl"
settings.out_path = "./simulation_output"
settings.database_name = "mese_database.pkl"

# Parameter ranges to explore
settings.t1_list = [0.8, 1.0, 1.2, 1.5, 1.8, 2.0, 2.5, 3.0]  # [seconds]
settings.t2_list = [30, 40, 50, 60, 70, 80, 90, 100, 110, 120]  # [milliseconds]
settings.b1_list = [0.8, 1.0, 1.2]  # B1+ scaling factors
settings.b0_list = [0, 20, 2]  # B0 offsets [start, stop, step] Hz

settings.use_gpu = True
settings.gpu_device = "0"
settings.visualize = True
```

### Step 3: Run Simulation

```python
from pymritools.simulation.emc.sequence.mese import MESE

# Create simulation object
mese_sim = MESE(params=params, settings=settings)

# Run simulation
mese_sim.simulate()

# Get database results
from pymritools.config.database import DB
db = mese_sim.get_db()

# Save results
mese_sim.save(save_db=True, save_config=True)

print(f"Simulation complete!")
print(f"Database contains {len(db.t1)} T1 values")
print(f"Database contains {len(db.t2)} T2 values")
print(fSignal tensor shape: {mese_sim.data.signal_mag.shape}")
```

## Understanding the Simulation Output

### Database Structure

```python
from pymritools.config.database import DB

# Load saved database
db = DB.load("mese_database.pkl")

# Access parameter arrays
print(f"T1 values [s]: {db.t1}")
print(f"T2 values [ms]: {db.t2}")
print(f"B1 values: {db.b1}")
print(f"B0 values [Hz]: {db.b0}")
print(f"Echo train length: {db.etl}")

# Access signal data
print(f"Signal magnitude shape: {len(db.m_h)} entries")
print(f"Each entry shape: {db.m_h[0].shape}")  # [total_params, etl]
```

### Signal Data Access

```python
# Access magnitude and phase signals
magnitude_signals = db.m_h  # List of signal arrays
phase_signals = db.p_h

# Each entry contains signals for all parameter combinations
total_sims = len(magnitude_signals)
params_per_sim = magnitude_signals[0].shape[0]  # T1 × T2 × B1 × B0
etl = magnitude_signals[0].shape[1]

print(f"Total simulations: {total_sims}")
print(f"Parameters per simulation: {params_per_sim} (T1 × T2 × B1 × B0)")
print(f"Echo train length: {etl}")
```

### Visualizing Simulation Results

```python
import matplotlib.pyplot as plt
import numpy as np

# Sample a few parameter combinations
sample_indices = [0, params_per_sim // 2, params_per_sim // 2 * params_per_sim // 2]

for idx in sample_indices:
    mag = magnitude_signals[idx]  # Shape: [total_params, etl]
    
    # Calculate total parameter index
    t1_idx = idx // (len(settings.t2_list) * len(settings.b1_list) * len(settings.b0_list))
    t2_idx = (idx // (len(settings.b1_list) * len(settings.b0_list))) % len(settings.t2_list)
    b1_idx = (idx // len(settings.b0_list)) % len(settings.b1_list)
    b0_idx = idx % len(settings.b0_list)
    
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(settings.tes, mag.mean(axis=0), 'o-')
    plt.xlabel('Echo Time [s]')
    plt.ylabel('Signal Magnitude')
    plt.title(f'T1={settings.t1_list[t1_idx]:.1f}s, T2={settings.t2_list[t2_idx]:.0f}ms, '
             f'B1={settings.b1_list[b1_idx]:.1f}, B0={settings.b0_list[b0_idx]:.0f}Hz')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    for echo in range(etl):
        parameter_idx = echo % params_per_sim
        plt.plot(settings.tes, mag[parameter_idx, :], label=f'Echo {echo+1}')
    plt.xlabel('Echo Time [s]')
    plt.ylabel('Signal Magnitude')
    plt.title('Echo Decay Curves')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
```

## Advanced: MEGESSE Simulation

### Symmetric MEGESSE

```python
from pymritools.simulation.emc.sequence import MEGESSE_SYM

# Configure for symmetric MEGESSE (5 echoes per RF pulse)
sym_params = EmcParameters()
sym_params.etl = 12  # Number of SE pulses
sym_params.esp = 15.0  # Echo spacing [ms]

sym_settings = EmcSimSettings()
sym_settings.sim_type = "megesse_siemens"
sym_settings.pulse_file = "rf_sinc.pkl"
sym_settings.out_path = "./megesse_sym_output"
sym_settings.database_name = "megesse_sym_db.pkl"

# Parameter ranges
sym_settings.t1_list = [1.0, 1.5, 2.0]
sym_settings.t2_list = [[40, 80, 5.0]]
sym_settings.b1_list = [0.8, 1.0, 1.2]

# Run simulation
megesse_sym = MEGESSE_SYM(params=sym_params, settings=sym_settings)
megesse_sym.simulate()
db_sym = megesse_sym.get_db()
megesse_sym.save()

print(f"Symmetric MEGESSE simulation complete!")
print(f"Total echoes: {sym_params.etl * 5}")  # 12 SE × 5 echoes/SE
```

### Asymmetric MEGESSE

```python
from pymritools.simulation.emc.sequence import MEGESSE_ASYM

# Configure for asymmetric MEGESSE (4 GRE + 1 SE pattern)
asym_params = EmcParameters()
asym_params.etl = 8
asym_params.esp = 12.0

asym_settings = EmcSimSettings()
asym_settings.sim_type = "megesse_siemens"
asym_settings.pulse_file = "rf_sinc.pkl"
asym_settings.database_name = "megesse_asym_db.pkl"

# Run simulation
megesse_asym = MEGESSE_ASYM(params=asym_params, settings=asym_settings)
megesse_asym.simulate()
db_asym = megesse_asym.get_db()
megesse_asym.save()

print(f"Asymmetric MEGESSE simulation complete!")
print(f"Total echoes: {asym_params.etl * 5}")  # 8 SE × 5 echoes/SE
```

## Working with Pre-computed Kernels

### Using Kernel Files

```python
# Use pre-computed RF and gradient kernels
kernel_params = EmcParameters()
kernel_params.etl = 16
kernel_params.tes = [20.0, 40.0, 60.0, 80.0, 100.0, 120.0, 140.0, 160.0,
              180.0, 200.0, 220.0, 240.0, 260.0, 280.0, 300.0, 320.0]

kernel_settings = EmcSimSettings()
kernel_settings.kernel_file = "precomputed_kernels.pkl"
kernel_settings.te_file = "echo_times.json"
kernel_settings.out_path = "./kernel_sim_output"

mese_kernel = MESE(params=kernel_params, settings=kernel_settings)
mese_kernel._prep_from_kernels(pathlib.Path("precomputed_kernels.pkl"))
mese_kernel.simulate()
db_kernel = mese_kernel.get_db()
```

## Pulse Profile Analysis

### Individual Pulse Simulation

```python
from pymritools.config.emc import PulseSimulationSettings
from pymritools.simulation.emc.sequence.pulse import Pulse

# Configure pulse simulation
pulse_settings = PulseSimulationSettings(
    pulse_name="test_excitation",
    sample_length=0.004,    # Slice length [m]
    num_samples=1000,       # Number of isochromats
    pulse_duration=2000.0,    # [ms]
    pulse_flip_angle=90.0    # [degrees]
)

# Run simulation
pulse_sim = Pulse(settings=pulse_settings)
pulse_sim.simulate()

# Analyze pulse profile
mag_profile = pulse_sim.data.magnetization_propagation.squeeze()
slice_position = pulse_sim.data.sample_axis

# Plot
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(slice_position * 1000, mag_profile[:, 0], label='Mx')
plt.plot(slice_position * 1000, mag_profile[:, 1], label='My')
plt.plot(slice_position * 1000, mag_profile[:, 2], label='Mz')
plt.xlabel('Slice Position [mm]')
plt.ylabel('Magnetization')
plt.title('Excitation Pulse Profile')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
flip_angle_profile = np.sqrt(mag_profile[:, 0]**2 + mag_profile[:, 1]**2) * 180/np.pi
plt.plot(slice_position * 1000, flip_angle_profile, label='Flip Angle')
plt.axhline(y=90, color='r', linestyle='--', label='Target 90°')
plt.xlabel('Slice Position [mm]')
plt.ylabel('Flip Angle [°]')
plt.title('Flip Angle Profile')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

print(f"Target flip angle: 90°")
print(f"Actual range: {flip_angle_profile.min():.1f}° to {flip_angle_profile.max():.1f}°")
```

## Parameter Space Exploration

### Systematic Parameter Study

```python
# Study effect of different T2 ranges
t2_ranges = [
    [[20, 40, 2.0]],   # Short T₂: white matter
    [[40, 80, 2.0]],   # Medium T₂: gray matter  
    [[80, 120, 2.0]],  # Long T₂: CSF
]

for i, t2_range in enumerate(t2_ranges):
    params = EmcParameters(etl=16, esp=9.0)
    settings = EmcSimSettings(
        t1_list=[1.0, 1.5, 2.0, 2.5],
        t2_list=t2_range,
        b1_list=[1.0],
        database_name=f"study_t2_range_{i}.pkl"
    )
    
    sim = MESE(params=params, settings=settings)
    sim.simulate()
    db = sim.get_db()
    
    print(f"T2 range {t2_range} - {len(db.m_h)} simulations completed")
```

### B1+ Inhomogeneity Effects

```python
# Study B0 frequency offset effects
b0_options = [
    [-100, 100, 10],  # -100 to 100 Hz in 10 Hz steps
    [-50, 50, 5],    # -50 to 50 Hz in 5 Hz steps
    [-25, 25, 2.5],   # -25 to 25 Hz in 2.5 Hz steps
]

for i, b0_range in enumerate(b0_options):
    settings = EmcSimSettings(
        t1_list=[1.0, 1.5, 2.0],
        t2_list=[[50, 70, 5.0]],
        b1_list=[1.0],
        b0_list=b0_range,
        database_name=f"study_b0_resolution_{i}.pkl",
        b0_batch_size=5,  # Process in batches
        visualize=False
    )
    
    sim = MESE(params=params, settings=settings)
    sim.simulate()
    db = sim.get_db()
```

## GPU Memory Management

### Calculating Memory Requirements

```python
# Estimate required GPU memory
def estimate_memory(settings: EmcSimSettings):
    """Estimate required GPU memory for simulation"""
    num_t1 = len(settings.t1_list)
    num_t2 = sum([len(np.arange(*t2)) for t2 in settings.t2_list])
    num_b1 = len(settings.b1_list)
    num_b0 = sum([len(np.arange(*b0)) for b0 in settings.b0_list])
    
    num_samples = settings.sample_number
    batch_size_b0 = settings.b0_batch_size
    
    # Float32 tensor: 4 bytes per element
    elements_per_sim = num_samples * 4  # [Mx, My, Mz, M0]
    
    # Calculate for batch
    memory_per_batch = batch_size_b0 * num_t1 * num_t2 * num_b1 * elements_per_sim * 4
    
    # Add signal tensor [t1, t2, b1, b0, etl]
    signal_memory = num_t1 * num_t2 * num_b1 * num_b0 * settings.params.etl * 8  # Complex float
    
    total_memory = memory_per_batch + signal_memory
    
    print(f"Estimated memory: {total_memory / 1024**2:.2f} MB")
    print(f"Memory per batch: {memory_per_batch / 1024**2:.2f} MB")
    print(f"Signal tensor: {signal_memory / 1024**2:.2f} MB")
    
    return total_memory

# Test memory estimation
estimate_memory(settings)
```

### Optimizing Batch Sizes

```python
# Adjust batch sizes based on available GPU memory
import torch

# Check available GPU memory
if torch.cuda.is_available():
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**2  # MB
    print(f"Available GPU memory: {gpu_memory:.0f} MB")
    
    # Set batch size to use ~80% of GPU memory
    sim_memory = estimate_memory(settings)
    batch_size = int((gpu_memory * 0.8) / (sim_memory / len(settings.b0_options)))
    
    print(f"Recommended batch size: {batch_size}")
    settings.b0_batch_size = batch_size
```

## Creating Custom Databases

### Specialized Tissue Parameters

```python
# Brain tissue parameters (literature values)
brain_params = {
    'White Matter (WM)': {'t1': [0.6, 0.8], 't2': [60, 80], 'b1': [1.0]},
    'Gray Matter (GM)': {'t1': [1.1, 1.4], 't2': [90, 110], 'b1': [1.0]},
    'CSF': {'t1': [4.0, 4.5], 't2': [1800, 2200], 'b1': [0.9]},
}

# Expand ranges for database creation
t1_list = []
t2_list = []
b1_list = []

for tissue, params in brain_params.items():
    t1_list.extend(np.linspace(params['t1'][0], params['t1'][1], 5))
    t2_list.extend(np.linspace(params['t2'][0], params['t2'][1], 5))
    b1_list.extend(np.arange(params['b1'][0], params['b1'][1], 0.05))

# Remove duplicates and sort
t1_list = sorted(set(t1_list))
t2_list = sorted(set(t2_list))
b1_list = sorted(set(b1_list))

settings = EmcSimSettings(
    t1_list=t1_list,
    t2_list=[[min(t2_list), max(t2_list), step] for step in [10, 15, 20]],
    b1_list=b1_list,
    database_name="brain_tissue_database.pkl"
)
```

## Database Management

### Combining Multiple Databases

```python
from pymritools.config.database import DB

# Load multiple databases
db1 = DB.load("database1.pkl")
db2 = DB.load("database2.pkl")

# Combine databases
combined_db = DB()
combined_db.t1 = list(set(db1.t1 + db2.t1))
combined_db.t2 = list(set(db1.t2 + db2.t2))
combined_db.b1 = list(set(db1.b1 + db2.b1))
combined_db.b0 = list(set(db1.b0 + db2.b0))
combined_db.etl = db1.etl  # Assuming same ETL

# Combine signals
combined_db.m_h = db1.m_h + db2.m_h
combined_db.p_h = db1.p_h + db2.p_h

# Save combined database
combined_db.save("combined_database.pkl")
```

### Subsetting Databases

```python
# Create subset database focused on specific T1 range
db_full = DB.load("full_database.pkl")

db_subset = DB()
db_subset.t1 = [t1 for t1 in db_full.t1 if 1.0 <= t1 <= 2.0]  # 1.0-2.0s
db_subset.t2 = db_full.t2
db_subset.b1 = db_full.b1
db_subset.b0 = db_full.b0
db_subset.etl = db_full.etl

# Find matching signal indices
indices = [i for i, t1 in enumerate(db_full.t1) if 1.0 <= t1 <= 2.0]
db_subset.m_h = [db_full.m_h[i] for i in indices]
db_subset.p_h = [db_full.p_h[i] for i in indices]

db_subset.save("t1_subset_database.pkl")
```

## Common Workflows

### Workflow 1: Database Creation + Dictionary Fitting

```python
# Step 1: Generate database
from pymritools.simulation.emc.sequence.mese import MESE

sim_params = EmcParameters(etl=16, esp=9.0)
sim_settings = EmcSimSettings(
    t1_list=[0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0],
    t2_list=[[30, 100, 5.0]],
    b1_list=[0.8, 1.0, 1.2],
    database_name="custom_me_database.pkl"
)

sim = MESE(params=sim_params, settings=sim_settings)
sim.simulate()
db = sim.get_db()

# Step 2: Use database for fitting
from pymritools.modeling.dictionary.grid_search import fit

fit_settings = EmcFitSettings(
    input_data="test_me_data.nii",
    input_database="custom_me_database.pkl",
    batch_size=2000,
    out_path="./custom_fit_output"
)

fit(fit_settings)
```

### Workflow 2: Parameter Sensitivity Analysis

```python
# Systematically vary T2 and observe signal changes
base_t1 = 1.5
fixed_t2_values = [30, 50, 70, 90, 110, 130, 150]

for t2_val in fixed_t2_values:
    settings = EmcSimSettings(
        t1_list=[base_t1],
        t2_list=[[t2_val - 5, t2_val + 5, 1.0]],  # Narrow range around each value
        b1_list=[1.0],
        database_name=f"sensitivity_t2_{t2_val}.pkl"
    )
    
    sim = MESE(params=params, settings=settings)
    sim.simulate()
    db = sim.get_db()
    
    # Analyze signal sensitivity to T2
    print(f"T2 = {t2_val} ms: Signal range = {np.array(db.m_h[0][:, 0]).max():.2e3}")
```

### Workflow 3: B0 Offset Study

```python
# Study B0 frequency offset effects on signal and phase
settings = EmcSimSettings(
    t1_list=[1.0, 1.5, 2.0],
    t2_list=[[40, 60, 2.0]],
    b1_list=[1.0],
    b0_list=[-200, 200, 10],  # Wide B0 range
    database_name="b0_study.pkl",
    visualize=False
)

sim = MESE(params=params, settings=settings)
sim.simulate()
db = sim.get_db()

# Analyze phase sensitivity to B0
phase_data = np.array(db.p_h)  # [total_sims, etl]
for i, b0_offset in enumerate(range(-200, 201, 10)):
    phase_range = np.max(phase_data[i], axis=0) - np.min(phase_data[i], axis=0)
    print(f"B0 = {b0_offset:3d} Hz: Phase range = {phase_range:.2f} rad")
```

## Troubleshooting

### Common Issues

**Issue 1: Out of GPU Memory**
```python
# Reduce sample resolution
params.sample_number = 600  # Instead of 1200

# Reduce parameter ranges
settings.t1_list = [1.0, 1.5, 2.0]  # Instead of extensive range

# Reduce batch size
settings.b0_batch_size = 2  # Instead of default
```

**Issue 2: Long Simulation Time**
```python
# Disable visualization
settings.visualize = False

# Use smaller echo train length
params.etl = 8  # Instead of 16

# Use GPU acceleration
settings.use_gpu = True
settings.gpu_device = "0"
```

**Issue 3: Signal Trajectory Problems**
```python
# Check pulse and gradient specifications
print(f"Excitation pulse: amplitude={params.excitation_angle}°, "
      f"duration={params.duration_excitation}µs, "
      f"gradient={params.gradient_excitation}mT/m")

print(f"Refocusing pulse: angles={params.refocus_angle}°, "
      f"duration={params.duration_refocus}µs, "
      f"gradient={params.gradient_refocus}mT/m")

# Check timing
print(f"Echo spacing: {params.esp}ms")
print(f"Echo times: {params.tes}s")
```

## CLI Usage

```bash
# MESE simulation
python -m pymritools.simulation.emc.sequence.mese simulate \
    -p emc_params.json \
    -s emc_settings.json \
    -o ./output

# With GPU specification
python -m pymritools.simulation.emc.sequence.mese simulate \
    -p emc_params.json \
    -s emc_settings.json \
    -gpu true \
    -gpud 0

# Pulse profile simulation
python -m pymritools.simulation.emc.sequence.pulse simulate \
    -n test_excitation \
    -s pulse_settings.json \
    -o ./profiles
```

## Next Steps

- Try different sequence parameters (ETL, ESP, flip angles)
- Generate custom databases for your specific applications
- Visualize and analyze magnetization profiles
- Combine simulation results with dictionary fitting
- Use the generated database for quantitative parameter estimation
- Read the [Simulation documentation](simulation.md) for detailed theory
- Explore [Sequence Programming](sequence_programming.md) to create custom pulses

This tutorial provides practical guidance for EMC simulation and database generation for quantitative MRI parameter estimation.