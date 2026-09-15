# API Reference

## Complete API Structure

This reference provides detailed documentation of all classes, methods, and functions available in PyMRITools, organized by module.

---

## pymritools.recon.loraks - LORAKS Reconstruction

### Classes and Enums

#### `LoraksImplementation` (Enum)
```python
class LoraksImplementation(Enum):
    """Specifies which LORAKS variant to use (P-LORAKS vs AC-LORAKS)"""
    P_LORAKS = auto()    # Penalized LORAKS
    AC_LORAKS = auto()   # Auto-calibration LORAKS
```

#### `OperatorType` (Enum)
```python
class OperatorType(Enum):
    """Specifies neighborhood matrix type"""
    C = auto()  # C-matrix (correlation only)
    S = auto()  # S-matrix (symmetry enforcement, preferred for magnitude)
```

#### `RankReductionMethod` (Enum)
```python
class RankReductionMethod(Enum):
    """Strategy for cutting ranks to achieve low-rank representation"""
    HARD_CUTOFF = auto()              # Hard rank cutoff (int: rank position)
    RELU_SHIFT = auto()                # ReLU shift threshold (float: shift value)
    RELU_SHIFT_AUTOMATIC = auto()      # Automatic percentage (float: 0-1 cumulative)
```

#### `RankReduction` (Dataclass)
```python
@dataclass
class RankReduction(Serializable):
    """Configuration for rank reduction strategies"""
    method: RankReductionMethod                    # Reduction method
    value: Optional[float | int] = None           # Method-specific value
```

#### `LoraksOptions` (Dataclass)
```python
@dataclass
class LoraksOptions(Serializable):
    """Base configuration for all LORAKS algorithms"""
    loraks_type: LoraksImplementation = LoraksImplementation.AC_LORAKS
    loraks_neighborhood_size: int = 5              # Side length of square neighborhood
    loraks_matrix_type: OperatorType = OperatorType.S
    rank: RankReduction = field(
        default_factory=lambda: RankReduction(method=HARD_CUTOFF, value=150))
    regularization_lambda: float = 0.0              # 0 = strict, >0 = penalized
    max_num_iter: int = 30                         # Maximum optimization iterations
    device: torch.device = field(default_factory=lambda: torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
```

#### `Loraks` (Factory Class)
```python
class Loraks:
    """Factory for creating LORAKS implementations"""
    
    @staticmethod
    def create(options: LoraksOptionsType) -> 'LoraksBase':
        """Return configured AcLoraks or PLoraks instance"""
```

#### `LoraksBase` (Abstract Base Class)
```python
class LoraksBase(ABC):
    """Abstract interface that all LORAKS implementations must follow"""
    
    @abstractmethod
    def configure(self, options: LoraksOptionsType):
        """Configure the solver with given parameters"""
    
    @abstractmethod
    def reconstruct_batch(self, k_space_batch: torch.Tensor, idx_batch: int = 0) -> torch.Tensor:
        """Reconstruct single batch - main computational method"""
    
    @abstractmethod
    def _initialize(self, k_space: torch.Tensor) -> None:
        """Initialize k-space dependent structures"""
    
    @final
    def reconstruct(self, k_space: torch.Tensor) -> torch.Tensor:
        """Main reconstruction method - handles batching"""
```

#### `AcLoraks` (Class)
```python
class AcLoraks(LoraksBase):
    """Auto-calibration LORAKS - fast reconstruction with ACS region"""
    
    def __init__(self):
        """Initializes with default NotImplemented values"""
    
    def configure(self, options: 'AcLoraksOptions' = None):
        """Configure AC-LORAKS with parameters"""
    
    def reconstruct_batch(self, batch: torch.Tensor, idx_batch: int = 0) -> torch.Tensor:
        """Reconstruct single batch - main computational method"""
    
    def get_autograd_stats(self):
        """Return (losses, learning_rates) if using autograd solver"""
```

**Private Methods:**
```python
def _initialize(self, k_space):
    """Setup operator/indexing, shape-dependent"""

def _prepare_batch(self, batch):
    """Extract AC region, compute nullspace, prepare v kernels"""

def _solve_cgd(self, k_sampled_points, vs, vc):
    """Conjugate gradient solver for Mf = b"""

def _solve_autograd(self, k_sampled_points, vs, vc):
    """Autograd optimization with adaptive learning rate"""

def _get_ac_matrix(self, batch):
    """Extract auto-calibration region from k-space"""

def _complex_subspace_representation(self, v):
    """Convert nullspace vectors to complex representation"""
```

#### `PLoraks` (Class)
```python
class PLoraks(LoraksBase):
    """Penalized LORAKS - direct optimization without AC region"""
    
    def __init__(self, **kwargs):
        """Initializes with defaults - use builder methods or configure()"""
    
    # Builder Methods (fluent interface)
    def with_patch_shape(self, patch_shape: Tuple) -> 'PLoraks':
        """Set spatial patch dimensions"""
    
    def with_device(self, device: Union[torch.device, str]) -> 'PLoraks':
        """Set computation device"""
    
    def with_s_matrix(self) -> 'PLoraks':
        """Use S-matrix operator"""
    
    def with_sv_hard_cutoff(self, rank: int) -> 'PLoraks':
        """Set hard rank cutoff"""
    
    def with_regularization_lambda(self, regularization_lambda: float) -> 'PLoraks':
        """Set regularization parameter"""
    
    def with_max_iterations(self, max_iter: int) -> 'PLoraks':
        """Set maximum iterations"""
    
    def configure(self, options: 'PLoraksOptions'):
        """Configure from PLoraksOptions dataclass"""
    
    def reconstruct_batch(self, k_space_batch: torch.Tensor, idx_batch: int = 0) -> torch.Tensor:
        """Optimize using autograd - main reconstruction method"""
```

**Additional Builder Methods:**
```python
def with_sample_directions(self, sample_directions: Tuple) -> 'PLoraks'
    """Set sampling direction flags"""

def with_c_matrix(self) -> 'PLoraks'  
    """Use C-matrix operator"""

def with_sv_soft_cutoff(self, tau: float) -> 'PLoraks'
    """Set ReLU shift threshold"""

def with_rank_reduction(self, reduction_algorithm: RankReduction) -> 'PLoraks':
    """Set rank reduction method"""

def with_sv_auto_soft_cutoff(self, percentage: float) -> 'PLoraks':
    """Set automatic ReLU cutoff percentage"""

def with_constant_learning_rate(self, learning_rate: float) -> 'PLoraks':
    """Fixed learning rate"""

def with_linear_learning_rate(self, min_lr, max_lr) -> 'PLoraks':
    """Linearly decaying learning rate"""

def with_lowrank_algorithm(self, algorithm_type, q: int, niter: int = 2):
    """Set SVD algorithm with q rank and niter iterations"""
```

### Functions

#### Matrix Operators

```python
def get_m_operator_fft(k_space: torch.Tensor, indices_op: torch.Tensor, 
                       matrix_shape: Tuple[int, int]) -> torch.Tensor:
    """Get C/S operator with FFT"""

def get_b_vector_fft(v_cplx_embed: torch.Tensor, batch_shape: Tuple, 
                     operator_type: OperatorType, k_rpsc: torch.Tensor,
                     indices: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Get b-vector with FFT"""
```

#### Data Preparation

```python
def prepare_k_space_to_batches(k_space_rpsct: torch.Tensor, batch_channel_indices: torch.Tensor) -> torch.Tensor:
    """Transform k-space to batched format: [bc*ns, ncb*nt, np, nr]"""

def unprepare_batches_to_k_space(k_batched: torch.Tensor, batch_channel_indices: torch.Tensor,
                                original_shape: Tuple) -> torch.Tensor:
    """Reverse batching to original dimensions"""

def pad_input(k_space: torch.Tensor, sampling_dims: Tuple[int, ...] = (-2, -1)) -> torch.Tensor:
    """Pad to ensure odd dimensions"""

def extract_channel_batch_indices(k_data: torch.Tensor, batch_size_channels: int) -> torch.Tensor:
    """K-means clustering for channel batching based on correlation"""
```

---

## pymritools.modeling

### B1+ AFI Mapping

#### `Settings` (Dataclass)
```python
@dataclass
class Settings(BaseClass):
    """B1+ AFI mapping configuration"""
    input_file: str = ""              # Input AFI echo images
    input_affine: str = ""            # Affine matrix for .pt files
    flip_angle: float = 60.0          # Flip angle in degrees
    ratio_tr2_tr1: int = 5           # TR₂/TR₁ ratio
    smoothing_kernel: float = 3      # Gaussian smoothing kernel size
    noise_mask: str = ""              # Noise mask file path
    input_in_image_domain: bool = True  # Input domain flag
```

#### Functions

```python
def calculate_b1(b1_data: torch.Tensor, r_tr21: float, smoothing_kernel: float = 3,
                flip_angle_set_deg: float = 60.0) -> torch.Tensor:
    """Core B1+ mapping calculation from AFI signals"""

def calculate_error_map(b1_data: torch.Tensor, mask: torch.Tensor, flip_angle_set_deg: float,
                       ratio_tr_n: float, path_visuals: pathlib.Path = None) -> torch.Tensor:
    """Compute comprehensive error map for B1+ estimation"""

def smooth_b1(alpha: torch.Tensor, smoothing_kernel: float) -> torch.Tensor:
    """Apply Gaussian smoothing to B1+ map"""

def processing(settings: Settings) -> None:
    """Main processing pipeline for B1+ AFI mapping"""
```

### Dictionary Fitting

#### `EmcFitSettings` (Dataclass)
```python
@dataclass
class FitSettings(BaseClass):
    """EMC dictionary fitting configuration"""
    input_data: str = ""
    input_database: str = ""
    input_b1: str = ""
    input_b1_err: str = ""
    input_affine: str = ""
    out_name: str = ""
    batch_size: int = 3000
    process_slice: bool = False
    rsos_channel_combine: bool = False
    input_in_image_space: bool = False
    low_rank_regularisation: int = None
    b1_error_cutoff: float = 0.15
    tx_factor: float = 1.0
```

#### Functions

```python
def fit_r2_fn(
    input_data: torch.Tensor,
    db_mag: torch.Tensor,
    t2_vals: torch.Tensor,
    b1_vals: torch.Tensor,
    t1_vals: torch.Tensor = torch.tensor([1.5]),
    b1_data: torch.Tensor = None,
    batch_size: int = 1000,
    device: torch.device = torch.get_default_device()
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Comprehensive R2 fitting using dictionary search with B1+ regularization

    Returns:
        (T2 map, B1 map, L2 error map)
    """

def fit(settings: FitSettings) -> None:
    """Main fitting pipeline for R2 parameter estimation"""

def megesse(settings: EmcFitSettings) -> None:
    """MEGESSE-specific fitting for combined R2 and R2* estimation"""
```

### Mono-exponential Decay Fitting

#### `MEXPSettings` (Dataclass)
```python
@dataclass
class MEXPSettings(BaseClass):
    """Mono-exponential decay fitting configuration"""
    input_data: str = ""                    # Input file (.pt or .nii)
    input_affine: str = ""                  # Affine file for .pt input
    input_in_img_space: bool = True         # Input domain flag
    echo_times: list = [0]                  # Echo times in seconds
```

#### Functions

```python
def fit(data: torch.Tensor, te: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Fits mono-exponential decay model S(t) = S₀·exp(-R·t)

    Returns:
        (S₀ map, R map) where T₂ = 1/R
    """

def process(settings: MEXPSettings) -> None:
    """Main processing pipeline for mono-exponential fitting"""
```

### ESPIRiT Coil Sensitivity Estimation

#### Functions

```python
def map_estimation(
    k_rpsc: torch.Tensor,
    kernel_size: int,
    num_ac_lines: int,
    rank_fraction_ac_matrix: float = 0.01,
    eigenvalue_cutoff: float = 0.99,
    device: torch.device = torch.get_default_device()
) -> torch.Tensor:
    """Main ESPIRiT sensitivity map estimation algorithm

    Returns:
        ESPIRiT sensitivity maps (n_maps, n_read, n_phase, n_slice, n_channels)
    """

def get_hankel_indices(nx: int, ny: int, nz: int, kernel_size: int) -> torch.Tensor:
    """Generate indexing for Hankel matrix construction"""

def built_hankel(k_ac: torch.Tensor, kernel_size: int) -> torch.Tensor:
    """Construct Hankel matrix from calibration region"""

def get_ker_imgs(kernels: torch.Tensor, kernel_size: int) -> torch.Tensor:
    """Convert k-space kernels to image space"""
```

---

## pymritools.processing

### Coil Compression

#### Functions

```python
def compress_channels_2d(
    input_k_space: torch.tensor,
    sampling_pattern: torch.tensor,
    num_compressed_channels: int,
    use_ac_data: bool = True,
    visualize: bool = False,
    device: torch.device = torch.get_default_device()
) -> torch.tensor:
    """2D k-space channel compression using geometric coil compression

    Returns:
        Compressed k-space data with reduced channel dimension
    """

def shape_input(
    input_k_space: torch.Tensor,
    sampling_pattern: torch.Tensor
):
    """Validate and shape input tensor dimensions

    Returns:
        (shaped k-space, shaped pattern, read_direction)
    """

def get_sampled_data(
    input_k_space: torch.Tensor,
    sampling_pattern: torch.Tensor,
    use_ac_data: bool = True
):
    """Extract sampled/ACS data from k-space
    Algorithm: Expands from center to find fully sampled region
    """
```

### Coil Sensitivity Estimation

#### `CoilSensConfig` (Dataclass)
```python
@dataclass
class CoilSensConfig(BaseClass):
    """Coil sensitivity map estimation configuration"""
    input_data: str = ""          # Input file path (.nii or .pt)
    input_affine: str = ""        # Affine matrix file (for .pt files)
    input_in_img_space: bool = True  # Whether input is already in image space
    coil_dimension: int = 3       # Coil dimension index (0-based)
    smoothing_kernel: int = 10    # Low-pass filter kernel size
```

#### Functions

```python
def smap(settings: CoilSensConfig) -> None:
    """Main entry point for coil sensitivity map estimation

    Process:
        1. Load input data (NIfTI or PyTorch format)
        2. Process slice-by-slice
        3. Convert to image space if needed
        4. Compute rSoS reference
        5. Normalize and smooth coil sensitivities
        6. Save results as NIfTI file
        7. Generate covariance plot and JSON
    """
```

### LCPCA Denoising

#### `SettingsLCPCA` (Dataclass)
```python
@dataclass
class SettingsLCPCA(BaseClass):
    """LCPCA denoising configuration"""
    in_path: str = "./examples/processing/semc_r0p9_echo_mag.nii.gz"
    in_affine: str = ""              # Affine matrix for .pt files
    file_prefix: str = "d"           # Output file prefix
    normalize: bool = False          # Normalize data before PCA
    input_image_data: bool = True    # Input is image space data
    noise_bias_correction: bool = False  # Enable noise bias correction
    fixed_p: int = 0                # Fixed number of singular values (0 = use MP inequality)
    noise_bias_mask: str = ""        # Pre-computed noise mask for NBC
```

#### `SettingsNBC` (Dataclass)
```python
@dataclass
class SettingsNBC(BaseClass):
    """Noise bias correction configuration"""
    input_data: str = ""             # Original noisy data
    denoised_data: str = ""         # Previously denoised data
    in_affine: str = ""             # Affine matrix
    file_prefix: str = "d"          # Output prefix
    noise_bias_mask: str = ""       # Noise mask for statistics
    batch_size: int = 200           # Processing batch size
    noise_histogram_depth: int = 100  # Noise histogram sampling resolution
```

#### Functions

```python
def denoise_lcpca(
    input_data: torch.Tensor,
    p: int = 1,
    fixed_cube_side_length: int = 1,
    device: torch.device = torch.get_default_device()
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Core LCPCA denoising implementation

    Parameters:
        input_data: 4D+ tensor [x, y, z, coils, additional_dims]
        p: Number of singular values to keep per patch (0 = automatic via MP)
        fixed_cube_side_length: Force specific patch size (1 = default)
        device: Computational device

    Returns:
        (denoised_data, access_mask, patch_counts)
    """

def noise_bias_correction(
    denoised_data: torch.Tensor,
    sigma: float,
    num_channels: int
):
    """Correct noise bias assuming non-central chi distribution

    Formula: √(max(|x|² - 2·n·σ², 0))
    """

def extract_noise_mask(
    input_data: torch.Tensor,
    erode_iter: int = 2
) -> torch.Tensor:
    """Automatic noise mask generation using autodmri

    Parameters:
        erode_iter: Number of binary erosion iterations
    """

def extract_noise_stats_from_mask(
    input_data: torch.Tensor | np.ndarray,
    mask: torch.Tensor | np.ndarray,
    path_visuals: pathlib.Path | str = None
) -> tuple[float, int]:
    """Estimate noise parameters from masked voxels

    Returns:
        (sigma, effective_number_of_channels)
    """
```

### Gibbs Unringing

#### Configuration

```python
@dataclass
class Settings(BaseClass):
    """Gibbs unringing configuration"""
    input_file: str = ""              # Input NIfTI file
    num_shifts_per_voxel: int = 100   # Number of sub-voxel shifts
    voxel_neighborhood_size: int = 3  # Neighborhood size for optimization
```

#### Functions

```python
def gibbs_unring_nd(
    image_data_nd: torch.Tensor,
    m: int = 100,
    k: int = 3,
    visualize: bool = True,
    gpu: bool = False,
    gpu_device: int = 0
) -> torch.Tensor:
    """Multidimensional Gibbs unringing (2D-4D)

    Parameters:
        image_data_nd: Input image data (2-4 dimensions)
        m: Number of shifted images per voxel (higher = more precise)
        k: Neighborhood size for optimization
        visualize: Enable progress bars
        gpu: Enable GPU acceleration

    Returns:
        Unringed image data
    """

def gibbs_unring_1d(
    image_data_1d: torch.Tensor,
    m: int = 100,
    k: int = 3,
    visualize: bool = True,
    gpu: bool = False,
    gpu_device: int = 0
) -> torch.Tensor:
    """1D implementation used internally by gibbs_unring_nd()

    Processes: Last dimension of input tensor
    """
```

---

## pymritools.simulation

### Core Classes

#### `GradPulse` (Class)
```python
class GradPulse:
    """Gradient pulse with RF and gradient data"""

    def __init__(
        self,
        pulse_type: str = 'Excitation',
        pulse_number: int = 0,
        num_sampling_points: int = 512,
        dt_sampling_steps_us: float = 5.0,
        duration_us: float = 0.0
    ):
        """Initialize gradient pulse"""

    def set_device(self, device: torch.device) -> None:
        """Push data to torch device (GPU or CPU)"""

    def assign_data(self, grad: torch.tensor, pulse_cplx: torch.tensor) -> None:
        """Assign gradient and pulse data to object"""

    @classmethod
    def prep_from_pulseq_kernel(cls, kernel: 'Kernel', name: str, b1s: torch.Tensor,
                              flip_angle_rad: float, device: torch.device,
                              pulse_number: int = 0, dt_set_sampling_steps_us: int = 5):
        """Prepare GradPulse from pypulseq Kernel object"""

    @classmethod
    def prep_grad_pulse_excitation(cls, pulse: 'RFPulse', params: 'EmcParameters',
                                 settings: 'EmcSimSettings', b1_vals: torch.Tensor):
        """Prepare slice-selective excitation gradient pulse"""

    @classmethod
    def prep_grad_pulse_refocus(cls, pulse: 'RFPulse', params: 'EmcParameters',
                               settings: 'EmcSimSettings', b1_vals: torch.Tensor,
                               refocus_pulse_number: int, force_sym_spoil: bool = False):
        """Prepare slice-selective refocusing gradient pulse"""

    @classmethod
    def prep_acquisition(cls, params: 'EmcParameters'):
        """Prepare gradient pulse representing acquisition step"""

    @staticmethod
    def build_pulse_grad_shapes(pulse: torch.tensor, grad_amp_slice_select: float,
                                duration_pulse_slice_select_us: float, grad_amp_post: float,
                                duration_post_us: float, duration_pre_us: float = 0.0,
                                grad_amp_pre: float = 0.0):
        """Build RF pulse and gradient shapes"""
```

#### `SimulationData` (Class)
```python
class SimulationData:
    """Data container for EMC simulation"""

    def __init__(self, params: 'EmcParameters', settings: 'EmcSimSettings', device):
        """Initialize simulation data object"""

    # Key tensors:
    # - t1_vals: torch.Tensor          # T1 values to simulate
    # - t2_vals: torch.Tensor          # T2 values to simulate
    # - b1_vals: torch.Tensor          # B1+ values to simulate
    # - b0_vals: torch.Tensor          # B0 values to simulate
    # - sample_axis: torch.Tensor      # Spatial positions along z-axis
    # - magnetization_propagation: torch.Tensor  # [t1, t2, b1, b0, samples, 4]
    # - signal_mag: torch.Tensor       # Magnitude [t1, t2, b1, b0, etl]
    # - signal_phase: torch.Tensor     # Phase [t1, t2, b1, b0, etl]

    def set_device(self, device: torch.device = None):
        """Move all tensors to specified device"""
```

### Configuration Classes

#### `EmcParameters` (Dataclass)
```python
@dataclass
class Parameters(Serializable):
    """EMC simulation parameters"""
    etl: int = 16
    esp: float = 9.0
    bw: float = 349
    excitation_angle: float = 90.0
    excitation_phase: float = 90.0
    gradient_excitation: float = -18.5
    duration_excitation: float = 2560.0
    refocus_angle: list = [140.0]
    refocus_phase: list = [0.0]
    gradient_refocus: float = -36.2
    duration_refocus: float = 3584.0
    tes: list = [0.0]
    sample_number: int = 1000
    length_z: float = 0.005
    acquisition_number: int = 50
```

#### `EmcSimSettings` (Dataclass)
```python
@dataclass
class SimulationSettings(BaseClass):
    """EMC simulation settings"""
    emc_params_file: str = "./examples/simulation/emc/emc_params.json"
    kernel_file: str = "./examples/simulation/kernels/kernels"
    te_file: str = ""
    pulse_file: str = "./examples/simulation/emc/pulse_pypulseq_default_gauss.pkl"
    database_name: str = "database_test.pkl"
    t1_list: list = [1.5]
    t2_list: list = [[25, 30, 0.5], [30, 35, 1]]
    b1_list: list = [0.6, 1.6, 0.1]
    b0_list: list = [[0, 1, 2]]
    signal_fourier_sampling: bool = False
    resample_pulse_to_dt_us: float = 5.0
    b0_batch_size: int = 1
    b1_batch_size: int = -1
```

---

## pymritools.seqprog

### Core Events

#### `RF` (Class)
```python
class RF:
    """RF pulse event implementation with various pulse types"""

    @classmethod
    def load_from_pypsi_pulse(cls, fname: str, flip_angle_rad: float, phase_rad: float,
                              system: 'Opts', duration_s: float = 2e-3):
        """Load RF pulse from file and calibrate flip angle"""

    @classmethod
    def make_sinc_pulse(cls, flip_angle_rad: float, system: 'Opts', phase_rad: float = 0.0,
                        duration_s: float = 2e-3, freq_offset_hz: float = 0.0,
                        time_bw_prod: float = 2, apodization: float = 0.0):
        """Generate sinc-shaped RF pulse"""

    @classmethod
    def make_gauss_pulse(cls, flip_angle_rad: float, system: 'Opts', phase_rad: float = 0.0,
                         duration_s: float = 2e-3, time_bw_prod: float = 2):
        """Generate Gaussian-shaped RF pulse"""

    def resample_to_new_duration(self, duration_s: float) -> None:
        """Resample pulse to new duration"""

    def set_on_raster(self, input_value: float | np.ndarray) -> float | np.ndarray:
        """Place timing on RF raster (typically 1 μs)"""

    def get_duration(self) -> float:
        """Return total duration: delay + duration + ringdown"""

    def to_simple_ns(self):
        """Convert to pypulseq-compatible namespace"""

    def plot(self) -> None:
        """Plot RF amplitude and phase"""
```

#### `GRAD` (Class)
```python
class GRAD:
    """Gradient event with comprehensive gradient shape creation"""

    @classmethod
    def make_trapezoid(cls, channel: str, system: 'Opts', amplitude: float = None,
                       area: float = None, delay_s: float = 0.0, duration_s: float = None):
        """Create trapezoidal gradient with specified area/amplitude"""

    @classmethod
    def make_slice_selective(cls, pulse_bandwidth_hz: float, slice_thickness_m: float,
                            duration_s: float, system: 'Opts', rephase: float = 0.0,
                            excitation: bool = False):
        """Create slice-selective gradient with rephasing and spoiling"""

    @classmethod
    def sym_grad(cls, system: 'Opts', channel: str = 'x', area_lobe: float = None,
                 duration_lobe: float = 0.0):
        """Create symmetric gradient pairs (e.g., for spoiling)"""

    def set_on_raster(self, value: float, return_delay: bool = False) -> float:
        """Place value on gradient raster (typically 10 μs)"""

    def get_duration(self) -> float:
        """Return total duration including delay"""

    def calculate_asym_area(self, forward: bool = True) -> float:
        """Calculate asymmetric area of trapezoid"""
```

#### `ADC` (Class)
```python
class ADC:
    """ADC event for signal sampling"""

    @classmethod
    def make_adc(cls, system: 'Opts', num_samples: int = 0, delay_s: float = 0,
                 duration_s: float = 0, dwell: float = 0):
        """Create ADC event with specified sampling parameters

        - num_samples: Number of samples (including oversampling)
        - dwell: Time between samples (seconds)
        - duration_s: Total duration = dwell × num_samples
        """

    def set_on_raster(self) -> None:
        """Place ADC timing on ADC raster (typically 100 ns)"""

    def get_duration(self) -> float:
        """Return total duration: delay + duration"""
```

#### `DELAY` (Class)
```python
class DELAY:
    """Timing delay event for sequence timing"""

    @classmethod
    def make_delay(cls, delay_s: float, system: 'Opts' = 'Opts()):
        """Create delay with specified duration"""

    def check_on_block_raster(self) -> bool:
        """Check if delay is compatible with gradient raster"""

    def set_on_block_raster(self) -> None:
        """Round delay to gradient raster"""

    def get_duration(self) -> float:
        """Return delay duration"""
```

### Sequence Configuration

#### `PulseqConfig` (Dataclass)
```python
@dataclass
class PulseqConfig(BaseClass):
    """General sequence configuration"""
    out_path: str = "."
    pulse_file: str = ""
    pulse_file_refocusing: str = ""
    visualize: bool = False
    version: str = "0.1"
```

#### `PulseqSystemSpecs` (Dataclass)
```python
@dataclass
class PulseqSystemSpecs(BaseClass):
    """Scanner specifications"""
    max_grad: float = 45.0              # mT/m
    max_slew: float = 200.0             # T/m/s
    grad_raster_time: float = 10e-6     # 10 μs
    rf_raster_time: float = 1e-6       # 1 μs
    adc_raster_time: float = 100e-9    # 100 ns
    gamma: float = 42.577              # MHz/T
    b_0: float = 3.0                   # Tesla
```

#### `PulseqParameters2D` (Dataclass)
```python
@dataclass
class PulseqParameters2D(BaseClass):
    """Sequence parameters"""
    resolution_n_read: int = 256
    resolution_n_phase: int = 256
    resolution_slice_num: int = 10
    resolution_voxel_size_read: float = 0.7      # mm
    bandwidth: float = 730.0                     # Hz/px
    etl: int = 16                               # Echo train length
    esp: float = 20.0                           # Echo spacing (ms)
    dwell: float = 2.0                          # Sampling dwell time (μs)
    excitation_rf_rad_fa: float = np.pi / 2
    excitation_duration: int = 2000              # μs
    refocusing_rf_rad_fa: list[float] = [np.pi]
    # ... additional parameters as documented in sequence_programming.md
```

### Raw Data Processing

#### Functions

```python
def pulseq_rd_to_torch(config: 'RD') -> None:
    """Load and process pulseq-style raw data

    Process:
        - Loads sampling configuration and sequence parameters
        - Sorts k-space data according to sampling pattern
        - Applies noise whitening
        - Removes oversampling
        - Saves as torch tensors for reconstruction
    """

def siemens_rd_to_torch(config: 'RD') -> None:
    """Load and process Siemens-style raw data

    Process:
        - Extracts noise, data, and reference scans
        - Handles 2D/3D acquisitions
        - Applies coordinate transforms
    """

def get_whitening_matrix(noise_data_n_samples_channel) -> np.ndarray:
    """Calculate channel decorrelation matrix from noise scans

    Returns: inverse Cholesky decomposition L^-1
    """

def get_affine(geom: 'Geometry', voxel_sizes_mm: np.ndarray, fov_mm: np.ndarray,
              vox_offset: float = -0.5, slice_gap_mm: float = 0.0) -> np.ndarray:
    """Calculate affine transformation matrix from scanner geometry"""

def remove_oversampling(data: np.ndarray | torch.Tensor, data_in_k_space: bool = True,
                        read_dir: int = 0, os_factor: int = 2) -> np.ndarray | torch.Tensor:
    """Remove oversampling from read direction"""
```

---

## pymritools.utils

### Data I/O

```python
def nifti_load(file_path: str) -> tuple:
    """Load NIfTI file

    Returns: (data_array, nifti_image_object)
    """

def nifti_save(data: np.ndarray, img_aff: np.ndarray, path_to_dir: str, file_name: str):
    """Save data as NIfTI file with affine matrix"""

def torch_load(file_path: str) -> torch.Tensor:
    """Load PyTorch tensor from .pt file"""

def torch_save(data: torch.Tensor, file_path: str):
    """Save PyTorch tensor to .pt file"""
```

### FFT Operations

```python
def fft_to_img(k_space: torch.Tensor) -> torch.Tensor:
    """FFT from k-space to image domain"""

def ifft_to_k(image: torch.Tensor) -> torch.Tensor:
    """IFFT from image domain to k-space"""

def fftshift(tensor: torch.Tensor, dim: int) -> torch.Tensor:
    """FFT shift operation on specified dimension"""

def ifftshift(tensor: torch.Tensor, dim: int) -> torch.Tensor:
    """Inverse FFT shift operation on specified dimension"""
```

### Utilities

```python
def root_sum_of_squares(data: torch.Tensor, coil_dim: int = -1) -> torch.Tensor:
    """Root sum-of-squares coil combination"""

def normalize_data(data: torch.Tensor) -> torch.Tensor:
    """Normalize data to range [0, 1]"""

def create_sample_profile(axis: torch.Tensor, extent: float, m: int = 1) -> torch.Tensor:
    """Generate sample profile using generalized normal distribution"""
```

---

## Common Patterns

### API Structure

**All major functions follow this pattern:**
```python
def main_functionname(input_data, parameter1, parameter2, **kwargs):
    """Main function with comprehensive parameters

    Parameters:
        input_data: Input tensor/data array
        parameter1: Description with typical range
        parameter2: Description with type

    Returns:
        output_data: Result with description

    Method:
        Brief algorithm description

    Examples:
        # Basic usage
        result = main_functionname(data, param1=value1)

        # Advanced usage
        result = main_functionname(data, param1=value1, param2=value2,
                                device='cuda', visualize=True)
    """
```

### Configuration Usage

**All config classes inherit from BaseClass:**
```python
@dataclass
class Settings(BaseClass):
    """Configuration with CLI and JSON support"""
    param1: str = "default"
    param2: int = 10
    use_gpu: bool = True
    gpu_device: int = 0
    out_path: str = "./output"

# Load from JSON
settings = Settings.from_json("config.json")

# Load from CLI
settings = Settings.from_cli(args, parser)

# Manual construction
settings = Settings(param1="value1", param2=20)
```

### Error Handling

**Consistent pattern for major functions:**
```python
import logging
log = logging.getLogger(__name__)

try:
    result = process_function(settings)
except Exception as e:
    log.error(f"Processing failed: {e}")
    raise
```

---

## See Also

- [Reconstruction](recon.md) - LORAKS theory and usage
- [Modeling](modeling.md) - B1+, dictionary fitting, mono-exp fitting
- [Processing](processing.md) - Denoising, unringing, coil compression
- [Simulation](simulation.md) - EMC simulation details
- [Sequence Programming](sequence_programming.md) - Event-based sequence building
- [Configuration System](../contribute/writing_code.md#configuration) - BaseClass usage guide