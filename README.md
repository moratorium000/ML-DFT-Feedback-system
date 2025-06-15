# ML-DFT Feedback System

A machine learning accelerated density functional theory (ML-DFT) framework that integrates quantum mechanical calculations with machine learning models for computational materials science applications.

## Project Objectives

The ML-DFT Feedback System aims to establish an automated computational pipeline that combines machine learning predictions with density functional theory calculations. The system facilitates materials property optimization through iterative structure modification and validation, enabling systematic exploration of chemical space with reduced computational overhead compared to traditional DFT-only approaches.

### Primary Goals
- **Structure-Property Optimization**: Modify atomic structures to achieve target material properties
- **ML-DFT Integration**: Establish feedback mechanisms between predictive models and quantum calculations
- **Computational Efficiency**: Reduce computational costs through intelligent sampling and prediction
- **Validation Framework**: Ensure physical and chemical consistency of generated structures

## System Components

### Structure Management
- **Prototype Registration**: Handle crystal structures and molecular systems in multiple formats
- **Property Targeting**: Define optimization objectives for electronic, mechanical, and thermodynamic properties
- **Convergence Monitoring**: Track optimization progress through quantitative metrics
- **Result Validation**: Verify structural and energetic consistency

### Machine Learning Pipeline
- **Graph Neural Networks**: Encode atomic structures using graph convolutional architectures
- **Path Prediction**: Predict optimization trajectories using sequence models
- **Property Prediction**: Multi-target regression for material properties
- **Uncertainty Estimation**: Quantify prediction confidence using statistical methods

### DFT Calculation Interface
- **Multi-code Support**: Compatible with VASP, Quantum ESPRESSO, and SIESTA
- **Job Management**: Automated calculation submission and monitoring
- **Output Processing**: Parse energies, forces, and electronic structure data
- **Quality Control**: Assess calculation convergence and reliability

### Mutation Operations
- **Atomic Modifications**: Displacement, substitution, and coordination changes
- **Lattice Transformations**: Strain application, rotation, and volume adjustment
- **Symmetry Operations**: Structure modifications with crystallographic constraints
- **Validation Checks**: Ensure physical feasibility and chemical stability

## Architecture

```
mldft/
├── core/                   # System orchestration and interfaces
│   ├── system.py          # Main system coordinator
│   ├── manager.py         # Component managers
│   ├── interfaces.py      # Data structure definitions
│   └── protocols.py       # Abstract interface specifications
├── models/                # Implementation modules
│   ├── ml/               # Machine learning components
│   │   ├── predictor.py  # Property and path prediction models
│   │   ├── trainer.py    # Model training procedures
│   │   └── evaluator.py  # Performance assessment
│   ├── dft/              # DFT calculation interface
│   │   ├── calculator.py # Calculation management
│   │   ├── parser.py     # Output file processing
│   │   └── converter.py  # Format conversion utilities
│   ├── mutation/         # Structure modification engine
│   │   ├── generator.py  # Mutation operators
│   │   ├── validator.py  # Constraint validation
│   │   └── optimizer.py  # Optimization algorithms
│   └── validation/       # Quality assurance
│       ├── metrics.py    # Evaluation metrics
│       ├── checker.py    # Structure validation
│       └── analyzer.py   # Statistical analysis
├── data/                 # Data management infrastructure
│   ├── database/         # Relational data storage
│   ├── cache/           # Multi-level caching
│   └── storage/         # File I/O operations
├── utils/               # Utility functions and helpers
├── config/              # System configuration
└── scripts/             # Command-line interfaces
    ├── train.py         # Model training
    ├── optimize.py      # Structure optimization
    └── analyze.py       # Result analysis
```

## Technical Implementation

### Programming Languages
- **Python**: Primary implementation language for all system components
- **SQL**: Database queries through SQLAlchemy ORM
- **JSON/YAML**: Configuration and data serialization formats
- **HDF5**: Scientific data storage format
- **Mermaid**: System documentation and workflow diagrams

### Core Dependencies
- **PyTorch**: Neural network implementation and training
- **PyTorch Geometric**: Graph neural network operations
- **SQLAlchemy**: Database abstraction and ORM
- **NumPy/SciPy**: Numerical computations and optimization
- **ASE**: Atomic structure manipulation
- **Pymatgen**: Materials science data processing
- **asyncio**: Asynchronous operation handling

### Data Management
- **Database Systems**: SQLite (development), PostgreSQL/MySQL (production)
- **Caching**: Multi-tier memory and disk caching
- **File Formats**: JSON, YAML, HDF5, CIF, POSCAR, XYZ
- **Storage Backends**: Local filesystem, AWS S3, Azure Blob, Google Cloud

## Installation and Setup

### System Requirements
- Python 3.8 or higher
- 8GB RAM minimum (16GB recommended)
- 50GB available storage space
- CUDA-compatible GPU (optional)

### DFT Software Requirements
At least one of the following DFT packages:
- VASP (Vienna Ab initio Simulation Package)
- Quantum ESPRESSO
- SIESTA

### Installation Steps

```bash
# Environment setup
python -m venv venv
source venv/bin/activate

# Dependency installation
pip install -r requirements.txt

# Configuration
cp config/default.yaml config/local.yaml
# Edit config/local.yaml as needed
```

## Usage Examples

### Command Line Interface

```bash
# Structure optimization
python scripts/optimize.py structure.cif properties.yaml \
    --config config/local.yaml \
    --output-dir results/ \
    --max-iterations 100

# Model training
python scripts/train.py \
    --data-dir training_data/ \
    --model-type property \
    --config config/local.yaml

# Result analysis
python scripts/analyze.py results/ \
    --analysis-type convergence \
    --output-dir analysis/
```

### Python API

```python
from core.system import MLDFTSystem
from config.settings import load_config

# System initialization
config = load_config('config/local.yaml')
system = MLDFTSystem(config)

# Structure optimization
structure = system.prototype_manager.load_structure('input.cif')
target_properties = {
    'band_gap': 2.0,
    'formation_energy': -1.5
}

result = await system.optimize_structure(
    structure=structure,
    target_properties=target_properties
)
```

## Configuration

### Basic Configuration Example

```yaml
dft:
  code: "vasp"
  parameters:
    xc_functional: "PBE"
    energy_cutoff: 520
    kpoints: [2, 2, 2]

ml:
  model_parameters:
    hidden_layers: [256, 128, 64]
    learning_rate: 0.001
    batch_size: 32

database:
  url: "sqlite:///mldft.db"

storage:
  base_dir: "./data"
  compression: true
```
