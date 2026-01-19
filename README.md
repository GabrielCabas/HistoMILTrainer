# HistoMILTrainer

A library for training Multi-Instance Learning (MIL) architectures from [MIL-Lab](https://github.com/mahmoodlab/MIL-Lab) on histology datasets. HistoMILTrainer provides a unified interface to train and evaluate various state-of-the-art MIL models for whole slide image (WSI) analysis.

## Overview

HistoMILTrainer offers a streamlined framework to train MIL architectures on histology data. It supports multiple architectures from MIL-Lab, including:

- **ABMIL** - Attention-based Multiple Instance Learning
- **CLAM** - Clustering-constrained Attention Multiple instance learning
- **DSMIL** - Dual-stream Multiple Instance Learning
- **DFTD** - Deep Feature-based Top-Down attention
- **ILRA** - Instance-Level Representation Aggregation
- **RRT** - Residual Regression Transformer
- **Transformer** - Transformer-based MIL
- **TransMIL** - Transductive Multiple Instance Learning
- **WIKG** - Weighted Instance Knowledge Graph

## Features

- **Unified Training Interface**: Train any supported MIL architecture with consistent parameters
- **Flexible Data Loading**: Support for variable number of patches per slide
- **Cross-Validation**: Built-in support for k-fold cross-validation
- **Feature Extraction Integration**: Works seamlessly with pre-extracted patch features (e.g., from TRIDENT)
- **Class Weighting**: Automatic class weight calculation for imbalanced datasets
- **Early Stopping**: Prevent overfitting with configurable early stopping
- **Case-Level Splitting**: Prevents data leakage by splitting at the case level

## Installation

HistoMILTrainer requires **Python 3.10**. Choose one of the following installation methods:

### Option 1: Using Conda (Recommended)

```bash
# Clone HistoMILTrainer
git clone https://github.com/digenoma-lab/HistoMILTrainer
cd HistoMILTrainer

# Create conda environment with all dependencies
conda env create -f environment.yml
conda activate histomil

# Install HistoMILTrainer in editable mode
pip install -e .
```

After installation, the CLI commands `histomil-splits` and `histomil-grid` will be available in your PATH.

The `environment.yml` file includes:
- Python 3.10
- MIL-Lab (from GitHub)
- smooth-topk (required for CLAM, from GitHub)
- All required dependencies (seaborn, matplotlib, pytest, etc.)

### Option 2: Using pip/Poetry

```bash
# Clone HistoMILTrainer
git clone https://github.com/digenoma-lab/HistoMILTrainer
cd HistoMILTrainer

# Install with pip (MIL-Lab will be installed automatically as a dependency)
pip install -e .
```

After installation, the CLI commands `histomil-splits` and `histomil-grid` will be available in your PATH.

**Note**: When installing with pip, dependencies are automatically installed from the `pyproject.toml` configuration. The package will install:
- MIL-Lab (from GitHub: `https://github.com/GabrielCabas/MIL-Lab.git`)
- smooth-topk (from GitHub: `https://github.com/oval-group/smooth-topk.git`, required for CLAM)
- seaborn
- matplotlib
- All other dependencies from MIL-Lab (torch, numpy, pandas, scikit-learn, tqdm, h5py, etc.)

### CLAM Support

The `smooth-topk` dependency (required for CLAM architecture) is automatically installed with HistoMILTrainer. No additional installation steps are required.

**Note**: MIL-Lab is not available on PyPI and is installed directly from GitHub. The installation process handles this automatically through the dependency configuration.

## Quick Start

### 1. Prepare Your Data

Organize your data in the following structure:
```
features/
  ├── slide1.h5
  ├── slide2.h5
  └── ...
```

Each H5 file should contain:
- `features`: Array of shape `(num_patches, feature_dim)`
- Optionally: `coords`: Array of patch coordinates

### 2. Create Dataset CSV

Create a CSV file with columns:
- `case_id`: Unique identifier for each case (patient)
- `slide_id`: Unique identifier for each slide
- `target`: Target label for classification (or specify custom column name with `--target`)

**Important**: Splits are created at the case level to prevent data leakage. Multiple slides from the same case will always be in the same split.

### 3. Generate Splits

```bash
histomil-splits \
  --csv_path ./data/dataset.csv \
  --splits_dir ./splits/ \
  --output_name my_task \
  --folds 10 \
  --test_frac 0.2 \
  --target target
```

**Note**: The `histomil-splits` command is available after installing the package. The `--target` argument specifies the column name for labels (default: "target"). Splits are stratified at the case level to ensure no data leakage between train/val/test sets.

### 4. Train Models with Grid Search

Use `histomil-grid` to perform hyperparameter optimization and train models across all folds:

```bash
histomil-grid \
  --features_path ./features/ \
  --splits_dir ./splits/my_task/ \
  --csv_path ./splits/my_task/dataset.csv \
  --mil abmil \
  --feature_extractor uni_v2 \
  --results_dir ./results/abmil/ \
  --epochs 20 \
  --learning_rate 4e-4 \
  --grid_params configs/abmil.json
```

**Note**: The `histomil-grid` command performs grid search across parameter combinations, trains models for all folds, selects the best parameters, and evaluates on the test set. The `--csv_path` should point to the `dataset.csv` file generated in the splits directory (e.g., `./splits/my_task/dataset.csv`).

### 5. Run Multiple Folds (SLURM)

Use the provided shell scripts for running multiple folds:

```bash
sbatch run_abmil.sh
sbatch run_clam.sh
# ... etc
```

## Usage

### Available CLI Commands

After installation, the following commands are available:

- `histomil-splits`: Generate train/validation/test splits
- `histomil-grid`: Perform grid search with hyperparameter optimization

### Split Generation (`histomil-splits`)

```bash
histomil-splits \
  --csv_path <path>                 # Path to dataset CSV (required)
  --output_name <name>              # Output directory name (required)
  --folds <int>                     # Number of folds (default: 10)
  --splits_dir <path>               # Output directory (default: ./splits)
  --test_frac <float>               # Test set fraction (default: 0.2)
  --target <column_name>            # Target column name (default: target)
```

### Grid Search (`histomil-grid`)

```bash
histomil-grid \
  --features_path <path>            # Path to H5 feature files directory (required)
  --splits_dir <path>               # Directory containing split files (required)
  --csv_path <path>                 # Path to dataset CSV (required)
  --mil <model_name>                # MIL architecture: abmil, clam, dsmil, dftd, etc. (default: abmil)
  --feature_extractor <name>        # Feature extractor: uni_v2, etc. (default: uni_v2)
  --results_dir <path>              # Output directory for results (default: ./temp_dir/)
  --epochs <int>                    # Number of training epochs (default: 10)
  --learning_rate <float>           # Learning rate (default: 4e-4)
  --folds <int>                     # Number of cross-validation folds (default: 10)
  --use_class_weights <bool>        # Use class weights (default: True)
  --grid_params <path>               # Path to grid search parameters JSON (default: configs/abmil.json)
```

### Supported Models

Use the `--mil` argument to specify the architecture:

- `abmil` - Attention-based MIL
- `clam` - CLAM architecture (requires batch_size=1)
- `dsmil` - Dual-stream MIL
- `dftd` - Deep Feature Top-Down
- `ilra` - Instance-Level Representation Aggregation
- `rrt` - Residual Regression Transformer
- `transformer` - Transformer-based MIL
- `transmil` - Transductive MIL
- `wikg` - Weighted Instance Knowledge Graph

**Note**: CLAM automatically sets batch_size to 1 during training.

## Project Structure

```
HistoMILTrainer/
├── histomil/           # Main library package
│   ├── models.py       # Model import functions
│   ├── datasets.py     # Dataset classes
│   ├── train.py        # Training and evaluation functions
│   ├── splits.py       # Split management
│   ├── grid_search.py  # Grid search class
│   ├── cli.py          # Command-line interface
│   └── utils.py        # Utility functions
├── tests/              # Test suite
│   └── test_import.py  # Import tests
├── configs/            # Model configuration files
├── environment.yml     # Conda environment configuration
└── pyproject.toml      # Package metadata and dependencies
```

## Output

### Training Output

Training produces:
- `{fold}-checkpoint.pt`: Best model checkpoint (saved in `results_dir`)
- `{fold}.csv`: Training and test metrics (AUC, accuracy, F1, precision, recall, etc.)

### Split Generation Output

The `histomil-splits` command generates:
- `dataset.csv`: Processed dataset with case_id, slide_id, and label columns
- `splits_{fold}_bool.csv`: Boolean splits for each fold (train/val/test columns)
- `splits_{fold}_descriptor.csv`: Summary statistics for each split

### Grid Search

You can perform grid search with hyperparameter optimization using:

```bash
histomil-grid \
  --features_path ./features/ \
  --splits_dir ./splits/my_task/ \
  --csv_path ./splits/my_task/dataset.csv \
  --mil abmil \
  --feature_extractor uni_v2 \
  --results_dir ./results/abmil/ \
  --epochs 20 \
  --learning_rate 4e-4 \
  --grid_params configs/abmil.json
```

The `histomil-grid` command performs:
- Grid search across all parameter combinations
- Cross-validation training for each combination
- Selection of best parameters based on validation AUC
- Testing of best models on test set
- Output of results, predictions, and best parameters


**Note**: MIL-Lab is automatically installed as a dependency when you install HistoMILTrainer. The `import_model` function relies on `src.builder.create_model` from MIL-Lab, which should be available after installation.

## Citation

If you use HistoMILTrainer in your research, please cite the original MIL-Lab paper and the specific architecture papers you use.

## License

See LICENSE file for details.
