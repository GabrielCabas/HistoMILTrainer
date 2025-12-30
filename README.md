# HistoMIL

A library for training Multi-Instance Learning (MIL) architectures from [MIL-Lab](https://github.com/mahmoodlab/MIL-Lab) on histology datasets. HistoMIL provides a unified interface to train and evaluate various state-of-the-art MIL models for whole slide image (WSI) analysis.

## Overview

HistoMIL offers a streamlined framework to train MIL architectures on histology data. It supports multiple architectures from MIL-Lab, including:

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

```bash
# Clone the repository
git clone <repository-url>
cd HistoMIL

# Install dependencies
pip install -r requirements.txt
```

**Note**: HistoMIL requires MIL-Lab to be installed and accessible. Make sure MIL-Lab is properly configured in your environment.

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
python make_splits.py \
  --csv_path ./data/dataset.csv \
  --splits_dir ./splits/ \
  --output_name my_task \
  --folds 10 \
  --test_frac 0.2 \
  --target target
```

**Note**: The `--target` argument specifies the column name for labels (default: "target"). Splits are stratified at the case level to ensure no data leakage between train/val/test sets.

### 4. Train a Model

```bash
python main.py \
  --fold 0 \
  --features_path ./features/ \
  --splits_dir ./splits/my_task/ \
  --csv_path ./splits/my_task/dataset.csv \
  --mil abmil \
  --feature_extractor uni_v2 \
  --results_dir ./results/abmil/ \
  --epochs 20 \
  --learning_rate 4e-4
```

**Note**: The `--csv_path` should point to the `dataset.csv` file generated in the splits directory (e.g., `./splits/my_task/dataset.csv`).

### 5. Run Multiple Folds (SLURM)

Use the provided shell scripts for running multiple folds:

```bash
sbatch run_abmil.sh
sbatch run_clam.sh
# ... etc
```

## Usage

### Command Line Arguments

```bash
python main.py \
  --fold <fold_number>              # Cross-validation fold (0-9)
  --features_path <path>            # Path to H5 feature files directory
  --splits_dir <path>               # Directory containing split files
  --csv_path <path>                 # Path to dataset CSV (usually in splits_dir)
  --mil <model_name>                # MIL architecture: abmil, clam, dsmil, dftd, etc.
  --feature_extractor <name>        # Feature extractor: uni_v2, etc.
  --results_dir <path>              # Output directory for results (default: ./temp_dir/)
  --epochs <int>                    # Number of training epochs (default: 3)
  --learning_rate <float>           # Learning rate (default: 4e-4)
  --use_class_weights <bool>        # Use class weights (default: True)
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
HistoMIL/
├── histomil/           # Main library package
│   ├── models.py       # Model import functions
│   ├── datasets.py     # Dataset classes
│   ├── train.py        # Training and evaluation functions
│   ├── splits.py       # Split management
│   └── utils.py        # Utility functions
├── main.py             # Main training script
├── make_splits.py      # Split generation script
```

## Output

### Training Output

Training produces:
- `{fold}-checkpoint.pt`: Best model checkpoint (saved in `results_dir`)
- `{fold}.csv`: Training and test metrics (AUC, accuracy, F1, precision, recall, etc.)

### Split Generation Output

The `make_splits.py` script generates:
- `dataset.csv`: Processed dataset with case_id, slide_id, and label columns
- `splits_{fold}_bool.csv`: Boolean splits for each fold (train/val/test columns)
- `splits_{fold}_descriptor.csv`: Summary statistics for each split

## Requirements

- Python 3.7+
- PyTorch
- NumPy
- Pandas
- scikit-learn
- tqdm
- h5py
- MIL-Lab (for model architectures)

**Note**: Make sure MIL-Lab is properly installed and accessible in your Python path. The `import_model` function relies on `src.builder.create_model` from MIL-Lab.

## Citation

If you use HistoMIL in your research, please cite the original MIL-Lab paper and the specific architecture papers you use.

## License

See LICENSE file for details.
