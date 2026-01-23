"""Grid search for MIL models with cross-validation."""
import json
import os
from itertools import product

import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
import logging

from histomil import (
    H5Dataset,
    seed_torch,
    get_weights,
    import_model,
    variable_patches_collate_fn,
)
from histomil.train import train, test

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GridSearch:
    """Grid search for MIL models with cross-validation."""
    
    SEED = 2
    BATCH_SIZE = 16
    
    def __init__(self, folds, features_path, splits_dir, csv_path, results_dir,
                 feature_extractor="uni_v2", epochs=10, learning_rate=4e-4,
                 mil="abmil", use_class_weights=True, grid_params_path="configs/abmil.json"):
        """
        Initialize GridSearch.
        
        Args:
            folds: Number of cross-validation folds
            features_path: Path to H5 feature files directory
            splits_dir: Directory containing split files
            csv_path: Path to dataset CSV
            results_dir: Output directory for results
            feature_extractor: Feature extractor name (default: "uni_v2")
            epochs: Number of training epochs (default: 10)
            learning_rate: Learning rate (default: 4e-4)
            mil: MIL architecture name (default: "abmil")
            use_class_weights: Whether to use class weights (default: True)
            grid_params_path: Path to JSON file with grid search parameters
        """
        self.folds = folds
        self.features_path = os.path.realpath(features_path)
        self.splits_dir = os.path.realpath(splits_dir)
        self.csv_path = os.path.realpath(csv_path)
        self.results_dir = os.path.realpath(results_dir)
        self.feature_extractor = feature_extractor
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.mil = mil
        self.use_class_weights = use_class_weights
        self.batch_size = 1 if mil == "clam" else self.BATCH_SIZE
        
        self.logger = logging.getLogger(__name__)
        
        self.logger.info(f"Initializing GridSearch with parameters:")
        self.logger.info(f"  - Folds: {folds}")
        self.logger.info(f"  - Features path: {self.features_path}")
        self.logger.info(f"  - Splits directory: {self.splits_dir}")
        self.logger.info(f"  - CSV path: {self.csv_path}")
        self.logger.info(f"  - Results directory: {self.results_dir}")
        self.logger.info(f"  - Feature extractor: {feature_extractor}")
        self.logger.info(f"  - Epochs: {epochs}")
        self.logger.info(f"  - Learning rate: {learning_rate}")
        self.logger.info(f"  - MIL model: {mil}")
        self.logger.info(f"  - Use class weights: {use_class_weights}")
        self.logger.info(f"  - Batch size: {self.batch_size}")
        self.logger.info(f"  - Device: {device}")
        
        # Load grid parameters
        self.logger.info(f"Loading grid search parameters from: {grid_params_path}")
        with open(grid_params_path, "r") as f:
            grid_params = json.load(f)
        self.logger.debug(f"Grid parameters: {grid_params}")
        
        self.param_combinations = [
            dict(zip(grid_params.keys(), combo))
            for combo in product(*grid_params.values())
        ]
        self.logger.info(f"Generated {len(self.param_combinations)} parameter combinations")
        
        # Set seed
        self.logger.debug(f"Setting random seed to: {self.SEED}")
        seed_torch(self.SEED)
        
        self.logger.info(f"Creating results directory: {self.results_dir}")
        os.makedirs(self.results_dir, exist_ok=True)
    
    def _convert_value(self, v):
        """Convert float to int if whole number (pandas casts ints to floats)."""
        if isinstance(v, float) and v.is_integer():
            return int(v)
        return v
    
    def _load_fold_data(self, fold_idx):
        """Load splits, descriptors and class weights for a fold."""
        self.logger.debug(f"Loading fold {fold_idx} data")
        splits_file = f"{self.splits_dir}/splits_{fold_idx}_bool.csv"
        self.logger.debug(f"Loading splits from: {splits_file}")
        splits = pd.read_csv(splits_file)
        splits.columns = ["slide_id", "train", "val", "test"]
        
        descriptors_file = f"{self.splits_dir}/splits_{fold_idx}_descriptor.csv"
        self.logger.debug(f"Loading descriptors from: {descriptors_file}")
        descriptors = pd.read_csv(descriptors_file, index_col=0)
        
        class_weights = None
        if self.use_class_weights:
            self.logger.debug("Computing class weights from training data")
            class_weights = get_weights(descriptors.train)
            self.logger.debug(f"Class weights: {class_weights}")
        else:
            self.logger.debug("Class weights disabled")
        
        self.logger.debug(f"Merging dataset CSV with splits")
        dataset_csv = pd.read_csv(self.csv_path).merge(splits, on="slide_id")
        self.logger.debug(f"Fold {fold_idx} data loaded: {len(dataset_csv)} slides")
        return dataset_csv, class_weights
    
    def _create_loader(self, dataset_csv, split, shuffle):
        """Create DataLoader for a given split."""
        split_data = dataset_csv[dataset_csv[split] == True]
        self.logger.debug(f"Creating DataLoader for {split} split: {len(split_data)} slides, shuffle={shuffle}")
        loader = DataLoader(
            H5Dataset(self.features_path, dataset_csv, split, variable_patches=True),
            batch_size=self.batch_size,
            shuffle=shuffle,
            collate_fn=variable_patches_collate_fn,
            worker_init_fn=lambda _: np.random.seed(self.SEED),
        )
        self.logger.debug(f"DataLoader created: {len(loader)} batches")
        return loader
    
    def _get_best_params(self, grid_search_results, params_columns):
        """Find parameter combination with highest mean val_auc."""
        self.logger.info("Finding best parameters by mean validation AUC")
        mean_grouped = (
            grid_search_results.groupby(params_columns)["val_auc"]
            .mean()
            .reset_index()
        )
        best_idx = mean_grouped["val_auc"].idxmax()
        best_params = mean_grouped.loc[best_idx, params_columns].to_dict()
        best_auc = mean_grouped.loc[best_idx, "val_auc"]
        self.logger.info(f"Best mean validation AUC: {best_auc:.4f}")
        self.logger.debug(f"Best parameters: {best_params}")
        return best_params
    
    def _filter_best_folds(self, grid_search_results, best_params_dict, params_columns):
        """Filter folds that use the best parameters."""
        mask = pd.Series([True] * len(grid_search_results))
        for col in params_columns:
            mask = mask & (grid_search_results[col] == best_params_dict[col])
        return grid_search_results[mask]
    
    def run(self):
        """Execute grid search."""
        self.logger.info("=" * 60)
        self.logger.info("Starting grid search")
        self.logger.info("=" * 60)
        self.logger.info(f"Using: {self.feature_extractor} with {self.mil}")
        self.logger.info(f"Generated {len(self.param_combinations)} parameter combinations")
        self.logger.info(f"Total combinations to train: {len(self.param_combinations) * self.folds}")
        
        # Grid search: train all param/fold combinations
        grid_search_results = []
        combination_idx = 0
        total_combinations = len(self.param_combinations) * self.folds
        
        for params in self.param_combinations:
            combination_idx += 1
            self.logger.info("-" * 60)
            self.logger.info(f"Parameter combination {combination_idx}/{len(self.param_combinations)}: {params}")
            
            for fold in range(self.folds):
                self.logger.info(f"Processing fold {fold}/{self.folds-1} for current parameters")
                dataset_csv, class_weights = self._load_fold_data(fold)
                train_loader = self._create_loader(dataset_csv, "train", shuffle=True)
                val_loader = self._create_loader(dataset_csv, "val", shuffle=True)
                
                self.logger.debug(f"Importing {self.mil} model with parameters: {params}")
                mil = import_model(
                    self.mil, self.feature_extractor, **params
                ).to(device)
                
                self.logger.info(f"Training model for fold {fold}")
                _, train_metrics, checkpoint = train(
                    mil,
                    train_loader,
                    val_loader,
                    self.results_dir,
                    self.learning_rate,
                    fold,
                    self.epochs,
                    class_weights=class_weights,
                    model_name=self.mil,
                    params=params,
                )
                
                self.logger.info(f"✓ Training completed for fold {fold}")
                self.logger.debug(f"  - Validation AUC: {train_metrics.get('val_auc', 'N/A'):.4f}")
                self.logger.debug(f"  - Checkpoint: {checkpoint}")
                
                train_metrics.update({
                    "model_checkpoint": checkpoint,
                    "fold": fold,
                    "params": params,
                })
                grid_search_results.append(train_metrics)
                
                progress = len(grid_search_results) / total_combinations * 100
                self.logger.info(f"Progress: {len(grid_search_results)}/{total_combinations} ({progress:.1f}%)")
        
        # Select best parameters by mean validation AUC
        self.logger.info("=" * 60)
        self.logger.info("Analyzing grid search results")
        self.logger.info("=" * 60)
        
        self.logger.debug("Normalizing grid search results")
        grid_search_results = pd.json_normalize(grid_search_results)
        params_columns = [
            col for col in grid_search_results.columns if col.startswith("params.")
        ]
        self.logger.debug(f"Parameter columns: {params_columns}")
        
        best_params_dict = self._get_best_params(grid_search_results, params_columns)
        best_folds = self._filter_best_folds(
            grid_search_results, best_params_dict, params_columns
        )
        self.logger.info(f"Best parameters: {best_params_dict}")
        self.logger.info(f"Folds with best params: {len(best_folds)}")
        
        clean_params = {
            k.replace("params.", ""): self._convert_value(v)
            for k, v in best_params_dict.items()
        }
        self.logger.debug(f"Cleaned parameters: {clean_params}")
        
        # Test best model on each fold
        self.logger.info("=" * 60)
        self.logger.info("Testing best models on test sets")
        self.logger.info("=" * 60)
        
        test_results = []
        for idx, (_, row) in enumerate(best_folds.iterrows(), 1):
            fold_idx = int(row["fold"])
            self.logger.info(f"Testing fold {fold_idx} ({idx}/{len(best_folds)})")
            
            dataset_csv, class_weights = self._load_fold_data(fold_idx)
            test_loader = self._create_loader(dataset_csv, "test", shuffle=False)
            
            self.logger.debug(f"Loading model checkpoint: {row['model_checkpoint']}")
            mil = import_model(
                self.mil, self.feature_extractor, **clean_params
            ).to(device)
            mil.load_state_dict(torch.load(row["model_checkpoint"]))
            
            self.logger.info(f"Evaluating model on test set")
            test_metrics, y_preds, y_true = test(
                mil, test_loader, class_weights=class_weights, model_name=self.mil
            )
            test_metrics["fold"] = fold_idx
            test_results.append(test_metrics)
            self.logger.info(f"✓ Fold {fold_idx} test results: AUC={test_metrics.get('test_auc', 'N/A'):.4f}, "
                           f"Acc={test_metrics.get('test_acc', 'N/A'):.4f}, "
                           f"F1={test_metrics.get('f1_macro', 'N/A'):.4f}")
            
            predictions_file = f"{self.results_dir}/predictions_{self.feature_extractor}.{self.mil}_{fold_idx}.csv"
            self.logger.debug(f"Saving predictions to: {predictions_file}")
            predictions = pd.DataFrame()
            predictions["y_pred"] = y_preds
            predictions["y_true"] = y_true
            predictions.to_csv(predictions_file, index=False)
        
        self.logger.info("=" * 60)
        self.logger.info("Compiling final test results")
        self.logger.info("=" * 60)
        
        test_results_df = pd.DataFrame(test_results)
        test_results_df["feature_extractor"] = self.feature_extractor
        test_results_df["mil"] = self.mil
        
        self.logger.info("Test results summary:")
        self.logger.info(f"\n{test_results_df}")
        
        test_results_file = f"{self.results_dir}/test_results_{self.feature_extractor}.{self.mil}.csv"
        self.logger.info(f"Saving test results to: {test_results_file}")
        test_results_df.to_csv(test_results_file, index=False)
        
        best_params_file = f"{self.results_dir}/best_params_{self.feature_extractor}.{self.mil}.json"
        self.logger.info(f"Saving best parameters to: {best_params_file}")
        json.dump(clean_params, open(best_params_file, "w"))
        
        self.logger.info("=" * 60)
        self.logger.info("✓ Grid search completed successfully")
        self.logger.info("=" * 60)

