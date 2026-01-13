"""Grid search for MIL models with cross-validation."""
import os
import argparse
import json
from itertools import product

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from histomil import (
    H5Dataset,
    seed_torch,
    get_weights,
    train,
    test,
    import_model,
    variable_patches_collate_fn,
)

SEED = 2
BATCH_SIZE = 16
seed_torch(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def convert_value(v):
    """Convert float to int if whole number (pandas casts ints to floats)."""
    if isinstance(v, float) and v.is_integer():
        return int(v)
    return v


def parse_args():
    parser = argparse.ArgumentParser(description="MIL Grid Search")
    parser.add_argument("--folds", type=int, default=10)
    parser.add_argument("--features_path", type=str, required=True)
    parser.add_argument("--splits_dir", type=str, required=True)
    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--results_dir", type=str, default="./temp_dir/")
    parser.add_argument("--feature_extractor", type=str, default="uni_v2")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=4e-4)
    parser.add_argument("--mil", type=str, default="abmil")
    parser.add_argument("--use_class_weights", type=bool, default=True)
    parser.add_argument("--grid_params", type=str, default="configs/abmil.json")
    return parser.parse_args()


def load_fold_data(split_dir, csv_path, fold_idx, use_class_weights):
    """Load splits, descriptors and class weights for a fold."""
    splits = pd.read_csv(f"{split_dir}/splits_{fold_idx}_bool.csv")
    splits.columns = ["slide_id", "train", "val", "test"]
    descriptors = pd.read_csv(
        f"{split_dir}/splits_{fold_idx}_descriptor.csv", index_col=0
    )
    class_weights = None
    if use_class_weights:
        class_weights = get_weights(descriptors.train)
    dataset_csv = pd.read_csv(csv_path).merge(splits, on="slide_id")
    return dataset_csv, class_weights


def create_loader(features_path, dataset_csv, split, batch_size, shuffle):
    """Create DataLoader for a given split."""
    return DataLoader(
        H5Dataset(features_path, dataset_csv, split, variable_patches=True),
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=variable_patches_collate_fn,
        worker_init_fn=lambda _: np.random.seed(SEED),
    )


def get_best_params(grid_search_results, params_columns):
    """Find parameter combination with highest mean val_auc."""
    mean_grouped = (
        grid_search_results.groupby(params_columns)["val_auc"]
        .mean()
        .reset_index()
        .rename(columns={"val_auc": "mean_val_auc"})
    )
    best_row = mean_grouped.loc[mean_grouped["mean_val_auc"].idxmax()]
    return best_row[params_columns].to_dict()


def filter_best_folds(grid_search_results, best_params_dict, params_columns):
    """Get all folds matching the best parameter combination."""
    mask = pd.Series([True] * len(grid_search_results))
    for col, val in best_params_dict.items():
        mask &= (grid_search_results[col] == val)
    return grid_search_results[mask]


if __name__ == "__main__":
    args = parse_args()

    features_path = os.path.realpath(args.features_path)
    split_dir = os.path.realpath(args.splits_dir)
    csv_path = os.path.realpath(args.csv_path)
    results_dir = os.path.realpath(args.results_dir)

    batch_size = 1 if args.mil == "clam" else BATCH_SIZE

    print(f"Using: {args.feature_extractor} with {args.mil}")
    os.makedirs(results_dir, exist_ok=True)

    with open(args.grid_params, "r") as f:
        grid_params = json.load(f)

    param_combinations = [
        dict(zip(grid_params.keys(), combo))
        for combo in product(*grid_params.values())
    ]
    print(f"Generated {len(param_combinations)} parameter combinations")

    # Grid search: train all param/fold combinations
    grid_search_results = []
    for params in param_combinations[:3]:
        print(f"Params: {params}")
        for fold in range(args.folds):
            print(f"Fold: {fold}")
            dataset_csv, class_weights = load_fold_data(
                split_dir, csv_path, fold, args.use_class_weights
            )
            train_loader = create_loader(
                features_path, dataset_csv, "train", batch_size, shuffle=True
            )
            val_loader = create_loader(
                features_path, dataset_csv, "val", batch_size, shuffle=True
            )

            mil = import_model(
                args.mil, args.feature_extractor, **params
            ).to(device)
            _, train_metrics, checkpoint = train(
                mil,
                train_loader,
                val_loader,
                results_dir,
                args.learning_rate,
                fold,
                args.epochs,
                class_weights=class_weights,
                model_name=args.mil,
                params=params,
            )
            train_metrics.update({
                "model_checkpoint": checkpoint,
                "fold": fold,
                "params": params,
            })
            grid_search_results.append(train_metrics)

    # Select best parameters by mean validation AUC
    grid_search_results = pd.json_normalize(grid_search_results)
    params_columns = [
        col for col in grid_search_results.columns if col.startswith("params.")
    ]
    best_params_dict = get_best_params(grid_search_results, params_columns)
    best_folds = filter_best_folds(
        grid_search_results, best_params_dict, params_columns
    )
    print(f"Best parameters: {best_params_dict}")
    print(f"Folds with best params: {len(best_folds)}")

    clean_params = {
        k.replace("params.", ""): convert_value(v)
        for k, v in best_params_dict.items()
    }

    # Test best model on each fold
    test_results = []
    for _, row in best_folds.iterrows():
        fold_idx = int(row["fold"])
        print(f"Testing fold: {fold_idx}")

        dataset_csv, class_weights = load_fold_data(
            split_dir, csv_path, fold_idx, args.use_class_weights
        )
        test_loader = create_loader(
            features_path, dataset_csv, "test", batch_size, shuffle=False
        )

        mil = import_model(
            args.mil, args.feature_extractor, **clean_params
        ).to(device)
        mil.load_state_dict(torch.load(row["model_checkpoint"]))

        test_metrics = test(
            mil, test_loader, class_weights=class_weights, model_name=args.mil
        )
        test_metrics["fold"] = fold_idx
        test_results.append(test_metrics)
        print(f"Fold {fold_idx}: {test_metrics}")

    test_results_df = pd.DataFrame(test_results)
    print("Test results:\n", test_results_df)
    test_results_df.to_csv(f"{results_dir}/test_results.csv", index=False)
