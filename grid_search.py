"""
Main file
"""
import os
import argparse
import pandas as pd
import torch
from torch.utils.data import DataLoader
import numpy as np
from histomil import H5Dataset, seed_torch, get_weights, train, test, import_model, variable_patches_collate_fn
import json
from itertools import product
import random


SEED = 2
BATCH_SIZE = 16
seed_torch(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CRAB-MIL Training Script")
    parser.add_argument("--folds", type=int, default=10)
    parser.add_argument("--features_path", type=str, required=True)
    parser.add_argument("--splits_dir", type=str, required=True)
    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--results_dir", type=str, default="./temp_dir/")
    parser.add_argument("--feature_extractor", type=str, default = "uni_v2")
    parser.add_argument("--epochs", type=int, default = 20)
    parser.add_argument("--learning_rate", type=float, default = 4e-4)
    parser.add_argument("--mil", type=str, default="abmil")
    parser.add_argument("--use_class_weights", type=bool, default=True)
    parser.add_argument("--grid_params", type=str, default="configs/abmil.json")
    args = parser.parse_args()
    

    features_path = os.path.realpath(args.features_path)
    split_dir = os.path.realpath(args.splits_dir)
    csv_path = os.path.realpath(args.csv_path)
    results_dir = os.path.realpath(args.results_dir)

    if args.mil == "clam": #CLAM needs it
        BATCH_SIZE = 1

    print("Using:", args.feature_extractor, "With: ", args.mil)
    os.makedirs(results_dir, exist_ok=True)

    with open(args.grid_params, "r") as f:
        grid_params = json.load(f)
    
    # Generate all parameter combinations
    param_names = list(grid_params.keys())
    param_values = list(grid_params.values())
    
    # Create list of dictionaries with all combinations
    param_combinations = []
    for combination in product(*param_values):
        param_dict = dict(zip(param_names, combination))
        param_combinations.append(param_dict)
    
    print(f"Generated {len(param_combinations)} parameter combinations")
    
    grid_search_results = []
    for params in param_combinations[:10]:
        print("Params:", params)
        trained_models = []
        for fold in range(args.folds):
            dataset_csv = pd.read_csv(csv_path)

            splits = pd.read_csv(f"{split_dir}/splits_{fold}_bool.csv")
            splits.columns = ["slide_id", "train", "val", "test"]

            descriptors = pd.read_csv(f"{split_dir}/splits_{fold}_descriptor.csv", index_col=0)
            print(descriptors)

            if args.use_class_weights:
                class_weights = get_weights(descriptors.train)
                print("Using class_weights:", class_weights)
            else:
                class_weights = None
                
            dataset_csv = dataset_csv.merge(splits, on="slide_id")
            print(dataset_csv)
            print("Create datasets")
            
            train_loader = DataLoader(H5Dataset(features_path, dataset_csv, "train", variable_patches=True),
                                    batch_size=BATCH_SIZE, shuffle=True,
                                    collate_fn=variable_patches_collate_fn,
                                    worker_init_fn=lambda _: np.random.seed(SEED))

            val_loader = DataLoader(H5Dataset(features_path, dataset_csv, "val", variable_patches=True), 
                                    batch_size=BATCH_SIZE, shuffle=True,
                                    collate_fn=variable_patches_collate_fn,
                                    worker_init_fn=lambda _: np.random.seed(SEED))

            test_loader = DataLoader(H5Dataset(features_path, dataset_csv, "test", variable_patches=True),
                                    batch_size=BATCH_SIZE, shuffle=False,
                                    collate_fn=variable_patches_collate_fn,
                                    worker_init_fn=lambda _: np.random.seed(SEED))

            print("Batches train:",len(train_loader))
            print("Batches val:", len(val_loader))
            print("Batches test:", len(test_loader))
            print("Importing model")

            print("Fold:", fold)
            mil = import_model(args.mil, args.feature_extractor, **params).to(device)
            trained_model, train_metrics, model_output_name = train(mil, train_loader,
                                                val_loader, results_dir,
                                                args.learning_rate,
                                                fold, args.epochs,
                                                class_weights = class_weights,
                                                model_name = args.mil)
            train_metrics["model_checkpoint"] = model_output_name
            train_metrics["fold"] = fold
            train_metrics["params"] = params
            grid_search_results.append(train_metrics)
    grid_search_results = pd.json_normalize(grid_search_results)
    params_columns = [col for col in grid_search_results.columns if col.startswith("params.")]
    
    # Calculate mean val_auc per parameter combination
    mean_grouped = grid_search_results.groupby(params_columns)["val_auc"].mean().reset_index()
    mean_grouped = mean_grouped.rename(columns={"val_auc": "mean_val_auc"})
    # Get the parameter combination with the highest mean val_auc
    best_params_row = mean_grouped.loc[mean_grouped["mean_val_auc"].idxmax()]
    
    # Get all folds that match the best parameter combination
    best_params_dict = best_params_row[params_columns].to_dict()
    mask = pd.Series([True] * len(grid_search_results))
    for col, val in best_params_dict.items():
        mask &= (grid_search_results[col] == val)
    
    best_estimator_folds = grid_search_results[mask]
    print(f"All {len(best_estimator_folds)} folds with best parameters:")
    print(best_estimator_folds)

    # Remove "params." prefix from keys and convert floats to int where applicable
    def convert_value(v):
        """Convert float to int if it's a whole number (pandas often converts ints to floats)"""
        if isinstance(v, float) and v.is_integer():
            return int(v)
        return v
    
    clean_params = {k.replace("params.", ""): convert_value(v) for k, v in best_params_dict.items()}
    
    # Test each fold with its corresponding checkpoint and test data
    test_results = []
    for _, row in best_estimator_folds.iterrows():
        fold_idx = int(row["fold"])
        checkpoint_path = row["model_checkpoint"]
        
        print(f"Testing fold: {fold_idx}")
        
        # Reload the test data for this specific fold
        splits = pd.read_csv(f"{split_dir}/splits_{fold_idx}_bool.csv")
        splits.columns = ["slide_id", "train", "val", "test"]
        descriptors = pd.read_csv(f"{split_dir}/splits_{fold_idx}_descriptor.csv", index_col=0)
        
        if args.use_class_weights:
            fold_class_weights = get_weights(descriptors.train)
        else:
            fold_class_weights = None
        
        fold_dataset_csv = pd.read_csv(csv_path).merge(splits, on="slide_id")
        fold_test_loader = DataLoader(
            H5Dataset(features_path, fold_dataset_csv, "test", variable_patches=True),
            batch_size=BATCH_SIZE, shuffle=False,
            collate_fn=variable_patches_collate_fn,
            worker_init_fn=lambda _: np.random.seed(SEED)
        )
        
        # Create a new model with the correct parameters and load checkpoint
        mil = import_model(args.mil, args.feature_extractor, **clean_params).to(device)
        mil.load_state_dict(torch.load(checkpoint_path))
        
        test_metrics = test(mil, fold_test_loader, class_weights=fold_class_weights, model_name=args.mil)
        test_metrics["fold"] = fold_idx
        test_results.append(test_metrics)
        print(f"Fold {fold_idx} metrics:", test_metrics)
    
    # Save all test results
    test_results_df = pd.DataFrame(test_results)
    print("All test results:")
    print(test_results_df)
    test_results_df.to_csv(f"{results_dir}/test_results.csv", index=False)