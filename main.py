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

SEED = 2
BATCH_SIZE = 16
seed_torch(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CRAB-MIL Training Script")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--features_path", type=str, required=True)
    parser.add_argument("--splits_dir", type=str, required=True)
    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--results_dir", type=str, default="./temp_dir/")
    parser.add_argument("--feature_extractor", type=str, default = "uni_v2")
    parser.add_argument("--epochs", type=int, default = 3)
    parser.add_argument("--learning_rate", type=float, default = 4e-4)
    parser.add_argument("--mil", type=str, default="abmil")
    parser.add_argument("--use_class_weights", type=bool, default=True)
    args = parser.parse_args()
    

    features_path = os.path.realpath(args.features_path)
    split_dir = os.path.realpath(args.splits_dir)
    csv_path = os.path.realpath(args.csv_path)
    results_dir = os.path.realpath(args.results_dir)

    if args.mil == "clam": #CLAM needs it
        BATCH_SIZE = 1

    print("Using:", args.feature_extractor, "With: ", args.mil)
    os.makedirs(results_dir, exist_ok=True)

    dataset_csv = pd.read_csv(csv_path)
    splits = pd.read_csv(f"{split_dir}/splits_{args.fold}_bool.csv")
    splits.columns = ["slide_id", "train", "val", "test"]

    descriptors = pd.read_csv(f"{split_dir}/splits_{args.fold}_descriptor.csv", index_col=0)
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

    mil = import_model(args.mil, args.feature_extractor).to(device)

    print(mil)
    print("Training")
    trained_model, train_metrics = train(mil, train_loader,
                                         val_loader, results_dir,
                                         args.learning_rate,
                                         args.fold, args.epochs,
                                         class_weights = class_weights,
                                         model_name = args.mil)
    print("Testing")
    test_metrics = test(trained_model, test_loader, class_weights = class_weights, model_name = args.mil)
    print("Exporting metrics")
    train_metrics = pd.json_normalize(train_metrics)
    test_metrics = pd.json_normalize(test_metrics)
    metrics = pd.concat([train_metrics, test_metrics], axis = 1)

    results_path = f"{results_dir}/{args.fold}.csv"
    metrics.to_csv(results_path, index=False)