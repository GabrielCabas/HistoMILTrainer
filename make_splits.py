"""Make splits script"""
import argparse
import os
from histomil.splits import SplitManager

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HistoMIL Make Splits Script")
    parser.add_argument("--folds", type=int, default=10)
    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--splits_dir", type=str, default="./splits")
    parser.add_argument("--test_frac", type=float, default=0.2)
    parser.add_argument("--target", type=str, default="target")
    parser.add_argument("--output_name", type=str, required=True)
    args = parser.parse_args()

    split_manager = SplitManager(
        csv_path=args.csv_path,
        output_name=args.output_name,
        folds=args.folds,
        splits_dir=args.splits_dir,
        test_frac=args.test_frac,
        target=args.target,
    )
    split_manager.create_splits()