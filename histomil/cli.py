"""Command-line interface for HistoMILTrainer."""
import argparse
import os
def make_splits():
    from histomil.splits import SplitManager
    """CLI entry point for making splits."""
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


def grid_search():
    from histomil.grid_search import GridSearch
    """CLI entry point for grid search."""
    parser = argparse.ArgumentParser(description="MIL Grid Search")
    parser.add_argument("--folds", type=int, default=10)
    parser.add_argument("--features_path", type=str, required=True)
    parser.add_argument("--splits_dir", type=str, required=True)
    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--results_dir", type=str, default="./temp_dir/")
    parser.add_argument("--feature_extractor", type=str, default="uni_v2")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=4e-4)
    parser.add_argument("--mil", type=str, default="abmil")
    parser.add_argument("--use_class_weights", type=bool, default=True)
    parser.add_argument("--grid_params", type=str, default=None)
    args = parser.parse_args()

    # Determine grid_params_path: use provided path or default to package config
    if args.grid_params:
        grid_params_path = args.grid_params
    else:
        # Try to get config from installed package or development directory
        grid_params_path = None
        
        # First, try relative path from current directory (for development)
        dev_path = os.path.join("histomil", "configs", f"{args.mil}.json")
        if os.path.exists(dev_path):
            grid_params_path = dev_path
        else:
            # Try from installed package location
            try:
                import histomil
                package_dir = os.path.dirname(histomil.__file__)
                package_path = os.path.join(package_dir, "configs", f"{args.mil}.json")
                if os.path.exists(package_path):
                    grid_params_path = package_path
                else:
                    raise FileNotFoundError(
                        f"Config file {args.mil}.json not found. "
                        f"Tried: {dev_path}, {package_path}"
                    )
            except ImportError:
                raise FileNotFoundError(
                    f"Config file {args.mil}.json not found and histomil package not installed."
                )

    # Create GridSearch instance and run
    grid_search = GridSearch(
        folds=args.folds,
        features_path=args.features_path,
        splits_dir=args.splits_dir,
        csv_path=args.csv_path,
        results_dir=args.results_dir,
        feature_extractor=args.feature_extractor,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        mil=args.mil,
        use_class_weights=args.use_class_weights,
        grid_params_path=grid_params_path,
    )
    grid_search.run()

def predict():
    """CLI entry point for grid search."""
    from histomil.predict import Predictor
    parser = argparse.ArgumentParser(description="MIL Predict")
    parser.add_argument("--features_path", type=str, required=True)
    parser.add_argument("--weights_path", type=str, required=True)
    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--results_dir", type=str, default="./")
    parser.add_argument("--feature_extractor", type=str, default="virchow2")
    parser.add_argument("--mil", type=str, default="abmil")
    parser.add_argument("--params_path", type=str, required=True)
    args = parser.parse_args()
    predictor = Predictor(
        csv_path=args.csv_path,
        weights_path=args.weights_path,
        features_path=args.features_path,
        feature_extractor=args.feature_extractor,
        results_dir=args.results_dir,
        mil=args.mil,
        params_path=args.params_path,
    )
    predictor.run()