"""
Splits manager module
"""
import os
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
import logging

class SplitManager:
    """Splits manager class"""
    def __init__(self, csv_path, output_name, folds=10, splits_dir="./splits", 
                 test_frac=0.2, target="target"):
        """
        Initialize SplitManager.
        
        Args:
            csv_path: Path to the CSV file containing the dataset
            output_name: Name for the output directory
            folds: Number of cross-validation folds (default: 10)
            splits_dir: Directory where splits will be saved (default: "./splits")
            test_frac: Fraction of data to use for testing (default: 0.2)
            target: Name of the target column in the CSV (default: "target")
        """
        self.logger = logging.getLogger(__name__)
        self.csv_path = csv_path
        self.target = target
        self.test_frac = test_frac
        self.splits_dir = splits_dir
        self.output_name = output_name
        self.folds = folds
        self.output_path = f"{self.splits_dir}/{self.output_name}/"
        
        self.logger.info(f"Initializing SplitManager with parameters:")
        self.logger.info(f"  - CSV path: {csv_path}")
        self.logger.info(f"  - Output name: {output_name}")
        self.logger.info(f"  - Folds: {folds}")
        self.logger.info(f"  - Splits directory: {splits_dir}")
        self.logger.info(f"  - Test fraction: {test_frac}")
        self.logger.info(f"  - Target column: {target}")
        
        self.logger.info(f"Creating output directory: {self.output_path}")
        os.makedirs(self.output_path, exist_ok=True)
        self.__check_csv()
        
    def __create_split(self, fold_idx):
        """Creates train, val, and test splits using KFold for train/val"""
        self.logger.info(f"Creating split for fold {fold_idx}/{self.folds-1}")
        data = self.__load_dataset()
        
        # First, separate test set at case level to avoid data leakage
        self.logger.debug("Grouping data by case_id and label to avoid data leakage")
        grouped = data.groupby(by=["case_id", "label"], as_index=False).first()[["case_id", "label"]]
        self.logger.debug(f"Unique grouped cases: {len(grouped)}")
        
        self.logger.info(f"Separating test set ({self.test_frac*100:.1f}% of cases)")
        train_val_grouped, test_grouped = train_test_split(
            grouped, test_size=self.test_frac, random_state=42
        )
        self.logger.info(f"  - Train+val cases: {len(train_val_grouped)}")
        self.logger.info(f"  - Test cases: {len(test_grouped)}")
        
        # Get all slides for test cases
        test_cases = set(test_grouped["case_id"])
        test_mask = data["case_id"].isin(test_cases)
        test_slides = set(data[test_mask]["slide_id"].values)
        self.logger.debug(f"Test slides: {len(test_slides)}")
        
        # Get train+val cases (as array for KFold)
        train_val_cases = train_val_grouped["case_id"].unique()
        
        # Use KFold to split train+val into train and val
        self.logger.debug(f"Applying KFold (n_splits={self.folds}) to split train+val into train and val")
        kf = KFold(n_splits=self.folds, shuffle=True, random_state=42)
        train_val_indices = list(kf.split(train_val_cases))[fold_idx]
        train_case_indices, val_case_indices = train_val_indices
        
        train_cases = set(train_val_cases[train_case_indices])
        val_cases = set(train_val_cases[val_case_indices])
        self.logger.info(f"  - Train cases: {len(train_cases)}")
        self.logger.info(f"  - Val cases: {len(val_cases)}")
        
        # Get slides for train and val cases
        train_mask = data["case_id"].isin(train_cases)
        val_mask = data["case_id"].isin(val_cases)
        train_slides = set(data[train_mask]["slide_id"].values)
        val_slides = set(data[val_mask]["slide_id"].values)
        self.logger.info(f"  - Train slides: {len(train_slides)}")
        self.logger.info(f"  - Val slides: {len(val_slides)}")
        
        # Create splits DataFrame with slide_id as index
        # Get unique slide_id -> label mapping (in case of duplicates, take first)
        self.logger.debug("Creating splits DataFrame with slide_id as index")
        slide_label_map = data[["slide_id", "label"]].drop_duplicates(subset="slide_id").set_index("slide_id")["label"]
        
        all_slides = data["slide_id"].unique()
        splits_df = pd.DataFrame(index=all_slides)
        splits_df["train"] = splits_df.index.isin(train_slides)
        splits_df["val"] = splits_df.index.isin(val_slides)
        splits_df["test"] = splits_df.index.isin(test_slides)
        splits_df["label"] = slide_label_map
        
        # Log final distribution
        train_count = splits_df["train"].sum()
        val_count = splits_df["val"].sum()
        test_count = splits_df["test"].sum()
        self.logger.info(f"✓ Split fold {fold_idx} created: train={train_count}, val={val_count}, test={test_count}")
        
        return splits_df

    def __check_csv(self):
        """Checks if the CSV file contains the required columns"""
        self.logger.info(f"Checking CSV structure: {self.csv_path}")
        data_check = pd.read_csv(self.csv_path, nrows=1)
        required_columns = ["case_id", "slide_id", self.target]
        self.logger.debug(f"Required columns: {required_columns}")
        self.logger.debug(f"Found columns: {list(data_check.columns)}")
        missing_columns = [col for col in required_columns if col not in data_check.columns]
        if missing_columns:
            error_msg = f"CSV file must contain the columns: {', '.join(missing_columns)}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        self.logger.info("✓ CSV validation completed: all required columns are present")

    def create_splits(self):
        """Creates splits for the dataset"""
        self.logger.info("=" * 60)
        self.logger.info("Starting dataset splits creation")
        self.logger.info("=" * 60)
        
        output_path = f"{self.splits_dir}/{self.output_name}/"
        os.makedirs(output_path, exist_ok=True)
        self.logger.info(f"Output directory: {output_path}")
        
        for i in range(self.folds):
            self.logger.info("-" * 60)
            splits_bool = self.__create_split(i)
            
            # Save boolean splits
            bool_file = f"{output_path}/splits_{i}_bool.csv"
            self.logger.debug(f"Saving boolean splits to: {bool_file}")
            splits_bool.drop(columns=["label"]).to_csv(bool_file)
            self.logger.debug(f"✓ File saved: {bool_file}")
            
            # Create summary
            self.logger.debug("Generating descriptive summary of the split")
            summary = splits_bool.value_counts().reset_index()
            summary.loc[summary.train, "split"] = "train"
            summary.loc[summary.val, "split"] = "val"
            summary.loc[summary.test, "split"] = "test"
            summary = summary[["split", "label", "count"]]
            summary = summary.sort_values(by=["split", "label"])
            summary = summary.pivot(index="label", columns="split", values="count").reset_index()
            summary = summary[["label", "train", "val", "test"]]
            summary = summary.rename(columns={"label": ""})
            
            descriptor_file = f"{output_path}/splits_{i}_descriptor.csv"
            self.logger.debug(f"Saving split descriptor to: {descriptor_file}")
            summary.to_csv(descriptor_file, index=False)
            self.logger.debug(f"✓ File saved: {descriptor_file}")
        
        self.logger.info("=" * 60)
        self.logger.info(f"✓ Process completed: {self.folds} folds created successfully")
        self.logger.info(f"Files saved to: {output_path}")
        self.logger.info("=" * 60)

    def __load_dataset(self):
        """Load the dataset and save it in the output path"""
        self.logger.info(f"Loading dataset from: {self.csv_path}")
        data = pd.read_csv(self.csv_path)
        self.logger.info(f"Dataset loaded: {len(data)} rows, {len(data.columns)} columns")
        
        self.logger.debug(f"Renaming column '{self.target}' to 'label'")
        data = data.rename(columns={self.target: "label"})
        
        self.logger.debug("Selecting columns: case_id, slide_id, label")
        data = data[["case_id", "slide_id", "label"]]
        
        dataset_output_path = f"{self.output_path}/dataset.csv"
        self.logger.info(f"Saving processed dataset to: {dataset_output_path}")
        data.to_csv(dataset_output_path, index=False)
        self.logger.info(f"✓ Dataset saved: {len(data)} rows")
        
        # Log basic statistics
        unique_cases = data["case_id"].nunique()
        unique_slides = data["slide_id"].nunique()
        unique_labels = data["label"].nunique()
        self.logger.info(f"Dataset statistics: {unique_cases} unique cases, {unique_slides} unique slides, {unique_labels} classes")
        
        return data
