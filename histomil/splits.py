"""
Splits manager module
"""
import os
import pandas as pd
from sklearn.model_selection import train_test_split, KFold


class SplitManager:
    """Splits manager class"""
    def __init__(self, args):
        self.csv_path = args.csv_path
        self.target = args.target
        self.test_frac = args.test_frac
        self.splits_dir = args.splits_dir
        self.output_name = args.output_name
        self.folds = args.folds
        self.output_path = f"{self.splits_dir}/{self.output_name}/"
        os.makedirs(self.output_path, exist_ok=True)
        self.__check_csv()
        
    def __create_split(self, fold_idx):
        """Creates train, val, and test splits using KFold for train/val"""
        data = self.__load_dataset()
        
        # First, separate test set at case level to avoid data leakage
        grouped = data.groupby(by=["case_id", "label"], as_index=False).first()[["case_id", "label"]]
        train_val_grouped, test_grouped = train_test_split(
            grouped, test_size=self.test_frac, random_state=42
        )
        
        # Get all slides for test cases
        test_cases = set(test_grouped["case_id"])
        test_mask = data["case_id"].isin(test_cases)
        test_slides = set(data[test_mask]["slide_id"].values)
        
        # Get train+val cases (as array for KFold)
        train_val_cases = train_val_grouped["case_id"].unique()
        
        # Use KFold to split train+val into train and val
        kf = KFold(n_splits=self.folds, shuffle=True, random_state=42)
        train_val_indices = list(kf.split(train_val_cases))[fold_idx]
        train_case_indices, val_case_indices = train_val_indices
        
        train_cases = set(train_val_cases[train_case_indices])
        val_cases = set(train_val_cases[val_case_indices])
        
        # Get slides for train and val cases
        train_mask = data["case_id"].isin(train_cases)
        val_mask = data["case_id"].isin(val_cases)
        train_slides = set(data[train_mask]["slide_id"].values)
        val_slides = set(data[val_mask]["slide_id"].values)
        
        # Create splits DataFrame with slide_id as index
        # Get unique slide_id -> label mapping (in case of duplicates, take first)
        slide_label_map = data[["slide_id", "label"]].drop_duplicates(subset="slide_id").set_index("slide_id")["label"]
        
        all_slides = data["slide_id"].unique()
        splits_df = pd.DataFrame(index=all_slides)
        splits_df["train"] = splits_df.index.isin(train_slides)
        splits_df["val"] = splits_df.index.isin(val_slides)
        splits_df["test"] = splits_df.index.isin(test_slides)
        splits_df["label"] = slide_label_map
        
        return splits_df

    def __check_csv(self):
        """Checks if the CSV file contains the required columns"""
        data_check = pd.read_csv(self.csv_path, nrows=1)
        required_columns = ["case_id", "slide_id", self.target]
        missing_columns = [col for col in required_columns if col not in data_check.columns]
        if missing_columns:
            raise ValueError(f"CSV file must contain the columns: {', '.join(missing_columns)}")

    def create_splits(self):
        """Creates splits for the dataset"""
        output_path = f"{self.splits_dir}/{self.output_name}/"
        os.makedirs(output_path, exist_ok=True)
        
        for i in range(self.folds):
            splits_bool = self.__create_split(i)
            splits_bool.drop(columns=["label"]).to_csv(f"{output_path}/splits_{i}_bool.csv")
            
            # Create summary
            summary = splits_bool.value_counts().reset_index()
            summary.loc[summary.train, "split"] = "train"
            summary.loc[summary.val, "split"] = "val"
            summary.loc[summary.test, "split"] = "test"
            summary = summary[["split", "label", "count"]]
            summary = summary.sort_values(by=["split", "label"])
            summary = summary.pivot(index="label", columns="split", values="count").reset_index()
            summary = summary[["label", "train", "val", "test"]]
            summary = summary.rename(columns={"label": ""})
            summary.to_csv(f"{output_path}/splits_{i}_descriptor.csv", index=False)

    def __load_dataset(self):
        """Load the dataset and save it in the output path"""
        data = pd.read_csv(self.csv_path)
        data = data.rename(columns={self.target: "label"})
        data = data[["case_id", "slide_id", "label"]]
        data.to_csv(f"{self.output_path}/dataset.csv", index=False)
        return data
