"""
Splits manager module
"""
import os
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split

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
        
    def __create_split(self):
        """Extracts train, test and val from the same dataset"""
        data = self.__load_dataset()
        grouped = data.groupby(by=["case_id", "label"], as_index=False).first()[["case_id", "label"]]
        train_grouped, test_grouped = train_test_split(grouped, test_size=self.test_frac)
        train_grouped, val_grouped = train_test_split(train_grouped, test_size=self.test_frac)
        train = train_grouped.merge(data, on=["case_id", "label"])
        val = val_grouped.merge(data, on=["case_id", "label"])
        test = test_grouped.merge(data, on=["case_id", "label"])

        train.index = train.slide_id
        val.index = val.slide_id
        test.index = test.slide_id

        train["train"] = True
        val["val"] = True
        test["test"] = True
        
        data = pd.concat([train, val, test])
        data = data[["train", "val", "test", "label"]]
        data = data.fillna(False)
        data.index.name = None
        return data

    def __check_csv(self):
        """Checks if the CSV file contains the required columns"""
        data_check = pd.read_csv(self.csv_path, nrows=1)
        required_columns = ["case_id", "slide_id", self.target]
        missing_columns = [col for col in required_columns if col not in data_check.columns]
        if missing_columns:
            raise ValueError(f"CSV file must contain the columns: {', '.join(missing_columns)}")

    def create_splits(self):
        """Creates splits for the dataset"""
        for i in tqdm(range(self.folds)):
            #We have two options
            output_path = f"{self.splits_dir}/{self.output_name}/"
            splits_bool = self.__create_split()
            os.makedirs(output_path, exist_ok=True) #Creates folder
            splits_bool.drop(columns=["label"]).to_csv(f"{output_path}/splits_{i}_bool.csv")
            summary = splits_bool.value_counts().reset_index()
            summary.loc[summary.train, "split"] = "train"
            summary.loc[summary.val, "split"] = "val"
            summary.loc[summary.test, "split"] = "test"
            summary = summary[["split", "label", "count"]]
            summary = summary.sort_values(by=["split", "label"])
            summary = summary.pivot(index = "label", columns="split", values="count"
                                    ).reset_index()
            summary = summary[["label", "train", "val", "test"]]
            summary = summary.rename(columns={"label": ""})
            print(summary)
            summary.to_csv(f"{output_path}/splits_{i}_descriptor.csv", index=False)

    def __load_dataset(self):
        """Load the dataset and save it in the output path"""
        data = pd.read_csv(self.csv_path)
        data = data.rename(columns={self.target: "label"})
        data = data[["case_id", "slide_id", "label"]]
        data.to_csv(f"{self.output_path}/dataset.csv", index=False)
        return data