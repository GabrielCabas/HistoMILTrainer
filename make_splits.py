"""Make splits script"""
import argparse
import os
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split

def create_simple(args):
    """Extracts train, test and val from the same dataset"""
    data = pd.read_csv(args.csv_path)
    data = data.rename(columns={args.target: "label"})
    grouped = data.groupby(by=["case_id", "label"], as_index=False).first()[["case_id", "label"]]
    train_grouped, test_grouped = train_test_split(grouped, test_size=args.test_frac)
    train_grouped, val_grouped = train_test_split(train_grouped, test_size=args.test_frac)
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HistoMIL Make Splits Script")
    parser.add_argument("--folds", type=int, default=10)
    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--splits_dir", type=str, default="./splits/")
    parser.add_argument("--test_frac", type=float, default = 0.2)
    parser.add_argument("--target", type=str, default="target")
    parser.add_argument("--external_test_csv", type=str, default=None) #Optional, if is provided, it's used as test set.
    args = parser.parse_args()
    csv_path = os.path.basename(args.csv_path).replace(".csv", "")
    # INSERT_YOUR_CODE
    # Validar que el dataset tiene las columnas "case_id" y "slide_id"
    data_check = pd.read_csv(args.csv_path, nrows=1)
    required_columns = ["case_id", "slide_id"]
    missing_columns = [col for col in required_columns if col not in data_check.columns]
    if missing_columns:
        raise ValueError(f"CSV file must contain the columns: {', '.join(missing_columns)}")

    for i in tqdm(range(args.folds)):
        #We have two options
        output_path = f"{args.splits_dir}/{csv_path}/"
        splits_bool = create_simple(args)
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
