import argparse
import pandas as pd
from dask.dataframe import read_csv
import json
import os

parser = argparse.ArgumentParser(description="Prepare data for downstream tasks")
parser.add_argument("--data-path", type=str, help="Path to raw data file the data should be csv")
parser.add_argument("--output-path", type=str, help="Path to save the processed data")
parser.add_argument("--embedded-split", help="Whether the split is already embedded in the data the values are train, val, test", default=False, action='store_true')
parser.add_argument("--split-column", type=str, help="Column name for the split data")
parser.add_argument("--split-ratio", nargs="+", type=list, help="Ratio to split the data")
parser.add_argument("--random-states", nargs="+", type=list, help="Random states for splitting the data")
parser.add_argument("--data-column", type=str, help="Column name for the sequence data")
parser.add_argument("--target-column", type=str, help="Column name for the target data")
parser.add_argument("--task", choices=["regression", "segmentation", "classification"], help="Type of downstream task")
parser.add_argument("--metric", choices=["spearman", "accuracy"], help="Evaluation metric")
parser.add_argument("--loss", choices=["xe", "mse"], help="Loss function")
args = parser.parse_args()

if __name__ == "__main__":
    file_name = os.path.basename(args.data_path).split(".")[0]
    if args.embedded_split:
        df = pd.read_csv(args.data_path)
        df_train = df[df[args.split_column] == "train"]
        df_val = df[df[args.split_column] == "val"]
        df_test = df[df[args.split_column] == "test"]
        df_train.to_csv(f"{args.output_path}/{file_name}_train.csv", index=False)
        df_val.to_csv(f"{args.output_path}/{file_name}_val.csv", index=False)
        df_test.to_csv(f"{args.output_path}/{file_name}_test.csv", index=False)
        paths = {"data": {
            "path_train": f"./{file_name}_train.csv",
            "path_val": f"./{file_name}_val.csv",
            "path_test": f"./{file_name}_test.csv"
        }}
    else:
        from dask.dataframe import read_csv
        df = read_csv(args.data_path)
        paths = {}
        random_states = [int(''.join(sublist)) for sublist in args.random_states]
        split_ratio = [float(''.join(sublist)) for sublist in args.split_ratio]
        for random_state in random_states:
            a, b, c = df.random_split(split_ratio, random_state=random_state)
            a.to_csv(f"{args.output_path}/{file_name}_train{random_state}", index=False)
            b.to_csv(f"{args.output_path}/{file_name}_val{random_state}", index=False)
            c.to_csv(f"{args.output_path}/{file_name}_test{random_state}", index=False)
            path =  {
                "path_train": f"./{file_name}_train{random_state}/0.part",
                "path_val": f"./{file_name}_val{random_state}/0.part",
                "path_test": f"./{file_name}_test{random_state}/0.part"
            }
            paths[f"{file_name}_seed{random_state}"] = path

    config = {
        "path": paths,
        "data_column": args.data_column,
        "target_column": args.target_column,
        "task": args.task,
        "metric": args.metric,
        "loss": args.loss
    }
    json.dump(config, open(f"{args.output_path}/config.json", "w"))