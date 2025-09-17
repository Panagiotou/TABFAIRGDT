from tabular_datasets.dataset import Dataset
import os
import numpy as np
import argparse









def main(args):

    dataset_loader = Dataset(args.dataset)
    dataset_name = dataset_loader.dataset_name
    protected_attribute = args.protected_attribute
    target = dataset_loader.target


    df = dataset_loader.original_dataframe

    print(f"size, {df.shape}")
    # Unique value percentages for target
    target_counts = df[target].value_counts(normalize=True) * 100
    print(f"Target column '{target}' unique values and percentages:\n{target_counts}\n")

    # Unique value percentages for protected attribute
    protected_counts = df[protected_attribute].value_counts(normalize=True) * 100
    print(f"Protected attribute '{protected_attribute}' unique values and percentages:\n{protected_counts}\n")

    # Unique value percentages for the intersection of target and protected attribute
    intersection_counts = df.groupby([protected_attribute, target]).size() / len(df) * 100
    print(f"Intersection of '{protected_attribute}' and '{target}' unique values and percentages:\n{intersection_counts}\n")


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Generation')

    args = parser.parse_args()