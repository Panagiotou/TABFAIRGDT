import os
import numpy as np
import argparse
import pandas as pd
import json
import time


from evaluation_framework.utils_eval import get_generative_method


def load_existing_results(results_file):
    """ Load existing results if the file exists, otherwise return an empty dictionary. """
    if os.path.exists(results_file):
        with open(results_file, "r") as f:
            return json.load(f)
    return {}

def save_results(results, results_file):
    """ Save updated results to the JSON file. """
    with open(results_file, "w") as f:
        json.dump(results, f, indent=4)

def generate_random_dataframe(num_features, num_rows=10000, min_unique=2, max_unique=15):
    """ Generate random DataFrame with numerical and categorical features, including 'sex' and 'target'. """
    dataframes, dtype_maps = [], []
    
    for n in num_features:
        num_numerical = (n - 2) // 2
        num_categorical = n - num_numerical - 2  
        data, dtype_map = {}, {}

        # Numerical features
        for i in range(num_numerical):
            col_name = f'num_{i}'
            data[col_name] = np.random.randint(0, 100, size=num_rows)
            dtype_map[col_name] = 'int'

        # Categorical features
        for i in range(num_categorical):
            col_name = f'cat_{i}'
            num_categories = np.random.randint(min_unique, max_unique + 1)
            categories = [chr(65 + j) for j in range(num_categories)]
            data[col_name] = pd.Categorical(np.random.choice(categories, size=num_rows))
            dtype_map[col_name] = 'category'

        # Protected and target attributes
        data['sex'] = pd.Categorical(np.random.choice(['Male', 'Female'], size=num_rows, p=[0.8, 0.2]))  
        data['target'] = pd.Categorical(np.random.choice([0, 1], size=num_rows, p=[0.85, 0.15]))  

        dtype_map['sex'] = 'category'
        dtype_map['target'] = 'category'

        df = pd.DataFrame(data)
        dataframes.append(df)
        dtype_maps.append(dtype_map)
    
    return dataframes, dtype_maps

def main(args):
    """ Runs experiments only for missing cases and updates the JSON results file incrementally. """
    results_file = os.path.join(args.results_folder, "generation_timings.json")
    existing_results = load_existing_results(results_file)

    df_sizes = [10, 100, 500]  # Feature sizes
    sample_sizes = [1000, 10000, 50000]  # Number of rows per dataset

    no_lambda_methods = ["fsmote", "prefair", "cuts"]
    num_repeats = args.num_repeats
    verbose = args.verbose

    for num_rows in sample_sizes:
        dfs, dtype_maps = generate_random_dataframe(df_sizes, num_rows=num_rows, min_unique=2, max_unique=15)

        for generative_method in args.methods:
            
            if generative_method == "real":
                continue

            if generative_method not in existing_results:
                existing_results[generative_method] = {}

            for df_size, train_df, dtype_map in zip(df_sizes, dfs, dtype_maps):
                if str(df_size) in existing_results[generative_method] and str(num_rows) in existing_results[generative_method][str(df_size)]:
                    print(f"Experiment, {generative_method}, {df_size}, {num_rows} already evaluated") 
                    continue  # Skip if experiment already exists

                print(f"Running experiment for {generative_method} on dataset size {df_size} with {num_rows} samples...")

                fit_times, generate_times = [], []
                generative_method_arguments = {
                    "declaration_args": {"verbose": verbose, "protected_attribute": "sex", "target": "target"},
                    "fit_args": {},
                    "generate_args": {},
                }

                if generative_method == "tab_fair_gan":
                    generative_method_arguments["declaration_args"].update({
                        "target_class_desired": 1,
                        "protected_attribute_under_represented": "Female"
                    })
                if generative_method == "prefair":
                    generative_method_arguments["declaration_args"]["target_class_desired"] = 1

                for _ in range(num_repeats):
                    generative_seed = np.random.randint(0, 10000)
                    if generative_method not in no_lambda_methods:
                        generative_method_arguments["fit_args"]["lamda"] = 1.0

                    synthesizer = get_generative_method(
                        generative_method, generative_seed, dtype_map, generative_method_arguments, train_df, len(train_df)
                    )

                    start_fit = time.time()
                    synthesizer.fit(train_df, **generative_method_arguments["fit_args"])
                    end_fit = time.time()

                    start_generate = time.time()
                    synthesizer.generate(len(train_df), **generative_method_arguments["generate_args"])
                    end_generate = time.time()

                    fit_times.append(end_fit - start_fit)
                    generate_times.append(end_generate - start_generate)

                # Store results for this method, dataset size, and sample size
                if str(df_size) not in existing_results[generative_method]:
                    existing_results[generative_method][str(df_size)] = {}

                existing_results[generative_method][str(df_size)][str(num_rows)] = {
                    "fit_time_avg": np.mean(fit_times),
                    "fit_time_std": np.std(fit_times),
                    "generate_time_avg": np.mean(generate_times),
                    "generate_time_std": np.std(generate_times),
                    "total_time_avg": np.mean(np.array(fit_times) + np.array(generate_times)),
                    "total_time_std": np.std(np.array(fit_times) + np.array(generate_times)),
                }

                # Save intermediate results immediately
                save_results(existing_results, results_file)
                print(f"Saved results for {generative_method} on dataset size {df_size} with {num_rows} samples.")

    print("All missing experiments completed and results updated.")

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Generation')

    args = parser.parse_args()