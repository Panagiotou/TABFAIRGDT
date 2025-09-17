from sklearn.model_selection import RepeatedKFold
import os 
import numpy as np
import glob
import pandas as pd
from atom import ATOMClassifier
from tqdm import tqdm
import contextlib
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
import json

from sklearn.metrics import roc_auc_score




def get_generative_method(generative_method, generative_seed, dtype_map, generative_method_arguments,
                          train_df, synthetic_size, dataset_name=""):
    if "tabfairgdt" == generative_method:
        from tabfairgdt_src.tabfairgdt import TABFAIRGDT
        synthesizer = TABFAIRGDT(seed=generative_seed, dtype_map=dtype_map, **generative_method_arguments["declaration_args"])
    elif "tab_fair_gan" == generative_method:
        from competitors.TabFairGAN_main.TabFairGAN import TabFairGAN
        synthesizer = TabFairGAN(seed=generative_seed, dtype_map=dtype_map, **generative_method_arguments["declaration_args"])
    elif "fsmote" == generative_method:
        from competitors.FSMOTE.FSMOTE import FSMOTE
        synthesizer = FSMOTE(seed=generative_seed, dtype_map=dtype_map, **generative_method_arguments["declaration_args"])
    elif "tabular_argn" == generative_method:
        from competitors.TabularARGN.TabularARGN import TabularARGN
        synthesizer = TabularARGN(seed=generative_seed, dtype_map=dtype_map, **generative_method_arguments["declaration_args"])
    elif "prefair" == generative_method:
        from competitors.PreFair.PreFair import PreFair
        synthesizer = PreFair(seed=generative_seed, dtype_map=dtype_map, **generative_method_arguments["declaration_args"])
    elif "cuts" == generative_method:
        from competitors.CuTS.CuTS import CuTS
        generative_method_arguments["fit_args"]["dataset_name"] = dataset_name
        synthesizer = CuTS(seed=generative_seed, dtype_map=dtype_map, **generative_method_arguments["declaration_args"])
    elif "tabfairgdt_fair_splitting_criterion" == generative_method:
        print("Warning, running TABFAIRGDT using a fair splitting criterion, this is not recomended!")
        from tabfairgdt_src.tabfairgdt import TABFAIRGDT_FAIR_SPLITTING_CRITERION
        generative_method_arguments["fit_args"]["dtypes"] = {col: dtype for col, dtype in dtype_map.items() if col in train_df.columns} 
        synthesizer = TABFAIRGDT_FAIR_SPLITTING_CRITERION(seed=generative_seed, dtype_map=dtype_map, **generative_method_arguments["declaration_args"])
    else:
        raise NotImplementedError(f"Generative method '{generative_method}' is not implemented.")
    return synthesizer


def get_files(root_dir, methods, lamda_tradeoff_values, metrics, data_metrics, data_metrics_diff, downstream_model, no_lambda_methods):
    # Initialize a dictionary to store the file paths
    files_dict = {}

    # List of subfolders
    subfolders = ['avg', 'std']

    # Iterate through each subfolder
    for method in methods:
        method_name = method
        files_dict[method_name] = {}
        files_dict[method_name]["avg"] = {}
        files_dict[method_name]["std"] = {}

        for metric in metrics:
            files_dict[method_name]["avg"][metric] = []
            files_dict[method_name]["std"][metric] = []
        for metric in data_metrics:
            files_dict[method_name]["avg"][metric] = []
            files_dict[method_name]["std"][metric] = []
        for metric in data_metrics_diff:
            files_dict[method_name]["avg"][metric] = []
            files_dict[method_name]["std"][metric] = []
        if method=="real" or method in no_lambda_methods: # here are the methods that do not have a "lamda" hyperparam
            filename = f"{method}.json"
            avg_file = os.path.join(root_dir, method, "avg", filename)
            std_file = os.path.join(root_dir, method, "std", filename)

            avg = pd.read_json(avg_file).T
            std = pd.read_json(std_file).T

            for metric in metrics:
                avg_metrics = avg.loc[downstream_model, metric]
                std_metrics = std.loc[downstream_model, metric]

                files_dict[method_name]["avg"][metric].append(avg_metrics)

                files_dict[method_name]["std"][metric].append(std_metrics)
            
            if method in no_lambda_methods:
                for metric in data_metrics:
                    avg_metrics = avg.loc["Data Metrics", metric]
                    std_metrics = std.loc["Data Metrics", metric]

                    files_dict[method_name]["avg"][metric].append(avg_metrics)

                    files_dict[method_name]["std"][metric].append(std_metrics)
                for metric in data_metrics_diff:
                    avg_metrics = avg.loc["Data Metrics", metric]
                    std_metrics = std.loc["Data Metrics", metric]

                    files_dict[method_name]["avg"][metric].append(avg_metrics)

                    files_dict[method_name]["std"][metric].append(std_metrics)
        else:
            for lamda in lamda_tradeoff_values:
                if "asc" in method or "desc" in method:
                    tmp = method.split("_")[0]
                    filename = f"{tmp}_lamda_{100*lamda:.0f}.json"
                else:
                    filename = f"{method}_lamda_{100*lamda:.0f}.json"
                avg_file = os.path.join(root_dir, method, "avg", filename)
                std_file = os.path.join(root_dir, method, "std", filename)

                avg = pd.read_json(avg_file).T
                std = pd.read_json(std_file).T

                for metric in metrics:
                    avg_metrics = avg.loc[downstream_model, metric]
                    std_metrics = std.loc[downstream_model, metric]

                    files_dict[method_name]["avg"][metric].append(avg_metrics)

                    files_dict[method_name]["std"][metric].append(std_metrics)

                for metric in data_metrics:
                    avg_metrics = avg.loc["Data Metrics", metric]
                    std_metrics = std.loc["Data Metrics", metric]

                    files_dict[method_name]["avg"][metric].append(avg_metrics)

                    files_dict[method_name]["std"][metric].append(std_metrics)
                for metric in data_metrics_diff:
                    avg_metrics = avg.loc["Data Metrics", metric]
                    std_metrics = std.loc["Data Metrics", metric]

                    files_dict[method_name]["avg"][metric].append(avg_metrics)

                    files_dict[method_name]["std"][metric].append(std_metrics)
    return files_dict

def gather_results(results_path_avg, results_path_std, competitor_order):

    all_avgs = []
    all_std = []
    for competitor in competitor_order:
        avg_file = os.path.join(results_path_avg, f"{competitor}.json")
        std_file = os.path.join(results_path_std, f"{competitor}.json")

        avg_metrics = pd.read_json(avg_file).T.to_numpy()
        std_metrics = pd.read_json(std_file).T.to_numpy()

        all_avgs.append(avg_metrics)
        all_std.append(std_metrics)

    all_averages = np.stack(all_avgs, axis=1)
    all_stds = np.stack(all_std, axis=1)

    return all_averages, all_stds





def generate_train_test_splits(seed, dataset_loader, num_repeats, num_folds, dataset_path, train_path, test_path):

    target = dataset_loader.target

    if not os.path.exists(train_path):
        print("Generating train-test splits")
        os.makedirs(dataset_path)


        os.makedirs(train_path)
        os.makedirs(test_path)

        all_data = dataset_loader.original_dataframe.copy()
        rkf = RepeatedKFold(n_splits=num_folds, n_repeats=num_repeats, random_state=seed)
        splits = []
        for i, (train_index, test_index) in enumerate(rkf.split(all_data)):    
            data_train, data_test = all_data.loc[train_index], all_data.loc[test_index]

            splits.append((data_train, data_test))

            train_split_name = os.path.join(train_path, f"train_{i}.json")
            test_split_name = os.path.join(test_path, f"test_{i}.json")

            data_train[target] = data_train[target].astype("category")
            data_test[target] = data_test[target].astype("category")

            data_train.to_json(train_split_name, orient='records', lines=True)
            data_test.to_json(test_split_name, orient='records', lines=True)


    splits = []

    # Check if splits exists
    print(f"Found train-test split files in {train_path}")
    train_json_files = glob.glob(os.path.join(train_path, "*.json"))
    test_json_files = glob.glob(os.path.join(test_path, "*.json"))

    train_json_files.sort()
    test_json_files.sort()
    zipped = zip(train_json_files, test_json_files)
    for train, test in zipped:
        # Load the .npz file
        data_train = pd.read_json(train, lines=True)
        data_test =  pd.read_json(test, lines=True)
        splits.append((data_train, data_test))
        data_train = data_train.astype(dataset_loader.dtype_map)
        data_test = data_test.astype(dataset_loader.dtype_map)

    return splits





def process_split(index, data_train, synthetic_path, generative_method, dataset_loader, generative_method_arguments, num_synthetic, experiment_name):
    split_path = os.path.join(synthetic_path, f"split_{index}")
    os.makedirs(split_path, exist_ok=True)  # Ensure the directory exists

    train_columns = data_train.columns
    train_df = pd.DataFrame(data_train, columns=train_columns)
    synthetic_size = len(train_df)

    synthetic_dfs = []
    for j in range(num_synthetic):
        generative_seed = hash((index, j)) % (2**32 - 1)

        synthesizer = get_generative_method(
            generative_method,
            generative_seed,
            dataset_loader.dtype_map,
            generative_method_arguments,
            train_df,
            synthetic_size,
            dataset_name=dataset_loader.dataset_name
        )

        print(generative_method_arguments["fit_args"])

        synthesizer.fit(train_df, **generative_method_arguments["fit_args"])

        synthetic_data = synthesizer.generate(
            int(synthetic_size), **generative_method_arguments["generate_args"]
        )

        if not (generative_method == "fsmote" or generative_method=="prefair"): #fsmote is augmentation, so size of synthetic will be different!
            assert len(synthetic_data) == synthetic_size, "Different generation size than expected"

        synthetic_json_file = os.path.join(split_path, f"{experiment_name}_{j}.json")

        synthetic_data.to_json(synthetic_json_file, orient='records', lines=True)

        synthetic_dfs.append(synthetic_data)

    return synthetic_dfs

def sequential_process_splits(splits, synthetic_path, generative_method, dataset_loader, generative_method_arguments, num_synthetic, experiment_name):
    len_splits = len(splits)
    synthetic_per_split = []

    # Process each split sequentially
    for i, (data_train, _) in enumerate(splits):
        # Process the current split
        synthetic_dfs = process_split(
            i,
            data_train,
            synthetic_path,
            generative_method,
            dataset_loader,
            generative_method_arguments,
            num_synthetic,
            experiment_name
        )
        synthetic_per_split.append((i, synthetic_dfs))

    # Sort results by split index to maintain order
    synthetic_per_split.sort(key=lambda x: x[0])
    return [x[1] for x in synthetic_per_split]

# Parallel processing for all splits
def parallel_process_splits(splits, synthetic_path, generative_method, dataset_loader, generative_method_arguments, num_synthetic, experiment_name):
    len_splits = len(splits)
    synthetic_per_split = []

    # Use ThreadPoolExecutor for parallelism

    with ThreadPoolExecutor() as executor:
        # Submit all splits for parallel processing
        future_to_index = {
            executor.submit(
                process_split,
                i,
                data_train,
                synthetic_path,
                generative_method,
                dataset_loader,
                generative_method_arguments,
                num_synthetic,
                experiment_name
            ): i
            for i, (data_train, _) in enumerate(splits)
        }

        for future in as_completed(future_to_index):
            i = future_to_index[future]
            try:
                synthetic_dfs = future.result()
                synthetic_per_split.append((i, synthetic_dfs))
            except Exception as e:
                print(f"Error processing split {i}: {e}")

    # Sort results by split index to maintain order
    synthetic_per_split.sort(key=lambda x: x[0])
    return [x[1] for x in synthetic_per_split]

def generate_synthetic_datasets(generative_method, generative_method_arguments, num_synthetic, dataset_loader, splits,
                                synthetic_path, results_path_avg, protected_attribute="sex", parallel=True):

    if "lamda" in generative_method_arguments["fit_args"]:
        lamda = generative_method_arguments["fit_args"]["lamda"]
        experiment_name = f"{generative_method}_lamda_{100*lamda:.0f}"
        print(lamda, experiment_name)
    else:
        experiment_name = f"{generative_method}"

    if os.path.isfile(os.path.join(results_path_avg, f"{experiment_name}.json")):
        print(f"\t\tExperiment {experiment_name} already evaluated")
        return

    len_splits = len(splits)

    for i in range(len_splits):
        split_path = os.path.join(synthetic_path, f"split_{i}")
        if not os.path.exists(split_path):
            os.makedirs(split_path)

    data_exist = os.path.isfile(os.path.join(synthetic_path, "split_0", f"{experiment_name}_0.json"))

    synthetic_per_split = []

    if data_exist:
        print(f"Found synthetic data for method: {generative_method}")
        for i in range(len_splits):
            split_path = os.path.join(synthetic_path, f"split_{i}")
            synthetic_json_files = glob.glob(os.path.join(split_path, f"{generative_method}*.json"))
            synthetic_json_files.sort()

            synthetic_dfs = []

            for synthetic_json_file in synthetic_json_files:
                synthetic_dataset = pd.read_json(synthetic_json_file, lines=True)
                synthetic_dfs.append(synthetic_dataset)

            synthetic_per_split.append(synthetic_dfs)
    else:
        if parallel:
            # Call the function
            synthetic_per_split = parallel_process_splits(
                splits=splits,
                synthetic_path=synthetic_path,
                generative_method=generative_method,
                dataset_loader=dataset_loader,
                generative_method_arguments=generative_method_arguments,
                num_synthetic=num_synthetic,
                experiment_name=experiment_name
            )
        else:
            synthetic_per_split = sequential_process_splits(
                splits=splits,
                synthetic_path=synthetic_path,
                generative_method=generative_method,
                dataset_loader=dataset_loader,
                generative_method_arguments=generative_method_arguments,
                num_synthetic=num_synthetic,
                experiment_name=experiment_name
            )
            

    return synthetic_per_split

def train_eval(models, X_train, y_train, test_data, metrics, fairness_metrics, device="cpu", engine="sklearnex", force_suppress_output=True, group_train_synthetic=None):

    X_test, y_test, group_test = test_data

    X_train = X_train.copy()
    X_test = X_test.copy()
    X_train.columns = [f"{col}_{i}" for i, col in enumerate(X_train.columns)]
    X_test.columns = [f"{col}_{i}" for i, col in enumerate(X_test.columns)]

    atom_x_y = ATOMClassifier(X_train, y_train, verbose=0)

    atom_x_y.encode()
    
    if group_train_synthetic is not None:
        
        group_mapping = {value: idx for idx, value in enumerate(group_train_synthetic.unique())}

        group_train_synthetic = np.array(group_train_synthetic.map(group_mapping)).astype(int)

        atom_x_s = ATOMClassifier(X_train, group_train_synthetic, verbose=0)

        atom_x_s.encode()
        if force_suppress_output:
            with suppress_stdout():
                atom_x_s.run(models=models, metric=lambda y_true, y_pred: 0, engine=engine, parallel=True, device=device)
        else:
            atom_x_s.run(models=models, metric=lambda y_true, y_pred: 0, engine=engine, parallel=True, device=device)


    if force_suppress_output:
        with suppress_stdout():
            atom_x_y.run(models=models, metric=lambda y_true, y_pred: 0, engine=engine, parallel=True, device=device)
    else:
        atom_x_y.run(models=models, metric=lambda y_true, y_pred: 0, engine=engine, parallel=True, device=device)




    metrics_for_all_clf = []
    e_fairness_metrics_res = []
    
    for model in models:
        y_pred = atom_x_y[model].predict(X_test)
        y_pred_proba = np.array(atom_x_y[model].predict_proba(X_test))[:, 1]

        metrics_res = compute_metrics(y_test, y_pred, y_pred_proba, metrics)

        fairness_metrics_res = compute_fairness_metrics(y_test, y_pred, group_test, fairness_metrics)

        if group_train_synthetic is not None:
            s_pred = np.array(atom_x_s[model].predict(X_test)).astype(int)
            s_true = np.array(group_test.map(group_mapping)).astype(int)

            e_fairness_metrics_res = compute_e_fairness_BER(s_true, s_pred)


        metrics_for_all_clf.append(metrics_res + fairness_metrics_res + e_fairness_metrics_res)

    return metrics_for_all_clf

def synthetic_results(generative_method, generative_method_arguments, models, dataset_loader, synthetic_per_split, test_set_per_split,
                      metrics, fairness_metrics, data_metrics,
                      results_path_avg, results_path_std, metric_names, data_metric_names, keep_protected_input=False, protected_attribute="sex",
                      device="cpu", engine="sklearnex"):


    if "lamda" in generative_method_arguments["fit_args"]:
        lamda = generative_method_arguments["fit_args"]["lamda"]
        experiment_name = f"{generative_method}_lamda_{100*lamda:.0f}.json"
    else:
        experiment_name = f"{generative_method}.json"

    if os.path.isfile(os.path.join(results_path_avg, experiment_name)):
        print(f"\t\tExperiment {experiment_name} already evaluated")
        return
    
    print(f"Evaluating for synthetic data ({generative_method})")
    target = dataset_loader.target

    len_splits = len(synthetic_per_split)
    len_models = len(models)


    metrics_for_all_splits = []
    data_metrics_for_all_splits = []

    for i, synthetic_data in enumerate(tqdm(synthetic_per_split, total=len_splits, desc="Processing splits")):

        data_test_real = test_set_per_split[i]
        data_test_real = data_test_real.astype(dataset_loader.dtype_map)

        X_test_original = data_test_real.copy().drop(target, axis=1)


        y_test = data_test_real[target].astype(int)
        X_test = data_test_real.drop(target, axis=1)

        group_test = X_test[protected_attribute].copy()

        if not keep_protected_input:
            X_test.drop(protected_attribute, axis=1, inplace=True)


        if not keep_protected_input:
            assert protected_attribute not in X_test.columns, (
                f"Error: '{protected_attribute}' is present in the DataFrame columns but keep_protected_input is False."
            )

        test_data = (X_test, y_test, group_test)

        synthetic_results_for_split = []
        data_results_for_split = []
        
        for data_train_synthetic in tqdm(synthetic_data, desc="\tProcessing synthetic datasets", leave=False):
            data_train_synthetic = data_train_synthetic.astype(dataset_loader.dtype_map)
            X_train_synthetic_original = data_train_synthetic.copy().drop(target, axis=1)

            y_train_synthetic = data_train_synthetic[target].astype(int)
            X_train_synthetic = data_train_synthetic.drop(target, axis=1)

            group_train_synthetic = None # no BER metric
            # group_train_synthetic = X_train_synthetic[protected_attribute].copy()


            if not keep_protected_input:
                X_train_synthetic.drop(protected_attribute, axis=1, inplace=True)
            
            metrics_for_all_clf = train_eval(models, X_train_synthetic, y_train_synthetic, test_data, metrics, fairness_metrics, device=device, engine=engine, group_train_synthetic=group_train_synthetic)

            synthetic_results_for_split.append(metrics_for_all_clf)

            data_results_for_split.append(compute_data_metrics(real=X_test_original, fake=X_train_synthetic_original, y_fake=y_train_synthetic, data_metrics=data_metrics, protected_attribute=protected_attribute))


        average_synthetic_results_for_split = np.mean(synthetic_results_for_split, axis=0)

        average_data_results_for_split = np.mean(data_results_for_split, axis=0)

        # This is equivalent to a result from one split for the real data for example (just averaged for N synthetic datasets)

        metrics_for_all_splits.append(average_synthetic_results_for_split)
        data_metrics_for_all_splits.append(average_data_results_for_split)

    average = np.mean(metrics_for_all_splits, axis=0)
    std = np.std(metrics_for_all_splits, axis=0)

    average_data_metrics = np.mean(data_metrics_for_all_splits, axis=0)
    std_data_metrics = np.std(data_metrics_for_all_splits, axis=0)

    # Create the DataFrames
    avg_df = pd.DataFrame(average, index=models, columns=metric_names)
    std_df = pd.DataFrame(std, index=models, columns=metric_names)

    avg_dict = avg_df.to_dict(orient="index")
    std_dict = std_df.to_dict(orient="index")

    # print(average_data_metrics)
    # print(data_metric_names)
    # exit(1)

    avg_dict["Data Metrics"] = dict(zip(data_metric_names, average_data_metrics))
    std_dict["Data Metrics"] = dict(zip(data_metric_names, std_data_metrics))

    with open(os.path.join(results_path_avg, experiment_name), 'w') as file:
        json.dump(avg_dict, file, indent=4) 

    with open(os.path.join(results_path_std, experiment_name), 'w') as file:
        json.dump(std_dict, file, indent=4) 

    # avg_df.to_json(os.path.join(results_path_avg, experiment_name), orient='index', indent=4)
    # std_df.to_json(os.path.join(results_path_std, experiment_name), orient='index', indent=4)


def real_results(models, dataset_loader, splits, metrics, fairness_metrics, results_path_avg,
                      results_path_std, metric_names, keep_protected_input=False, protected_attribute=["sex"],
                      device="cpu", engine="sklearnex"):
    
    experiment_name = f"real.json"


    if os.path.isfile(os.path.join(results_path_avg, experiment_name)):
        print(f"\t\tExperiment for real data is already evaluated")
        return



    print("Evaluating for real data")
    target = dataset_loader.target
    len_splits = len(splits)
    len_models = len(models)


    metrics_for_all_splits = []

    for i, (data_train, data_test) in enumerate(tqdm(splits, total=len_splits, desc="Processing splits")):

        y_train = data_train[target].astype(int)
        X_train = data_train.drop(target, axis=1)

        y_test = data_test[target].astype(int)
        X_test = data_test.drop(target, axis=1)

        group_train = X_train[protected_attribute].copy()
        group_test = X_test[protected_attribute].copy()

        if not keep_protected_input:
            X_train.drop(protected_attribute, axis=1, inplace=True)
            X_test.drop(protected_attribute, axis=1, inplace=True)

        test_data = (X_test, y_test, group_test)


        group_train = None # no BER

        metrics_for_all_clf = train_eval(models, X_train, y_train, test_data, metrics, fairness_metrics, device=device, engine=engine, group_train_synthetic=group_train)

        metrics_for_all_splits.append(metrics_for_all_clf)

    metrics_for_all_splits = np.array(metrics_for_all_splits)

    average = np.mean(metrics_for_all_splits, axis=0)
    std = np.std(metrics_for_all_splits, axis=0)

    # Create the DataFrames
    avg_df = pd.DataFrame(average, index=models, columns=metric_names)
    std_df = pd.DataFrame(std, index=models, columns=metric_names)

    avg_df.to_json(os.path.join(results_path_avg, experiment_name), orient='index', indent=4)
    std_df.to_json(os.path.join(results_path_std, experiment_name), orient='index', indent=4)

def compute_metrics(y_test, y_pred, y_pred_proba, metrics):
    res_metrics = []
    for metric in metrics:

        if metric == roc_auc_score:
            metric_value = metric(y_test, y_pred_proba)
        else:
            metric_value = metric(y_test, y_pred)
        # try:
        #     metric_value = metric(y_test, y_pred)
        # except:
        #     metric_value = metric(y_test, y_pred_proba)
        res_metrics.append(metric_value)
    
    return res_metrics

def compute_fairness_metrics(y_test, y_pred, group, fairness_metrics):
    return [m(y_test, y_pred, group) for m in fairness_metrics]    

def compute_continuous_fairness_metrics(y_pred_proba, group, fairness_metrics):
    return [m(y_pred_proba, group) for m in fairness_metrics]  

def get_data_for_eval(real: pd.DataFrame,
                      fake: pd.DataFrame,
                      protected_attribute: str):
    """
    Split the data according to the protected attribute.
    Example: if PA is sex and the values are male and female, returns 4 dataframes: male_real, female_real, male_fake, female_fake.
    """
    uniques = real[protected_attribute].unique().tolist()
    if len(uniques) != 2:
        raise ValueError("Protected attribute must have 2 unique values.")
    
    df_real_0 = real[real[protected_attribute] == uniques[0]]
    df_real_1 = real[real[protected_attribute] == uniques[1]]
    df_fake_0 = fake[fake[protected_attribute] == uniques[0]]
    df_fake_1 = fake[fake[protected_attribute] == uniques[1]]

    return df_real_0, df_real_1, df_fake_0, df_fake_1

def compute_data_metrics(real, fake, y_fake, data_metrics, protected_attribute:str):
    ret = []
    # ret_diffs = []

    # Get the minimum length for pairing
    min_size = min(len(real), len(fake))


    # Apply the same logic to the other DataFrame pairs

    real_sampled = real.sample(n=min_size) if len(real) > min_size else real
    fake_sampled = fake.sample(n=min_size) if len(fake) > min_size else fake


    # df_real_s_0, df_real_s_1, df_fake_s_0, df_fake_s_1 = get_data_for_eval(real, fake, protected_attribute=protected_attribute)

    # min_size_0 = min(len(df_real_s_0), len(df_fake_s_0))
    # df_real_s_0_sampled = df_real_s_0.sample(n=min_size_0) if len(df_real_s_0) > min_size_0 else df_real_s_0
    # df_fake_s_0_sampled = df_fake_s_0.sample(n=min_size_0) if len(df_fake_s_0) > min_size_0 else df_fake_s_0

    # min_size_1 = min(len(df_real_s_1), len(df_fake_s_1))
    # df_real_s_1_sampled = df_real_s_1.sample(n=min_size_1) if len(df_real_s_1) > min_size_1 else df_real_s_1
    # df_fake_s_1_sampled = df_fake_s_1.sample(n=min_size_1) if len(df_fake_s_1) > min_size_1 else df_fake_s_1


    for m in data_metrics: 
        ret += m(real=real_sampled, fake=fake_sampled)
        # metric_s_0 = m(df_real_s_0_sampled, df_fake_s_0_sampled)
        # metric_s_1 = m(df_real_s_1_sampled, df_fake_s_1_sampled)
        # abs_diffs = [abs(x-y) for x,y in zip(metric_s_0, metric_s_1)] 
        # ret_diffs += abs_diffs
    return ret #+ ret_diffs
  




def compute_e_fairness_BER(s_true, s_pred):
    # Calculate False Negative Rate (FNR): P(f(X) = 0 | S = 1)
    false_negative = np.sum((s_pred == 0) & (s_true == 1))
    total_positive = np.sum(s_true == 1)
    fnr = false_negative / total_positive if total_positive > 0 else 0

    # Calculate False Positive Rate (FPR): P(f(X) = 1 | S = 0)
    false_positive = np.sum((s_pred == 1) & (s_true == 0))
    total_negative = np.sum(s_true == 0)
    fpr = false_positive / total_negative if total_negative > 0 else 0

    # Compute Balanced Error Rate (BER)
    return [(fnr + fpr) / 2]



@contextlib.contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout

