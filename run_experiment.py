from tabular_datasets.dataset import Dataset
import os
import numpy as np
import argparse


import warnings

from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score, precision_score, recall_score

from evaluation_framework.fairness_metrics import eq_odd, stat_par, eq_opp, dpr
from evaluation_framework.fairness_metrics import hgr
from evaluation_framework.eval_dcr import DCR


from evaluation_framework.eval_synth_data import tv_score, ks_score, prdc_score, detection_score

from evaluation_framework.utils_eval import generate_train_test_splits, generate_synthetic_datasets, synthetic_results, real_results

# Suppress LightGBM categorical_feature warning
warnings.filterwarnings("ignore", category=UserWarning, message="categorical_feature keyword has been found*")
warnings.filterwarnings("ignore", category=UserWarning, message="categorical_feature in param dict is overridden*")

# Ignore FutureWarnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def generate_folders(dataset_name, generative_method, results_folder, prefix=""):

    if prefix:
        synthetic_splits_folder = f"{results_folder}/{dataset_name}_{prefix}"
    else:
        synthetic_splits_folder = f"{results_folder}/{dataset_name}"

    synthetic_path = os.path.join(synthetic_splits_folder, generative_method, "synthetic_train_splits")
    results_path = os.path.join(synthetic_splits_folder, generative_method)
    results_path_avg = os.path.join(results_path, "avg")
    results_path_std = os.path.join(results_path, "std")

    if not os.path.exists(results_path): 
        os.makedirs(results_path)
    if not os.path.exists(results_path_avg):
        os.makedirs(results_path_avg)
        os.makedirs(results_path_std)

    if not (os.path.exists(synthetic_path) and os.path.isdir(synthetic_path)):
        os.makedirs(synthetic_path)

    return synthetic_path, results_path_avg, results_path_std

def main(args):

    dataset_loader = Dataset(args.dataset)
    dataset_name = dataset_loader.dataset_name

    target = dataset_loader.target
    target_class_desired = dataset_loader.favorable_target
    protected_attribute_under_represented = dataset_loader.protected_attribute_under_represented

    seed = args.seed
    device = args.device
    engine = args.engine
    generative_methods = args.methods   


    lamda_tradeoff_values = args.lamda_tradeoff_values
    acc_threshold = args.acc_threshold





    num_repeats = args.num_repeats
    num_folds = args.num_folds
    num_synthetic = args.num_synthetic
    protected_attribute = args.protected_attribute
    verbose = args.verbose
    downstream_models = args.downstream_models

    if protected_attribute != "sex":

        dataset_name += f"_{protected_attribute}"



    # Create the synthetic methods dictionary with different alpha/lamda parameters


    generative_methods_arguments = [{
        "declaration_args": {"verbose": verbose, "protected_attribute":protected_attribute,
                            "target": target},
        "fit_args": {},
        "generate_args": {},
    } for i in range(len(generative_methods))]



    dataset_path = os.path.join("tabular_datasets", dataset_name, "train_test_splits")
    train_path = os.path.join(dataset_path, "train")
    test_path = os.path.join(dataset_path, "test")




    no_lambda_methods = ["fsmote", "prefair", "cuts"]



    splits = generate_train_test_splits(seed, dataset_loader, num_repeats, num_folds, dataset_path, train_path, test_path)


    metrics = [accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score, precision_score, recall_score]
    fairness_metrics = [eq_odd, stat_par, eq_opp, dpr]
    data_metrics = [DCR, tv_score, ks_score, prdc_score, detection_score]

    # data_metrics = [tv_score, ks_score, detection_score]

    metric_names = ["accuracy_score", "balanced_accuracy_score", "f1_score", "roc_auc_score", "precission", "recall", "eq_odd", "stat_par", "eq_opp", "dpr"]
    data_metric_names = ["DCR", "TV-score (categorical)", "KS-score (continuous)", "PRDC precision", "PRDC recall", "PRDC density", "PRDC coverage", "detection_score"] 
    # data_metric_names = ["TV-score (categorical)", "KS-score (continuous)", "detection_score"] 

    # data_metric_names += [name + "_diff" for name in data_metric_names]

    for generative_method, generative_method_arguments in zip(generative_methods, generative_methods_arguments):


        if generative_method == "real":
            real_path, results_path_avg, results_path_std = generate_folders(dataset_name, generative_method, args.results_folder, args.prefix)
        
                              
            real_results(downstream_models, dataset_loader, splits, metrics, fairness_metrics, results_path_avg,
                      results_path_std, metric_names, device=device, protected_attribute=protected_attribute) 
               
        else:

            synthetic_path, results_path_avg, results_path_std = generate_folders(dataset_name, generative_method, args.results_folder, args.prefix)

            if generative_method == "tab_fair_gan":
                generative_method_arguments["declaration_args"]["target_class_desired"] = target_class_desired
                generative_method_arguments["declaration_args"]["protected_attribute_under_represented"] = protected_attribute_under_represented

            if args.reorder and generative_method == "tabfairgdt":
                generative_method_arguments["declaration_args"]["re_order"] = args.reorder
                
            if generative_method == "prefair":
                generative_method_arguments["declaration_args"]["target_class_desired"] = target_class_desired
            
            if "acc_threshold" in generative_method:
                generative_method_arguments["declaration_args"]["acc_threshold"] = acc_threshold



            if generative_method in no_lambda_methods: # no lamda hyperparam that controls fairness in any way 
                lamda_tradeoff_values = [-1]
            else:
                lamda_tradeoff_values = args.lamda_tradeoff_values


            for lamda_tradeoff in lamda_tradeoff_values:

                lamda_tradeoff = float(lamda_tradeoff)

                if lamda_tradeoff < 0:
                    print("No Lamda hyperparameter is used in this method, running only 1 experiment")            
                else:
                    print("Current Lamda", lamda_tradeoff)
                    generative_method_arguments["fit_args"]["lamda"] = lamda_tradeoff

                # This generates num_synthetic synthetic copies of each train split in train_test_splits and saves them.

                synthetic_per_split = generate_synthetic_datasets(generative_method, generative_method_arguments, num_synthetic,
                                                                    dataset_loader, splits, synthetic_path, results_path_avg, 
                                                                    protected_attribute=protected_attribute, parallel=args.parallel)
                
                if synthetic_per_split is None: # experiment already evaluated
                    continue

                test_set_per_split = [split[-1] for split in splits]

                synthetic_results(generative_method, generative_method_arguments, downstream_models, dataset_loader, synthetic_per_split, 
                                test_set_per_split, metrics, fairness_metrics, data_metrics, results_path_avg, results_path_std, metric_names, data_metric_names, 
                                device=device, engine=engine, protected_attribute=protected_attribute) 

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Generation')

    args = parser.parse_args()