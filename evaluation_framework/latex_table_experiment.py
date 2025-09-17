import os
import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from matplotlib.gridspec import GridSpec
from tabular_datasets.dataset import Dataset
import matplotlib.ticker as ticker


from evaluation_framework.utils_eval import get_files
import argparse


def generate_split_latex_table(metrics, names, method_names, dataset_names, dataset_names_latex, 
                           generative_methods, args, lamda_tradeoff_values, downstream_model, 
                           no_lambda_methods, datasets_per_row=2):
    """
    Generate a single LaTeX table with datasets arranged in rows of 'datasets_per_row'.
    The methods are repeated for each row of datasets.
    
    Parameters:
    -----------
    metrics : list
        List of metric names
    names : dict
        Dictionary mapping metric names to their LaTeX display names
    method_names : dict
        Dictionary mapping method keys to their LaTeX display names
    dataset_names : list
        List of dataset names
    dataset_names_latex : list
        List of dataset LaTeX display names
    generative_methods : list
        List of method names to include in the table
    args : object
        Arguments containing results_folder and other parameters
    lamda_tradeoff_values : list
        Lambda values for trade-off
    downstream_model : str
        Name of the downstream model
    no_lambda_methods : list
        Methods that don't use lambda
    datasets_per_row : int
        Number of datasets to include in each row
        
    Returns:
    --------
    str
        LaTeX table as a string
    """
    import os
    import numpy as np
    
    # Calculate number of columns and rows of datasets
    num_metrics = len(metrics)
    num_datasets = len(dataset_names)
    num_dataset_rows = (num_datasets + datasets_per_row - 1) // datasets_per_row  # Ceiling division
    
    # Collect data for all datasets
    all_data = {}
    for method in generative_methods:
        all_data[method] = []
    
    for dataset_idx, dataset_name in enumerate(dataset_names):
        results_path = os.path.join(args.results_folder, dataset_name)
        
        # Initialize empty lists for each dataset
        data_metrics = []
        data_metrics_diff = []
        
        # Get data for this dataset
        files_dict = get_files(results_path, generative_methods, lamda_tradeoff_values, metrics, 
                              data_metrics, data_metrics_diff, downstream_model, no_lambda_methods)
        
        # Store the data for each method and metric
        for method in generative_methods:
            method_data = []
            for metric in metrics:
                if method in files_dict and metric in files_dict[method]["avg"]:
                    avg = files_dict[method]["avg"][metric]
                    std = files_dict[method]["std"][metric]
                    
                    # Convert to scalar if it's a list with a single element
                    avg_val = avg[0] if isinstance(avg, list) and len(avg) == 1 else avg
                    std_val = std[0] if isinstance(std, list) and len(std) == 1 else std
                    
                    method_data.append((avg_val, std_val))
                else:
                    method_data.append((float('nan'), float('nan')))
            
            all_data[method].append(method_data)
    
    # Start table
    result = "\\begin{table}[ht]\n\\centering\n\\caption{Results across datasets}\n"
    
    # Generate table for each row of datasets
    for row_idx in range(num_dataset_rows):
        start_idx = row_idx * datasets_per_row
        end_idx = min(start_idx + datasets_per_row, num_datasets)
        current_datasets = dataset_names[start_idx:end_idx]
        current_dataset_names_latex = dataset_names_latex[start_idx:end_idx]
        
        # Create tabular environment with appropriate columns for this row
        # One column for method names + (num metrics columns * num datasets in this row)
        result += f"\\begin{{tabular}}{{l|{num_metrics * len(current_datasets) * 'c'}}}\n\\hline\n"
        
        # Create the multicolumn headers for datasets
        result += "\\textbf{Method}"
        for dataset_name in current_dataset_names_latex:
            result += f" & \\multicolumn{{{num_metrics}}}{{c}}{{{dataset_name}}}"
        result += " \\\\\n"
        
        # Create the subheaders for metrics
        result += " "  # Space for method column
        for _ in current_datasets:
            for metric in metrics:
                metric_name = names[metric]
                result += f" & {metric_name}"
        result += " \\\\\n\\hline\n"
        
        # Add rows for each method
        for method in generative_methods:
            method_name = method_names[method]
            row = [method_name]
            
            for dataset_idx in range(start_idx, end_idx):
                for metric_idx, _ in enumerate(metrics):
                    # Get the avg and std values
                    avg, std = all_data[method][dataset_idx][metric_idx]
                    
                    # Format the cell as avg (no std), handling NaN values
                    if np.isnan(avg) or np.isnan(std):
                        cell = "N/A"
                    else:
                        cell = f"{avg:.3f} {{\\tiny $\\pm$ {std:.3f}}}"
                    
                    row.append(cell)
            
            result += f"{' & '.join(row)} \\\\\n"
        
        # End this tabular
        result += "\\hline\n\\end{tabular}"
        
        # Add space between dataset rows
        if row_idx < num_dataset_rows - 1:
            result += "\n\n\\vspace{1cm}\n\n"
    
    # End table
    result += "\n\\end{table}"
    return result

# Example usage:
# latex_table = generate_split_latex_table(metrics, names, method_names, dataset_names, dataset_names_latex, 
#                                        generative_methods, args, lamda_tradeoff_values, downstream_model, 
#                                        no_lambda_methods, datasets_per_row=2)
# print(latex_table)

# Example usage:
# latex_tables = generate_latex_table_string(metrics, names, method_names, dataset_names, 
#                                          dataset_names_latex, files_dict, generative_methods)
# print(latex_tables)
# 
def generate_percentile_diff_table(metrics, names, method_names, dataset_names, 
                              generative_methods, args, lamda_tradeoff_values, 
                              downstream_model, no_lambda_methods, real_method):
    """
    Generate a LaTeX table showing the percentile difference of each method compared to a 
    reference 'real' method, averaged across all datasets.
    
    Parameters:
    -----------
    metrics : list
        List of metric names
    names : dict
        Dictionary mapping metric names to their LaTeX display names
    method_names : dict
        Dictionary mapping method keys to their LaTeX display names
    dataset_names : list
        List of dataset names
    generative_methods : list
        List of method names to include in the table
    args : object
        Arguments containing results_folder and other parameters
    lamda_tradeoff_values : list
        Lambda values for trade-off
    downstream_model : str
        Name of the downstream model
    no_lambda_methods : list
        Methods that don't use lambda
    real_method : str
        The name of the reference method to compare against
        
    Returns:
    --------
    str
        LaTeX table as a string
    """
    import os
    import numpy as np
    
    # Check if real_method is in generative_methods
    if real_method not in generative_methods:
        raise ValueError(f"Reference method '{real_method}' not found in generative_methods")
    
    num_metrics = len(metrics)
    num_datasets = len(dataset_names)
    
    # Initialize dictionaries to store data
    method_data = {method: [] for method in generative_methods}
    real_method_data = []
    
    # Collect data for all datasets
    for dataset_idx, dataset_name in enumerate(dataset_names):
        results_path = os.path.join(args.results_folder, dataset_name)
        
        # Initialize empty lists for dataset metrics
        data_metrics = []
        data_metrics_diff = []
        
        # Get data for this dataset
        files_dict = get_files(results_path, generative_methods, lamda_tradeoff_values, metrics, 
                              data_metrics, data_metrics_diff, downstream_model, no_lambda_methods)
        
        # Extract reference method data for this dataset
        real_dataset_data = []
        for metric in metrics:
            if real_method in files_dict and metric in files_dict[real_method]["avg"]:
                avg = files_dict[real_method]["avg"][metric]
                # Convert to scalar if it's a list with a single element
                avg_val = avg[0] if isinstance(avg, list) and len(avg) == 1 else avg
                real_dataset_data.append(avg_val)
            else:
                real_dataset_data.append(float('nan'))
        
        real_method_data.append(real_dataset_data)
        
        # Store data for each method
        for method in generative_methods:
            method_dataset_data = []
            for metric_idx, metric in enumerate(metrics):
                if method in files_dict and metric in files_dict[method]["avg"]:
                    avg = files_dict[method]["avg"][metric]
                    # Convert to scalar if it's a list with a single element
                    avg_val = avg[0] if isinstance(avg, list) and len(avg) == 1 else avg
                    method_dataset_data.append(avg_val)
                else:
                    method_dataset_data.append(float('nan'))
            
            method_data[method].append(method_dataset_data)
    
    # Calculate percentile differences and average across datasets
    percentile_diffs = {}
    for method in generative_methods:
        percentile_diffs[method] = []
        
        for metric_idx in range(num_metrics):
            valid_diffs = []
            
            for dataset_idx in range(num_datasets):
                method_val = method_data[method][dataset_idx][metric_idx]
                real_val = real_method_data[dataset_idx][metric_idx]

                if method=="cuts" and dataset_names[dataset_idx] == "dutch":
                    continue
                
                if method=="prefair" and (dataset_names[dataset_idx] == "bank" or dataset_names[dataset_idx] == "census"):
                    continue

                            
                if method=="fsmote" and (dataset_names[dataset_idx] == "adult" or dataset_names[dataset_idx] == "bank" or dataset_names[dataset_idx] == "census"):
                    continue
                # Skip if either value is NaN
                if np.isnan(method_val) or np.isnan(real_val) or real_val == 0:
                    continue
                
                # Calculate percentile difference
                # (method_val - real_val) / real_val * 100
                percent_diff = ((method_val - real_val) / abs(real_val)) * 100
                valid_diffs.append(percent_diff)
            
            # Calculate average percentile difference across datasets for this metric
            if valid_diffs:
                avg_diff = np.mean(valid_diffs)
                std_diff = np.std(valid_diffs)
                percentile_diffs[method].append((avg_diff, std_diff))
            else:
                percentile_diffs[method].append((float('nan'), float('nan')))
    
    # Generate LaTeX table
    result = "\\begin{table}[ht]\n\\centering\n"
    result += f"\\caption{{Percentile difference compared to {method_names[real_method]}, averaged across all datasets}}\n"
    
    # Create tabular environment
    result += f"\\begin{{tabular}}{{l|{num_metrics * 'c'}}}\n\\hline\n"
    
    # Header row with metrics
    result += "\\textbf{Method}"
    for metric in metrics:
        metric_name = names[metric]
        result += f" & {metric_name}"
    result += " \\\\\n\\hline\n"
    
    # Skip the real method in the table since its percentile diff with itself is always 0
    for method in generative_methods:
        if method == real_method:
            continue
            
        method_name = method_names[method]
        row = [method_name]
        
        for metric_idx in range(num_metrics):
            avg_diff, std_diff = percentile_diffs[method][metric_idx]
            
            # Format the cell, handling NaN values
            if np.isnan(avg_diff) or np.isnan(std_diff):
                cell = "N/A"
            else:
                # Use +/- for the sign of the difference
                sign = "+" if avg_diff > 0 else ""
                cell = f"{sign}{avg_diff:.2f}\\% {{\\tiny $\\pm$ {std_diff:.2f}\\%}}"
            
            row.append(cell)
        
        result += f"{' & '.join(row)} \\\\\n"
    
    # End table
    result += "\\hline\n\\end{tabular}\n\\end{table}"
    return result
   
def main(args):
    # We now assume only one downstream model
    downstream_model = args.downstream_models[0]
    protected_attribute = args.protected_attribute


    if protected_attribute != "sex":
        dataset_name += f"_{protected_attribute}"


    generative_methods = args.methods   
    colors = args.colors

    lamda_tradeoff_values = [1.0]

    plot_folder = args.plot_folder

    if not os.path.exists(plot_folder): 
        os.makedirs(plot_folder)

    metrics = ["roc_auc_score", "stat_par"]
    names = {"roc_auc_score": r'ROC AUC $\uparrow$' "", "stat_par":r"Stat. Par. $\downarrow$"}

    method_names = {"real": 'Real Data', "tabfairgdt": "\\us", "tab_fair_gan": '\\tabfairgan', "tabular_argn": "\mostlyai", "cuts":'\cuts', "fsmote": '\\fsmote', "prefair": '\prefair'}

    max_cols_per_metric = len(metrics)  # Renamed from max_cols_per_model

    no_lambda_methods = ["fsmote", "prefair", "cuts"]

    data_metrics = []
    data_metrics_diff = []

    dataset_names = ["adult", "dutch", "bank", "census", "acsiUT", "acsiAL"]
    # dataset_names = ["adult", "dutch"]
    dataset_names_latex = ["Adult", "Dutch Census", "Bank Marketing", "KDD Census", "ACS-I Utah", "ACS-I Alabama"]
    
    table = generate_split_latex_table(metrics, names, method_names, dataset_names, dataset_names_latex, 
                              generative_methods, args, lamda_tradeoff_values, downstream_model, 
                              no_lambda_methods)

    print(table)
    print("\n\n ============== \n\n")

    overall_table = generate_percentile_diff_table(metrics, names, method_names, dataset_names, 
                              generative_methods, args, lamda_tradeoff_values, 
                              downstream_model, no_lambda_methods, real_method="real")
    print(overall_table)
    
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Generation')

    args = parser.parse_args()