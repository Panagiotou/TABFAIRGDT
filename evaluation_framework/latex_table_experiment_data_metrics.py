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
                           no_lambda_methods, datasets_per_row=1):
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
        data_metrics_diff = []
        
        # Get data for this dataset
        files_dict = get_files(results_path, generative_methods, lamda_tradeoff_values, [], 
                              metrics, data_metrics_diff, downstream_model, no_lambda_methods)
        
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
                    if np.isnan(avg) or np.isnan(std) or avg==[]:
                        cell = "-"
                    else:
                        cell = f"{avg:.2f}"
                    
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

def generate_averaged_latex_table(metrics, names, method_names, dataset_names, 
                               generative_methods, args, lamda_tradeoff_values, 
                               downstream_model, no_lambda_methods):
    """
    Generate a LaTeX table with results averaged across all datasets.
    Each method will have a single row with mean ± standard deviation across datasets.
    
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
        
    Returns:
    --------
    str
        LaTeX table as a string with averaged results
    """
    import os
    import numpy as np
    
    # Initialize data structure to hold results for each method and metric
    # For each method, we'll store a list of (avg, std) pairs for each dataset
    method_results = {method: {metric: [] for metric in metrics} for method in generative_methods}
    
    # Collect data for all datasets
    for dataset_name in dataset_names:
        results_path = os.path.join(args.results_folder, dataset_name)
        
        # Initialize empty lists for each dataset
        data_metrics_diff = []
        
        # Get data for this dataset
        files_dict = get_files(results_path, generative_methods, lamda_tradeoff_values, [], 
                              metrics, data_metrics_diff, downstream_model, no_lambda_methods)
        
        # Store the data for each method and metric
        for method in generative_methods:
            if method not in files_dict:
                continue
                
            for metric in metrics:
                if metric in files_dict[method]["avg"]:
                    avg = files_dict[method]["avg"][metric]
                    std = files_dict[method]["std"][metric]
                    
                    # Convert to scalar if it's a list with a single element
                    avg_val = avg[0] if isinstance(avg, list) and len(avg) == 1 else avg
                    std_val = std[0] if isinstance(std, list) and len(std) == 1 else std
                    
                    # Only append non-NaN values
                    if not (np.isnan(avg_val) or np.isnan(std_val) or avg_val == []):
                        method_results[method][metric].append(avg_val)
    
    # Calculate the average and standard deviation across datasets for each method and metric
    averaged_results = {}
    for method in generative_methods:
        averaged_results[method] = {}
        for metric in metrics:
            values = method_results[method][metric]
            if values:  # Only calculate if we have values
                mean = np.mean(values)
                std = np.std(values, ddof=1)  # Sample standard deviation
                averaged_results[method][metric] = (mean, std)
            else:
                averaged_results[method][metric] = (float('nan'), float('nan'))
    
    # Start building the LaTeX table
    result = "\\begin{table}[ht]\n\\centering\n\\caption{Results averaged across all datasets}\n"
    
    # Create tabular environment with appropriate columns
    # One column for method names + one column for each metric
    result += f"\\begin{{tabular}}{{l|{len(metrics) * 'c'}}}\n\\hline\n"
    
    # Create the header row with metric names
    result += "\\textbf{Method}"
    for metric in metrics:
        metric_name = names[metric]
        result += f" & \\textbf{{{metric_name}}}"
    result += " \\\\\n\\hline\n"
    
    # Add rows for each method
    for method in generative_methods:
        method_name = method_names[method]
        row = [method_name]
        
        for metric in metrics:
            # Get the avg and std values
            mean, std = averaged_results[method][metric]
            
            # Format the cell as mean ± std, handling NaN values
            if np.isnan(mean) or np.isnan(std):
                cell = "-"
            else:
                cell = f"{mean:.2f} {{\\tiny$\\pm${std:.1f}}}"
            
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
    generative_methods.remove("real")
    colors = args.colors

    lamda_tradeoff_values = [1.0]

    plot_folder = args.plot_folder

    if not os.path.exists(plot_folder): 
        os.makedirs(plot_folder)

    data_metrics = ["detection_score", "KS-score (continuous)", "TV-score (categorical)", "PRDC precision", "PRDC recall", "PRDC density", "PRDC coverage", "DCR"] 
    names = {"detection_score": r'Det. Sc. $\approx 0.5$', "KS-score (continuous)": r"KS $\uparrow$", "TV-score (categorical)": r"TV $\uparrow$",
              "PRDC precision": r"Precision $\uparrow$", "PRDC recall": r"Recall $\uparrow$", "PRDC density": r"Density $\uparrow$", "PRDC coverage": r"Coverage $\uparrow$", "DCR": "DCR"}

    method_names = {"tabfairgdt": "\\us", "tab_fair_gan": '\\tabfairgan', "tabular_argn": "\mostlyai", "cuts":'\cuts', "fsmote": '\\fsmote', "prefair": '\prefair',
                    "tabfairgdt_asc_target": "Asc. Target", "tabfairgdt_desc_target": "Desc. Target", "tabfairgdt_asc_protected": "Asc. PA", "tabfairgdt_desc_protected": "Desc. PA",
                    # "tabfairgdt": "Original"
                    }

    max_cols_per_metric = len(data_metrics)  # Renamed from max_cols_per_model

    no_lambda_methods = ["fsmote", "prefair", "cuts"]

    data_metrics_diff = []

    dataset_names = ["adult", "dutch", "bank", "census", "acsiUT", "acsiAL"]
    # dataset_names = ["adult", "dutch"]
    dataset_names_latex = ["Adult", "Dutch Census", "Bank Marketing", "KDD Census", "ACS-I Utah", "ACS-I Alabama"]
    
    table = generate_split_latex_table(data_metrics, names, method_names, dataset_names, dataset_names_latex, 
                              generative_methods, args, lamda_tradeoff_values, downstream_model, 
                              no_lambda_methods)

    print(table)

    overall_table = generate_averaged_latex_table(data_metrics, names, method_names, dataset_names, 
                              generative_methods, args, lamda_tradeoff_values, 
                              downstream_model, no_lambda_methods)
    print(overall_table)
    
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Generation')

    args = parser.parse_args()