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
def main(args):
    # We now assume only one downstream model
    downstream_model = args.downstream_models[0]
    protected_attribute = args.protected_attribute


    if protected_attribute != "sex":
        dataset_name += f"_{protected_attribute}"


    generative_methods = ["real", "fair_cart_leaf_relab_lamda", "fair_cart_target_only_fair" ]   
    colors = args.colors

    lamda_tradeoff_values = args.lamda_tradeoff_values

    plot_folder = args.plot_folder

    if not os.path.exists(plot_folder): 
        os.makedirs(plot_folder)

    metrics = ["roc_auc_score", "stat_par"]
    names = {"roc_auc_score": r'ROC AUC $\uparrow$' "", "stat_par":r"Stat. Par. $\downarrow$"}

    method_names = {"real": r'Real Data', "fair_cart_leaf_relab_lamda": r"$\mathbf{T}$$_{\mathrm{\mathbf{AB}}}$$\mathbf{F}$$_{\mathrm{\mathbf{AIR}}}$$\mathbf{GDT}$", "fair_cart_target_only_fair": r"T$_{\mathrm{AB}}$F$_{\mathrm{AIR}}$GDT (Fair Splitting Criterion)"}

    colors = ['black', 'blue', 'cyan']

    max_cols_per_metric = len(metrics)  # Renamed from max_cols_per_model

    no_lambda_methods = ["fsmote", "prefair", "cuts"]

    data_metrics = []
    data_metrics_diff = []

    dataset_names = ["adult"]
    dataset_names_latex = ["Adult Census"]
    
    # Create one big figure for all datasets
    total_rows = 1
    fig = plt.figure(figsize=(max_cols_per_metric*6, total_rows*5), constrained_layout=True)
    # Create a grid for the entire figure
    main_gs = fig.add_gridspec(total_rows, 1)
    
    # Create dummy handles for the legend
    handles = []
    labels = []
    for method, color in zip(generative_methods, colors):
        if method == "real":
            line = plt.Line2D([], [], color=color, label=method, linestyle='--', linewidth=2)
        elif method in no_lambda_methods:
            line = plt.Line2D([], [], color=color, label=method, linestyle='-', linewidth=2)
        else:
            line = plt.Line2D([], [], color=color, label=method, marker='D', linestyle='-', linewidth=2)
        handles.append(line)
        labels.append(method_names[method])
    
    # Process each dataset
    for dataset_idx, dataset_name in enumerate(dataset_names):
        results_path = os.path.join(args.results_folder, dataset_name)
        
        dataset_subfig = fig.add_subfigure(main_gs[dataset_idx, 0])
        
        # Add the legend above the first dataset's title
        if dataset_idx == 0:
            # Calculate the position for the legend
            legend_y_position = 1.1  # Just above the dataset title
            dataset_subfig.legend(handles=handles, 
                          labels=labels, 
                          loc='upper center',  # Position the legend at the top center
                          fontsize=17,
                          bbox_to_anchor=(0.5, legend_y_position),  # Position centered, above the title
                          ncol=len(generative_methods),  # Make the legend horizontal
                          frameon=True,  # Add a frame around the legend
                          borderaxespad=0.)
                        #   title="Methods",  # Add title to the legend
                        #   title_fontsize=15) 
            
        dataset_subfig.suptitle(f'{dataset_names_latex[dataset_idx]}', fontsize=18, fontweight='bold')
        
        # Determine how many rows are needed for the metrics
        num_metric_rows = (len(metrics) + max_cols_per_metric - 1) // max_cols_per_metric
        
        # Create a grid for metrics inside the dataset subfigure
        gs = dataset_subfig.add_gridspec(num_metric_rows, max_cols_per_metric)
        
        # Plot metrics in the grid
        for metric_idx, metric in enumerate(metrics):
            metric_row = metric_idx // max_cols_per_metric
            metric_col = metric_idx % max_cols_per_metric
            ax = dataset_subfig.add_subplot(gs[metric_row, metric_col])
            
            files_dict = get_files(results_path, generative_methods, lamda_tradeoff_values, metrics, 
                                  data_metrics, data_metrics_diff, downstream_model, no_lambda_methods)
            
            for method, color in zip(generative_methods, colors):
                if dataset_name == "census" and method == "tab_fair_gan":
                    continue
                method_name = method
                avg = np.array(files_dict[method_name]["avg"][metric])
                std = np.array(files_dict[method_name]["std"][metric])
                if method == "real":
                    ax.hlines(avg, xmin=0, xmax=1, linestyle='--', label=method_name, color=color, alpha=1, linewidth=2)
                elif method in no_lambda_methods:
                    lower_line = avg - std
                    upper_line = avg + std
                    ax.hlines(avg, xmin=0, xmax=1, linestyle='-', label=method_name, color=color, alpha=0.5, linewidth=2)
                    ax.hlines(lower_line, xmin=0, xmax=1, linestyle='-', color=color, alpha=0.2)
                    ax.hlines(upper_line, xmin=0, xmax=1, linestyle='-', color=color, alpha=0.2)
                    ax.fill_betweenx([lower_line[0], upper_line[0]], 0, 1, color=color, alpha=0.2)
                else:
                    ax.plot(lamda_tradeoff_values, avg, label=method_name, marker='D', color=color, alpha=0.5)
                    ax.fill_between(lamda_tradeoff_values, avg - std, avg + std, color=color, alpha=0.2)
            
            ax.set_title(f'{names[metric]}', fontsize=20)
            ax.grid(True)
            
            # Add y-label to first column only
            # if metric_col == 0:
            #     ax.set_ylabel('Score')
            
            # Add x-label to bottom row only if method uses lambda
            if metric_row == num_metric_rows - 1:
                ax.set_xlabel(r'$\lambda$', fontsize=20)
            
            ax.tick_params(axis='both', which='major', labelsize=15)
            ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))

    
    # Save the single combined figure with all datasets
    plt.savefig(os.path.join(plot_folder, "fair_splitting.png"), dpi=300, bbox_inches='tight')
    plt.close(fig)  # Close the figure to free memory
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Generation')

    args = parser.parse_args()