import os
import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import matplotlib.colors as mcolors
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

    generative_methods = args.methods   
    colors = args.colors

    lamda_tradeoff_values = args.lamda_tradeoff_values

    plot_folder = args.plot_folder

    if not os.path.exists(plot_folder): 
        os.makedirs(plot_folder)

    metrics = ["roc_auc_score", "stat_par"]
    names = {"roc_auc_score": r'ROC AUC $\uparrow$', "stat_par": r"Statistical Parity $\downarrow$"}

    method_names = {"real": r'Real Data', "tabfairgdt": r"$\mathbf{T}$$_{\mathrm{\mathbf{AB}}}$$\mathbf{F}$$_{\mathrm{\mathbf{AIR}}}$$\mathbf{GDT}$", "tabular_argn": r'TabularARGN', "tab_fair_gan": r'TabFairGAN'}

    no_lambda_methods = ["fsmote", "prefair", "cuts"]

    data_metrics = []
    data_metrics_diff = []

    dataset_names = ["adult", "acsiLA", "dutch", "census", "bank"]
    dataset_names_latex = ["Adult", "ACS-I Louisiana", "Dutch Census", "KDD Census", "Bank Marketing"]
    
    # Create one big figure for all datasets - now with 1 plot per dataset
    total_rows = len(dataset_names)//2 + 1
    fig = plt.figure(figsize=(12, total_rows*5), constrained_layout=True)
    # Create a grid for the entire figure
    main_gs = fig.add_gridspec(total_rows, 2)
    
    # Create dummy handles for the legend
    handles = []
    labels = []
    for method, color in zip(generative_methods, colors):
        if method == "real":
            marker = plt.Line2D([], [], color=color, label=method, marker='*', markersize=12, linestyle='', linewidth=2)
        elif method in no_lambda_methods:
            marker = plt.Line2D([], [], color=color, label=method, marker='o', markersize=8, linestyle='', linewidth=2)
        else:
            marker = plt.Line2D([], [], color=color, label=method, marker='D', markersize=8, linestyle='-', linewidth=2, alpha=0.7)
        handles.append(marker)
        labels.append(method_names[method])
    
    # Process each dataset
    for dataset_idx, dataset_name in enumerate(dataset_names):
        results_path = os.path.join(args.results_folder, dataset_name)
        
        if dataset_idx >= 2:
            # Create a subfigure for this dataset
            dataset_subfig = fig.add_subfigure(main_gs[dataset_idx-2, 1])
        else:
            # Create a subfigure for this dataset
            dataset_subfig = fig.add_subfigure(main_gs[dataset_idx, 0])
        
        # Add the legend above the first dataset's title
        if dataset_idx == 0:
            # Calculate the position for the legend
            legend_y_position = 1.15  # Just above the dataset title
            dataset_subfig.legend(handles=handles, 
                          labels=labels, 
                          loc='upper center',  # Position the legend at the top center
                          fontsize=16,
                          bbox_to_anchor=(1.0, legend_y_position),  # Position centered, above the title
                          ncol=len(generative_methods),  # Make the legend horizontal
                          frameon=True,  # Add a frame around the legend
                          borderaxespad=0.)
            
        dataset_subfig.suptitle(f'{dataset_names_latex[dataset_idx]}', fontsize=18, fontweight='bold')
        
        # Create a single subplot for the scatter plot
        ax = dataset_subfig.add_subplot(1, 1, 1)
        
        files_dict = get_files(results_path, generative_methods, lamda_tradeoff_values, metrics, 
                              data_metrics, data_metrics_diff, downstream_model, no_lambda_methods)
        
        for method, color in zip(generative_methods, colors):
            if dataset_name == "census" and method == "tab_fair_gan":
                continue
            if dataset_name == "bank" and method == "tab_fair_gan":
                continue
            
            method_name = method
            roc_auc_avg = np.array(files_dict[method_name]["avg"]["roc_auc_score"])
            roc_auc_std = np.array(files_dict[method_name]["std"]["roc_auc_score"])
            stat_par_avg = np.array(files_dict[method_name]["avg"]["stat_par"])
            stat_par_std = np.array(files_dict[method_name]["std"]["stat_par"])
            
            if method == "real":
                # Plot as a single point WITHOUT error bars
                ax.scatter(stat_par_avg, roc_auc_avg, 
                          marker='*', s=144, color=color,  # s=144 for markersize=12 equivalent
                          label=method_name, linewidth=2)
            elif method in no_lambda_methods:
                # Plot as a single point with thinner, more transparent error bars
                # Create a lighter color for error bars
                error_color = (*mcolors.to_rgb(color), 0.4)  # Same color but with alpha
                ax.errorbar(stat_par_avg, roc_auc_avg, 
                           xerr=stat_par_std, yerr=roc_auc_std,
                           marker='o', markersize=8, color=color, 
                           label=method_name, capsize=3, linewidth=2,
                           elinewidth=1, ecolor=error_color)
            else:
                # Plot as connected points with thinner, more transparent error bars for different lambda values
                # Create a lighter color for error bars
                error_color = (*mcolors.to_rgb(color), 0.4)  # Same color but with alpha
                ax.errorbar(stat_par_avg, roc_auc_avg, 
                           xerr=stat_par_std, yerr=roc_auc_std,
                           marker='D', markersize=6, color=color, alpha=0.7,
                           label=method_name, capsize=2, linewidth=2, linestyle='-',
                           elinewidth=1, ecolor=error_color)
        
        ax.set_xlabel(r'Statistical Parity $\downarrow$', fontsize=16)
        ax.set_ylabel(r'ROC AUC $\uparrow$', fontsize=16)
        ax.grid(True, alpha=0.3)
        
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.3f'))
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.3f'))
        
        # Add a note about the ideal region (high ROC AUC, low stat par)
        # ax.annotate('Ideal region\n(High ROC AUC, Low Stat. Par.)', 
        #             xy=(0, 1), xycoords='axes fraction',
        #             xytext=(-10, -10), textcoords='offset points',
        #             ha='right', va='top', fontsize=10, alpha=0.7,
        #             bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgray', alpha=0.5))

    
    # Save the single combined figure with all datasets
    plt.savefig(os.path.join(plot_folder, "all_datasets_scatter.png"), dpi=300, bbox_inches='tight')
    plt.close(fig)  # Close the figure to free memory

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generation')
    args = parser.parse_args()