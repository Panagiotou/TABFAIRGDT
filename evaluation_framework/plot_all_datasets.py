import os
import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from matplotlib.gridspec import GridSpec
from tabular_datasets.dataset import Dataset
import matplotlib.ticker as ticker
from matplotlib.ticker import MaxNLocator


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
    names = {"roc_auc_score": r'ROC AUC $\uparrow$' "", "stat_par":r"Stat. Par. $\downarrow$"}

    method_names = {"real": r'Real Data', "fair_cart_leaf_relab_lamda": r"$\mathbf{T}$$_{\mathrm{\mathbf{AB}}}$$\mathbf{F}$$_{\mathrm{\mathbf{AIR}}}$$\mathbf{GDT}$", "mostlyai": r'TabularARGN', "tab_fair_gan": r'TabFairGAN',
                    "tabfairgdt": r"$\mathbf{T}$$_{\mathrm{\mathbf{AB}}}$$\mathbf{F}$$_{\mathrm{\mathbf{AIR}}}$$\mathbf{GDT}$", "tabular_argn": r'TabularARGN',
                    "tabfairgdt_asc_target": "Asc. Target", "tabfairgdt_desc_target": "Desc. Target", "tabfairgdt_asc_protected": "Asc. PA", "tabfairgdt_desc_protected": "Desc. PA",
                    # "tabfairgdt": "Original"
                    }

    max_cols_per_metric = len(metrics)  # Renamed from max_cols_per_model

    no_lambda_methods = ["fsmote", "prefair", "cuts"]

    data_metrics = []
    data_metrics_diff = []

    dataset_names = ["adult", "dutch", "bank", "census", "acsiUT", "acsiAL"]
    # dataset_names = ["adult", "dutch"]
    dataset_names_latex = ["Adult", "Dutch Census", "Bank Marketing", "KDD Census", "ACS-I Utah", "ACS-I Alabama"]
    # dataset_names_latex = ["Adult", "Dutch Census"]

    # Create one big figure for all datasets
    total_rows = len(dataset_names)//2 + 1
    fig = plt.figure(figsize=(max_cols_per_metric*18, total_rows*9), constrained_layout=True)
    # Create a grid for the entire figure
    print(total_rows)
    main_gs = fig.add_gridspec(total_rows, 2)
    
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
        

        row = dataset_idx // 2
        col = dataset_idx % 2
        dataset_subfig = fig.add_subfigure(main_gs[row, col])
        # if dataset_idx >= 2:
            # Create a subfigure for this dataset
        # dataset_subfig = fig.add_subfigure(main_gs[1, dataset_idx])
        # else:
        #     # Create a subfigure for this dataset
        #     dataset_subfig = fig.add_subfigure(main_gs[0, dataset_idx])
        
        # Add the legend above the first dataset's title
        if dataset_idx == 0:
            # Calculate the position for the legend
            legend_y_position = 1.15  # Just above the dataset title
            dataset_subfig.legend(handles=handles, 
                          labels=labels, 
                          loc='upper center',  # Position the legend at the top center
                          fontsize=45,
                          bbox_to_anchor=(1.0, legend_y_position),  # Position centered, above the title
                          ncol=len(generative_methods),  # Make the legend horizontal
                          frameon=True,  # Add a frame around the legend
                          borderaxespad=0.)
                        #   title="Methods",  # Add title to the legend
                        #   title_fontsize=15) 
            
        dataset_subfig.suptitle(f'{dataset_names_latex[dataset_idx]}', fontsize=35, fontweight='bold')
        
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
                if dataset_name == "bank" and method == "tab_fair_gan":
                    continue
                method_name = method
                avg = np.array(files_dict[method_name]["avg"][metric])
                std = np.array(files_dict[method_name]["std"][metric])
                if method == "real":
                    ax.hlines(avg, xmin=0, xmax=1, linestyle='--', label=method_name, color=color, alpha=1, linewidth=3)
                elif method in no_lambda_methods:
                    lower_line = avg - std
                    upper_line = avg + std
                    ax.hlines(avg, xmin=0, xmax=1, linestyle='-', label=method_name, color=color, alpha=0.5, linewidth=2)
                    ax.hlines(lower_line, xmin=0, xmax=1, linestyle='-', color=color, alpha=0.2)
                    ax.hlines(upper_line, xmin=0, xmax=1, linestyle='-', color=color, alpha=0.2)
                    ax.fill_betweenx([lower_line[0], upper_line[0]], 0, 1, color=color, alpha=0.2)
                else:
                    ax.plot(lamda_tradeoff_values, avg, label=method_name, marker='D', color=color, alpha=0.5, linewidth=4, markersize=12)
                    ax.fill_between(lamda_tradeoff_values, avg - std, avg + std, color=color, alpha=0.2)
            
            ax.set_title(f'{names[metric]}', fontsize=35)
            ax.grid(True)

            # Remove the box around the plot
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            
            # Add y-label to first column only
            # if metric_col == 0:
            #     ax.set_ylabel('Score')
            
            # Add x-label to bottom row only if method uses lambda
            if metric_row == num_metric_rows - 1:
                ax.set_xlabel(r'$\lambda$', fontsize=35)
            
            
            # ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.3f'))
            y_values = ax.get_yticks()
            y_values_ticks = []
            for v in y_values:
                v = round(v, 2)
                if v not in y_values_ticks:
                    y_values_ticks.append(v)
            y_values = y_values_ticks
            y_values_ticks = []
            if len(y_values) > 4:
                for i in range(len(y_values)):
                    if i % 2 == 0:
                        if y_values[i] > 0:
                            y_values_ticks.append(y_values[i])
                        # else:
                        #     y_values_ticks.append(y_values[1])
            else:
                y_values_ticks = y_values
            ax.set_yticks(y_values_ticks)

            ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
            ax.set_ylim(min(y_values_ticks) - (y_values_ticks[1]-y_values_ticks[0])/2, max(y_values_ticks) + (y_values_ticks[1]-y_values_ticks[0])/2)
            # ax.yaxis.set_major_locator(MaxNLocator(nbins=5, prune='both'))

            ax.tick_params(axis='both', which='major', labelsize=35)

    
    # Save the single combined figure with all datasets
    # plt.savefig(os.path.join(plot_folder, "all_datasets_ordering_small.pdf"), bbox_inches='tight')
    plt.savefig(os.path.join(plot_folder, "all_datasets.pdf"), bbox_inches='tight')
    plt.close(fig)  # Close the figure to free memory
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Generation')

    args = parser.parse_args()