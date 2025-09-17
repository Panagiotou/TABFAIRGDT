import os
import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from matplotlib.gridspec import GridSpec
from tabular_datasets.dataset import Dataset


from evaluation_framework.utils_eval import get_files
import argparse
def main(args):
    dataset_name = args.dataset
    downstream_models = args.downstream_models
    protected_attribute = args.protected_attribute

    dataset_loader = Dataset(args.dataset)


    if protected_attribute != "sex":

        dataset_name += f"_{protected_attribute}"

    if protected_attribute in dataset_loader.continuous_input_cols:

        continuous_protected_attribute = True
    else:
        continuous_protected_attribute = False


    generative_methods = args.methods   
    colors = args.colors

    lamda_tradeoff_values = args.lamda_tradeoff_values

    plot_folder = args.plot_folder
    max_cols_per_model = args.max_cols_per_model  # New parameter

    if not os.path.exists(plot_folder): 
        os.makedirs(plot_folder)

    if continuous_protected_attribute:
        metrics = ["roc_auc_score", "f1_score", "precission", "recall", "HGR"]
        names = {"roc_auc_score": "ROC AUC", "f1_score": "F1", "precission": "precission", "recall":"recall","HGR":"HGR"}
        optim = {"roc_auc_score":"\u2191", "f1_score":"\u2191", "precission":"\u2191", "recall":"\u2191", "HGR": "\u2193"}

        data_metrics = ["TV-score (categorical)", "KS-score (continuous)", "PRDC precision", "PRDC recall", "PRDC density", "PRDC coverage", "detection_score"] 
        optim_data_metrics = {"TV-score (categorical)":"\u2191", "KS-score (continuous)":"\u2191", "PRDC precision":"\u2191", "PRDC recall":"\u2191", 
                "PRDC density": "\u2191", "PRDC coverage": "\u2191", "detection_score": "\u2248 0.5"}
        data_metrics_diff = [] 
        optim_data_metrics_diff = {}
    else:
        metrics = ["roc_auc_score", "balanced_accuracy_score", "f1_score", "precission", "recall", "stat_par", "eq_odd", "eq_opp", "dpr"]
        names = {"roc_auc_score": "ROC AUC", "balanced_accuracy_score": "BAC", "f1_score": "F1", "precission": "precission", "recall":"recall",
                "stat_par": "statistical parity", "eq_odd": "equalized odds", "eq_opp": "equal opportunity", "dpr": "demographic_parity_ratio"}
        optim = {"roc_auc_score":"\u2191", "balanced_accuracy_score":"\u2191", "f1_score":"\u2191", "precission":"\u2191", "recall":"\u2191", 
                "stat_par": "\u2193", "eq_odd": "\u2193", "eq_opp": "\u2193", "dpr":"\u2191",
                }

        data_metrics = ["TV-score (categorical)", "KS-score (continuous)", "PRDC precision", "PRDC recall", "PRDC density", "PRDC coverage", "detection_score", "DCR"] 
        optim_data_metrics = {"TV-score (categorical)":"\u2191", "KS-score (continuous)":"\u2191", "PRDC precision":"\u2191", "PRDC recall":"\u2191", 
                "PRDC density": "\u2191", "PRDC coverage": "\u2191", "detection_score": "\u2248 0.5",  "DCR": "\u2191",
                }
        data_metrics_diff = [] 
        optim_data_metrics_diff = {}
        # data_metrics_diff = ["TV-score (categorical)_diff", "KS-score (continuous)_diff", "PRDC precision_diff", "PRDC recall_diff", "PRDC density_diff", "PRDC coverage_diff", "detection_score_diff"] 
        # optim_data_metrics_diff = {"TV-score (categorical)_diff":"\u2193", "KS-score (continuous)_diff":"\u2193", "PRDC precision_diff":"\u2193", "PRDC recall_diff":"\u2193", 
        #         "PRDC density_diff": "\u2193", "PRDC coverage_diff": "\u2193", "detection_score_diff": "\u2193",
        #         }

    no_lambda_methods = ["fsmote", "prefair", "cuts"]

    results_path = os.path.join(args.results_folder, dataset_name)



    if len(data_metrics_diff)>0:
        fig = plt.figure(figsize=(max_cols_per_model*5, (len(downstream_models)+2)*10 + 2), constrained_layout=True)
        fig.suptitle(f'Dataset "{dataset_name}"', fontsize=18, fontweight='bold')
        outer_gs = fig.add_gridspec(len(downstream_models) + 3, 1, height_ratios=[1]*(len(downstream_models)+2) + [0.1])
        # Define GridSpec layout for rows of subfigures
    else:
        fig = plt.figure(figsize=(max_cols_per_model*5, (len(downstream_models)+1)*10 + 1), constrained_layout=True)
        fig.suptitle(f'Dataset "{dataset_name}"', fontsize=18, fontweight='bold')
        outer_gs = fig.add_gridspec(len(downstream_models) + 2, 1, height_ratios=[1]*(len(downstream_models)+1) + [0.1])


    # Create each SubFigure within GridSpec
    for row, downstream_model in enumerate(downstream_models):
        subfig = fig.add_subfigure(outer_gs[row, 0])
        subfig.suptitle(f'Downstream model "{downstream_model}"', fontsize=16, fontweight='bold')

        # Determine how many rows are needed for the metrics
        num_metric_rows = (len(metrics) + max_cols_per_model - 1) // max_cols_per_model

        # Create a grid for metrics inside the subfigure
        gs = subfig.add_gridspec(num_metric_rows, max_cols_per_model)

        # Plot metrics in the grid
        for metric_idx, metric in enumerate(metrics):
            metric_row = metric_idx // max_cols_per_model
            metric_col = metric_idx % max_cols_per_model
            ax = subfig.add_subplot(gs[metric_row, metric_col])

            files_dict = get_files(results_path, generative_methods, lamda_tradeoff_values, metrics, data_metrics, data_metrics_diff, downstream_model, no_lambda_methods)

            for method, color in zip(generative_methods, colors):
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

                ax.set_title(f'{names[metric]} {optim[metric]}')
                ax.grid(True)


    if len(data_metrics_diff)>0:   
        subfig_data_metrics = fig.add_subfigure(outer_gs[-3, 0])
    else:
        subfig_data_metrics = fig.add_subfigure(outer_gs[-2, 0])

    subfig_data_metrics.suptitle(f'Data metrics', fontsize=16, fontweight='bold')

    # Determine how many rows are needed for the metrics
    num_metric_rows = (len(data_metrics) + max_cols_per_model - 1) // max_cols_per_model

    # Create a grid for metrics inside the subfigure
    gs = subfig_data_metrics.add_gridspec(num_metric_rows, max_cols_per_model)

    # Plot metrics in the grid
    for metric_idx, metric in enumerate(data_metrics):
        metric_row = metric_idx // max_cols_per_model
        metric_col = metric_idx % max_cols_per_model
        ax = subfig_data_metrics.add_subplot(gs[metric_row, metric_col])

        files_dict = get_files(results_path, generative_methods, lamda_tradeoff_values, metrics, data_metrics, data_metrics_diff, downstream_model, no_lambda_methods)

        for method, color in zip(generative_methods, colors):

            if method != "real":
                method_name = method
                avg = np.array(files_dict[method_name]["avg"][metric])
                std = np.array(files_dict[method_name]["std"][metric])
                # if method == "real":
                #     ax.hlines(avg, xmin=0, xmax=1, linestyle='--', label=method_name, color=color, alpha=1, linewidth=2)
                if method in no_lambda_methods:
                    lower_line = avg - std
                    upper_line = avg + std
                    ax.hlines(avg, xmin=0, xmax=1, linestyle='-', label=method_name, color=color, alpha=0.5, linewidth=2)
                    ax.hlines(lower_line, xmin=0, xmax=1, linestyle='-', color=color, alpha=0.2)
                    ax.hlines(upper_line, xmin=0, xmax=1, linestyle='-', color=color, alpha=0.2)
                    ax.fill_betweenx([lower_line[0], upper_line[0]], 0, 1, color=color, alpha=0.2)
                else:
                    ax.plot(lamda_tradeoff_values, avg, label=method_name, marker='D', color=color, alpha=0.5)
                    ax.fill_between(lamda_tradeoff_values, avg - std, avg + std, color=color, alpha=0.2)

            ax.set_title(f'{metric} {optim_data_metrics[metric]}')
            ax.grid(True)





    if len(data_metrics_diff)>0:

        subfig_data_metrics_diff = fig.add_subfigure(outer_gs[-2, 0])
        subfig_data_metrics_diff.suptitle(f'Data metrics diff', fontsize=16, fontweight='bold')
        # Determine how many rows are needed for the metrics
        num_metric_rows = (len(data_metrics_diff) + max_cols_per_model - 1) // max_cols_per_model

        # Create a grid for metrics inside the subfigure
        gs = subfig_data_metrics_diff.add_gridspec(num_metric_rows, max_cols_per_model)

        # Plot metrics in the grid
        for metric_idx, metric in enumerate(data_metrics_diff):
            metric_row = metric_idx // max_cols_per_model
            metric_col = metric_idx % max_cols_per_model
            ax = subfig_data_metrics_diff.add_subplot(gs[metric_row, metric_col])

            files_dict = get_files(results_path, generative_methods, lamda_tradeoff_values, metrics, data_metrics, data_metrics_diff, downstream_model, no_lambda_methods)

            for method, color in zip(generative_methods, colors):

                if method != "real":
                    method_name = method
                    avg = np.array(files_dict[method_name]["avg"][metric])
                    std = np.array(files_dict[method_name]["std"][metric])
                    # if method == "real":
                    #     ax.hlines(avg, xmin=0, xmax=1, linestyle='--', label=method_name, color=color, alpha=1, linewidth=2)
                    if method in no_lambda_methods:
                        lower_line = avg - std
                        upper_line = avg + std
                        ax.hlines(avg, xmin=0, xmax=1, linestyle='-', label=method_name, color=color, alpha=0.5, linewidth=2)
                        ax.hlines(lower_line, xmin=0, xmax=1, linestyle='-', color=color, alpha=0.2)
                        ax.hlines(upper_line, xmin=0, xmax=1, linestyle='-', color=color, alpha=0.2)
                        ax.fill_betweenx([lower_line[0], upper_line[0]], 0, 1, color=color, alpha=0.2)
                    else:
                        ax.plot(lamda_tradeoff_values, avg, label=method_name, marker='D', color=color, alpha=0.5)
                        ax.fill_between(lamda_tradeoff_values, avg - std, avg + std, color=color, alpha=0.2)

                ax.set_title(f'{metric} {optim_data_metrics_diff[metric]}')
                ax.grid(True)


    # Add a subplot for legends
    legend_ax = fig.add_subfigure(outer_gs[-1, 0])
    # legend_ax.axis('off')  # Hide the axes

    # Create a dummy plot to extract handles and labels
    handles = []
    labels = []
    for method, color in zip(generative_methods, colors):
        if method=="real":
            line = plt.Line2D([], [], color=color, label=method, linestyle='--')
        elif method in no_lambda_methods:
            line = plt.Line2D([], [], color=color, label=method, linestyle='-')
        else:
            line = plt.Line2D([], [], color=color, label=method, marker='D', linestyle='-')
        handles.append(line)
        labels.append(method)

    legend_ax.legend(handles=handles, labels=labels, loc='center', fontsize=12, ncol=len(generative_methods))

    plt.savefig(os.path.join(plot_folder, f"{dataset_name}.png"), dpi=300, bbox_inches='tight')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Generation')

    args = parser.parse_args()