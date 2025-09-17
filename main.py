import argparse
import numpy as np 

from run_experiment import main as run
from evaluation_framework.plot_results import main as plot
from evaluation_framework.plot_all_datasets import main as plot_all
from evaluation_framework.plot_all_datasets_pareto import main as plot_all_pareto
from evaluation_framework.plot_fair_splitting import main as fair_splitting
from evaluation_framework.run_computational_time import main as time
from evaluation_framework.run_computational_time_all import main as time_all
from evaluation_framework.dataset_stats import main as stats
from evaluation_framework.latex_table_experiment import main as table
from evaluation_framework.latex_table_experiment_data_metrics import main as table_data

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    
def execute_function(method):
    main_fn = eval(f'{method}')
    return main_fn

def get_args():
    parser = argparse.ArgumentParser(description='Pipeline')


    parser.add_argument('--methods', type=str, default=['real', 'tabfairgdt', 'tabular_argn', 'tab_fair_gan', 'cuts', 'fsmote', 'prefair'], help='Methods to execute', nargs='+')
    parser.add_argument('--colors', type=str, default=['black', 'blue', 'orange', 'red', 'green', 'brown', 'yellow'], help='One color for each generative method', nargs="+")



    parser.add_argument('--downstream_models', type=str, default=["LGB"], help='Downstream models for evaluation', nargs='+')


    # General configs
    parser.add_argument('--dataset', type=str, default='adult', help='Name of dataset.')
    parser.add_argument('--mode', type=str, default="run", help='Run eperiment or plot results')
    parser.add_argument('--verbose', type=str2bool, default=False, help='Verbose')
    parser.add_argument('--results-folder', type=str, default="results", help='Output folder')
    parser.add_argument('--prefix', type=str, default="", help='prefix to results folder')
    parser.add_argument('--plot-folder', type=str, default="plots", help='Output plot folder')
    
    parser.add_argument('--protected_attribute', type=str, default="sex", help='protected attribute')

    parser.add_argument('--reorder', type=str, default="", help='Change order of features')




    # Experiment configs
    parser.add_argument('--seed', default=2024, help='Random seed for split generation', type=int)
    parser.add_argument('--num_repeats', default=1, help='num_repeats', type=int)
    parser.add_argument('--num_folds', default=3, help='num_folds', type=int)
    parser.add_argument('--num_synthetic', default=1, help='number of synthetic train sets for split', type=int)
    parser.add_argument('--device', default="cpu", help='Device', type=str)
    parser.add_argument('--engine', default="sklearnex", help='ATOM engine', type=str)
    parser.add_argument('--parallel', default=True, help='Run in parallel', type=str2bool)
    
    # parser.add_argument('--lamda_tradeoff_values', type=list, default=[0], help='Lamda values')
    parser.add_argument('--lamda_tradeoff_values', type=list, default=np.arange(0, 1.05, 0.05), help='Lamda values')
    


    parser.add_argument('--acc_threshold', default=-1, help='Max acc drop percentage for relabeling', type=float)

    parser.add_argument('--max_cols_per_model', default=4, help='For plotting', type=int)



    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = get_args()

    main_fn = execute_function(args.mode)

    main_fn(args)