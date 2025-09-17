import json
import numpy as np

# Load the JSON data
with open('results/generation_timings.json', 'r') as f:
    data = json.load(f)

methods = list(data.keys())
df_sizes = [10, 100, 500]  # Feature sizes
sample_sizes = [1000, 10000, 50000]  # Number of rows per dataset
metrics = ["fit_time_avg", "generate_time_avg", "total_time_avg"]
metric_names = ["Fit", "Gen", "Total"]

# Format method names nicely
method_display_names = {
    "fair_cart_leaf_relab_lamda": "CART-Leaf",
    "mostlyai": "MostlyAI"
}

# Function to create LaTeX table
def create_latex_table():
    # Start LaTeX table
    latex = "\\begin{table}[htbp]\n"
    latex += "\\centering\n"
    latex += "\\caption{Performance comparison of synthetic data generation methods (time in seconds)}\n"
    latex += "\\label{tab:generation_times}\n"
    
    # Load packages needed
    latex += "% Requires \\usepackage{booktabs}, \\usepackage{multirow}\n"
    
    # Calculate column width - methods as columns
    num_method_cols = len(methods) * len(metrics)
    col_spec = "lrr" + "r" * num_method_cols
    
    latex += "\\begin{tabular}{" + col_spec + "}\n"
    latex += "\\toprule\n"
    
    # First header row: Methods
    latex += "& & & "
    for method in methods:
        method_name = method_display_names.get(method, method.replace("_", " ").title())
        latex += "\\multicolumn{" + str(len(metrics)) + "}{c}{" + method_name + "} & "
    latex = latex[:-2]  # Remove the last "& "
    latex += "\\\\\n"
    
    # Second header row: Metrics
    latex += "Dim & $n$ & & "
    for _ in methods:
        for metric in metric_names:
            latex += metric + " & "
    latex = latex[:-2]  # Remove the last "& "
    latex += "\\\\ \\midrule\n"
    
    # Find the best (lowest) values for each configuration and metric
    best_values = {}
    for d in df_sizes:
        for s in sample_sizes:
            for metric in metrics:
                values = []
                for method in methods:
                    try:
                        values.append(data[method][str(d)][str(s)][metric])
                    except KeyError:
                        values.append(float('inf'))
                best_values[(d, s, metric)] = min(values)
    
    # Data rows
    for i, d in enumerate(df_sizes):
        for j, s in enumerate(sample_sizes):
            # Use multirow for dimensions
            if j == 0:
                latex += "\\multirow{" + str(len(sample_sizes)) + "}{*}{$" + str(d) + "$} & "
            else:
                latex += " & "
                
            latex += "$" + str(s) + "$ & & "
            
            for method in methods:
                for k, metric in enumerate(metrics):
                    try:
                        value = data[method][str(d)][str(s)][metric]
                        # Check if this is the best value for this config and metric
                        if abs(value - best_values[(d, s, metric)]) < 1e-6:
                            latex += "\\textbf{" + f"{value:.2f}" + "} & "
                        else:
                            latex += f"{value:.2f} & "
                    except KeyError:
                        latex += "- & "
            
            latex = latex[:-2]  # Remove the last "& "
            latex += " \\\\\n"
            
            # Add a small gap after each dimension group
            if j == len(sample_sizes) - 1 and i < len(df_sizes) - 1:
                latex += "\\midrule\n"
    
    # Add improvement percentage row
    latex += "\\midrule\n"
    latex += "\\multicolumn{3}{l}{Improvement (\\%)} & "
    
    # Calculate improvement percentages for each metric
    improvement_percentages = {metric: [] for metric in metrics}
    
    for method_idx, method in enumerate(methods):
        if method == "fair_cart_leaf_relab_lamda":  # Our method
            for metric_idx, metric in enumerate(metrics):
                percentages = []
                for d in df_sizes:
                    for s in sample_sizes:
                        try:
                            our_value = data["fair_cart_leaf_relab_lamda"][str(d)][str(s)][metric]
                            competitor_value = data["mostlyai"][str(d)][str(s)][metric]
                            # Calculate improvement: (competitor - ours) / competitor * 100
                            # Positive percentage means our method is faster
                            percent_improvement = (competitor_value - our_value) / competitor_value * 100
                            percentages.append(percent_improvement)
                        except (KeyError, ZeroDivisionError):
                            pass
                
                # Calculate average and std of improvement percentages
                if percentages:
                    avg_improvement = np.mean(percentages)
                    std_improvement = np.std(percentages)
                    latex += f"{avg_improvement:.2f}{{\\tiny$\\pm${int(std_improvement)}}} & "
                else:
                    latex += "- & "
        else:  # Competitor method
            # Leave empty cells for the competitor
            for _ in metrics:
                latex += "& "
    
    latex = latex[:-2]  # Remove the last "& "
    latex += " \\\\\n"
    
    # End LaTeX table
    latex += "\\bottomrule\n"
    latex += "\\end{tabular}\n"
    latex += "\\end{table}\n"
    
    return latex

# Generate the LaTeX table
latex_table = create_latex_table()
print(latex_table)

# Save to file
with open('results/generation_times_table.tex', 'w') as f:
    f.write(latex_table)

print("LaTeX table saved to 'generation_times_table.tex'")