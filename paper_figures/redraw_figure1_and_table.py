#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Redraw Figure 1 with improvements and create a comprehensive comparison table
for the paper on lesion boundary segmentation models.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
import matplotlib.patches as mpatches

# Set high-quality plotting style for publication
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.5)
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Computer Modern Roman', 'Times New Roman']

# Create directory for paper figures
output_dir = Path("paper_figures")
output_dir.mkdir(exist_ok=True)

# Load JSON data
with open("model_comparison_data.json", "r") as f:
    data = json.load(f)

# Extract model data
models = data["models"]

# Create a DataFrame for the comparison table
table_data = []
for name, model_data in models.items():
    # Format name for display
    display_name = name.replace("_", " ")
    
    # Get performance metrics
    perf = model_data["performance_metrics"]
    res = model_data["resource_metrics"]
    eff = model_data["efficiency_metrics"]
    
    # Format parameters for display
    if isinstance(res["parameters"], str):
        params = "Combined"
        params_val = 0  # For plotting
    else:
        params = res["parameters_readable"]
        params_val = res["parameters"] / 1_000_000  # Convert to millions for plotting
    
    # Add to table data
    table_data.append({
        "Model": display_name,
        "Dice": perf["dice"],
        "IoU": perf["iou"],
        "Boundary IoU": perf["boundary_iou"],
        "Precision": perf["precision"],
        "Sensitivity": perf["sensitivity"],
        "Specificity": perf["specificity"],
        "Parameters": params,
        "Size (MB)": res["model_size_mb"],
        "GPU Memory (GB)": res["gpu_memory_gb"],
        "Inference (s)": res["inference_time_sec"],
        "FLOPs": res["flops_readable"],
        "Efficiency Score": eff["efficiency_score"]
    })

# Convert to DataFrame and sort by Dice score
df = pd.DataFrame(table_data)
df = df.sort_values(by="Dice", ascending=False)

# Save as markdown and LaTeX table
with open(output_dir / "model_comparison_table.md", "w") as f:
    f.write(df.to_markdown(index=False))

# Save as LaTeX table with formatting
latex_table = df.to_latex(index=False, float_format="%.4f", escape=False)

# Enhance LaTeX table formatting
latex_table = latex_table.replace("tabular", "tabular*{\\textwidth}")
latex_table = latex_table.replace("\\begin{tabular", "\\begin{tabular")
latex_table = latex_table.replace("\\toprule", "\\toprule\n\\rowcolor{lightgray}")

with open(output_dir / "model_comparison_table.tex", "w") as f:
    f.write(latex_table)

# Extract data for plotting
model_names = df["Model"].tolist()
dice_scores = df["Dice"].tolist()
iou_scores = df["IoU"].tolist()
params = []
for name in df["Model"].tolist():
    model_name = name.replace(" ", "_")
    if model_name == "Enhanced_Ensemble":
        params.append(0)  # For ensemble
    else:
        params.append(models[model_name]["resource_metrics"]["parameters"] / 1_000_000)

gpu_memory = df["GPU Memory (GB)"].tolist()
efficiency = df["Efficiency Score"].tolist()

# Now redraw Figure 1 with improvements
plt.figure(figsize=(12, 9))

# Create the bubble chart with enhanced visualization
fig, ax = plt.subplots(figsize=(12, 9))

# Define a more scientific color palette
colors = sns.color_palette("viridis", len(model_names))

# Create custom bubble sizes based on GPU memory (scaled)
sizes = [m*300 for m in gpu_memory]

# Plot each model
for i, (name, dice, param, size, color, eff) in enumerate(zip(
    model_names, dice_scores, params, sizes, colors, efficiency
)):
    
    # Special case for Ensemble which has "Combined" parameters
    if param == 0:
        param = 62.1  # Sum of component models (57.8 + 4.3)
    
    # Plot the bubble
    ax.scatter(
        param, dice, 
        s=size, 
        color=color, 
        alpha=0.75, 
        edgecolors='white', 
        linewidth=1.5,
        zorder=3
    )
    
    # Add efficiency score as a text label inside the bubble
    ax.annotate(
        f"{eff:.1f}",
        (param, dice),
        fontsize=11,
        ha='center',
        va='center',
        color='white',
        weight='bold'
    )
    
    # Add model name
    if name == "Enhanced Ensemble":
        # Position the ensemble label differently to avoid overlap
        ax.annotate(
            name,
            (param + 5, dice + 0.001),
            fontsize=14,
            ha='left',
            va='center',
            weight='bold'
        )
    else:
        ax.annotate(
            name,
            (param, dice + 0.0025),
            fontsize=14,
            ha='center',
            va='center',
            weight='bold'
        )

# Add trend line to show parameter-performance relationship
z = np.polyfit(params, dice_scores, 1)
p = np.poly1d(z)
x_trend = np.linspace(0, max(params)+5, 100)
ax.plot(x_trend, p(x_trend), "--", color="gray", alpha=0.7, zorder=1)

# Set axis labels and title with enhanced formatting
ax.set_xlabel('Number of Parameters (Millions)', fontsize=20, labelpad=10)
ax.set_ylabel('Dice Similarity Coefficient', fontsize=20, labelpad=10)
ax.set_title('Model Performance vs. Complexity', fontsize=20, pad=20)

# Add a subtitle with explanation
plt.figtext(0.5, 0.01, 
           "Bubble size represents GPU memory usage; number inside shows efficiency score (1-10)",
           ha='center', fontsize=14, fontstyle='italic')

# Add legend for bubble sizes
legend_sizes = [2.8, 4.2, 6.8, 8.0]
legend_labels = [f"{s} GB" for s in legend_sizes]
legend_handles = [
    plt.scatter([], [], s=s*300, color='gray', alpha=0.6, edgecolors='white', linewidth=1.5)
    for s in legend_sizes
]

# Add custom legend for efficiency score
dummy_handles = []
dummy_labels = ["Efficiency Score", "1-10 scale"]
legend1 = ax.legend(legend_handles, legend_labels, 
                  title="GPU Memory Usage",
                  frameon=True, fancybox=True, 
                  loc='upper center', bbox_to_anchor=(0.25, 0.04),
                  fontsize=14, title_fontsize=16,
                  handletextpad=2, columnspacing=2, 
                  ncol=2)

# Add the legend manually
ax.add_artist(legend1)

# Set grid and tick customization
ax.grid(True, linestyle='--', alpha=0.7, zorder=0)
ax.tick_params(axis='both', which='major', labelsize=14)

# Improve the appearance of the plot
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Focus y-axis on the relevant range with a bit more padding
ax.set_ylim(0.84, 0.885)

# Add a dashed reference line for the best performance
ax.axhline(y=max(dice_scores), color='darkgreen', linestyle='--', alpha=0.6, zorder=0)
ax.text(max(params)-5, max(dice_scores)+0.0005, f"Best: {max(dice_scores):.4f}", 
      fontsize=11, color='darkgreen', ha='right')

# Add annotations for key insights
ax.annotate(
    "Most efficient",
    (8.4, 0.877),  # Lightweight DuaSkinSeg
    xytext=(15, 0.874),
    fontsize=11,
    arrowprops=dict(arrowstyle="->", color='black', alpha=0.7),
    bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="orange", alpha=0.8)
)

ax.annotate(
    "Best performance",
    (31.2, 0.8785),  # DuaSkinSeg
    xytext=(40, 0.882),
    fontsize=11,
    arrowprops=dict(arrowstyle="->", color='black', alpha=0.7),
    bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="orange", alpha=0.8)
)

# Save the enhanced figure
fig.tight_layout(rect=[0, 0.05, 1, 0.95])  # Adjust for the subtitle
fig.savefig(output_dir / "fig1_enhanced_performance_vs_parameters.png", dpi=300, bbox_inches='tight')
fig.savefig(output_dir / "fig1_enhanced_performance_vs_parameters.pdf", format='pdf', bbox_inches='tight')

print(f"Enhanced Figure 1 saved to {output_dir}!")
print(f"Comprehensive comparison table saved as:")
print(f"  - {output_dir}/model_comparison_table.md (Markdown format)")
print(f"  - {output_dir}/model_comparison_table.tex (LaTeX format)")