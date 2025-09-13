#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Generate high-resolution publication-ready figures from model comparison data.
These figures are specifically designed for academic paper publication.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
import os
from pathlib import Path

# Set high-quality plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.2)
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['savefig.pad_inches'] = 0.05
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

# Extract model names and metrics
model_names = []
dice_scores = []
iou_scores = []
parameters = []
gpu_memory = []
inference_time = []
boundary_iou = []
hausdorff = []
efficiency_scores = []

for name, model_data in models.items():
    # Format name for display
    display_name = name.replace("_", " ")
    if display_name == "Enhanced Ensemble":
        display_name = "Ensemble"
        
    model_names.append(display_name)
    dice_scores.append(model_data["performance_metrics"]["dice"])
    iou_scores.append(model_data["performance_metrics"]["iou"])
    
    if isinstance(model_data["resource_metrics"]["parameters"], str):
        parameters.append(0)  # For ensemble
    else:
        parameters.append(model_data["resource_metrics"]["parameters"] / 1_000_000)  # Convert to millions
        
    gpu_memory.append(model_data["resource_metrics"]["gpu_memory_gb"])
    inference_time.append(model_data["resource_metrics"]["inference_time_sec"])
    boundary_iou.append(model_data["performance_metrics"]["boundary_iou"])
    hausdorff.append(model_data["performance_metrics"]["hausdorff"])
    efficiency_scores.append(model_data["efficiency_metrics"]["efficiency_score"])

# Sort by Dice score for consistent ordering
indices = np.argsort(dice_scores)[::-1]  # Descending order
model_names = [model_names[i] for i in indices]
dice_scores = [dice_scores[i] for i in indices]
iou_scores = [iou_scores[i] for i in indices]
parameters = [parameters[i] for i in indices]
gpu_memory = [gpu_memory[i] for i in indices]
inference_time = [inference_time[i] for i in indices]
boundary_iou = [boundary_iou[i] for i in indices]
hausdorff = [hausdorff[i] for i in indices]
efficiency_scores = [efficiency_scores[i] for i in indices]

# Colors for each model - using a scientific color palette
colors = sns.color_palette("muted", len(model_names))

# =============================================================================
# Figure 1: Performance vs Resource Usage
# =============================================================================
plt.figure(figsize=(10, 8))

# Set up bubble chart
fig, ax = plt.subplots(figsize=(10, 8))
scatter = ax.scatter(
    parameters, 
    dice_scores, 
    s=[m*300 for m in gpu_memory],  # Size by GPU memory
    c=colors,
    alpha=0.7,
    edgecolors='white',
    linewidths=1.5
)

# Add model name annotations
for i, name in enumerate(model_names):
    if name == "Ensemble":
        # Special handling for ensemble
        ax.annotate(
            name, 
            (parameters[i] + 2, dice_scores[i]),
            fontsize=11,
            weight='bold',
            ha='left'
        )
    else:
        ax.annotate(
            name, 
            (parameters[i], dice_scores[i] + 0.0015),
            fontsize=11,
            weight='bold',
            ha='center'
        )

# Set axis labels and title
ax.set_xlabel('Number of Parameters (Millions)', fontsize=14)
ax.set_ylabel('Dice Similarity Coefficient', fontsize=14)
ax.set_title('Model Performance vs Parameter Count', fontsize=16)

# Add legend for bubble size
sizes = [2.8, 4.2, 6.8, 8.0]
labels = [f"{s} GB" for s in sizes]
leg = ax.legend(
    handles=[
        plt.scatter([], [], s=s*300, ec="none", color="gray", alpha=0.7) 
        for s in sizes
    ],
    labels=labels,
    title="GPU Memory Usage",
    frameon=True,
    fancybox=True,
    scatterpoints=1,
    loc="lower left",
    fontsize=10,
    title_fontsize=12
)

# Grid and tick customization
ax.grid(True, linestyle='--', alpha=0.7)
ax.tick_params(axis='both', which='major', labelsize=12)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Set y-axis limits to focus on the relevant range
ax.set_ylim(0.84, 0.885)

# Save the figure
fig.tight_layout()
fig.savefig(output_dir / "fig1_performance_vs_parameters.png", dpi=300, bbox_inches='tight')
fig.savefig(output_dir / "fig1_performance_vs_parameters.pdf", format='pdf', bbox_inches='tight')

# =============================================================================
# Figure 2: Model Performance Metrics Radar Chart
# =============================================================================
fig = plt.figure(figsize=(12, 9))

# Prepare radar chart data
categories = ['Dice', 'IoU', 'Boundary IoU', 'Specificity', 'Sensitivity', 'Precision']

# Get top 4 models by dice score
top_model_indices = indices[:4]
top_model_names = [model_names[i] for i in top_model_indices]
top_model_colors = [colors[i] for i in top_model_indices]

# Prepare data for radar chart
radar_data = []
for i in top_model_indices:
    model_name = list(models.keys())[i]
    model = models[model_name]
    metrics = model["performance_metrics"]
    # Normalize boundary IoU which has smaller values
    boundary = metrics["boundary_iou"] * 5  # Scale up for visibility
    radar_data.append([
        metrics["dice"], 
        metrics["iou"],
        boundary,
        metrics["specificity"], 
        metrics["sensitivity"],
        metrics["precision"]
    ])

# Number of variables
N = len(categories)

# Compute angle for each axis
angles = [n / N * 2 * np.pi for n in range(N)]
angles += angles[:1]  # Close the loop

# Initialize the radar chart
ax = plt.subplot(111, polar=True)

# Draw one axis per variable and add labels
plt.xticks(angles[:-1], categories, size=14)

# Draw the y-axis labels (0.8 to 1.0)
ax.set_ylim(0.7, 1.0)
ax.set_yticks([0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0])
ax.set_yticklabels(['0.7', '0.75', '0.8', '0.85', '0.9', '0.95', '1.0'], fontsize=12)

# Plot each model
for i, (model, color) in enumerate(zip(radar_data, top_model_colors)):
    values = model.copy()
    values += values[:1]  # Close the loop
    ax.plot(angles, values, linewidth=2, color=color, label=top_model_names[i])
    ax.fill(angles, values, color=color, alpha=0.1)

# Add legend
plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1), fontsize=14)

# Add title
plt.title('Performance Metrics Comparison', size=16, y=1.1)

# Save the radar chart
plt.tight_layout()
plt.savefig(output_dir / "fig2_model_metrics_radar.png", dpi=300, bbox_inches='tight')
plt.savefig(output_dir / "fig2_model_metrics_radar.pdf", format='pdf', bbox_inches='tight')

# =============================================================================
# Figure 3: Efficiency Analysis
# =============================================================================
fig, ax = plt.subplots(figsize=(12, 6))

# Prepare data for grouped bar chart
metrics = ["Dice Score", "Parameters (M)", "GPU Memory (GB)", "Inference Time (s)"]
model_display_names = model_names.copy()

# Normalize values for better visualization
norm_dice = [d for d in dice_scores]
norm_params = [min(p/10, 10) for p in parameters]  # Limit to 10 for display
norm_memory = [m for m in gpu_memory]
norm_time = [t*5 for t in inference_time]  # Scale up for visibility

# Set width of bars
barWidth = 0.2
positions = np.arange(len(model_display_names))

# Create bars
bars1 = ax.bar(positions - barWidth*1.5, norm_dice, barWidth, label='Dice Score', color='#1f77b4', alpha=0.8)
bars2 = ax.bar(positions - barWidth/2, norm_params, barWidth, label='Parameters (M)', color='#ff7f0e', alpha=0.8)
bars3 = ax.bar(positions + barWidth/2, norm_memory, barWidth, label='GPU Memory (GB)', color='#2ca02c', alpha=0.8)
bars4 = ax.bar(positions + barWidth*1.5, norm_time, barWidth, label='Inference Time (s)', color='#d62728', alpha=0.8)

# Add x-axis ticks
plt.xlabel('Model', fontsize=14)
plt.ylabel('Normalized Value', fontsize=14)
plt.title('Model Efficiency Comparison', fontsize=16)
plt.xticks(positions, model_display_names, rotation=30, ha='right', fontsize=12)

# Create legend
plt.legend(fontsize=12)

# Create twin axis for efficiency score
ax2 = ax.twinx()
ax2.plot(positions, efficiency_scores, 'o-', color='purple', linewidth=2, markersize=8, label='Efficiency Score')
ax2.set_ylabel('Efficiency Score (1-10)', color='purple', fontsize=14)
ax2.tick_params(axis='y', labelcolor='purple')
ax2.set_ylim(5, 10)

# Combine legends from both axes
lines, labels = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines + lines2, labels + labels2, loc='upper right', fontsize=12)

# Set grid
ax.grid(axis='y', linestyle='--', alpha=0.7)

# Save the figure
plt.tight_layout()
plt.savefig(output_dir / "fig3_efficiency_comparison.png", dpi=300, bbox_inches='tight')
plt.savefig(output_dir / "fig3_efficiency_comparison.pdf", format='pdf', bbox_inches='tight')

# =============================================================================
# Figure 4: Performance vs GPU Memory Usage
# =============================================================================
plt.figure(figsize=(10, 6))

# Create scatter plot with size representing model parameters
fig, ax = plt.subplots(figsize=(10, 6))

# Size bubbles by parameters
size_factor = 50000
sizes = []
for p in parameters:
    if p == 0:  # Handle ensemble special case
        sizes.append(2000)  # Give it a fixed size
    else:
        sizes.append(p * size_factor)

scatter = ax.scatter(
    gpu_memory,
    dice_scores,
    s=sizes,
    c=colors,
    alpha=0.7,
    edgecolors='white',
    linewidths=1.5
)

# Add model name annotations
for i, name in enumerate(model_names):
    ax.annotate(
        name,
        (gpu_memory[i], dice_scores[i] + 0.0015),
        fontsize=11,
        weight='bold',
        ha='center'
    )

# Add efficiency score as text annotations
for i, score in enumerate(efficiency_scores):
    ax.annotate(
        f"Eff: {score}",
        (gpu_memory[i], dice_scores[i] - 0.0025),
        fontsize=9,
        ha='center',
        color='darkblue'
    )

# Set axis labels and title
ax.set_xlabel('GPU Memory Usage (GB)', fontsize=14)
ax.set_ylabel('Dice Similarity Coefficient', fontsize=14)
ax.set_title('Model Performance vs GPU Memory Usage', fontsize=16)

# Set y-axis limits to focus on the relevant range
ax.set_ylim(0.84, 0.885)

# Grid and tick customization
ax.grid(True, linestyle='--', alpha=0.7)
ax.tick_params(axis='both', which='major', labelsize=12)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Save the figure
plt.tight_layout()
plt.savefig(output_dir / "fig4_performance_vs_memory.png", dpi=300, bbox_inches='tight')
plt.savefig(output_dir / "fig4_performance_vs_memory.pdf", format='pdf', bbox_inches='tight')

# =============================================================================
# Figure 5: Combined Performance-Resource-Efficiency Visualization
# =============================================================================
fig = plt.figure(figsize=(15, 10))
gs = GridSpec(2, 3, figure=fig)

# 1. Main bubble chart (spans 2 columns)
ax_main = fig.add_subplot(gs[0, :2])

# Set up the main bubble chart
param_exp = [np.log10(p) if p > 0 else 0 for p in parameters]  # Log scale for parameters
scatter = ax_main.scatter(
    gpu_memory,
    dice_scores,
    s=[p*size_factor/2 for p in parameters],
    c=[plt.cm.viridis(e/10) for e in efficiency_scores],  # Color by efficiency
    alpha=0.8,
    edgecolors='white',
    linewidths=1.5
)

# Add model name annotations
for i, name in enumerate(model_names):
    ax_main.annotate(
        name,
        (gpu_memory[i], dice_scores[i] + 0.0015),
        fontsize=11,
        weight='bold',
        ha='center'
    )

# Set axis labels and title
ax_main.set_xlabel('GPU Memory Usage (GB)', fontsize=14)
ax_main.set_ylabel('Dice Similarity Coefficient', fontsize=14)
ax_main.set_title('Performance vs Resource Usage (Color = Efficiency)', fontsize=16)

# Set y-axis limits to focus on the relevant range
ax_main.set_ylim(0.84, 0.885)

# Grid and tick customization
ax_main.grid(True, linestyle='--', alpha=0.7)
ax_main.tick_params(axis='both', which='major', labelsize=12)

# Colorbar for efficiency scores
cbar = plt.colorbar(plt.cm.ScalarMappable(cmap=plt.cm.viridis, 
                                         norm=plt.Normalize(6, 10)),
                   ax=ax_main)
cbar.set_label('Efficiency Score', fontsize=12)

# 2. Parameter breakdown - pie charts
ax_pie = fig.add_subplot(gs[0, 2])

# Select the two DuaSkinSeg models for component breakdown
duaskin_data = models["DuaSkinSeg"]["architecture"]["component_breakdown"]
lightweight_data = models["Lightweight_DuaSkinSeg"]["architecture"]["component_breakdown"]

# Extract data for pie charts
labels1 = list(duaskin_data.keys())
values1 = [component["percentage"] for component in duaskin_data.values()]
labels2 = list(lightweight_data.keys())
values2 = [component["percentage"] for component in lightweight_data.values()]

# Create pie chart
ax_pie.pie(values1, labels=None, autopct='%1.1f%%', startangle=90, colors=plt.cm.tab10.colors[:4], 
         wedgeprops={'edgecolor': 'w', 'linewidth': 1, 'alpha': 0.7})
ax_pie.add_artist(plt.Circle((0, 0), 0.3, fc='white'))
ax_pie.set_title('DuaSkinSeg Architecture Breakdown', fontsize=14)

# Add custom legend
ax_pie.legend(labels1, title="Components", loc="center", bbox_to_anchor=(0.5, 0.5), fontsize=10)

# 3. Bar chart comparing metrics across models
ax_metrics = fig.add_subplot(gs[1, :])

# Metrics to include
metrics_names = ["dice", "iou", "precision", "sensitivity", "boundary_iou"]
metrics_display = ["Dice", "IoU", "Precision", "Sensitivity", "Boundary IoU"]

# Prepare data
metrics_data = []
for name, model_data in models.items():
    model_metrics = [model_data["performance_metrics"][m] for m in metrics_names]
    metrics_data.append(model_metrics)

# Transpose to get metrics by type
metrics_data = np.array(metrics_data).T

# Create positions for bars
bar_positions = np.arange(len(metrics_display))
width = 0.12

# Plot bars for each model
for i, (name, color) in enumerate(zip(model_names, colors)):
    position = bar_positions + (i - len(model_names)/2 + 0.5) * width
    ax_metrics.bar(position, metrics_data[:, i], width=width, color=color, label=name, alpha=0.8)

# Customize bar chart
ax_metrics.set_xticks(bar_positions)
ax_metrics.set_xticklabels(metrics_display, fontsize=12)
ax_metrics.set_ylabel("Score", fontsize=14)
ax_metrics.set_title("Performance Metrics Comparison", fontsize=16)
ax_metrics.legend(fontsize=10, loc='upper right')
ax_metrics.set_ylim(0, 1.0)
ax_metrics.grid(axis='y', linestyle='--', alpha=0.7)

# Set overall title
fig.suptitle("Comprehensive Model Analysis for Lesion Boundary Segmentation", fontsize=18, y=0.98)

# Adjust layout
fig.tight_layout()
fig.subplots_adjust(top=0.9)

# Save the figure
plt.savefig(output_dir / "fig5_combined_analysis.png", dpi=300, bbox_inches='tight')
plt.savefig(output_dir / "fig5_combined_analysis.pdf", format='pdf', bbox_inches='tight')

print(f"All figures successfully saved to {output_dir}!")
print("The following figures were created:")
print("  1. fig1_performance_vs_parameters.png/pdf - Bubble chart showing Dice vs Parameters")
print("  2. fig2_model_metrics_radar.png/pdf - Radar chart of performance metrics")
print("  3. fig3_efficiency_comparison.png/pdf - Bar chart comparing efficiency metrics")
print("  4. fig4_performance_vs_memory.png/pdf - Bubble chart showing Dice vs GPU Memory")
print("  5. fig5_combined_analysis.png/pdf - Comprehensive multi-panel visualization")
print("\nThese figures are ready for publication in your paper!")