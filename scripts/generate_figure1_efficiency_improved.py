#!/usr/bin/env python3
"""
Generate an improved Figure 1: Performance vs Parameters bubble chart for the paper.
This script creates a bubble chart showing model performance (Dice) vs parameters,
with bubble size representing GPU memory usage and clear, non-overlapping labels.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib.patheffects as PathEffects

# Create the output directory if it doesn't exist
os.makedirs("paper_figures/keep", exist_ok=True)

# Model data: name, dice, parameters (M), GPU memory (GB), efficiency score
model_data = [
    ("DuaSkinSeg", 87.85, 31.2, 6.8, 7.1),
    ("Lightweight DuaSkinSeg", 87.72, 8.4, 4.2, 9.2),
    ("Custom U-Net", 86.30, 4.3, 3.5, 8.8),
    ("MONAI U-Net", 84.50, 2.6, 2.8, 8.5),
    ("UNetMamba", 81.61, 2.4, 3.2, 7.8),
    ("Enhanced Ensemble", 87.53, 69.2, 8.0, 6.8),
]

# Extract data
names = [item[0] for item in model_data]
dice = [item[1] for item in model_data]
params = [item[2] for item in model_data]
memory = [item[3] for item in model_data]
efficiency = [item[4] for item in model_data]

# Create a nice color palette - using a more distinctive palette
colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c']

# Create the figure with higher resolution and better size
plt.figure(figsize=(12, 8), dpi=300)  # Increased figure size for better readability

# Create the scatter plot with improved aesthetics
# Use proper area scaling for memory bubbles (area should be proportional to value)
# Scale factor controls overall bubble size
scale_factor = 80
memory_sizes = [scale_factor * (m**2) for m in memory]

# Main bubble scatter plot with improved visibility
scatter = plt.scatter(params, dice, s=memory_sizes, c=colors, alpha=0.8, 
                     edgecolors='white', linewidths=2.0)

# Add model name labels with improved positioning and readability
label_offsets = {
    "DuaSkinSeg": (-5, 0.15),
    "Lightweight DuaSkinSeg": (0.5, 0.25),
    "Custom U-Net": (0, -0.3),
    "MONAI U-Net": (0, 0.25),
    "UNetMamba": (-0.2, -0.3),
    "Enhanced Ensemble": (-20, 0.15)
}

# Add model name labels
for i, name in enumerate(names):
    offset_x, offset_y = label_offsets.get(name, (0, 0))
    txt = plt.annotate(name, 
                      (params[i] + offset_x, dice[i] + offset_y),
                      fontsize=12, fontweight='bold',  # Increased font size
                      ha='center')
    # Add stronger outline to labels for better readability
    txt.set_path_effects([PathEffects.withStroke(linewidth=4, foreground='white')])

# Add efficiency scores with better positioning to avoid overlap
for i, score in enumerate(efficiency):
    eff_x = params[i]
    eff_y = dice[i] - 0.6  # Position further below the bubble
    
    # Special cases to avoid overlaps - improved positioning
    if names[i] == "Lightweight DuaSkinSeg":
        eff_x += 3
    elif names[i] == "DuaSkinSeg":
        eff_x -= 4
    elif names[i] == "Enhanced Ensemble":
        eff_x -= 15
    elif names[i] == "UNetMamba":
        eff_x -= 1
    elif names[i] == "MONAI U-Net":
        eff_x += 1
        
    plt.text(eff_x, eff_y, f"Efficiency: {score}/10", fontsize=10, 
             ha='center', va='top', fontweight='medium',
             bbox=dict(facecolor='white', alpha=0.9, boxstyle='round,pad=0.4',
                      edgecolor='gray', linewidth=0.5))

# Create a custom legend for the memory usage - using actual memory values
# Use min, median, max for better representation
memory_sizes_legend = [min(memory), np.median(memory), max(memory)]
memory_labels = [f"{size:.1f} GB" for size in memory_sizes_legend]

# Create legend elements for memory with proper size scaling
legend_elements = []
for size in memory_sizes_legend:
    legend_elements.append(
        plt.scatter([], [], s=scale_factor * (size**2), c='gray', alpha=0.7, 
                   edgecolors='white', linewidths=1.5)
    )

# Create two separate legends in the bottom right corner

# First legend for model colors (at the bottom)
model_patches = [Patch(facecolor=colors[i], edgecolor='white', linewidth=0.5, 
                      label=names[i]) for i in range(len(names))]
legend_models = plt.legend(handles=model_patches, 
                         title="Model Types",
                         loc='lower right',
                         bbox_to_anchor=(0.98, 0.12),  # Position higher than before
                         title_fontsize=11,
                         frameon=True,
                         fontsize=9,
                         ncol=2,  # Use 2 columns for models
                         framealpha=0.9)
legend_models.get_frame().set_edgecolor('gray')
legend_models.get_frame().set_linewidth(0.5)

# Add first legend to the plot
plt.gca().add_artist(legend_models)

# Second legend for GPU Memory (above the model legend)
legend_gpu = plt.legend(legend_elements, memory_labels, 
                      title="GPU Memory Usage", 
                      loc='lower right',
                      bbox_to_anchor=(0.98, 0.32),  # Position above model legend
                      title_fontsize=11,
                      frameon=True,
                      labelspacing=1.0,
                      handletextpad=2,
                      framealpha=0.9)
legend_gpu.get_frame().set_edgecolor('gray')
legend_gpu.get_frame().set_linewidth(0.5)

# Labels and title with improved styling
plt.xlabel('Model Parameters (Millions)', fontsize=14, fontweight='bold')
plt.ylabel('Dice Coefficient (%)', fontsize=14, fontweight='bold')
plt.title('Performance vs. Model Size Trade-off', fontsize=18, fontweight='bold')

# Set axis ranges with some padding for readability
plt.xlim(1.5, 100)  # Adjusted to start higher for better log scale
plt.ylim(81, 89)

# Use log scale for x-axis to better visualize the parameter differences
plt.xscale('log')

# Add grid for better readability
plt.grid(True, linestyle='--', alpha=0.5)

# Add subtle box around plot area
plt.box(True)

# Add a horizontal dashed line at the highest performance for reference
plt.axhline(y=max(dice), color='gray', linestyle='--', alpha=0.5)
plt.text(plt.xlim()[0]*1.1, max(dice) + 0.1, 'Best Performance', 
        fontsize=9, fontstyle='italic', color='gray')

# Customize tick labels
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# Format x-axis ticks to show actual values despite log scale
# Add custom ticks at specific parameter values
custom_ticks = [2, 5, 10, 20, 50, 70]
plt.xticks(custom_ticks, [f"{x}" for x in custom_ticks], fontsize=12)

# Add annotation for the best efficiency model
best_eff_idx = np.argmax(efficiency)
best_eff_name = names[best_eff_idx]
best_eff_score = efficiency[best_eff_idx]
plt.annotate(f"Highest Efficiency: {best_eff_score}/10",
            xy=(params[best_eff_idx], dice[best_eff_idx]),
            xytext=(params[best_eff_idx]*0.5, dice[best_eff_idx]-1.5),
            arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=7, alpha=0.7),
            fontsize=10, fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.9))

# Adjust the figure margins to ensure the legends are within the canvas
plt.subplots_adjust(right=0.72, bottom=0.12)

# Save the figure with high quality in multiple formats
plt.savefig("paper_figures/keep/figure1_performance_vs_parameters_improved.png", dpi=300, bbox_inches='tight', pad_inches=0.2)
plt.savefig("paper_figures/keep/figure1_performance_vs_parameters_improved.pdf", bbox_inches='tight', pad_inches=0.2)
# Save as TIFF format with high resolution
plt.savefig("paper_figures/keep/figure1_performance_vs_parameters_improved.tiff", dpi=600, format='tiff', bbox_inches='tight', pad_inches=0.2)

print("Improved Figure 1 has been generated and saved to paper_figures/keep/")