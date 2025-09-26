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

# Create a nice color palette
colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c']

# Create the figure with higher resolution and better size
plt.figure(figsize=(10, 7), dpi=300)

# Create the scatter plot with improved aesthetics
# Scale memory bubbles for better visibility (sqrt scaling makes area proportional)
memory_sizes = [m * 100 for m in memory]  

# Main bubble scatter plot
scatter = plt.scatter(params, dice, s=memory_sizes, c=colors, alpha=0.7, edgecolors='white', linewidths=1.5)

# Add model name labels with improved positioning and readability
label_offsets = {
    "DuaSkinSeg": (-2, 0.1),
    "Lightweight DuaSkinSeg": (0, 0.2),
    "Custom U-Net": (0, -0.2),
    "MONAI U-Net": (0, 0.2),
    "UNetMamba": (0, -0.2),
    "Enhanced Ensemble": (-5, 0.1)
}

for i, name in enumerate(names):
    offset_x, offset_y = label_offsets.get(name, (0, 0))
    txt = plt.annotate(name, 
                      (params[i] + offset_x, dice[i] + offset_y),
                      fontsize=10, fontweight='bold',
                      ha='center')
    # Add subtle outline to labels for better readability
    txt.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='white')])

# Add efficiency scores with careful positioning to avoid overlap
for i, score in enumerate(efficiency):
    eff_x = params[i]
    eff_y = dice[i] - 0.5  # Position below the bubble
    
    # Special cases to avoid overlaps
    if names[i] == "Lightweight DuaSkinSeg":
        eff_x += 2
    elif names[i] == "DuaSkinSeg":
        eff_x -= 2
    elif names[i] == "Enhanced Ensemble":
        eff_x -= 5
        
    plt.text(eff_x, eff_y, f"Eff: {score}/10", fontsize=9, 
             ha='center', va='top', 
             bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.3'))

# Create a custom legend for the memory usage
memory_sizes_legend = [3, 5, 7]
memory_labels = [f"{size} GB" for size in memory_sizes_legend]

# Create legend elements
legend_elements = []
for i, size in enumerate(memory_sizes_legend):
    legend_elements.append(
        plt.scatter([], [], s=size*100, c='gray', alpha=0.7, 
                   edgecolors='white', linewidths=1)
    )

# Add the legend with better placement
legend1 = plt.legend(legend_elements, memory_labels, 
                    title="GPU Memory", 
                    loc='upper right', 
                    title_fontsize=11,
                    frameon=True,
                    labelspacing=1.5,
                    handletextpad=2)

# Add legend for model colors
legend_patches = [Patch(facecolor=colors[i], edgecolor='white', 
                       label=names[i]) for i in range(len(names))]
legend2 = plt.legend(handles=legend_patches, 
                    loc='lower right', 
                    frameon=True,
                    fontsize=9)

# Add the first legend back after adding the second one
plt.gca().add_artist(legend1)

# Labels and title with improved styling
plt.xlabel('Parameters (Million)', fontsize=14, fontweight='bold')
plt.ylabel('Dice Coefficient (%)', fontsize=14, fontweight='bold')
plt.title('Performance vs. Model Size Trade-off', fontsize=16, fontweight='bold')

# Set axis ranges with some padding for readability
plt.xlim(-0.5, 75)
plt.ylim(81, 89)

# Use log scale for x-axis to better visualize the parameter differences
plt.xscale('log')

# Add grid for better readability
plt.grid(True, linestyle='--', alpha=0.7)

# Add subtle box around plot area
plt.box(True)

# Add a horizontal dashed line at the highest performance for reference
plt.axhline(y=max(dice), color='gray', linestyle='--', alpha=0.5)

# Customize tick labels to be more readable
plt.xticks(fontsize=11)
plt.yticks(fontsize=11)

# Tight layout to maximize figure size within canvas
plt.tight_layout()

# Save the figure with high quality
plt.savefig("paper_figures/keep/figure1_performance_vs_parameters.png", dpi=300, bbox_inches='tight')
plt.savefig("paper_figures/keep/figure1_performance_vs_parameters.pdf", bbox_inches='tight')

print("Figure 1 has been generated and saved to paper_figures/keep/")