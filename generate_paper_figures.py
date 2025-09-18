#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Generate high-quality figures and tables for lesion boundary segmentation research paper.
This script creates publication-quality visualizations based on the model comparison data.

Author: Moon1570
Date: 2023-11-18
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch
from matplotlib.gridspec import GridSpec
import matplotlib.ticker as mtick

# Set style for publication quality figures
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['figure.titlesize'] = 16
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['savefig.pad_inches'] = 0.1

# Create output directory if it doesn't exist
os.makedirs('paper_figures', exist_ok=True)

# Load model comparison data
with open('model_comparison_data.json', 'r') as f:
    data = json.load(f)

# Create mapping for display names (replacing underscores with spaces)
model_display_names = {
    'DuaSkinSeg': 'DuaSkinSeg',
    'Lightweight_DuaSkinSeg': 'Lightweight DuaSkinSeg',
    'Enhanced_Ensemble': 'Enhanced Ensemble',
    'Attention_U-Net': 'Attention U-Net',
    'Custom_U-Net': 'Custom U-Net',
    'MONAI_U-Net': 'MONAI U-Net',
    'UNetMamba': 'UNetMamba'  # In case this is added
}


def create_performance_vs_parameters_figure():
    """Create a figure showing model performance vs. parameter count."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Get models from data
    models = list(data['models'].keys())
    
    # Extract metric data
    dice_scores = [data["models"][m]["performance_metrics"]["dice"] for m in models]
    param_counts = []
    for m in models:
        param = data["models"][m]["resource_metrics"]["parameters"]
        if param == "Combined":
            # For ensembles, use a reasonable value for visualization
            param_counts.append(100.0)  # 100M parameters as placeholder
        else:
            param_counts.append(float(param) / 1e6)
    gpu_memory = [data["models"][m]["resource_metrics"]["gpu_memory_gb"] for m in models]
    efficiency = [data["models"][m]["efficiency_metrics"]["efficiency_score"] for m in models]
    
    # Create custom colormap
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(models)))
    
    # Create scatter plot with size proportional to GPU memory
    scatter = ax.scatter(param_counts, dice_scores, s=[m*50 for m in gpu_memory], 
                c=efficiency, cmap='viridis', alpha=0.8, edgecolors='black', linewidths=1)
    
    # Add reference line for best performance
    ax.axhline(max(dice_scores), color='#ff7f0e', linestyle='--', alpha=0.5, 
               label=f"Best Dice Score: {max(dice_scores):.4f}")
    
    # Annotate each point with model name
    for i, model in enumerate(models):
        # Get display name
        display_name = model_display_names.get(model, model)
        
        # Customize annotation position for each model to prevent overlap
        if model == "DuaSkinSeg":
            xytext = (10, -25)
        elif model == "Lightweight_DuaSkinSeg":
            xytext = (-20, 15)
        elif model == "Enhanced_Ensemble":
            xytext = (10, 15)
        elif model == "MONAI_U-Net":
            xytext = (-20, -25)
        else:
            xytext = (10, 10)
            
        ax.annotate(display_name, 
                    (param_counts[i], dice_scores[i]),
                    xytext=xytext, 
                    textcoords='offset points',
                    fontsize=10,
                    bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.1'))
    
    # Add colorbar to show efficiency score
    cbar = plt.colorbar(scatter)
    cbar.set_label('Efficiency Score')
    
    # Add a legend for bubble size
    size_legend_values = [3, 5, 7]
    legend_elements = [
        Patch(facecolor='none', edgecolor='gray', 
              label=f'{s} GB GPU Memory') for s in size_legend_values
    ]
    size_legend = ax.legend(handles=legend_elements, 
                          loc='upper right',
                          title='Memory Usage',
                          frameon=True)
    ax.add_artist(size_legend)
    
    # Set labels and title
    ax.set_xlabel('Parameters (Millions)')
    ax.set_ylabel('Dice Score')
    ax.set_title('Model Performance vs. Parameter Count')
    
    # Format y-axis as percentage
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    
    # Adjust x-axis to log scale for better visualization of parameter differences
    ax.set_xscale('log')
    ax.set_xlim(left=max(0.5, min(param_counts)*0.5), right=max(param_counts)*1.5)
    
    # Set y-axis limits for better visualization
    y_min = min(dice_scores) * 0.95
    y_max = min(1.0, max(dice_scores) * 1.02)  # Cap at 1.0 since dice score can't exceed 1
    ax.set_ylim(y_min, y_max)
    
    # Save figure
    plt.tight_layout()
    plt.savefig('paper_figures/fig1_performance_vs_parameters.png')
    plt.savefig('paper_figures/fig1_performance_vs_parameters.pdf')
    plt.close()


def create_model_metrics_radar_chart():
    """Create radar chart comparing multiple metrics across models."""
    # Metrics to include in radar chart
    metrics = ['dice', 'iou', 'sensitivity', 'specificity', 'precision']
    
    # Get models from data
    models = list(data['models'].keys())
    
    # Select a subset of models to avoid overcrowding the chart (top 5)
    if len(models) > 5:
        # Sort models by dice score and take top 5
        model_dice = [(m, data["models"][m]["performance_metrics"]["dice"]) for m in models]
        model_dice.sort(key=lambda x: x[1], reverse=True)
        selected_models = [m[0] for m in model_dice[:5]]
    else:
        selected_models = models
    
    # Extract data
    metric_values = {
        model: [data["models"][model]["performance_metrics"][metric] for metric in metrics]
        for model in selected_models
    }
    
    # Create radar chart
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    
    # Calculate the angle for each metric
    angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # Close the polygon
    
    # Add metric labels
    metric_labels = [m.capitalize() for m in metrics]
    metric_labels += metric_labels[:1]  # Close the polygon
    
    # Set the labels at the correct angles
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_labels[:-1], fontsize=14)
    
    # Set y-axis limits
    ax.set_ylim(0.7, 1.0)
    ax.set_yticks([0.75, 0.8, 0.85, 0.9, 0.95, 1.0])
    ax.set_yticklabels(['0.75', '0.80', '0.85', '0.90', '0.95', '1.00'], fontsize=10)
    
    # Plot each model's metrics
    for i, model in enumerate(selected_models):
        # Get display name
        display_name = model_display_names.get(model, model)
        
        values = metric_values[model]
        values += values[:1]  # Close the polygon
        
        ax.plot(angles, values, 'o-', linewidth=2, label=display_name, markersize=6)
        ax.fill(angles, values, alpha=0.1)
    
    # Add legend
    ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1), fontsize=10)
    
    plt.title('Model Performance Metrics Comparison', fontsize=16, pad=20)
    plt.tight_layout()
    plt.savefig('paper_figures/fig2_model_metrics_radar.png')
    plt.savefig('paper_figures/fig2_model_metrics_radar.pdf')
    plt.close()


def create_efficiency_comparison_figure():
    """Create a figure comparing efficiency metrics across models."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    models = list(data["models"].keys())
    
    # Extract efficiency data
    efficiency_scores = [data["models"][m]["efficiency_metrics"]["efficiency_score"] for m in models]
    inference_times = [data["models"][m]["resource_metrics"]["inference_time_sec"] for m in models]
    dice_scores = [data["models"][m]["performance_metrics"]["dice"] for m in models]
    
    # Sort models by efficiency score
    sorted_indices = np.argsort(efficiency_scores)
    sorted_models = [models[i] for i in sorted_indices]
    sorted_efficiency = [efficiency_scores[i] for i in sorted_indices]
    sorted_inference = [inference_times[i] for i in sorted_indices]
    sorted_dice = [dice_scores[i] for i in sorted_indices]
    
    # Get display names for sorted models
    sorted_display_names = [model_display_names.get(m, m) for m in sorted_models]
    
    # Set up bar positions
    x = np.arange(len(models))
    width = 0.35
    
    # Create bars for efficiency score
    bars1 = ax.bar(x - width/2, sorted_efficiency, width, label='Efficiency Score', 
                  color=plt.cm.viridis(np.linspace(0.2, 0.8, len(sorted_efficiency))))
    
    # Add efficiency values on bars
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{height:.1f}', ha='center', va='bottom', fontsize=9)
    
    # Create secondary y-axis for inference time
    ax2 = ax.twinx()
    bars2 = ax2.bar(x + width/2, sorted_inference, width, label='Inference Time (s)', 
                   color=plt.cm.plasma(np.linspace(0.2, 0.8, len(sorted_inference))))
    
    # Add inference time values on bars
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{height:.2f}s', ha='center', va='bottom', fontsize=9)
    
    # Add dice score as text label at bottom
    for i, dice in enumerate(sorted_dice):
        ax.text(x[i], -0.5, f'Dice: {dice:.3f}', ha='center', va='top', 
                rotation=90, fontsize=9, color='darkblue')
    
    # Set labels and title
    ax.set_xlabel('Model')
    ax.set_ylabel('Efficiency Score')
    ax2.set_ylabel('Inference Time (seconds)')
    plt.title('Model Efficiency Comparison')
    
    # Set x-tick labels to model display names
    ax.set_xticks(x)
    ax.set_xticklabels(sorted_display_names, rotation=45, ha='right')
    
    # Add legends
    ax.legend(loc='upper left')
    ax2.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig('paper_figures/fig3_efficiency_comparison.png')
    plt.savefig('paper_figures/fig3_efficiency_comparison.pdf')
    plt.close()


def create_performance_vs_memory_figure():
    """Create a figure showing performance vs. memory usage."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    models = list(data["models"].keys())
    
    # Extract data
    dice_scores = [data["models"][m]["performance_metrics"]["dice"] for m in models]
    gpu_memory = [data["models"][m]["resource_metrics"]["gpu_memory_gb"] for m in models]
    model_sizes = [data["models"][m]["resource_metrics"]["model_size_mb"] for m in models]
    efficiency = [data["models"][m]["efficiency_metrics"]["efficiency_score"] for m in models]
    
    # Create scatter plot
    scatter = ax.scatter(gpu_memory, dice_scores, s=[size/2 for size in model_sizes], 
                        c=efficiency, cmap='plasma', alpha=0.8, edgecolors='black', linewidths=1)
    
    # Add reference lines
    ax.axhline(max(dice_scores), color='#ff7f0e', linestyle='--', alpha=0.5,
               label=f"Best Dice Score: {max(dice_scores):.4f}")
    
    # Annotate each point with model name
    for i, model in enumerate(models):
        # Get display name
        display_name = model_display_names.get(model, model)
        
        # Customize annotation position
        if model == "Enhanced_Ensemble":
            xytext = (10, -20)
        elif model == "DuaSkinSeg":
            xytext = (-20, 15)
        else:
            xytext = (10, 10)
            
        ax.annotate(display_name, 
                    (gpu_memory[i], dice_scores[i]),
                    xytext=xytext, 
                    textcoords='offset points',
                    fontsize=10,
                    bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.1'))
    
    # Add colorbar for efficiency score
    cbar = plt.colorbar(scatter)
    cbar.set_label('Efficiency Score')
    
    # Add legend for bubble size
    size_legend_values = [20, 100, 200]  # MB
    legend_elements = [
        Patch(facecolor='none', edgecolor='gray', 
              label=f'{s} MB Model Size') for s in size_legend_values
    ]
    size_legend = ax.legend(handles=legend_elements, 
                           loc='upper right', 
                           title='Model Size',
                           frameon=True)
    ax.add_artist(size_legend)
    
    # Set labels and title
    ax.set_xlabel('GPU Memory Usage (GB)')
    ax.set_ylabel('Dice Score')
    ax.set_title('Model Performance vs. GPU Memory Usage')
    
    # Format y-axis as percentage
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    
    # Set y-axis limits
    y_min = min(dice_scores) * 0.95
    y_max = min(1.0, max(dice_scores) * 1.02)
    ax.set_ylim(y_min, y_max)
    
    plt.tight_layout()
    plt.savefig('paper_figures/fig4_performance_vs_memory.png')
    plt.savefig('paper_figures/fig4_performance_vs_memory.pdf')
    plt.close()


def create_combined_analysis_figure():
    """Create a figure with combined analysis."""
    fig = plt.figure(figsize=(12, 9))
    gs = GridSpec(2, 2, figure=fig)
    
    models = list(data["models"].keys())
    
    # Extract metrics
    dice_scores = [data["models"][m]["performance_metrics"]["dice"] for m in models]
    iou_scores = [data["models"][m]["performance_metrics"]["iou"] for m in models]
    boundary_iou = [data["models"][m]["performance_metrics"]["boundary_iou"] for m in models]
    param_counts = []
    for m in models:
        param = data["models"][m]["resource_metrics"]["parameters"]
        if param == "Combined":
            param_counts.append(100.0)  # 100M parameters as placeholder for ensemble
        else:
            param_counts.append(float(param) / 1e6)
    gpu_memory = [data["models"][m]["resource_metrics"]["gpu_memory_gb"] for m in models]
    inference_time = [data["models"][m]["resource_metrics"]["inference_time_sec"] for m in models]
    efficiency = [data["models"][m]["efficiency_metrics"]["efficiency_score"] for m in models]
    
    # Plot 1: Dice vs IoU with Boundary IoU as color
    ax1 = fig.add_subplot(gs[0, 0])
    scatter1 = ax1.scatter(iou_scores, dice_scores, s=100, c=boundary_iou, 
                         cmap='YlGnBu', alpha=0.8, edgecolors='black', linewidths=1)
    
    # Annotate points
    for i, model in enumerate(models):
        # Get display name
        display_name = model_display_names.get(model, model)
        
        ax1.annotate(display_name, 
                    (iou_scores[i], dice_scores[i]),
                    xytext=(10, 5), 
                    textcoords='offset points',
                    fontsize=9,
                    bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7))
    
    ax1.set_xlabel('IoU Score')
    ax1.set_ylabel('Dice Score')
    ax1.set_title('Dice vs. IoU with Boundary IoU Intensity')
    cbar1 = plt.colorbar(scatter1, ax=ax1)
    cbar1.set_label('Boundary IoU')
    
    # Plot 2: Parameter count vs GPU memory with efficiency as color
    ax2 = fig.add_subplot(gs[0, 1])
    scatter2 = ax2.scatter(param_counts, gpu_memory, s=80, c=efficiency, 
                         cmap='viridis', alpha=0.8, edgecolors='black', linewidths=1)
    
    # Annotate points
    for i, model in enumerate(models):
        # Get display name
        display_name = model_display_names.get(model, model)
        
        ax2.annotate(display_name, 
                    (param_counts[i], gpu_memory[i]),
                    xytext=(10, 5), 
                    textcoords='offset points',
                    fontsize=9,
                    bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7))
    
    ax2.set_xlabel('Parameters (Millions)')
    ax2.set_ylabel('GPU Memory (GB)')
    ax2.set_title('Parameters vs. GPU Memory with Efficiency')
    ax2.set_xscale('log')
    cbar2 = plt.colorbar(scatter2, ax=ax2)
    cbar2.set_label('Efficiency Score')
    
    # Plot 3: Efficiency vs Dice Score with inference time as bubble size
    ax3 = fig.add_subplot(gs[1, 0])
    scatter3 = ax3.scatter(efficiency, dice_scores, s=[t*200 for t in inference_time], 
                         c=range(len(models)), cmap='tab10', alpha=0.7, edgecolors='black', linewidths=1)
    
    # Annotate points
    for i, model in enumerate(models):
        # Get display name
        display_name = model_display_names.get(model, model)
        
        ax3.annotate(display_name, 
                    (efficiency[i], dice_scores[i]),
                    xytext=(10, 5), 
                    textcoords='offset points',
                    fontsize=9,
                    bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7))
    
    ax3.set_xlabel('Efficiency Score')
    ax3.set_ylabel('Dice Score')
    ax3.set_title('Efficiency vs. Dice with Inference Time')
    
    # Add legend for bubble size
    size_legend_values = [0.1, 0.3, 0.5]  # seconds
    legend_elements = [
        Patch(facecolor='none', edgecolor='gray', 
              label=f'{s}s Inference') for s in size_legend_values
    ]
    ax3.legend(handles=legend_elements, loc='lower right', title='Inference Time')
    
    # Plot 4: Model Type Analysis - Bar chart of performance by architecture
    ax4 = fig.add_subplot(gs[1, 1])
    
    # Group models by type
    model_types = {
        'Transformer-Based': [m for m in models if m in ['DuaSkinSeg', 'Lightweight_DuaSkinSeg']],
        'Ensemble': [m for m in models if m in ['Enhanced_Ensemble']],
        'Attention-Based': [m for m in models if m in ['Attention_U-Net']],
        'Standard CNN': [m for m in models if m in ['Custom_U-Net', 'MONAI_U-Net']],
        'State-Space Models': [m for m in models if m in ['UNetMamba']]
    }
    
    # Filter out empty model types
    model_types = {t: models_list for t, models_list in model_types.items() if models_list}
    
    type_dice = {t: np.mean([data["models"][m]["performance_metrics"]["dice"] 
                           for m in models if m in models_in_type]) 
                for t, models_in_type in model_types.items()}
    
    type_efficiency = {t: np.mean([data["models"][m]["efficiency_metrics"]["efficiency_score"] 
                                 for m in models if m in models_in_type]) 
                      for t, models_in_type in model_types.items()}
    
    # Create grouped bar chart
    types = list(model_types.keys())
    dice_means = [type_dice[t] for t in types]
    efficiency_means = [type_efficiency[t] for t in types]
    
    x = np.arange(len(types))
    width = 0.35
    
    bars1 = ax4.bar(x - width/2, dice_means, width, label='Avg. Dice Score')
    bars2 = ax4.bar(x + width/2, efficiency_means, width, label='Avg. Efficiency')
    
    ax4.set_xlabel('Model Architecture Type')
    ax4.set_ylabel('Score')
    ax4.set_title('Performance by Architecture Type')
    ax4.set_xticks(x)
    ax4.set_xticklabels(types, rotation=45, ha='right')
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig('paper_figures/fig5_combined_analysis.png')
    plt.savefig('paper_figures/fig5_combined_analysis.pdf')
    plt.close()


def create_model_comparison_table():
    """Create a comprehensive model comparison table."""
    # Create DataFrame for the table
    df = pd.DataFrame(columns=['Model', 'Dice', 'IoU', 'Boundary IoU', 'Precision', 
                              'Sensitivity', 'Specificity', 'Parameters', 
                              'Size (MB)', 'GPU Memory (GB)', 'Inference (s)', 
                              'FLOPs', 'Efficiency Score'])
    
    models = list(data["models"].keys())
    
    # Fill the DataFrame
    for i, model in enumerate(models):
        model_data = data["models"][model]
        display_name = model_display_names.get(model, model)
        
        df.loc[i] = [
            display_name,
            model_data["performance_metrics"]["dice"],
            model_data["performance_metrics"]["iou"],
            model_data["performance_metrics"]["boundary_iou"],
            model_data["performance_metrics"]["precision"],
            model_data["performance_metrics"]["sensitivity"],
            model_data["performance_metrics"]["specificity"],
            model_data["resource_metrics"]["parameters_readable"],
            model_data["resource_metrics"]["model_size_mb"],
            model_data["resource_metrics"]["gpu_memory_gb"],
            model_data["resource_metrics"]["inference_time_sec"],
            model_data["resource_metrics"]["flops_readable"],
            model_data["efficiency_metrics"]["efficiency_score"]
        ]
    
    # Sort by Dice score (descending)
    df = df.sort_values('Dice', ascending=False).reset_index(drop=True)
    
    # Save as Markdown
    with open('paper_figures/model_comparison_table.md', 'w') as f:
        f.write(df.to_markdown(index=False))
    
    # Save as LaTeX
    with open('paper_figures/model_comparison_table.tex', 'w') as f:
        f.write(df.to_latex(index=False, float_format="%.4f"))


def create_enhanced_visualization():
    """Create an enhanced performance vs parameters figure with advanced styling."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    models = list(data["models"].keys())
    
    # Extract metrics
    dice_scores = [data["models"][m]["performance_metrics"]["dice"] for m in models]
    param_counts = []
    for m in models:
        param = data["models"][m]["resource_metrics"]["parameters"]
        if param == "Combined":
            param_counts.append(100.0)  # 100M parameters as placeholder for ensemble
        else:
            param_counts.append(float(param) / 1e6)
    gpu_memory = [data["models"][m]["resource_metrics"]["gpu_memory_gb"] for m in models]
    efficiency = [data["models"][m]["efficiency_metrics"]["efficiency_score"] for m in models]
    
    # Handle flops that might be "Combined" for ensemble
    flops = []
    for m in models:
        flop_val = data["models"][m]["resource_metrics"]["flops"]
        if flop_val == "Combined":
            flops.append(120.0)  # 120G flops as placeholder
        else:
            flops.append(float(flop_val) / 1e9)
    
    # Create custom colormap
    colors = plt.cm.viridis(np.linspace(0, 1, 256))
    custom_cmap = LinearSegmentedColormap.from_list('custom_viridis', colors)
    
    # Create scatter plot with enhanced styling
    scatter = ax.scatter(param_counts, dice_scores, 
                       s=[m*100 for m in gpu_memory],  # Size based on GPU memory
                       c=efficiency,  # Color based on efficiency
                       cmap=custom_cmap, 
                       alpha=0.85, 
                       edgecolors='white',
                       linewidths=1.5,
                       zorder=10)
    
    # Add connecting lines between points with opacity based on parameter difference
    for i in range(len(models)):
        for j in range(i+1, len(models)):
            # Calculate opacity based on parameter similarity
            param_diff = abs(np.log10(param_counts[i]) - np.log10(param_counts[j]))
            opacity = max(0.05, min(0.3, 1.0 - param_diff/3))
            
            ax.plot([param_counts[i], param_counts[j]], 
                   [dice_scores[i], dice_scores[j]], 
                   'o-', color='gray', alpha=opacity, linewidth=1.0, 
                   markersize=0, zorder=1)
    
    # Add best performance reference line with gradient fill
    y_max = max(dice_scores)
    ax.axhline(y_max, color='#ff7f0e', linestyle='-', linewidth=2, alpha=0.6,
              label=f"Best Dice Score: {y_max:.4f}")
    
    # Fill area between best score and axis
    x_min, x_max = ax.get_xlim()
    ax.fill_between([x_min, x_max], y_max, y_max*0.95, color='#ff7f0e', alpha=0.05)
    
    # Add annotations with enhanced styling
    for i, model in enumerate(models):
        # Get display name
        display_name = model_display_names.get(model, model)
        
        # Customize annotation position for each model to prevent overlap
        if model == "DuaSkinSeg":
            xytext = (10, -30)
        elif model == "Lightweight_DuaSkinSeg":
            xytext = (-90, 15)
        elif model == "Enhanced_Ensemble":
            xytext = (15, 25)
        elif model == "MONAI_U-Net":
            xytext = (-40, -40)
        else:
            xytext = (10, 15)
            
        # Add FLOPs info to annotation
        ax.annotate(f"{display_name}\n{efficiency[i]:.1f} efficiency\n{flops[i]:.1f}G FLOPs", 
                   xy=(param_counts[i], dice_scores[i]),
                   xytext=xytext, 
                   textcoords='offset points',
                   fontsize=9,
                   bbox=dict(boxstyle='round,pad=0.4', fc='white', ec='gray', alpha=0.9),
                   arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.2', 
                                  color='gray', alpha=0.7))
    
    # Add colorbar with enhanced styling
    cbar = plt.colorbar(scatter, ax=ax, pad=0.02)
    cbar.set_label('Efficiency Score', fontsize=14)
    cbar.ax.tick_params(labelsize=10)
    
    # Add a legend for bubble size
    size_legend_values = [2, 5, 8]
    legend_elements = [
        plt.scatter([], [], s=s*100, fc='gray', ec='white', alpha=0.5,
                   label=f'{s} GB') for s in size_legend_values
    ]
    ax.legend(handles=legend_elements, 
             loc='upper right',
             title='GPU Memory',
             frameon=True,
             fontsize=10)
    
    # Set grid and background styling
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_facecolor('#f8f9fa')
    
    # Set labels and title with enhanced styling
    ax.set_xlabel('Model Parameters (Millions)', fontsize=14)
    ax.set_ylabel('Dice Score', fontsize=14)
    ax.set_title('Enhanced Performance vs. Resource Analysis', fontsize=16, fontweight='bold')
    
    # Format y-axis as percentage
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    
    # Use log scale for x-axis and add grid lines
    ax.set_xscale('log')
    ax.set_xlim(left=max(0.5, min(param_counts)*0.5), right=max(param_counts)*1.5)
    
    # Set y-axis limits with some padding
    y_min = max(0.7, min(dice_scores) * 0.95)
    y_max = min(1.0, max(dice_scores) * 1.02)
    ax.set_ylim(y_min, y_max)
    
    # Add a text box with key observations
    observations = (
        "Key Observations:\n"
        "• Lightweight DuaSkinSeg offers best efficiency\n"
        "• DuaSkinSeg achieves highest dice score\n"
        "• Small models (<10M params) show strong performance\n"
        "• Enhanced Ensemble trades efficiency for robustness"
    )
    ax.text(0.02, 0.02, observations, transform=ax.transAxes, fontsize=10,
           bbox=dict(boxstyle='round,pad=0.5', fc='#f0f0f0', ec='gray', alpha=0.9))
    
    # Save high-quality figure
    plt.tight_layout()
    plt.savefig('paper_figures/fig1_enhanced_performance_vs_parameters.png')
    plt.savefig('paper_figures/fig1_enhanced_performance_vs_parameters.pdf')
    plt.close()


if __name__ == "__main__":
    print("Generating paper figures and tables...")
    
    # Generate all figures
    create_performance_vs_parameters_figure()
    create_model_metrics_radar_chart()
    create_efficiency_comparison_figure()
    create_performance_vs_memory_figure()
    create_combined_analysis_figure()
    create_enhanced_visualization()
    
    # Generate tables
    create_model_comparison_table()
    
    print("All figures and tables generated successfully!")