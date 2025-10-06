# Visualization Guide for Lesion Boundary Segmentation Project

This document provides a guide to all the visualizations and tables generated for your skin lesion boundary segmentation research paper.

## Main Figures (in `paper_figures/`)

The main figures provide primary visualizations for your research paper, including:

1. **Performance vs Parameters**: Scatter plot showing the relationship between model performance (Dice scores) and model size (parameters)
2. **Radar Charts**: Multi-axis visualization of performance metrics across all models
3. **Efficiency Comparison**: Bar charts showing efficiency metrics across models
4. **Model Comparison Table**: Comprehensive table with all performance and resource metrics

## Additional Figures (in `paper_figures/additional_figures/`)

Additional specialized visualizations are available for supplementary material or specific sections:

### 1. Baseline Model Analysis
- **Baseline Results Table**: `baseline_results_table.md`/`.tex`
- **Baseline Results Figure**: `baseline_results_figure.png`/`.pdf`
  - Shows performance metrics and resource usage for standard U-Net variants

### 2. Ensemble Model Analysis
- **Ensemble Components Table**: `ensemble_components_table.md`
- **Ensemble Results Table**: `ensemble_results_table.md`/`.tex`
- **Ensemble Results Figure**: `ensemble_results_figure.png`/`.pdf`
  - Shows performance of the Enhanced Ensemble model compared to its components
  - Includes radar chart comparing ensemble with best-performing models

### 3. Resource Usage Analysis
- **Resource Usage Table**: `resource_usage_table.md`/`.tex`
- **Resource Tradeoffs Table**: `resource_tradeoffs_table.md`
- **Resource Tradeoffs Figure**: `resource_tradeoffs_figure.png`/`.pdf`
  - Bubble plot showing the tradeoff between inference time, GPU memory, and model size
  - Quadrant analysis for identifying optimal models based on resource constraints

### 4. Failure Mode Analysis
- **Failure Mode Table**: `failure_mode_table.md`
- **Failure Mode Analysis Figure**: `failure_mode_analysis.png`/`.pdf`
  - Qualitative comparison of how models handle different failure scenarios
  - Rating scale: Very Good, Good, Fair, Poor based on performance metrics

### 5. Comprehensive Comparisons
- **Comprehensive Comparison Table**: `comprehensive_comparison_table.md`/`.tex`
  - Complete table with all models and all metrics
- **Model Comparison Figures**: `model_comparison.png`/`.pdf`
  - Multi-plot figure showing:
    - Dice score comparison across all models
    - IoU vs. Boundary IoU scatter plot
    - Sensitivity vs. Specificity plot (colored by Dice score)
    - Performance vs. Model Size log plot with trend line

## Using the Visualizations

### For Publication
- **PDF Formats**: All figures are provided in PDF format for high-quality publication
- **LaTeX Tables**: `.tex` files can be directly included in LaTeX documents
- **Markdown Tables**: `.md` files are available for easy preview and inclusion in documentation

### For Presentation
- **PNG Formats**: All figures are provided in PNG format for presentations
- **High Resolution**: All images are generated at 300 DPI for clear display

### Key Findings Highlighted

1. **DuaSkinSeg Performance**: The DuaSkinSeg model achieves the highest Dice score (0.8785) and Boundary IoU (0.1512)
2. **Lightweight Model Efficiency**: The Lightweight DuaSkinSeg provides the best efficiency score (9.2/10) with only a minimal performance drop
3. **Ensemble Model**: The Enhanced Ensemble provides a 0.36% improvement over the best component model
4. **Resource Tradeoffs**: Clear visualization of how different models balance performance vs. resource usage
5. **Failure Modes**: Qualitative analysis shows that DuaSkinSeg and Lightweight DuaSkinSeg handle irregular boundaries and artifacts significantly better than other models

## Scripts

Two Python scripts were created to generate all these visualizations:

1. `generate_paper_figures.py`: Creates main publication figures
2. `generate_additional_figures.py`: Creates specialized supplementary visualizations

Both scripts read data from `model_comparison_data.json` and generate figures with publication-quality styling using matplotlib and seaborn.

## Future Visualization Enhancements

Consider these potential enhancements for future work:

1. **Interactive visualizations**: Convert key figures to interactive web-based versions
2. **Sample image comparisons**: Add visual examples of model outputs on challenging cases
3. **Ablation study visualizations**: Create figures showing impact of different components
4. **Training curve comparisons**: Add visualizations of training progress for different models