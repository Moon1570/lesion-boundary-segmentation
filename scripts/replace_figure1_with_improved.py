#!/usr/bin/env python3
"""
Replace the original Figure 1 files with the improved versions.
"""

import os
import shutil

print("Replacing original Figure 1 with improved version...")

# Define the source and destination file paths
source_png = "paper_figures/keep/figure1_performance_vs_parameters_improved.png"
source_pdf = "paper_figures/keep/figure1_performance_vs_parameters_improved.pdf"
source_tiff = "paper_figures/keep/figure1_performance_vs_parameters_improved.tiff"
dest_png = "paper_figures/keep/figure1_performance_vs_parameters.png"
dest_pdf = "paper_figures/keep/figure1_performance_vs_parameters.pdf"
dest_tiff = "paper_figures/keep/figure1_performance_vs_parameters.tiff"

# Backup the original files (optional)
backup_dir = "paper_figures/keep/backup"
os.makedirs(backup_dir, exist_ok=True)
if os.path.exists(dest_png):
    shutil.copy2(dest_png, os.path.join(backup_dir, "figure1_performance_vs_parameters_original.png"))
if os.path.exists(dest_pdf):
    shutil.copy2(dest_pdf, os.path.join(backup_dir, "figure1_performance_vs_parameters_original.pdf"))
if os.path.exists(dest_tiff):
    shutil.copy2(dest_tiff, os.path.join(backup_dir, "figure1_performance_vs_parameters_original.tiff"))

# Replace the original files with the improved ones
shutil.copy2(source_png, dest_png)
shutil.copy2(source_pdf, dest_pdf)
if os.path.exists(source_tiff):
    shutil.copy2(source_tiff, dest_tiff)

print("Original Figure 1 files have been replaced with improved versions.")
print("Original files were backed up to paper_figures/keep/backup/ directory.")