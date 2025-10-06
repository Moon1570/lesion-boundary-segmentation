# Development Directory

This directory contains development utilities, debugging scripts, and maintenance tools used during the development of the lesion segmentation project.

## Structure

- **debug/** - Debugging scripts and diagnostic tools
  - Various debug_*.py scripts for troubleshooting data loading, preprocessing, and model issues

- **utilities/** - Development utilities and analysis tools
  - Analysis scripts (analyze_*, compare_*, eda_*)
  - Monitoring and inspection tools (monitor_*, inspect_*, check_*)
  - Setup and optimization utilities (setup_*, optimize_*)
  - TensorBoard utilities

- **maintenance/** - Code maintenance and fix scripts
  - Scripts for fixing issues discovered during development
  - Cleanup utilities for checkpoints and data
  - Bug fixes and corrections

## Usage

These scripts are primarily intended for development and debugging purposes. They are not part of the core training/inference pipeline but were used during development to identify and resolve issues.

Most scripts in this directory can be run standalone for specific debugging or analysis tasks.