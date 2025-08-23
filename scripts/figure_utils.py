#!/usr/bin/env python3
"""
Figure management utilities for ISIC2018 dataset implementation.
Provides centralized figure saving and organization.

Author: GitHub Copilot
"""

import os
import shutil
from pathlib import Path
from typing import Optional
import matplotlib.pyplot as plt

def ensure_figs_dir(subdir: Optional[str] = None) -> Path:
    """
    Ensure runs/figs directory (and optional subdirectory) exists.
    
    Args:
        subdir: Optional subdirectory within runs/figs
        
    Returns:
        Path to the figures directory
    """
    base_dir = Path("runs/figs")
    
    if subdir:
        figs_dir = base_dir / subdir
    else:
        figs_dir = base_dir
    
    figs_dir.mkdir(parents=True, exist_ok=True)
    return figs_dir

def save_figure(filename: str, subdir: Optional[str] = None, dpi: int = 150, **kwargs):
    """
    Save current matplotlib figure to runs/figs directory.
    
    Args:
        filename: Name of the figure file (with extension)
        subdir: Optional subdirectory within runs/figs
        dpi: DPI for saved figure
        **kwargs: Additional arguments passed to plt.savefig
    """
    figs_dir = ensure_figs_dir(subdir)
    save_path = figs_dir / filename
    
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight', **kwargs)
    print(f"📊 Figure saved: {save_path}")
    return save_path

def get_figure_path(filename: str, subdir: Optional[str] = None) -> Path:
    """
    Get the full path for a figure file in runs/figs.
    
    Args:
        filename: Name of the figure file
        subdir: Optional subdirectory within runs/figs
        
    Returns:
        Full path to the figure file
    """
    figs_dir = ensure_figs_dir(subdir)
    return figs_dir / filename

def organize_existing_figures():
    """
    Move any existing figures from root directory to runs/figs.
    """
    root_dir = Path(".")
    figs_dir = ensure_figs_dir()
    
    # Common figure patterns to look for
    figure_patterns = [
        "*.png", "*.jpg", "*.jpeg", "*.pdf", "*.svg"
    ]
    
    moved_files = []
    
    for pattern in figure_patterns:
        for fig_file in root_dir.glob(pattern):
            # Skip if it's already in runs/ directory
            if "runs" in str(fig_file):
                continue
                
            # Skip if it's in data/ directory (these are dataset files)
            if "data" in str(fig_file):
                continue
            
            # Skip if it's a source file or not a figure
            skip_files = {
                "README.md", "requirements.txt", ".gitignore",
                "__pycache__", ".venv", ".git"
            }
            
            if any(skip in str(fig_file) for skip in skip_files):
                continue
            
            # Check if it looks like a figure based on filename
            figure_keywords = [
                "demo", "augmentation", "dataset", "visualization", 
                "plot", "graph", "chart", "sample", "statistics"
            ]
            
            if any(keyword in fig_file.stem.lower() for keyword in figure_keywords):
                dest_path = figs_dir / fig_file.name
                try:
                    shutil.move(str(fig_file), str(dest_path))
                    moved_files.append((fig_file.name, dest_path))
                    print(f"📁 Moved: {fig_file.name} -> {dest_path}")
                except Exception as e:
                    print(f"⚠️ Could not move {fig_file.name}: {e}")
    
    if moved_files:
        print(f"\n✅ Organized {len(moved_files)} existing figures into runs/figs/")
    else:
        print("ℹ️ No existing figures found to organize")
    
    return moved_files

def list_figures(subdir: Optional[str] = None) -> list:
    """
    List all figures in runs/figs directory.
    
    Args:
        subdir: Optional subdirectory to list
        
    Returns:
        List of figure file paths
    """
    figs_dir = ensure_figs_dir(subdir)
    
    if not figs_dir.exists():
        print(f"📁 Figures directory not found: {figs_dir}")
        return []
    
    # Find all image files
    image_extensions = ["*.png", "*.jpg", "*.jpeg", "*.pdf", "*.svg"]
    figure_files = []
    
    for ext in image_extensions:
        figure_files.extend(figs_dir.glob(ext))
    
    # Sort by name
    figure_files.sort()
    
    if figure_files:
        print(f"📊 Found {len(figure_files)} figures in {figs_dir}:")
        for fig_file in figure_files:
            file_size = fig_file.stat().st_size / 1024  # KB
            print(f"  📄 {fig_file.name} ({file_size:.1f} KB)")
    else:
        print(f"📁 No figures found in {figs_dir}")
    
    return figure_files

def clean_figures(subdir: Optional[str] = None, confirm: bool = True) -> int:
    """
    Clean (delete) all figures in runs/figs directory.
    
    Args:
        subdir: Optional subdirectory to clean
        confirm: Whether to ask for confirmation
        
    Returns:
        Number of files deleted
    """
    figure_files = list_figures(subdir)
    
    if not figure_files:
        return 0
    
    if confirm:
        response = input(f"🗑️ Delete {len(figure_files)} figures? (y/N): ")
        if response.lower() != 'y':
            print("❌ Cancelled")
            return 0
    
    deleted_count = 0
    for fig_file in figure_files:
        try:
            fig_file.unlink()
            deleted_count += 1
            print(f"🗑️ Deleted: {fig_file.name}")
        except Exception as e:
            print(f"⚠️ Could not delete {fig_file.name}: {e}")
    
    print(f"✅ Deleted {deleted_count} figures")
    return deleted_count

def create_index_html():
    """
    Create an HTML index file to view all figures in a browser.
    """
    figs_dir = ensure_figs_dir()
    figure_files = list_figures()
    
    if not figure_files:
        print("📁 No figures found to index")
        return
    
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>ISIC2018 Dataset Figures</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .figure {{ margin: 20px 0; border: 1px solid #ddd; padding: 10px; }}
        .figure img {{ max-width: 100%; height: auto; }}
        .figure h3 {{ margin-top: 0; color: #333; }}
        .stats {{ background: #f5f5f5; padding: 10px; margin: 10px 0; }}
    </style>
</head>
<body>
    <h1>🎯 ISIC2018 Dataset - Generated Figures</h1>
    <div class="stats">
        <strong>📊 Total Figures:</strong> {len(figure_files)}<br>
        <strong>📁 Directory:</strong> {figs_dir.absolute()}<br>
        <strong>🕒 Generated:</strong> Now
    </div>
"""
    
    for fig_file in figure_files:
        relative_path = fig_file.relative_to(figs_dir.parent.parent)  # Relative to project root
        file_size = fig_file.stat().st_size / 1024  # KB
        
        html_content += f"""
    <div class="figure">
        <h3>📄 {fig_file.name}</h3>
        <p><strong>Size:</strong> {file_size:.1f} KB</p>
        <img src="{relative_path}" alt="{fig_file.stem}">
    </div>
"""
    
    html_content += """
</body>
</html>
"""
    
    index_path = figs_dir / "index.html"
    with open(index_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"📄 HTML index created: {index_path}")
    print(f"🌐 Open in browser: file://{index_path.absolute()}")
    return index_path

def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Manage figures for ISIC2018 dataset")
    parser.add_argument("action", choices=["list", "organize", "clean", "index"], 
                       help="Action to perform")
    parser.add_argument("--subdir", help="Subdirectory within runs/figs")
    parser.add_argument("--no-confirm", action="store_true", 
                       help="Skip confirmation for destructive actions")
    
    args = parser.parse_args()
    
    if args.action == "list":
        list_figures(args.subdir)
    elif args.action == "organize":
        organize_existing_figures()
    elif args.action == "clean":
        clean_figures(args.subdir, confirm=not args.no_confirm)
    elif args.action == "index":
        create_index_html()

if __name__ == "__main__":
    main()
