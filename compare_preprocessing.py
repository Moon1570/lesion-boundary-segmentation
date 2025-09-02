#!/usr/bin/env python3
"""
Model Comparison Script for Original vs Advanced Preprocessed Data

This script trains multiple models on both original and advanced preprocessed data
to determine which preprocessing approach yields better results.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from scripts.dataset import create_data_loaders, ISIC2018Dataset
from train import LesionSegmentationTrainer
from models.unet import create_unet
from models.unetmamba_efficient import create_unetmamba
from models.enhanced_unet import EnhancedUNet


class ModelComparator:
    """
    Compare different models on original vs advanced preprocessed data.
    """
    
    def __init__(self, output_dir: str = "runs/preprocessing_comparison"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.results = []
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
    
    def create_model(self, model_name: str) -> nn.Module:
        """Create model by name."""
        if model_name == "unet":
            return create_unet()
        elif model_name == "unetmamba":
            return create_unetmamba()
        elif model_name == "enhanced_unet":
            return EnhancedUNet()
        else:
            raise ValueError(f"Unknown model: {model_name}")
    
    def train_model(self, model_name: str, data_dir: str, 
                   epochs: int = 30, batch_size: int = 8) -> dict:
        """
        Train a model on specified dataset.
        
        Args:
            model_name: Name of the model to train
            data_dir: Directory containing the dataset
            epochs: Number of epochs to train
            batch_size: Batch size for training
            
        Returns:
            Dictionary with training results
        """
        print(f"\n🚀 Training {model_name} on {data_dir}")
        print("=" * 60)
        
        try:
            # Create data loaders
            train_loader, val_loader, _ = create_data_loaders(
                data_dir=data_dir,
                batch_size=batch_size,
                num_workers=4
            )
            
            # Create model
            model = self.create_model(model_name)
            model = model.to(self.device)
            
            # Create unique run directory
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            dataset_name = Path(data_dir).name
            run_name = f"{model_name}_{dataset_name}_{timestamp}"
            run_dir = self.output_dir / run_name
            run_dir.mkdir(parents=True, exist_ok=True)
            
            # Create trainer
            trainer = LesionSegmentationTrainer(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                device=self.device,
                output_dir=str(run_dir)
            )
            
            # Train model
            best_dice, training_time = trainer.train(epochs=epochs)
            
            # Get final validation metrics
            final_metrics = trainer.validate()
            
            result = {
                'model': model_name,
                'dataset': dataset_name,
                'best_dice': best_dice,
                'final_dice': final_metrics['dice'],
                'final_loss': final_metrics['loss'],
                'final_iou': final_metrics['iou'],
                'training_time': training_time,
                'epochs': epochs,
                'run_dir': str(run_dir),
                'timestamp': timestamp
            }
            
            print(f"✅ {model_name} on {dataset_name} completed!")
            print(f"   Best Dice: {best_dice:.4f}")
            print(f"   Final Metrics: Dice={final_metrics['dice']:.4f}, IoU={final_metrics['iou']:.4f}")
            print(f"   Training Time: {training_time:.2f} hours")
            
            return result
            
        except Exception as e:
            print(f"❌ Error training {model_name} on {dataset_name}: {e}")
            return {
                'model': model_name,
                'dataset': dataset_name,
                'error': str(e),
                'best_dice': 0.0,
                'final_dice': 0.0,
                'training_time': 0.0
            }
    
    def run_comparison(self, models: list, datasets: list, 
                      epochs: int = 30, batch_size: int = 8):
        """
        Run comprehensive comparison across models and datasets.
        
        Args:
            models: List of model names to test
            datasets: List of dataset directories
            epochs: Number of epochs per training
            batch_size: Batch size for training
        """
        print("🔬 Starting Comprehensive Model & Preprocessing Comparison")
        print("=" * 80)
        print(f"Models: {models}")
        print(f"Datasets: {datasets}")
        print(f"Epochs per run: {epochs}")
        print(f"Batch size: {batch_size}")
        print("=" * 80)
        
        total_runs = len(models) * len(datasets)
        current_run = 0
        
        for model_name in models:
            for dataset_dir in datasets:
                current_run += 1
                print(f"\n📊 Run {current_run}/{total_runs}")
                
                result = self.train_model(
                    model_name=model_name,
                    data_dir=dataset_dir,
                    epochs=epochs,
                    batch_size=batch_size
                )
                
                self.results.append(result)
                
                # Save intermediate results
                self.save_results()
        
        # Create final analysis
        self.analyze_results()
    
    def save_results(self):
        """Save results to JSON file."""
        results_file = self.output_dir / "comparison_results.json"
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"💾 Results saved to: {results_file}")
    
    def analyze_results(self):
        """Analyze and visualize results."""
        if not self.results:
            print("No results to analyze")
            return
        
        print("\n📈 Analyzing Results...")
        
        # Convert to DataFrame
        df = pd.DataFrame(self.results)
        
        # Remove failed runs
        df = df[df['best_dice'] > 0]
        
        if df.empty:
            print("No successful runs to analyze")
            return
        
        # Create summary table
        print("\n📊 Summary Results:")
        print("=" * 80)
        
        summary = df.groupby(['model', 'dataset']).agg({
            'best_dice': 'mean',
            'final_dice': 'mean',
            'final_iou': 'mean',
            'training_time': 'mean'
        }).round(4)
        
        print(summary)
        
        # Save detailed results
        df.to_csv(self.output_dir / "detailed_results.csv", index=False)
        summary.to_csv(self.output_dir / "summary_results.csv")
        
        # Create visualizations
        self.create_visualizations(df)
        
        # Find best performing combinations
        self.find_best_combinations(df)
    
    def create_visualizations(self, df):
        """Create comparison visualizations."""
        print("\n🎨 Creating visualizations...")
        
        # 1. Dice Score Comparison
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Best Dice by Model and Dataset
        pivot_dice = df.pivot_table(values='best_dice', index='model', columns='dataset', aggfunc='mean')
        pivot_dice.plot(kind='bar', ax=axes[0,0], title='Best Dice Score by Model and Dataset')
        axes[0,0].set_ylabel('Dice Score')
        axes[0,0].legend(title='Dataset')
        axes[0,0].tick_params(axis='x', rotation=45)
        
        # IoU Comparison
        pivot_iou = df.pivot_table(values='final_iou', index='model', columns='dataset', aggfunc='mean')
        pivot_iou.plot(kind='bar', ax=axes[0,1], title='IoU by Model and Dataset')
        axes[0,1].set_ylabel('IoU Score')
        axes[0,1].legend(title='Dataset')
        axes[0,1].tick_params(axis='x', rotation=45)
        
        # Training Time Comparison
        pivot_time = df.pivot_table(values='training_time', index='model', columns='dataset', aggfunc='mean')
        pivot_time.plot(kind='bar', ax=axes[1,0], title='Training Time by Model and Dataset')
        axes[1,0].set_ylabel('Training Time (hours)')
        axes[1,0].legend(title='Dataset')
        axes[1,0].tick_params(axis='x', rotation=45)
        
        # Dice vs Training Time Scatter
        for dataset in df['dataset'].unique():
            subset = df[df['dataset'] == dataset]
            axes[1,1].scatter(subset['training_time'], subset['best_dice'], 
                            label=dataset, alpha=0.7, s=100)
        
        axes[1,1].set_xlabel('Training Time (hours)')
        axes[1,1].set_ylabel('Best Dice Score')
        axes[1,1].set_title('Performance vs Training Time')
        axes[1,1].legend(title='Dataset')
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "comparison_visualizations.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Visualizations saved to: {self.output_dir / 'comparison_visualizations.png'}")
    
    def find_best_combinations(self, df):
        """Find and report best performing combinations."""
        print("\n🏆 Best Performing Combinations:")
        print("=" * 50)
        
        # Best overall dice score
        best_dice = df.loc[df['best_dice'].idxmax()]
        print(f"🥇 Highest Dice Score: {best_dice['best_dice']:.4f}")
        print(f"   Model: {best_dice['model']}")
        print(f"   Dataset: {best_dice['dataset']}")
        print(f"   Training Time: {best_dice['training_time']:.2f} hours")
        
        # Best efficiency (dice/time ratio)
        df['efficiency'] = df['best_dice'] / (df['training_time'] + 0.01)  # Avoid division by zero
        best_efficiency = df.loc[df['efficiency'].idxmax()]
        print(f"\n⚡ Best Efficiency: {best_efficiency['efficiency']:.3f} (Dice/Hour)")
        print(f"   Model: {best_efficiency['model']}")
        print(f"   Dataset: {best_efficiency['dataset']}")
        print(f"   Dice: {best_efficiency['best_dice']:.4f}")
        print(f"   Time: {best_efficiency['training_time']:.2f} hours")
        
        # Dataset comparison
        print(f"\n📈 Dataset Performance Comparison:")
        dataset_performance = df.groupby('dataset')['best_dice'].mean().sort_values(ascending=False)
        for dataset, avg_dice in dataset_performance.items():
            print(f"   {dataset}: {avg_dice:.4f} average Dice")
        
        # Model comparison
        print(f"\n🤖 Model Performance Comparison:")
        model_performance = df.groupby('model')['best_dice'].mean().sort_values(ascending=False)
        for model, avg_dice in model_performance.items():
            print(f"   {model}: {avg_dice:.4f} average Dice")


def main():
    """Main function for model comparison."""
    parser = argparse.ArgumentParser(description="Compare models on original vs advanced preprocessed data")
    parser.add_argument("--models", nargs='+', default=['unet', 'enhanced_unet'],
                      help="Models to test")
    parser.add_argument("--datasets", nargs='+', 
                      default=['data/ISIC2018_proc', 'data/ISIC2018_advanced'],
                      help="Dataset directories to test")
    parser.add_argument("--epochs", type=int, default=25,
                      help="Number of epochs per training")
    parser.add_argument("--batch-size", type=int, default=8,
                      help="Batch size for training")
    parser.add_argument("--output-dir", type=str, default="runs/preprocessing_comparison",
                      help="Output directory for results")
    
    args = parser.parse_args()
    
    # Verify datasets exist
    for dataset in args.datasets:
        if not Path(dataset).exists():
            print(f"❌ Dataset not found: {dataset}")
            return
    
    print("🔬 Model & Preprocessing Comparison Study")
    print("=" * 60)
    print(f"Models to test: {args.models}")
    print(f"Datasets to test: {args.datasets}")
    print(f"Training epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Output directory: {args.output_dir}")
    print("=" * 60)
    
    # Create comparator and run comparison
    comparator = ModelComparator(output_dir=args.output_dir)
    
    comparator.run_comparison(
        models=args.models,
        datasets=args.datasets,
        epochs=args.epochs,
        batch_size=args.batch_size
    )
    
    print("\n✅ Comparison study completed!")
    print(f"📁 Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
