#!/usr/bin/env python3
"""
Quick DuaSkinSeg Fine-tuning Script

This script fine-tunes the existing DuaSkinSeg model using your current training infrastructure.
Baseline: 0.8785 Dice Score
Target: Improve by 0.5-1% through advanced fine-tuning techniques.
"""

import os
import json
import shutil
from datetime import datetime

def create_finetune_experiment():
    """Create fine-tuning experiment based on existing DuaSkinSeg training."""
    
    # Create timestamp for experiment
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    experiment_name = f"duaskinseg_finetune_{timestamp}"
    
    print(f"🚀 Creating fine-tuning experiment: {experiment_name}")
    print(f"📊 Baseline DuaSkinSeg Dice Score: 0.8785")
    print(f"🎯 Target: Improve by 0.5-1.0% through fine-tuning")
    
    # Create experiment directory
    exp_dir = f"runs/{experiment_name}"
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(f"{exp_dir}/checkpoints", exist_ok=True)
    os.makedirs(f"{exp_dir}/logs", exist_ok=True)
    
    # Load base DuaSkinSeg configuration
    base_config_path = "configs/duaskinseg_advanced.json"
    with open(base_config_path, 'r') as f:
        base_config = json.load(f)
    
    # Create fine-tuning configuration with optimizations
    finetune_config = base_config.copy()
    
    # Fine-tuning specific modifications
    finetune_config.update({
        "experiment_name": experiment_name,
        "base_model_path": "runs/duaskinseg_advanced/checkpoints/best_model_20250902_211024_dice_0.8785.pth",
        "baseline_dice": 0.8785,
        
        # Reduced learning rate for fine-tuning
        "optimizer": {
            "name": "adamw",
            "lr": 2e-5,  # Much lower LR for fine-tuning
            "weight_decay": 0.005,  # Reduced weight decay
            "betas": [0.9, 0.999]
        },
        
        # Smaller batch size for stability
        "data": {
            **base_config["data"],
            "batch_size": 2,  # Smaller batch for fine-tuning
            "enhanced_augmentation": True
        },
        
        # Enhanced loss function
        "loss": {
            "name": "enhanced_combined",
            "weights": {
                "bce": 0.25,
                "dice": 0.35,
                "focal": 0.20,
                "boundary": 0.20
            },
            "focal_alpha": 0.75,
            "focal_gamma": 2.5,
            "boundary_weight": 4.0,
            "sigmoid": True
        },
        
        # Cosine annealing with warm restarts
        "scheduler": {
            "name": "cosine_restart",
            "T_0": 10,
            "T_mult": 2,
            "eta_min": 1e-8
        },
        
        # Reduced epochs with early stopping
        "training": {
            "epochs": 40,
            "use_amp": True,
            "gradient_clip_val": 0.5
        },
        
        # More sensitive early stopping
        "callbacks": {
            "early_stopping": {
                "patience": 12,
                "min_delta": 0.0001,
                "monitor": "val_dice"
            },
            "reduce_lr_on_plateau": {
                "factor": 0.5,
                "patience": 5,
                "min_lr": 1e-8
            }
        },
        
        # Fine-tuning strategy
        "finetune_strategy": {
            "freeze_encoder_epochs": 5,  # Freeze encoders for first 5 epochs
            "layer_wise_lr": {
                "mobilenet_encoder": 5e-6,
                "vit_encoder": 1e-5, 
                "decoder": 2e-5,
                "fusion": 3e-5
            }
        }
    })
    
    # Save fine-tuning configuration
    config_path = f"{exp_dir}/finetune_config.json"
    with open(config_path, 'w') as f:
        json.dump(finetune_config, f, indent=2)
    
    print(f"💾 Fine-tuning config saved: {config_path}")
    
    # Create training script
    training_script = f"""#!/usr/bin/env python3
'''
DuaSkinSeg Fine-tuning Training Script
Generated automatically for experiment: {experiment_name}
'''

import os
import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train import main as train_main

def load_pretrained_model(model, checkpoint_path):
    '''Load pretrained DuaSkinSeg weights.'''
    print(f"Loading pretrained weights from: {{checkpoint_path}}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    model.load_state_dict(state_dict, strict=False)
    print("Pretrained weights loaded successfully")
    return model

def setup_layerwise_optimizer(model, config):
    '''Setup optimizer with layer-wise learning rates.'''
    base_lr = config['optimizer']['lr']
    layerwise_config = config.get('finetune_strategy', {{}}).get('layer_wise_lr', {{}})
    
    param_groups = []
    
    # MobileNet encoder parameters (lower LR)
    mobilenet_params = [p for name, p in model.named_parameters() if 'mobilenet_encoder' in name]
    if mobilenet_params:
        param_groups.append({{
            'params': mobilenet_params,
            'lr': layerwise_config.get('mobilenet_encoder', base_lr * 0.1)
        }})
    
    # ViT encoder parameters (medium LR) 
    vit_params = [p for name, p in model.named_parameters() if 'vit_encoder' in name]
    if vit_params:
        param_groups.append({{
            'params': vit_params,
            'lr': layerwise_config.get('vit_encoder', base_lr * 0.5)
        }})
    
    # Decoder and fusion parameters (full LR)
    other_params = [p for name, p in model.named_parameters() 
                   if not any(enc in name for enc in ['mobilenet_encoder', 'vit_encoder'])]
    if other_params:
        param_groups.append({{
            'params': other_params,
            'lr': layerwise_config.get('decoder', base_lr)
        }})
    
    optimizer = optim.AdamW(
        param_groups,
        weight_decay=config['optimizer']['weight_decay'],
        betas=config['optimizer']['betas']
    )
    
    return optimizer

if __name__ == '__main__':
    print("🚀 Starting DuaSkinSeg Fine-tuning")
    print(f"📊 Baseline Dice Score: 0.8785")
    print(f"🎯 Target Improvement: +0.5-1.0%")
    print("-" * 50)
    
    # Load configuration
    config_path = 'finetune_config.json'
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Modify training configuration for fine-tuning
    original_train = train_main
    
    def finetune_main():
        # Set up fine-tuning specific parameters
        os.environ['EXPERIMENT_NAME'] = config['experiment_name']
        os.environ['FINETUNE_MODE'] = '1'
        os.environ['PRETRAINED_PATH'] = config['base_model_path']
        os.environ['BASELINE_DICE'] = str(config['baseline_dice'])
        
        # Run training with fine-tuning config
        return original_train()
    
    # Start fine-tuning
    try:
        result = finetune_main()
        print("Fine-tuning completed successfully!")
        print(f"Check results in: runs/{experiment_name}/")
    except Exception as e:
        print(f"❌ Fine-tuning failed: {{e}}")
        raise
"""
    
    # Save training script
    script_path = f"{exp_dir}/run_finetune.py"
    with open(script_path, 'w', encoding='utf-8') as f:
        f.write(training_script)
    
    print(f"🐍 Training script created: {script_path}")
    
    # Create run command
    run_command = f"""# DuaSkinSeg Fine-tuning Commands

## Method 1: Direct fine-tuning (Recommended)
cd runs/{experiment_name}
python run_finetune.py

## Method 2: Using main training script
python finetune_duaskinseg_simple.py --config configs/finetune_duaskinseg_advanced.json

## Method 3: Manual training with config
python train.py --config runs/{experiment_name}/finetune_config.json --finetune --pretrained runs/duaskinseg_advanced/checkpoints/best_model_20250902_211024_dice_0.8785.pth

## Expected Results:
# Baseline DuaSkinSeg: 0.8785 Dice
# Target Fine-tuned:   0.8835-0.8855 Dice (+0.5-0.7%)
# Best Case Scenario:  0.8880+ Dice (+1.0%+)

## Fine-tuning Strategy:
# 1. Layer-wise learning rates (encoders: low, decoders: high)
# 2. Progressive unfreezing (freeze encoders initially)
# 3. Enhanced data augmentation
# 4. Advanced loss combination (BCE + Dice + Focal + Boundary)
# 5. Cosine annealing with warm restarts
# 6. Gradient clipping for stability
"""
    
    readme_path = f"{exp_dir}/README.md"
    with open(readme_path, 'w') as f:
        f.write(run_command)
    
    print(f"📖 Instructions saved: {readme_path}")
    
    # Summary
    print(f"\\n✅ Fine-tuning experiment created successfully!")
    print(f"📁 Experiment directory: runs/{experiment_name}/")
    print(f"⚙️  Configuration: {config_path}")
    print(f"🐍 Training script: {script_path}")
    print(f"📖 Instructions: {readme_path}")
    
    print(f"\\n🚀 To start fine-tuning, run:")
    print(f"   cd runs/{experiment_name}")
    print(f"   python run_finetune.py")
    
    print(f"\\n📊 Expected Results:")
    print(f"   Baseline: 0.8785 Dice Score")
    print(f"   Target:   0.8835-0.8855 Dice Score (+0.5-0.7%)")
    print(f"   Optimistic: 0.8880+ Dice Score (+1.0%+)")
    
    return experiment_name

if __name__ == '__main__':
    experiment_name = create_finetune_experiment()
    
    # Ask user if they want to start training immediately
    response = input(f"\\n🚀 Start fine-tuning immediately? (y/n): ").lower().strip()
    
    if response == 'y':
        import subprocess
        import os
        
        exp_dir = f"runs/{experiment_name}"
        print(f"\\n🎯 Starting fine-tuning in {exp_dir}...")
        
        try:
            # Change to experiment directory and run training
            os.chdir(exp_dir)
            result = subprocess.run(['python', 'run_finetune.py'], check=True, capture_output=True, text=True)
            
            print("✅ Fine-tuning completed!")
            print("📊 Check results in the experiment directory")
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Fine-tuning failed: {e}")
            print("💡 Try running manually:")
            print(f"   cd {exp_dir}")
            print(f"   python run_finetune.py")
    else:
        print(f"\\n💡 To start fine-tuning later, run:")
        print(f"   cd runs/{experiment_name}")
        print(f"   python run_finetune.py")