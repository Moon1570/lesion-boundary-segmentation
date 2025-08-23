#!/usr/bin/env python3
"""
Start TensorBoard monitoring for training.
"""

import subprocess
import sys
import time
import webbrowser
from pathlib import Path

def start_tensorboard(logdir: str = "runs/ckpts/logs/tensorboard", port: int = 6006):
    """Start TensorBoard server."""
    
    logdir_path = Path(logdir)
    if not logdir_path.exists():
        print(f"❌ TensorBoard log directory not found: {logdir}")
        print("   Make sure training has started and created logs.")
        return False
    
    print(f"🚀 Starting TensorBoard...")
    print(f"   Log directory: {logdir}")
    print(f"   Port: {port}")
    
    try:
        # Start TensorBoard
        cmd = [
            sys.executable, "-m", "tensorboard.main",
            "--logdir", str(logdir),
            "--port", str(port),
            "--host", "0.0.0.0",
            "--reload_interval", "30"
        ]
        
        print(f"   Command: {' '.join(cmd)}")
        
        # Start TensorBoard in background
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Wait a moment for TensorBoard to start
        time.sleep(3)
        
        # Check if process is still running
        if process.poll() is None:
            url = f"http://localhost:{port}"
            print(f"✅ TensorBoard started successfully!")
            print(f"   URL: {url}")
            print(f"   Process ID: {process.pid}")
            
            # Try to open in browser
            try:
                webbrowser.open(url)
                print(f"   🌐 Opened in browser")
            except:
                print(f"   ℹ️  Manually open: {url}")
            
            print(f"\n📊 TensorBoard features available:")
            print(f"   • SCALARS: Loss curves, metrics, learning rate")
            print(f"   • GRAPHS: Model architecture visualization")
            print(f"   • HISTOGRAMS: Parameter and gradient distributions")
            print(f"   • IMAGES: Training sample predictions")
            
            print(f"\n⏸️  Press Ctrl+C to stop TensorBoard")
            
            try:
                # Keep the process running
                process.wait()
            except KeyboardInterrupt:
                print(f"\n🛑 Stopping TensorBoard...")
                process.terminate()
                process.wait()
                print(f"✅ TensorBoard stopped")
            
            return True
        else:
            stdout, stderr = process.communicate()
            print(f"❌ TensorBoard failed to start")
            print(f"   Error: {stderr}")
            return False
            
    except FileNotFoundError:
        print(f"❌ TensorBoard not found. Install with: pip install tensorboard")
        return False
    except Exception as e:
        print(f"❌ Failed to start TensorBoard: {e}")
        return False

def main():
    """Main function."""
    print("📊 TensorBoard Launcher for Lesion Segmentation")
    print("=" * 50)
    
    # Check for custom arguments
    if len(sys.argv) > 1:
        logdir = sys.argv[1]
    else:
        logdir = "runs/ckpts/logs/tensorboard"
    
    if len(sys.argv) > 2:
        port = int(sys.argv[2])
    else:
        port = 6006
    
    success = start_tensorboard(logdir, port)
    
    if not success:
        print(f"\n💡 Troubleshooting:")
        print(f"   1. Make sure training is running or has run")
        print(f"   2. Check log directory exists: {logdir}")
        print(f"   3. Try different port: python start_tensorboard.py {logdir} 6007")
        print(f"   4. Install tensorboard: pip install tensorboard")

if __name__ == "__main__":
    main()
