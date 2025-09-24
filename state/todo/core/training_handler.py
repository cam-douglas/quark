"""
Training Handler Module
=======================
Handles model training operations.
"""

import subprocess
import sys
import json
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime


class TrainingHandler:
    """Handles training pipeline operations."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.training_scripts = {
            'local': self.project_root / 'run_training_with_available_resources.py',
            'gcp': self.project_root / 'run_training_compute_engine.py',
            'simple': self.project_root / 'scripts' / 'simple_local_training.py'
        }
        self.state_file = self.project_root / 'state' / 'training_state.json'
        
    def route_command(self, action: str, params: Dict) -> int:
        """Route training commands."""
        if action == 'train':
            return self.start_training(params)
        elif action == 'status':
            return self.check_training_status()
        elif action == 'stop':
            return self.stop_training()
        elif action == 'resume':
            return self.resume_training(params)
        elif action == 'metrics':
            return self.show_training_metrics()
        else:
            return self.show_help()
    
    def start_training(self, params: Dict) -> int:
        """Start model training."""
        stage = params.get('stage', 1)
        mode = params.get('mode', 'local')
        
        print(f"\n🚀 Starting Training - Stage {stage}")
        print("=" * 50)
        print(f"Mode: {mode}")
        
        # Select appropriate training script
        if mode == 'gcp' and self.training_scripts['gcp'].exists():
            script = self.training_scripts['gcp']
        elif self.training_scripts['local'].exists():
            script = self.training_scripts['local']
        else:
            script = self.training_scripts.get('simple')
        
        if script and script.exists():
            cmd = [sys.executable, str(script)]
            
            # Add stage parameter
            cmd.extend(['--stage', str(stage)])
            
            # Add optional parameters
            if params.get('epochs'):
                cmd.extend(['--epochs', str(params['epochs'])])
            if params.get('batch_size'):
                cmd.extend(['--batch-size', str(params['batch_size'])])
            if params.get('checkpoint'):
                cmd.extend(['--checkpoint', params['checkpoint']])
            
            # Save training state
            self._save_training_state({
                'status': 'running',
                'stage': stage,
                'mode': mode,
                'started': datetime.now().isoformat(),
                'script': str(script)
            })
            
            return subprocess.run(cmd).returncode
        else:
            print("⚠️ No training script available")
            print("📝 Available modes: local, gcp, simple")
            return 1
    
    def check_training_status(self) -> int:
        """Check training status."""
        print("\n📊 Training Status")
        print("=" * 50)
        
        # Load saved state
        state = self._load_training_state()
        if state:
            print(f"Status: {state.get('status', 'unknown')}")
            print(f"Stage: {state.get('stage', 'N/A')}")
            print(f"Mode: {state.get('mode', 'N/A')}")
            if state.get('started'):
                print(f"Started: {state['started']}")
            if state.get('checkpoint'):
                print(f"Last checkpoint: {state['checkpoint']}")
        else:
            print("⚠️ No active training session")
        
        # Check for running processes
        try:
            result = subprocess.run(['pgrep', '-f', 'training'],
                                  capture_output=True, text=True)
            if result.returncode == 0:
                print("\n✅ Training process detected")
                print(f"   PID: {result.stdout.strip()}")
        except:
            pass
        
        # Check for recent checkpoints
        checkpoint_dir = self.project_root / 'data' / 'models' / 'checkpoints'
        if checkpoint_dir.exists():
            recent = sorted(checkpoint_dir.glob('*.pt'))[-3:]
            if recent:
                print("\n📁 Recent checkpoints:")
                for f in recent:
                    print(f"   • {f.name}")
        
        return 0
    
    def stop_training(self) -> int:
        """Stop current training."""
        print("\n⏹️ Stopping Training")
        print("=" * 50)
        
        try:
            # Find and kill training processes
            result = subprocess.run(['pkill', '-f', 'training'],
                                  capture_output=True)
            if result.returncode == 0:
                print("✅ Training stopped")
                self._save_training_state({'status': 'stopped'})
            else:
                print("⚠️ No training process found")
        except:
            print("❌ Could not stop training")
            return 1
        
        return 0
    
    def resume_training(self, params: Dict) -> int:
        """Resume training from checkpoint."""
        print("\n▶️ Resuming Training")
        print("=" * 50)
        
        state = self._load_training_state()
        if not state:
            print("⚠️ No training session to resume")
            return 1
        
        # Find latest checkpoint
        checkpoint_dir = self.project_root / 'data' / 'models' / 'checkpoints'
        if checkpoint_dir.exists():
            checkpoints = sorted(checkpoint_dir.glob('*.pt'))
            if checkpoints:
                latest = checkpoints[-1]
                print(f"📁 Resuming from: {latest.name}")
                
                params['checkpoint'] = str(latest)
                params['stage'] = state.get('stage', 1)
                params['mode'] = state.get('mode', 'local')
                
                return self.start_training(params)
        
        print("⚠️ No checkpoint found")
        return 1
    
    def show_training_metrics(self) -> int:
        """Show training metrics."""
        print("\n📈 Training Metrics")
        print("=" * 50)
        
        # Look for metrics files
        metrics_file = self.project_root / 'data' / 'experiments' / 'training_metrics.json'
        if metrics_file.exists():
            try:
                with open(metrics_file) as f:
                    metrics = json.load(f)
                
                print(f"Epoch: {metrics.get('epoch', 'N/A')}")
                print(f"Loss: {metrics.get('loss', 'N/A'):.4f}")
                print(f"Accuracy: {metrics.get('accuracy', 'N/A'):.2%}")
                print(f"Learning Rate: {metrics.get('lr', 'N/A')}")
            except:
                print("⚠️ Could not load metrics")
        else:
            print("⚠️ No metrics available")
            print("📝 Metrics will be saved during training")
        
        return 0
    
    def _save_training_state(self, state: Dict) -> None:
        """Save training state."""
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(self.state_file, 'w') as f:
                json.dump(state, f, indent=2)
        except:
            pass
    
    def _load_training_state(self) -> Optional[Dict]:
        """Load training state."""
        if self.state_file.exists():
            try:
                with open(self.state_file) as f:
                    return json.load(f)
            except:
                pass
        return None
    
    def show_help(self) -> int:
        """Show training help."""
        print("""
🎓 Training Commands:
  todo train model --stage 1   → Start training for stage 1
  todo train gcp --stage 2     → Train on GCP for stage 2  
  todo training status          → Check training status
  todo stop training            → Stop current training
  todo resume training          → Resume from checkpoint
  todo training metrics         → Show training metrics
""")
        return 0
