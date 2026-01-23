#!/usr/bin/env python3
"""
Export Drawing Model for Raspberry Pi Deployment

Combined script: PyTorch → ONNX → TFLite
Exports SAC Actor + Neural IK as a single inference pipeline.

Usage:
    python3 export_drawing_model.py
    python3 export_drawing_model.py --actor checkpoints/sac_drawing/actor_best.pth
"""

import os
import sys
import argparse
import numpy as np

# Add parent directory for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)

# Model dimensions for DRAWING task (18D state)
STATE_DIM = 18  # Drawing: joints(6) + EE(3) + target(3) + dist(3) + dist3d(1) + progress(1) + remaining(1)
ACTION_DIM = 3  # SAC outputs 3D action (direction deltas)
JOINT_DIM = 6   # Neural IK outputs 6 joint angles


class DrawingModelExporter:
    """Export SAC Actor + Neural IK for drawing deployment"""
    
    def __init__(self):
        self.actor_path = None
        self.neural_ik_path = None
        self.output_dir = None
        
    def find_latest_models(self):
        """Find latest trained drawing models"""
        checkpoints_dir = os.path.join(parent_dir, 'checkpoints')
        
        # Look for drawing-specific checkpoints
        drawing_dirs = ['sac_drawing', 'sac_drawing_neuralIK', 'sac_gazebo']
        
        for dir_name in drawing_dirs:
            dir_path = os.path.join(checkpoints_dir, dir_name)
            if os.path.exists(dir_path):
                # Look for actor model
                for fname in ['actor_best.pth', 'actor_sac_best.pth', 'actor_final.pth']:
                    actor_path = os.path.join(dir_path, fname)
                    if os.path.exists(actor_path):
                        self.actor_path = actor_path
                        self.output_dir = dir_path
                        print(f"✅ Found actor: {actor_path}")
                        break
                if self.actor_path:
                    break
        
        # Look for Neural IK model
        nik_path = os.path.join(checkpoints_dir, 'neural_ik.pth')
        if os.path.exists(nik_path):
            self.neural_ik_path = nik_path
            print(f"✅ Found Neural IK: {nik_path}")
        else:
            print(f"⚠️  Neural IK not found at {nik_path}")
            
        return self.actor_path is not None
    
    def export_to_onnx(self):
        """Step 1: Export PyTorch models to ONNX"""
        import torch
        import torch.nn as nn
        
        print("\n" + "="*70)
        print("🔄 Step 1: PyTorch → ONNX Export")
        print("="*70)
        
        # Define SAC Actor Network (matches training architecture)
        class SACActorNetwork(nn.Module):
            def __init__(self, state_dim, action_dim):
                super().__init__()
                self.l1 = nn.Linear(state_dim, 256)
                self.l2 = nn.Linear(256, 256)
                self.mean_linear = nn.Linear(256, action_dim)
            
            def forward(self, x):
                x = torch.relu(self.l1(x))
                x = torch.relu(self.l2(x))
                return torch.tanh(self.mean_linear(x))
        
        # Define Neural IK Network
        class NeuralIKNetwork(nn.Module):
            def __init__(self, input_dim=3, output_dim=6):
                super().__init__()
                self.l1 = nn.Linear(input_dim, 256)
                self.l2 = nn.Linear(256, 256)
                self.l3 = nn.Linear(256, output_dim)
            
            def forward(self, xyz):
                x = torch.relu(self.l1(xyz))
                x = torch.relu(self.l2(x))
                return self.l3(x)  # Joint angles in radians
        
        # Load and export Actor
        print(f"\n📦 Loading SAC Actor: {self.actor_path}")
        actor = SACActorNetwork(STATE_DIM, ACTION_DIM)
        try:
            state_dict = torch.load(self.actor_path, map_location='cpu')
            actor.load_state_dict(state_dict, strict=False)
            actor.eval()
            print("✅ Actor loaded")
        except Exception as e:
            print(f"❌ Failed to load actor: {e}")
            return None
        
        # Export Actor to ONNX
        actor_onnx = os.path.join(self.output_dir, 'actor_drawing.onnx')
        dummy_state = torch.randn(1, STATE_DIM)
        
        torch.onnx.export(
            actor, dummy_state, actor_onnx,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['state'],
            output_names=['action'],
            dynamic_axes={'state': {0: 'batch'}, 'action': {0: 'batch'}}
        )
        print(f"✅ Actor ONNX: {actor_onnx}")
        
        # Load and export Neural IK
        neural_ik_onnx = None
        if self.neural_ik_path:
            print(f"\n📦 Loading Neural IK: {self.neural_ik_path}")
            neural_ik = NeuralIKNetwork(3, 6)
            try:
                state_dict = torch.load(self.neural_ik_path, map_location='cpu')
                neural_ik.load_state_dict(state_dict, strict=False)
                neural_ik.eval()
                print("✅ Neural IK loaded")
                
                # Export Neural IK to ONNX
                neural_ik_onnx = os.path.join(self.output_dir, 'neural_ik.onnx')
                dummy_xyz = torch.randn(1, 3)
                
                torch.onnx.export(
                    neural_ik, dummy_xyz, neural_ik_onnx,
                    export_params=True,
                    opset_version=11,
                    do_constant_folding=True,
                    input_names=['xyz'],
                    output_names=['joints'],
                    dynamic_axes={'xyz': {0: 'batch'}, 'joints': {0: 'batch'}}
                )
                print(f"✅ Neural IK ONNX: {neural_ik_onnx}")
            except Exception as e:
                print(f"⚠️  Failed to load Neural IK: {e}")
        
        return actor_onnx, neural_ik_onnx
    
    def convert_to_tflite(self, actor_onnx, neural_ik_onnx=None):
        """Step 2: Convert ONNX to TFLite"""
        print("\n" + "="*70)
        print("🔄 Step 2: ONNX → TFLite Conversion")
        print("="*70)
        
        # Suppress TF warnings
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        
        try:
            import onnx
            from onnx_tf.backend import prepare
            import tensorflow as tf
            import tempfile
            import shutil
        except ImportError as e:
            print(f"❌ Missing package: {e}")
            print("   pip install onnx onnx-tf tensorflow")
            return None
        
        results = {}
        
        for name, onnx_path in [('actor', actor_onnx), ('neural_ik', neural_ik_onnx)]:
            if onnx_path is None or not os.path.exists(onnx_path):
                continue
                
            print(f"\n⚙️  Converting {name}...")
            
            try:
                # Load ONNX
                onnx_model = onnx.load(onnx_path)
                
                # Fix IR version if needed
                if onnx_model.ir_version > 9:
                    onnx_model.ir_version = 9
                
                # Convert to TensorFlow
                tf_rep = prepare(onnx_model)
                
                # Save to temp dir
                temp_dir = tempfile.mkdtemp()
                saved_model_path = os.path.join(temp_dir, "saved_model")
                tf_rep.export_graph(saved_model_path)
                
                # Convert to TFLite (Float32 - no quantization)
                converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_path)
                tflite_model = converter.convert()
                
                # Save TFLite
                tflite_path = onnx_path.replace('.onnx', '.tflite')
                with open(tflite_path, 'wb') as f:
                    f.write(tflite_model)
                
                # Cleanup
                shutil.rmtree(temp_dir)
                
                # Stats
                onnx_size = os.path.getsize(onnx_path)
                tflite_size = os.path.getsize(tflite_path)
                
                print(f"✅ {name} TFLite: {tflite_path}")
                print(f"   Size: {onnx_size/1024:.1f}KB → {tflite_size/1024:.1f}KB")
                
                results[name] = tflite_path
                
            except Exception as e:
                print(f"❌ {name} conversion failed: {e}")
                import traceback
                traceback.print_exc()
        
        return results
    
    def run(self, actor_path=None, neural_ik_path=None):
        """Run full export pipeline"""
        print("="*70)
        print("🎨 DRAWING MODEL EXPORT PIPELINE")
        print("   PyTorch → ONNX → TFLite (Float32)")
        print("="*70)
        
        # Set paths if provided
        if actor_path:
            self.actor_path = actor_path
            self.output_dir = os.path.dirname(actor_path)
        if neural_ik_path:
            self.neural_ik_path = neural_ik_path
        
        # Find models if not specified
        if not self.actor_path:
            if not self.find_latest_models():
                print("❌ No trained models found!")
                return False
        
        # Step 1: PyTorch → ONNX
        onnx_result = self.export_to_onnx()
        if onnx_result is None:
            return False
        
        actor_onnx, neural_ik_onnx = onnx_result
        
        # Step 2: ONNX → TFLite
        tflite_result = self.convert_to_tflite(actor_onnx, neural_ik_onnx)
        
        # Summary
        print("\n" + "="*70)
        print("✅ EXPORT COMPLETE!")
        print("="*70)
        
        if tflite_result:
            print("\n📦 Output Files:")
            for name, path in tflite_result.items():
                print(f"   {name}: {path}")
            
            print("\n📝 Deploy to Pi:")
            print("   scp checkpoints/sac_drawing/*.tflite pi@raspberrypi:~/robot/")
            print("   python3 deploy_drawing_on_pi.py")
        
        return True


def main():
    parser = argparse.ArgumentParser(description='Export Drawing Model for Pi Deployment')
    parser.add_argument('--actor', type=str, help='Path to SAC actor .pth file')
    parser.add_argument('--neural-ik', type=str, help='Path to Neural IK .pth file')
    
    args = parser.parse_args()
    
    exporter = DrawingModelExporter()
    success = exporter.run(args.actor, args.neural_ik)
    
    return 0 if success else 1


if __name__ == '__main__':
    exit(main())
