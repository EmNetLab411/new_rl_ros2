#!/usr/bin/env python3
"""
Export Trained RL Model to TensorFlow Lite for Raspberry Pi Deployment

This script converts trained PyTorch models (.pth) to TensorFlow Lite (.tflite) format
for efficient deployment on Raspberry Pi with limited resources.

IMPORTANT: Run this script in a FRESH terminal (not after importing PyTorch elsewhere)
to avoid protobuf conflicts.

Model Specs (6DOF Robot - Direct Joint Control):
    - State dim: 18 (joints(6) + robot_xyz(3) + target_xyz(3) + dist(4) + vel(2))
    - Action dim: 6 (joint angle deltas)

Usage:
    python3 export_tflite.py --model ../checkpoints/sac_gazebo/actor_sac_best.pth
    python3 export_tflite.py --model ../checkpoints/sac_gazebo/actor_sac_best.pth --quantize
    python3 export_tflite.py --model ../checkpoints/td3/actor_td3_best.pth --agent td3
"""

# IMPORTANT: Import TensorFlow FIRST to avoid protobuf conflicts
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TF warnings

import tensorflow as tf
import json
import sys
import argparse
import numpy as np

# Add paths for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)


def convert_to_tflite(model_path: str, output_path: str = None, quantize: bool = False, 
                      state_dim: int = 18, action_dim: int = 6, agent_type: str = 'sac'):
    """
    Convert trained actor model to TensorFlow Lite format
    
    Args:
        model_path: Path to model file (.pth or base path)
        output_path: Output .tflite file path (auto-generated if None)
        quantize: Apply post-training quantization for smaller model
        state_dim: State dimension (default: 18 for our 6DOF robot)
        action_dim: Action dimension (default: 6 for joint deltas)
        agent_type: 'sac' or 'td3'
    """
    print("="*70)
    print("🔄 TensorFlow Lite Model Conversion")
    print("="*70)
    
    # Handle .pth extension
    if model_path.endswith('.pth'):
        model_base = model_path[:-4]
    else:
        model_base = model_path
        model_path = model_path + '.pth'
    
    # 1. Load actor model weights
    print(f"\n🤖 Loading actor model from: {model_path}")
    print(f"   Agent type: {agent_type.upper()}")
    print(f"   State dim: {state_dim}")
    print(f"   Action dim: {action_dim}")
    
    if not os.path.exists(model_path):
        print(f"❌ ERROR: Actor weights not found at {model_path}")
        return False
    
    # 2. Recreate actor architecture and load weights
    print(f"\n🏗️  Recreating {agent_type.upper()} agent architecture...")
    
    try:
        import torch
        import torch.nn as nn
        
        # Define actor network matching training architecture
        if agent_type == 'sac':
            # SAC GaussianActor - simplified for inference (mean only)
            class ActorNetwork(nn.Module):
                def __init__(self, state_dim, action_dim, max_action=0.1):
                    super().__init__()
                    self.max_action = max_action
                    self.l1 = nn.Linear(state_dim, 256)
                    self.l2 = nn.Linear(256, 256)
                    self.mean_linear = nn.Linear(256, action_dim)
                
                def forward(self, x):
                    x = torch.relu(self.l1(x))
                    x = torch.relu(self.l2(x))
                    mean = self.mean_linear(x)
                    return torch.tanh(mean) * self.max_action
        else:
            # TD3 Actor
            class ActorNetwork(nn.Module):
                def __init__(self, state_dim, action_dim, max_action=0.1):
                    super().__init__()
                    self.max_action = max_action
                    self.l1 = nn.Linear(state_dim, 400)
                    self.l2 = nn.Linear(400, 300)
                    self.l3 = nn.Linear(300, action_dim)
                
                def forward(self, x):
                    x = torch.relu(self.l1(x))
                    x = torch.relu(self.l2(x))
                    return torch.tanh(self.l3(x)) * self.max_action
        
        actor_network = ActorNetwork(state_dim, action_dim, max_action=0.1)
        
        # Load trained weights
        try:
            state_dict = torch.load(model_path, map_location='cpu')
            actor_network.load_state_dict(state_dict, strict=False)
            actor_network.eval()
            actor_network.cpu()
            print("✅ Actor weights loaded successfully")
        except Exception as e:
            print(f"❌ ERROR: Could not load weights: {e}")
            import traceback
            traceback.print_exc()
            return False
        
    except Exception as e:
        print(f"❌ ERROR: Failed to load actor model: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 3. Convert PyTorch → ONNX → TensorFlow → TFLite
    print(f"\n⚙️  Converting PyTorch → ONNX → TensorFlow → TFLite...")
    print(f"   Quantization: {'ENABLED' if quantize else 'DISABLED'}")
    
    try:
        import onnx
        from onnx_tf.backend import prepare
        
        # Step 1: PyTorch → ONNX
        print("   Step 1/3: PyTorch → ONNX...")
        dummy_input = torch.randn(1, state_dim)
        onnx_path = f"{model_base}_temp.onnx"
        
        torch.onnx.export(
            actor_network,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
        )
        print("   ✅ ONNX export successful")
        
        # Step 2: ONNX → TensorFlow
        print("   Step 2/3: ONNX → TensorFlow...")
        onnx_model = onnx.load(onnx_path)
        tf_rep = prepare(onnx_model)
        print("   ✅ TensorFlow conversion successful")
        
        # Step 3: TensorFlow → TFLite
        print("   Step 3/3: TensorFlow → TFLite...")
        
        # Export to SavedModel first
        import tempfile
        temp_dir = tempfile.mkdtemp()
        saved_model_path = os.path.join(temp_dir, "saved_model")
        
        tf_rep.export_graph(saved_model_path)
        
        # Convert SavedModel to TFLite
        converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_path)
        
        # Apply optimizations
        if quantize:
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
        else:
            converter.optimizations = []
        
        tflite_model = converter.convert()
        print("   ✅ Conversion successful!")
        
        # Clean up temporary files
        import shutil
        if os.path.exists(onnx_path):
            os.remove(onnx_path)
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        
    except ImportError as e:
        print(f"\n❌ ERROR: Missing required package: {e}")
        print("\n📝 Please install required packages:")
        print("   pip install onnx onnx-tf tensorflow")
        return False
    except Exception as e:
        print(f"❌ ERROR: Conversion failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 4. Save TFLite model
    if output_path is None:
        suffix = "_quantized" if quantize else ""
        output_path = f"{model_base}{suffix}.tflite"
    
    print(f"\n💾 Saving TFLite model to: {output_path}")
    
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    
    # 5. Model verification and statistics
    print(f"\n📊 Conversion Statistics:")
    print("="*70)
    
    original_size = os.path.getsize(model_path)
    tflite_size = os.path.getsize(output_path)
    reduction = 100 * (1 - tflite_size / original_size)
    
    print(f"Original model (.pth):     {original_size / 1024:.2f} KB")
    print(f"Converted model (.tflite): {tflite_size / 1024:.2f} KB")
    print(f"Size reduction:            {reduction:.1f}%")
    
    # 6. Test inference
    print(f"\n🧪 Testing TFLite model inference...")
    
    try:
        # Load TFLite model
        interpreter = tf.lite.Interpreter(model_path=output_path)
        interpreter.allocate_tensors()
        
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        print(f"   Input shape: {input_details[0]['shape']}")
        print(f"   Output shape: {output_details[0]['shape']}")
        
        # Test with random input
        test_input = np.random.randn(1, state_dim).astype(np.float32)
        interpreter.set_tensor(input_details[0]['index'], test_input)
        interpreter.invoke()
        test_output = interpreter.get_tensor(output_details[0]['index'])
        
        print(f"   Test input: {test_input[0, :3]}... (shape: {test_input.shape})")
        print(f"   Test output: {test_output[0]} (shape: {test_output.shape})")
        print("✅ Inference test passed!")
        
    except Exception as e:
        print(f"⚠️  Warning: Could not test inference: {e}")
    
    print("\n" + "="*70)
    print("✅ CONVERSION COMPLETE!")
    print("="*70)
    print(f"\n📦 Model ready for deployment:")
    print(f"   {output_path}")
    print(f"\n📝 Next steps:")
    print(f"   1. Run: ./deploy_to_pi.sh")
    print(f"   2. On Pi: python3 deploy_on_pi.py --model {os.path.basename(output_path)}")
    print()
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Export trained RL model to TensorFlow Lite for Raspberry Pi',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert SAC model
  python3 export_tflite.py --model ../checkpoints/sac_gazebo/actor_sac_best.pth
  
  # Convert with quantization (recommended for Pi)
  python3 export_tflite.py --model ../checkpoints/sac_gazebo/actor_sac_best.pth --quantize
  
  # Convert TD3 model
  python3 export_tflite.py --model ../checkpoints/td3/actor_td3_best.pth --agent td3
  
  # Specify custom output path
  python3 export_tflite.py --model ../checkpoints/sac_gazebo/actor_sac_best.pth --output model.tflite

Note: Run this script in a FRESH terminal to avoid protobuf conflicts!
        """
    )
    
    parser.add_argument('--model', '-m', type=str, required=True,
                        help='Path to model (.pth file or base path)')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Output .tflite file path (auto-generated if not specified)')
    parser.add_argument('--agent', type=str, default='sac', choices=['sac', 'td3'],
                        help='Agent type: sac or td3 (default: sac)')
    parser.add_argument('--quantize', '-q', action='store_true',
                        help='Apply post-training quantization (reduces size ~4x, recommended)')
    parser.add_argument('--state-dim', type=int, default=18,
                        help='State dimension (default: 18)')
    parser.add_argument('--action-dim', type=int, default=6,
                        help='Action dimension (default: 6)')
    
    args = parser.parse_args()
    
    # Convert model
    success = convert_to_tflite(
        model_path=args.model,
        output_path=args.output,
        quantize=args.quantize,
        state_dim=args.state_dim,
        action_dim=args.action_dim,
        agent_type=args.agent
    )
    
    if success:
        print("✅ Export completed successfully!")
        return 0
    else:
        print("❌ Export failed!")
        return 1


if __name__ == '__main__':
    exit(main())
