#!/usr/bin/env python3
"""
Chẩn đoán layout output thật của file .tflite — dùng để tìm đúng lỗi
_normalize_output() trong run_pi4_ros2.py đang đoán sai (gây toạ độ 3D
sai hàng chục mét khi dùng TFLite thay ONNX).

Chạy trên Pi (trong ~/aeroscript):
    python3 scripts/inspect_tflite.py --model best_float32.tflite
"""
import argparse

import numpy as np

try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    from tensorflow.lite.python.interpreter import Interpreter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="best_float32.tflite")
    args = ap.parse_args()

    interpreter = Interpreter(model_path=args.model, num_threads=2)
    interpreter.allocate_tensors()

    print("=== INPUT DETAILS ===")
    for d in interpreter.get_input_details():
        print(d)

    print("\n=== OUTPUT DETAILS (tất cả output, không chỉ [0]) ===")
    outs = interpreter.get_output_details()
    for d in outs:
        print(d)

    inp = interpreter.get_input_details()[0]
    shape = inp["shape"]
    dummy = np.full(shape, 0.5, dtype=inp["dtype"])
    interpreter.set_tensor(inp["index"], dummy)
    interpreter.invoke()

    print(f"\n=== SỐ LƯỢNG OUTPUT TENSOR: {len(outs)} ===")
    for i, d in enumerate(outs):
        raw = interpreter.get_tensor(d["index"])
        print(f"\nOutput #{i}: shape={raw.shape} dtype={raw.dtype}")
        print(f"  min={raw.min():.4f} max={raw.max():.4f} mean={raw.mean():.4f}")
        flat = raw.reshape(-1)
        print(f"  10 giá trị đầu: {flat[:10]}")


if __name__ == "__main__":
    main()
