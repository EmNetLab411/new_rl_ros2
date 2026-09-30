#!/usr/bin/env python3
"""
So sánh tốc độ TFLite (XNNPACK) vs ONNX Runtime trên Pi 4 — 2 file
best_float16.tflite / best_float32.tflite đã có sẵn trong ~/aeroscript (ai
đó từng export thử trước đây, chưa dùng tới). XNNPACK đôi khi nhanh hơn
ONNX Runtime CPU EP đáng kể trên ARM Cortex-A72 cho model conv nhỏ.

Chạy trên Pi (cần: pip install tflite-runtime, hoặc "pip install tensorflow"
nếu tflite-runtime không có sẵn cho bản Python/OS đang dùng):
    python3 benchmark_tflite.py --model best_float16.tflite
    python3 benchmark_tflite.py --model best_float32.tflite --threads 4
"""
import argparse
import time

import numpy as np

try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    from tensorflow.lite.python.interpreter import Interpreter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="best_float16.tflite")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--n-runs", type=int, default=30)
    args = ap.parse_args()

    interpreter = Interpreter(model_path=args.model, num_threads=args.threads)
    interpreter.allocate_tensors()
    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]
    shape = inp["shape"]  # thường [1, H, W, 3] (NHWC) cho TFLite
    h, w = int(shape[1]), int(shape[2])
    dtype = inp["dtype"]

    print(f"Model: {args.model}  input_shape={list(shape)}  dtype={dtype}  threads={args.threads}")

    dummy = np.full((1, h, w, 3), 0.5, dtype=np.float32)
    if dtype != np.float32:
        dummy = dummy.astype(dtype)

    # Warmup
    for _ in range(3):
        interpreter.set_tensor(inp["index"], dummy)
        interpreter.invoke()
        interpreter.get_tensor(out["index"])

    times = []
    for _ in range(args.n_runs):
        t0 = time.perf_counter()
        interpreter.set_tensor(inp["index"], dummy)
        interpreter.invoke()
        interpreter.get_tensor(out["index"])
        times.append(time.perf_counter() - t0)

    times = np.array(times) * 1000  # ms
    print(f"Inference: mean={times.mean():.1f}ms  min={times.min():.1f}ms  "
          f"max={times.max():.1f}ms  -> ~{1000/times.mean():.1f} FPS thuần suy luận")


if __name__ == "__main__":
    main()
