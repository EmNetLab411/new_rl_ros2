#!/usr/bin/env python3
"""Đo tốc độ suy luận thuần của model ONNX trên Pi — bản song song của
benchmark_tflite.py, dùng để so TỐC ĐỘ GIỮA CÁC imgsz khác nhau.

Mục đích (xem PLAN.md mục 5c): trước khi bỏ hàng giờ train lại model ở
imgsz=224, cần biết chắc 224 thật sự nhanh hơn bao nhiêu trên chính con Pi
này. Tốc độ suy luận chỉ phụ thuộc kiến trúc + imgsz, KHÔNG phụ thuộc trọng
số, nên đo bằng model yolov8n-pose gốc của COCO là đủ đại diện.

Chạy trên Pi (trong ~/aeroscript), đo lần lượt từng kích thước:
    for f in probe_320.onnx probe_256.onnx probe_224.onnx probe_192.onnx; do
        python3 scripts/benchmark_onnx.py --model $f --threads 2
    done
"""
import argparse
import time

import numpy as np
import onnxruntime as ort


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--n-runs", type=int, default=30)
    args = ap.parse_args()

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = args.threads
    opts.inter_op_num_threads = 1
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    sess = ort.InferenceSession(args.model, sess_options=opts,
                                providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    shape = [d if isinstance(d, int) else 1 for d in inp.shape]
    dummy = np.full(shape, 0.5, dtype=np.float32)
    out_name = sess.get_outputs()[0].name

    # Warmup: lần chạy đầu luôn chậm bất thường (JIT + graph optimization)
    for _ in range(3):
        sess.run([out_name], {inp.name: dummy})

    times = []
    for _ in range(args.n_runs):
        t0 = time.perf_counter()
        sess.run([out_name], {inp.name: dummy})
        times.append(time.perf_counter() - t0)

    t = np.array(times) * 1000
    print(f"{args.model:18s} shape={shape}  threads={args.threads}  "
          f"mean={t.mean():6.1f}ms  min={t.min():6.1f}ms  max={t.max():6.1f}ms  "
          f"-> {1000/t.mean():4.1f} fps thuần suy luận")


if __name__ == "__main__":
    main()
