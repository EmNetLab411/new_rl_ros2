"""
benchmark_onnx.py — Đo chính xác thời gian từng bước, không qua ROS2.
Chạy trực tiếp trên Pi để biết bottleneck thật nằm ở đâu: session.run()
hay preprocess/decode/solvePnP.

Cách chạy:
    python3 benchmark_onnx.py --model best.onnx
    python3 benchmark_onnx.py --model best_int8.onnx --threads 4
"""
import argparse
import time
import numpy as np
import cv2
import onnxruntime as ort


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="best.onnx")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--n-runs", type=int, default=30)
    args = parser.parse_args()

    # Test với nhiều mức threads để so sánh trực tiếp
    thread_options = [1, 2, 4] if args.threads == 0 else [args.threads]

    # Tạo 1 ảnh giả 640x480 (giống camera thật) để benchmark nhất quán
    dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    for n_threads in thread_options:
        print(f"\n{'='*60}")
        print(f"  Threads = {n_threads}")
        print(f"{'='*60}")

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = n_threads
        opts.inter_op_num_threads = 1
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        session = ort.InferenceSession(args.model, sess_options=opts,
                                       providers=["CPUExecutionProvider"])
        input_name  = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        inp_shape   = session.get_inputs()[0].shape
        imgsz       = int(inp_shape[2]) if inp_shape[1] == 3 else int(inp_shape[1])
        is_nhwc     = inp_shape[1] != 3

        # Warmup — lần đầu luôn chậm hơn do lazy init, không tính vào benchmark
        for _ in range(3):
            canvas = np.full((imgsz, imgsz, 3), 114, np.uint8)
            rgb = canvas[:,:,::-1].astype(np.float32) / 255.0
            t = rgb[np.newaxis] if is_nhwc else np.transpose(rgb,(2,0,1))[np.newaxis]
            session.run([output_name], {input_name: t})

        # Đo preprocess riêng
        preprocess_times = []
        inference_times  = []

        for _ in range(args.n_runs):
            t0 = time.perf_counter()
            h0, w0 = dummy_frame.shape[:2]
            r = imgsz / max(h0, w0)
            nw, nh = int(w0*r), int(h0*r)
            canvas = np.full((imgsz, imgsz, 3), 114, np.uint8)
            dw, dh = (imgsz-nw)//2, (imgsz-nh)//2
            canvas[dh:dh+nh, dw:dw+nw] = cv2.resize(dummy_frame, (nw, nh))
            rgb = canvas[:,:,::-1].astype(np.float32) / 255.0
            tensor = rgb[np.newaxis] if is_nhwc else np.transpose(rgb,(2,0,1))[np.newaxis]
            t1 = time.perf_counter()

            session.run([output_name], {input_name: tensor})
            t2 = time.perf_counter()

            preprocess_times.append((t1 - t0) * 1000)
            inference_times.append((t2 - t1) * 1000)

        pp = np.array(preprocess_times)
        inf = np.array(inference_times)

        print(f"Preprocess : mean={pp.mean():.1f}ms  min={pp.min():.1f}ms  max={pp.max():.1f}ms")
        print(f"Inference  : mean={inf.mean():.1f}ms  min={inf.min():.1f}ms  max={inf.max():.1f}ms")
        total_ms = pp.mean() + inf.mean()
        print(f"Tổng       : {total_ms:.1f}ms/frame  →  FPS lý thuyết tối đa: {1000/total_ms:.1f}")

        del session


if __name__ == "__main__":
    main()