#!/usr/bin/env python3
"""
Đọc log lượt chạy của run_pi4_ros2.py (logs/run_*.csv) — chỉ cần numpy, chạy
được trên cả Pi lẫn laptop.

    # 1 lượt: bảng số liệu + các lần bút đứng yên (kiểm sai số so với bảng)
    python3 scripts/analyze_run.py logs/run_20261007_101500_360p_bang.csv

    # nhiều lượt: so sánh cạnh nhau (2 lượt thì có thêm cột chênh lệch)
    python3 scripts/analyze_run.py logs/run_*_360p_khongbang.csv logs/run_*_360p_bang.csv

    # lượt mới nhất trong thư mục logs/
    python3 scripts/analyze_run.py --last
    python3 scripts/analyze_run.py --last 4          # 4 lượt mới nhất, so sánh

    # thử kích thước bút khác (Tip→Tail, Tip→đường L-R, L↔R, mm) mà không cần chạy lại
    python3 scripts/analyze_run.py logs/run_x.csv --pen-dims-mm 43 29.5 15.5

    # thử file calib camera khác trên log đã có
    python3 scripts/analyze_run.py logs/run_x.csv --calib calib/c930e_720p.npz

Kiểm sai số: trong lúc chạy (có --board), chạm mũi bút vào dấu + và 4 góc
vùng vẽ, mỗi điểm GIỮ YÊN ít nhất 2 giây. Script tự tìm các đoạn bút đứng yên,
so với điểm chuẩn gần nhất, và tính xem khoảng cách bút bị đo lệch mấy lần.
"""
import argparse
import csv
import glob
import json
import os
import re
import sys

import numpy as np

REFS = {"dấu +": (0, 0), "góc trên-phải": (50, 50), "góc trên-trái": (-50, 50),
        "góc dưới-trái": (-50, -50), "góc dưới-phải": (50, -50)}
NUM_SKIP = {"brd_seen", "brd_corners"}


# ── đọc file ────────────────────────────────────────────────────────────────
def load(path):
    with open(path, newline="") as f:
        first = f.readline()
        meta = json.loads(first[1:]) if first.startswith("#") else {}
        if not first.startswith("#"):
            f.seek(0)
        rows = list(csv.DictReader(f))
    cols = {}
    for c in (rows[0].keys() if rows else []):
        if c in NUM_SKIP:
            cols[c] = [r[c] for r in rows]
        else:
            cols[c] = np.array([float(r[c]) if r[c] not in ("", None) else np.nan for r in rows])
    name = re.sub(r"^cmp\d+_", "", meta.get("tag") or os.path.basename(path)[4:-4])
    return {"path": path, "meta": meta, "c": cols, "n": len(rows), "name": name}


def quat_R(q):
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def board_T(c, i):
    """(R, t[mm]) của bảng ở dòng i, hoặc None."""
    if "brd_x" not in c or np.isnan(c["brd_x"][i]):
        return None
    return (quat_R([c[k][i] for k in ("brd_qx", "brd_qy", "brd_qz", "brd_qw")]),
            np.array([c["brd_x"][i], c["brd_y"][i], c["brd_z"][i]]))


def recompute(run, dims=None, calib=None):
    """Tính lại từ dữ liệu thô đã log, không cần chạy lại trên Pi:
      - `calib`: file .npz khác -> đổi K/dist (tự quy đổi theo độ phân giải),
        tính lại pose bảng từ góc marker đã log (giữ pose tới lần dò kế tiếp,
        không làm mượt) và XYZ bút từ điểm khớp;
      - `dims`: kích thước bút khác (Tip→Tail, Tip→đường L-R, L↔R).
    XYZ bút tính lại không qua bộ lọc Kalman 3D."""
    import cv2
    m, c = run["meta"], run["c"]
    if calib:
        d = np.load(calib)
        K = d["K"].astype(np.float64).copy()
        w0, h0 = (int(v) for v in d["image_size"])
        K[0] *= m["width"] / w0
        K[1] *= m["height"] / h0
        dist = d["dist"].astype(np.float64).ravel()
        m["K"] = [K[0, 0], K[1, 1], K[0, 2], K[1, 2]]
        m["dist"] = dist.tolist()
    fx, fy, cx, cy = m["K"]
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])
    dist = np.array(m.get("dist", [0] * 5), np.float64)
    if dims:
        L, D, W = dims
        m["pen_3d"] = [[0, 0, 0], [0, L, 0], [-W / 2, D, 0], [W / 2, D, 0]]
    pen = np.array(m.get("pen_3d", [[0, 0, 0], [0, 64, 0], [-11.5, 44, 0], [11.5, 44, 0]]), np.float64)
    bc = m.get("board_cfg") or {}
    o, h = bc.get("offset_mm", 75.0), bc.get("marker_mm", 30.0) / 2
    sq = lambda x, y: [[x - h, y + h, 0], [x + h, y + h, 0], [x + h, y - h, 0], [x - h, y - h, 0]]
    obj = {0: np.array(sq(-o, o), float), 1: np.array(sq(o, o), float),
           2: np.array(sq(o, -o), float), 3: np.array(sq(-o, -o), float)}
    redo_board = bool(calib) and "brd_corners" in c
    T = None
    for i in range(run["n"]):
        if redo_board:
            dc = parse_corners(c["brd_corners"][i])
            if len(dc) >= 2:
                ids = sorted(dc)
                ok, rv, tv = cv2.solvePnP(np.vstack([obj[k] for k in ids]), np.vstack([dc[k] for k in ids]),
                                          K, dist, flags=cv2.SOLVEPNP_IPPE)
                # chỉ nhận pose mới từ lần dò đủ 4 marker (hoặc khi chưa có pose nào)
                if ok and tv[2][0] > 0 and (len(dc) == 4 or T is None):
                    pr = cv2.projectPoints(np.vstack([obj[k] for k in ids]), rv, tv, K, dist)[0].reshape(-1, 2)
                    T = (cv2.Rodrigues(rv)[0], tv.ravel(),
                         float(np.sqrt(np.mean(np.sum((pr - np.vstack([dc[k] for k in ids])) ** 2, axis=1)))))
            if T is not None and not np.isnan(c["brd_x"][i]):
                R, t, e = T
                c["brd_x"][i], c["brd_y"][i], c["brd_z"][i] = t
                c["brd_err"][i] = e
                # quaternion từ R
                qw = np.sqrt(max(0.0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
                if qw > 1e-6:
                    q = [(R[2, 1] - R[1, 2]) / (4 * qw), (R[0, 2] - R[2, 0]) / (4 * qw),
                         (R[1, 0] - R[0, 1]) / (4 * qw), qw]
                    c["brd_qx"][i], c["brd_qy"][i], c["brd_qz"][i], c["brd_qw"][i] = q
        if np.isnan(c["tip_u"][i]):
            continue
        kp = np.array([[c[a + "_u"][i], c[a + "_v"][i]] for a in ("tip", "tail", "l", "r")])
        ok, _, t = cv2.solvePnP(pen, kp, K, dist, flags=cv2.SOLVEPNP_IPPE)
        if not ok or t[2][0] <= 0:
            continue
        c["x"][i], c["y"][i], c["z"][i] = t.ravel()
        Tb = (T[0], T[1]) if (redo_board and T is not None) else board_T(c, i)
        if Tb is not None and "bx" in c:
            c["bx"][i], c["by"][i], c["bz"][i] = Tb[0].T @ (t.ravel() - Tb[1])


# ── kiểm bảng: hình học bản in và tiêu cự camera ────────────────────────────
def parse_corners(sv):
    out = {}
    for part in (sv or "").split(";"):
        if ":" in part:
            i, v = part.split(":")
            out[int(i)] = np.array(v.split(), float).reshape(4, 2)
    return out


def board_check(run):
    """Từ góc marker đã log (các lần thấy đủ 4 marker), đo 2 thứ độc lập:
      1. Tỉ lệ (khoảng cách tâm 2 marker kề nhau) / (cạnh marker) của BẢN IN —
         không phụ thuộc camera hay khoảng cách. Bảng đúng: 150/30 = 5,00.
      2. Tiêu cự camera suy từ độ méo phối cảnh của bảng, so với tiêu cự trong
         file calib. Cần bảng nghiêng ≥ ~15 độ mới đo được."""
    import cv2
    c, m = run["c"], run["meta"]
    if "brd_corners" not in c:
        print("  Log này chưa ghi góc marker (bản node cũ) — không kiểm được bảng.")
        return
    dets = [d for d in (parse_corners(v) for v in c["brd_corners"]) if len(d) == 4]
    if len(dets) < 5:
        print(f"  Chỉ có {len(dets)} lần dò thấy đủ 4 marker — cần ít nhất 5 (bỏ tay ra cho camera thấy cả bảng).")
        return
    fx, fy, cx, cy = m["K"]
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])
    dist = np.array(m.get("dist", [0] * 5), float)
    bc = m.get("board_cfg") or {}
    o, h = bc.get("offset_mm", 75.0), bc.get("marker_mm", 30.0) / 2
    sq = lambda x, y: np.array([[x - h, y + h], [x + h, y + h], [x + h, y - h], [x - h, y - h]])
    obj = {0: sq(-o, o), 1: sq(o, o), 2: sq(o, -o), 3: sq(-o, -o)}
    unit = np.array([[-.5, .5], [.5, .5], [.5, -.5], [-.5, -.5]])       # 1 marker, đơn vị = cạnh marker
    use = dets[:: max(1, len(dets) // 80)]
    ids = sorted(obj)
    obj3 = np.vstack([np.column_stack([obj[i], np.zeros(4)]) for i in ids]) / 1000.0
    def plane_err(side):
        """Lệch (px) khi khớp 16 góc với mặt phẳng — không dùng tiêu cự camera —
        nếu cạnh marker là `side` mm (tâm marker giữ ở ±o)."""
        hh = side / 2
        sq2 = lambda x, y: [[x - hh, y + hh], [x + hh, y + hh], [x + hh, y - hh], [x - hh, y - hh]]
        src = np.array(sq2(-o, o) + sq2(o, o) + sq2(o, -o) + sq2(-o, -o), np.float64)
        e = []
        for d in use:
            dst = cv2.undistortPoints(np.vstack([d[i] for i in ids]).reshape(-1, 1, 2), K, dist, P=K).reshape(-1, 2)
            H, _ = cv2.findHomography(src, dst)
            pr = cv2.perspectiveTransform(src.reshape(-1, 1, 2), H).reshape(-1, 2)
            e.append(np.mean(np.sum((pr - dst) ** 2, axis=1)))
        return float(np.sqrt(np.mean(e)))

    sides = np.linspace(0.8, 1.2, 41) * 2 * h
    pe = np.array([plane_err(sd) for sd in sides])
    side_best, pe_best = float(sides[int(np.argmin(pe))]), float(pe.min())

    def fit_err(k):
        """Sai số chiếu lại trung bình (px) nếu tiêu cự thật = k × tiêu cự calib."""
        Kk = K.copy()
        Kk[0, 0] *= k
        Kk[1, 1] *= k
        e = []
        for d in use:
            img = np.vstack([d[i] for i in ids]).astype(np.float64)
            ok, rv, tv = cv2.solvePnP(obj3, img, Kk, dist, flags=cv2.SOLVEPNP_IPPE)
            if ok:
                pr = cv2.projectPoints(obj3, rv, tv, Kk, dist)[0].reshape(-1, 2)
                e.append(np.mean(np.sum((pr - img) ** 2, axis=1)))
        return float(np.sqrt(np.mean(e))) if e else np.nan

    e1 = fit_err(1.0)
    geom_ok = abs(side_best / (2 * h) - 1) <= 0.07
    print(f"  Dùng {len(dets)} lần dò thấy đủ 4 marker. Vị trí bảng tính ra lệch góc marker trên ảnh "
          f"{e1:.2f} px (tốt: khoảng 1 px trở xuống).")
    print(f"  1. Bản in — với tâm marker cách nhau {2 * o:.0f} mm, cạnh marker khớp ảnh nhất là {side_best:.1f} mm "
          f"(khai {2 * h:.0f} mm); khớp mặt phẳng lệch {pe_best:.2f} px.")
    if not geom_ok:
        print(f"     → LỆCH {100 * (side_best / (2 * h) - 1):+.0f}%: tỉ lệ cạnh marker / khoảng cách tâm của bản in không như khai. "
              f"Đo thước rồi chạy với\n       --board-marker-mm / --board-offset-mm. (Chưa kiểm được camera khi bảng còn sai.)")
        return
    print("     → khớp: tỉ lệ kích thước của bản in đúng như khai."
          + (" Nhưng lệch trên 1,5 px: giấy cong / chưa dán phẳng." if pe_best > 1.5 else ""))
    ks = np.linspace(0.6, 1.6, 41)
    es = np.array([fit_err(k) for k in ks])
    b = int(np.nanargmin(es))
    k_best, e_best = float(ks[b]), float(es[b])
    print(f"  2. Camera — tiêu cự khớp các lần dò nhất: {k_best:.2f} × tiêu cự trong file calib "
          f"(lệch còn {e_best:.2f} px, so với {e1:.2f} px khi dùng đúng file calib).")
    if e1 - e_best < 0.15 or e1 < 1.25 * e_best:
        if e1 > max(1.5, 1.5 * pe_best):
            print("     → đổi tiêu cự không làm khớp hơn, mà độ lệch vẫn lớn: nghi giấy cong / chưa dán phẳng, "
                  "hoặc bảng gần chính diện (nghiêng 25–40 độ rồi chạy lại).")
        else:
            print("     → khớp với file calib (hoặc bảng quá chính diện để phân biệt — nghiêng 25–40 độ nếu muốn chắc).")
    else:
        print(f"     → LỆCH {100 * (k_best - 1):+.0f}%: camera trên Pi đang có góc nhìn khác lúc calib (zoom / chế độ ảnh). "
              f"Mọi khoảng cách, cả bút lẫn bảng,\n       đang bị tính ra bằng khoảng {1 / k_best:.2f} lần giá trị thật. "
              f"Gửi kết quả `v4l2-ctl -d /dev/video0 --list-ctrls`.")
        if e_best > 1.0:
            print("       (Sau khi chỉnh tiêu cự vẫn còn lệch trên 1 px: có thể thêm giấy cong.)")


# ── các đoạn bút đứng yên ───────────────────────────────────────────────────
def still_segments(run, min_s=1.5, thr_px=5.0, gap_s=0.5):
    """Danh sách (i0, i1) các đoạn mà điểm Tip trên ảnh nằm yên trong bán kính
    thr_px (tính ở ảnh rộng 640) ít nhất min_s giây."""
    c = run["c"]
    if "tip_u" not in c:
        return []
    thr = thr_px * run["meta"].get("width", 640) / 640.0
    t, u, v = c["t"], c["tip_u"], c["tip_v"]
    segs, start, pts = [], None, []

    def close(end):
        if start is not None and t[end] - t[start] >= min_s:
            segs.append((start, end))

    last = None
    for i in range(run["n"]):
        if np.isnan(u[i]):
            continue
        p = np.array([u[i], v[i]])
        if start is not None and (t[i] - t[last] > gap_s
                                  or np.linalg.norm(p - np.mean(pts, axis=0)) > thr):
            close(last)
            start = None
        if start is None:
            start, pts = i, []
        pts.append(p)
        last = i
    if last is not None:
        close(last)
    return segs


def touch_report(run):
    c, segs = run["c"], still_segments(run)
    if not segs:
        print("  Không có đoạn nào bút đứng yên ≥ 1,5 giây.")
        return
    has_board = "bx" in c and np.any(~np.isnan(c["bx"]))
    print(f"  {len(segs)} đoạn bút đứng yên. Toạ độ mm.")
    if has_board:
        print("  'nếu đang chạm bảng' = kéo đầu bút dọc tia nhìn của camera tới mặt bảng: cho biết bút\n"
              "  đang chạm điểm nào, và khoảng cách bút bị đo lệch mấy lần (1,00 = đúng).\n")
        print(f"  {'':>2} {'':>7} {'':>5} | {'so với camera':^27} | {'so với bảng':^20} | {'nếu đang chạm bảng':^22} |")
        print(f"  {'#':>2} {'lúc(s)':>7} {'dài':>5} | {'X':>6} {'Y':>6} {'Z':>6} {'rung Z':>6} | "
              f"{'x':>6} {'y':>6} {'cách':>6} | {'x':>6} {'y':>6} {'đo/thật':>8} | điểm chuẩn gần nhất")
    else:
        print(f"  {'#':>2} {'lúc(s)':>7} {'dài':>5} | {'X':>6} {'Y':>6} {'Z':>6} (so với camera) | rung X / Y / Z")
    ks, errs_now, errs_touch = [], [], []
    for n, (i0, i1) in enumerate(segs, 1):
        sl = slice(i0, i1 + 1)
        ok = ~np.isnan(c["x"][sl])
        p = np.array([np.nanmean(c[k][sl]) for k in "xyz"])
        sd = np.array([np.nanstd(c[k][sl]) for k in "xyz"])
        head = f"  {n:>2} {c['t'][i0]:>7.1f} {c['t'][i1] - c['t'][i0]:>5.1f} | {p[0]:>+6.0f} {p[1]:>+6.0f} {p[2]:>6.0f}"
        if not has_board or np.all(np.isnan(c["bx"][sl])):
            print(head + f" | ±{sd[0]:.1f} / ±{sd[1]:.1f} / ±{sd[2]:.1f}")
            continue
        b = np.array([np.nanmean(c[k][sl]) for k in ("bx", "by", "bz")])
        mid = i0 + int(np.flatnonzero(~np.isnan(c["brd_x"][sl]))[len(np.flatnonzero(~np.isnan(c["brd_x"][sl]))) // 2])
        R, t = board_T(c, mid)
        nrm = R[:, 2]
        k = float(nrm @ t) / float(nrm @ p)          # tỉ lệ kéo dọc tia nhìn để chạm mặt bảng
        pt = R.T @ (k * p - t)
        name, ref = min(REFS.items(), key=lambda kv: np.hypot(pt[0] - kv[1][0], pt[1] - kv[1][1]))
        e_touch = np.hypot(pt[0] - ref[0], pt[1] - ref[1])
        e_now = np.linalg.norm(b - [ref[0], ref[1], 0])
        ks.append(1.0 / k); errs_now.append(e_now); errs_touch.append(e_touch)
        print(head + f" {sd[2]:>6.1f} | {b[0]:>+6.0f} {b[1]:>+6.0f} {b[2]:>6.0f} | "
              f"{pt[0]:>+6.0f} {pt[1]:>+6.0f} {1.0 / k:>8.2f} | {name}: lệch {e_now:.0f} mm "
              f"(nếu chạm: {e_touch:.0f} mm trên mặt bảng)")
    if ks:
        ks = np.array(ks)
        print(f"\n  Sai số so với điểm chuẩn gần nhất: trung bình {np.mean(errs_now):.0f} mm, "
              f"lớn nhất {np.max(errs_now):.0f} mm.")
        print(f"  Tỉ lệ khoảng cách đo/thật (giả định mọi đoạn trên là lúc mũi bút CHẠM bảng): "
              f"trung bình {ks.mean():.2f}, dao động {ks.min():.2f}–{ks.max():.2f}.")
        if len(ks) < 3:
            print("  (Cần ít nhất 3 lần giữ yên ở các điểm khác nhau mới kết luận được về kích thước bút.)")
        elif abs(ks.mean() - 1) > 0.05 and ks.std() < 0.08 * ks.mean():
            pen = np.array(run["meta"].get("pen_3d", [[0, 0, 0], [0, 64, 0], [-11.5, 44, 0], [11.5, 44, 0]]))
            dims = np.array([pen[1, 1], pen[2, 1], pen[3, 0] - pen[2, 0]]) / ks.mean()
            print(f"  → Tỉ lệ ổn định và khác 1: khả năng cao kích thước bút khai trong node lớn/nhỏ hơn bút thật\n"
                  f"    {ks.mean():.2f} lần. Kích thước cho khớp: --pen-dims-mm {dims[0]:.1f} {dims[1]:.1f} {dims[2]:.1f}\n"
                  f"    (kiểm lại bằng thước trước khi tin; thử ngay: thêm cờ đó vào lệnh analyze_run.py này).")
        elif abs(ks.mean() - 1) > 0.05:
            print("  → Tỉ lệ khác 1 nhưng KHÔNG ổn định giữa các điểm: không phải chỉ do kích thước bút — "
                  "model có thể đang đặt điểm khớp sai chỗ, hoặc có đoạn bút không chạm bảng.")


# ── bảng số liệu ────────────────────────────────────────────────────────────
def summary(run, skip_s):
    c, m = run["c"], run["meta"]
    t = c["t"]
    keep = t >= min(skip_s, t[-1] * 0.2) if run["n"] > 5 else np.ones(run["n"], bool)
    T = t[keep]
    dur = T[-1] - T[0] if len(T) > 1 else 0.0

    def col(k):
        return c[k][keep] if k in c else np.array([np.nan])

    def med(k):
        v = col(k)
        return np.nan if np.all(np.isnan(v)) else float(np.nanmedian(v))

    def p95(k):
        v = col(k)
        return np.nan if np.all(np.isnan(v)) else float(np.nanpercentile(v, 95))

    def mean(k):
        v = col(k)
        return np.nan if np.all(np.isnan(v)) else float(np.nanmean(v))

    out = {}
    out["Độ phân giải bắt ảnh"] = f"{m.get('width', '?')}x{m.get('height', '?')}"
    bc = m.get("board_cfg") or {}
    out["Dò bảng"] = f"có, mỗi {bc.get('every', '?')} frame" if m.get("board") else "không"
    out["Thời lượng tính (s)"] = dur
    out["Số frame xử lý"] = float(len(T))
    out["Tốc độ xử lý (fps)"] = (len(T) - 1) / dur if dur > 0 else np.nan
    dt = np.diff(T) * 1000
    out["Khoảng cách 2 frame, 5% tệ nhất (ms)"] = float(np.percentile(dt, 95)) if len(dt) else np.nan
    out["Bắt được bút (% frame)"] = float(np.nanmean(col("det")) * 100)
    out["Giải mã ảnh camera (ms)"] = med("dec_ms")
    out["Frame chờ tới lượt model (ms)"] = med("wait_ms")
    out["Model: thu nhỏ ảnh (ms)"] = med("pre_ms")
    out["Model: chạy (ms)"] = med("infer_ms")
    out["Trễ camera→XYZ, giữa (ms)"] = med("xyz_ms")
    out["Trễ camera→XYZ, 5% tệ nhất (ms)"] = p95("xyz_ms")
    out["Trễ camera→ảnh stream (ms)"] = med("img_ms")
    if m.get("board") and "aruco_n" in c:
        an = col("aruco_n")
        new = np.r_[False, np.diff(an) > 0]
        am = col("aruco_ms")[new]
        out["Dò bảng: mỗi lần (ms)"] = float(np.nanmedian(am)) if len(am) and not np.all(np.isnan(am)) else np.nan
        out["Dò bảng: số lần mỗi giây"] = (np.nanmax(an) - np.nanmin(an)) / dur if dur > 0 else np.nan
        seen = [s for s, k in zip(c["brd_seen"], keep) if k]
        out["Bảng: thấy đủ 4 marker (% frame)"] = 100.0 * sum(len(s) == 4 for s in seen) / max(len(seen), 1)
        out["Bảng: sai số chiếu lại (px)"] = med("brd_err")
    out["CPU cả máy (%)"] = mean("cpu_sys")
    out["CPU node (% của 1 nhân)"] = mean("cpu_node")
    out["RAM node (MB)"] = mean("rss_mb")
    out["Nhiệt độ trung bình (°C)"] = mean("temp")
    v = col("temp")
    out["Nhiệt độ cao nhất (°C)"] = np.nan if np.all(np.isnan(v)) else float(np.nanmax(v))
    segs = [s for s in still_segments(run) if t[s[0]] >= T[0]]
    if segs:
        sd = np.array([[np.nanstd(c[k][a:b + 1]) for k in "xyz"] for a, b in segs])
        out["Rung khi bút đứng yên: X (mm)"] = float(np.median(sd[:, 0]))
        out["Rung khi bút đứng yên: Y (mm)"] = float(np.median(sd[:, 1]))
        out["Rung khi bút đứng yên: Z (mm)"] = float(np.median(sd[:, 2]))
    return out


def fmt(v):
    if isinstance(v, str):
        return v
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "--"
    return f"{v:.0f}" if abs(v) >= 100 else f"{v:.1f}"


def print_table(runs, sums):
    keys = list(max(sums, key=len))                 # thứ tự dòng theo lượt có nhiều mục nhất
    for s in sums:
        keys += [k for k in s if k not in keys]
    w0 = max(len(k) for k in keys) + 1
    names = [r["name"][:22] for r in runs]
    w = max([14] + [len(n) + 2 for n in names] + [len(fmt(v)) + 2 for sm in sums for v in sm.values()])
    diff = len(runs) == 2
    print(f"{'':<{w0}}" + "".join(f"{n:>{w}}" for n in names) + (f"{'chênh (2−1)':>{w}}" if diff else ""))
    for k in keys:
        vals = [s.get(k) for s in sums]
        line = f"{k:<{w0}}" + "".join(f"{fmt(v):>{w}}" for v in vals)
        if diff and all(isinstance(v, float) and not np.isnan(v) for v in vals):
            d = vals[1] - vals[0]
            line += f"{('+' if d >= 0 else '') + fmt(d):>{w}}"
        print(line)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="*")
    ap.add_argument("--last", type=int, nargs="?", const=1, default=0,
                    help="Lấy N lượt mới nhất trong --log-dir (mặc định 1)")
    ap.add_argument("--log-dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs"))
    ap.add_argument("--skip", type=float, default=3.0, help="Bỏ N giây đầu khi tính số liệu (khởi động)")
    ap.add_argument("--pen-dims-mm", type=float, nargs=3, default=None, metavar=("DAI", "VI_TRI_DIA", "RONG"),
                    help="Tính lại XYZ từ điểm khớp đã log với kích thước bút này")
    ap.add_argument("--calib", default=None,
                    help="Tính lại pose bảng và XYZ bút với file calib (.npz) khác")
    args = ap.parse_args()

    files = list(args.files)
    if args.last:
        found = sorted(glob.glob(os.path.join(args.log_dir, "run_*.csv")))
        files += found[-args.last:]
    if not files:
        ap.error("chưa chỉ file log nào (đưa đường dẫn, hoặc dùng --last)")
    runs = [r for r in (load(f) for f in files) if r["n"] > 1]
    if not runs:
        sys.exit("Các file log đều rỗng.")
    if args.pen_dims_mm or args.calib:
        for r in runs:
            recompute(r, args.pen_dims_mm, args.calib)
        print("(Đã tính lại từ dữ liệu thô trong log"
              + (f", file calib {args.calib}" if args.calib else "")
              + (f", kích thước bút {args.pen_dims_mm} mm" if args.pen_dims_mm else "")
              + "; XYZ bút không qua bộ lọc.)\n")

    for i, r in enumerate(runs, 1):
        print(f"[{i}] {r['name']}: {r['path']}  (bắt đầu {r['meta'].get('start', '?')}, {r['n']} dòng)")
    print()
    print_table(runs, [summary(r, args.skip) for r in runs])
    for r in runs:
        if r["meta"].get("board"):
            print(f"\n── Kiểm bảng — {r['name']} " + "─" * 40)
            board_check(r)
        print(f"\n── Bút đứng yên — {r['name']} " + "─" * 40)
        touch_report(r)


if __name__ == "__main__":
    main()
