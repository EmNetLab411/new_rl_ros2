#!/usr/bin/env python3
"""
Gán nhãn 4 keypoint (Tip, Tail, L, R) trực tiếp trên máy, không cần Roboflow.

Chạy (từ thư mục CoVip):
    python3 scripts/label_local.py --dirs "nghieng_xa_*" --every 3
    python3 scripts/label_local.py --dirs "nghieng_xa_*" --every 3 --review  # duyệt/sửa ảnh đã gán
    python3 scripts/label_local.py --dirs "nghieng_xa_*" --every 3 --merge   # gộp vào dataset_split

--review: hiện từng ảnh đã gán kèm 4 điểm (mặc định phóng to quanh bút).
    Enter/Space : đúng (hoặc lưu bản đã sửa), sang ảnh tiếp
    kéo 1 điểm  : nhấn chuột gần điểm đó rồi kéo tới chỗ đúng
    r           : xoá cả 4 điểm, click lại Tip -> Tail -> L -> R
    d           : xoá nhãn (sai quá, không sửa được)    e : ảnh không có bút
    u           : lùi lại    f : phóng to / cả khung    q : thoát (lần sau tiếp tục)

--dirs chọn thư mục con trong datasets/frames (glob). --every N chỉ đưa ra 1
ảnh mỗi N ảnh (các ảnh liền nhau gần giống hệt, gán hết không thêm gì mấy).
Ảnh đưa ra xen kẽ giữa các video để dừng giữa chừng vẫn đủ mọi góc nghiêng.

Quy ước điểm (giống dataset gốc):
    Tip  = mũi kim loại nhọn         Tail = đáy khối nhựa xanh (phía sau)
    L, R = mép trái / phải của đĩa xanh, tính theo BÚT: tưởng tượng xoay ảnh
           cho mũi bút hướng lên trên, bên trái là L (khớp PEN_3D của solvePnP)

Điều khiển:
    Click chuột trái : đặt điểm tiếp theo (Tip -> Tail -> L -> R)
    o                : đánh dấu điểm VỪA đặt là bị che (vẫn click vị trí ước lượng)
    x                : điểm hiện tại không thấy/không đoán được -> bỏ, sang điểm kế
    z                : undo điểm vừa đặt
    Enter / n        : lưu ảnh này, sang ảnh tiếp
    e                : ảnh KHÔNG có bút -> lưu làm ảnh "trống" (dạy model không
                       nhận nhầm nền, vd cái thuyền xanh) rồi sang ảnh tiếp
    s                : bỏ ảnh (bút nhoè/khó quá), không lưu
    u                : quay lại ảnh trước để làm lại
    q / ESC          : lưu tiến độ và thoát; chạy lại sẽ tiếp tục từ chỗ dừng

--merge: chép ảnh đã gán vào dataset_split/{train,val}. Val lấy ~15% CUỐI mỗi
video (không bốc ngẫu nhiên — ảnh liền nhau gần giống hệt, bốc ngẫu nhiên sẽ
làm val chấm trên ảnh gần như đã train). Chạy lại make_split.py sẽ xoá phần
gộp này, khi đó chạy lại --merge.
"""
import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
FRAMES_DIR = ROOT / "datasets" / "frames"
LABEL_OUT = ROOT / "datasets" / "local_labels"
PROGRESS_FILE = LABEL_OUT / "_progress.json"
SPLIT_DIR = ROOT / "dataset_split"

NAMES = ["Tip", "Tail", "L", "R"]
COLORS = [(0, 255, 0), (0, 200, 255), (255, 100, 0), (100, 0, 255)]
DISPLAY_MAX_W = 1280


def load_progress():
    if PROGRESS_FILE.exists():
        p = json.loads(PROGRESS_FILE.read_text())
        p.setdefault("empty", [])
        p.setdefault("auto", [])
        return p
    return {"done": [], "skipped": [], "empty": [], "auto": []}


def save_progress(prog):
    LABEL_OUT.mkdir(parents=True, exist_ok=True)
    PROGRESS_FILE.write_text(json.dumps(prog, ensure_ascii=False, indent=0))


def forget(prog, stem):
    for k in ("done", "skipped", "empty", "auto"):
        if stem in prog[k]:
            prog[k].remove(stem)


def write_yolo_label(path, points, occluded, img_shape):
    h, w = img_shape[:2]
    pts = [p for p in points if p is not None]
    if len(pts) < 2:
        return False
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    bw, bh = max((x1 - x0) * 1.4, 20), max((y1 - y0) * 1.4, 20)
    parts = [0, cx / w, cy / h, bw / w, bh / h]
    for p, occ in zip(points, occluded):
        if p is None:
            parts += [0, 0, 0]
        else:
            parts += [p[0] / w, p[1] / h, 1 if occ else 2]
    path.write_text(" ".join(f"{v:.6f}" if isinstance(v, float) else str(v) for v in parts) + "\n")
    return True


class Labeler:
    def __init__(self, files):
        self.files = files
        self.idx = 0
        self.scale = 1.0
        self.img = None
        cv2.namedWindow("label", cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback("label", self.on_mouse)

    def reset_points(self):
        self.points = [None] * 4
        self.occluded = [False] * 4
        self.cur = 0          # điểm kế tiếp cần đặt (x có thể nhảy qua 1 điểm)

    def on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and self.cur < 4:
            self.points[self.cur] = (x / self.scale, y / self.scale)
            self.cur += 1
            self.redraw()

    def load(self):
        self.reset_points()
        self.img = cv2.imread(str(self.files[self.idx]))
        h, w = self.img.shape[:2]
        self.scale = min(DISPLAY_MAX_W / w, 1.0)
        self.redraw()

    def redraw(self):
        h, w = self.img.shape[:2]
        disp = cv2.resize(self.img, (int(w * self.scale), int(h * self.scale)))
        for i, p in enumerate(self.points):
            if p is None:
                continue
            x, y = int(p[0] * self.scale), int(p[1] * self.scale)
            marker = cv2.MARKER_TRIANGLE_UP if self.occluded[i] else cv2.MARKER_CROSS
            cv2.drawMarker(disp, (x, y), COLORS[i], marker, 22, 2)
            cv2.putText(disp, NAMES[i], (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS[i], 2)
        nxt = f"click {NAMES[self.cur]}" if self.cur < 4 else "du diem - Enter de luu"
        cv2.putText(disp, f"[{self.idx+1}/{len(self.files)}] {self.files[self.idx].parent.name[:16]}  {nxt}",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
        cv2.putText(disp, f"[{self.idx+1}/{len(self.files)}] {self.files[self.idx].parent.name[:16]}  {nxt}",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        help_ = "click=diem  o=bi che  x=bo diem  z=undo  Enter=luu  e=KHONG co but  s=bo anh  u=lui  q=thoat"
        cv2.putText(disp, help_, (10, disp.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
        cv2.putText(disp, help_, (10, disp.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.imshow("label", disp)

    def run(self):
        prog = load_progress()
        seen = set(prog["done"]) | set(prog["skipped"]) | set(prog["empty"]) | set(prog["auto"])
        LABEL_OUT.mkdir(parents=True, exist_ok=True)
        while 0 <= self.idx < len(self.files):
            f = self.files[self.idx]
            if f.stem in seen:
                self.idx += 1
                continue
            self.load()
            while True:
                key = cv2.waitKey(20) & 0xFF
                if key == ord('o') and self.cur > 0 and self.points[self.cur - 1] is not None:
                    self.occluded[self.cur - 1] = not self.occluded[self.cur - 1]
                    self.redraw()
                elif key == ord('z') and self.cur > 0:
                    self.cur -= 1
                    self.points[self.cur] = None
                    self.occluded[self.cur] = False
                    self.redraw()
                elif key == ord('x') and self.cur < 4:
                    self.cur += 1           # để None = không gán được điểm này
                    self.redraw()
                elif key in (13, 10, ord('n')):
                    if write_yolo_label(LABEL_OUT / (f.stem + ".txt"), self.points,
                                        self.occluded, self.img.shape):
                        shutil.copy(f, LABEL_OUT / f.name)
                        prog["done"].append(f.stem); seen.add(f.stem)
                        save_progress(prog)
                        self.idx += 1
                        break
                    print("Cần ít nhất 2 điểm để lưu (hoặc bấm e nếu không có bút, s để bỏ).")
                elif key == ord('e'):
                    (LABEL_OUT / (f.stem + ".txt")).write_text("")
                    shutil.copy(f, LABEL_OUT / f.name)
                    prog["empty"].append(f.stem); seen.add(f.stem)
                    save_progress(prog)
                    self.idx += 1
                    break
                elif key == ord('s'):
                    prog["skipped"].append(f.stem); seen.add(f.stem)
                    save_progress(prog)
                    self.idx += 1
                    break
                elif key == ord('u') and self.idx > 0:
                    # Lùi về ảnh trước và xoá nhãn cũ của nó để làm lại
                    self.idx -= 1
                    prev = self.files[self.idx]
                    forget(prog, prev.stem); seen.discard(prev.stem)
                    for ext in (".txt", prev.suffix):
                        (LABEL_OUT / (prev.stem + ext)).unlink(missing_ok=True)
                    save_progress(prog)
                    break
                elif key in (27, ord('q')):
                    save_progress(prog)
                    cv2.destroyAllWindows()
                    print(f"Đã lưu tiến độ: {len(prog['done'])} ảnh có bút, "
                          f"{len(prog['empty'])} ảnh trống, {len(prog['skipped'])} bỏ qua.")
                    return
        cv2.destroyAllWindows()
        print(f"Xong danh sách. Có bút: {len(prog['done'])}, trống: {len(prog['empty'])}, "
              f"bỏ qua: {len(prog['skipped'])}")


def read_yolo_label(path, img_shape):
    """-> (points, occluded) theo pixel; None nếu file trống/không có."""
    h, w = img_shape[:2]
    v = path.read_text().split() if path.exists() else []
    if len(v) < 17:
        return None
    pts, occ = [], []
    for i in range(4):
        x, y, vis = float(v[5 + 3 * i]), float(v[6 + 3 * i]), int(float(v[7 + 3 * i]))
        pts.append(None if vis == 0 else (x * w, y * h))
        occ.append(vis == 1)
    return pts, occ


class Reviewer:
    """Duyệt ảnh ĐÃ gán (nhãn tự động + gán tay): xem 4 điểm, kéo sửa nếu lệch.

    Mặc định phóng to quanh bút (bút trong ảnh mới thường nhỏ) — f để xem cả
    khung. Kéo 1 điểm: nhấn chuột gần điểm đó rồi kéo. r: xoá hết, click lại
    4 điểm từ đầu. Ảnh sửa xong chuyển từ danh sách "auto" sang "done".
    """
    VIEW_W, VIEW_H = 1280, 720

    def __init__(self, files, prog):
        self.files, self.prog = files, prog
        self.idx = 0
        self.zoom = True
        cv2.namedWindow("review", cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback("review", self.on_mouse)

    # --- toạ độ: ảnh gốc <-> màn hình (khung nhìn = vùng cắt x0,y0 + tỉ lệ s)
    def set_view(self):
        h, w = self.img.shape[:2]
        pts = [p for p in self.points if p is not None]
        if self.zoom and pts:
            xs, ys = [p[0] for p in pts], [p[1] for p in pts]
            size = max(max(xs) - min(xs), max(ys) - min(ys), 60) * 2.2
            cw = min(max(size * 16 / 9, 320), w); ch = cw * 9 / 16
            cx, cy = (max(xs) + min(xs)) / 2, (max(ys) + min(ys)) / 2
            self.x0 = min(max(cx - cw / 2, 0), w - cw); self.y0 = min(max(cy - ch / 2, 0), h - ch)
            self.s = self.VIEW_W / cw
        else:
            self.x0 = self.y0 = 0; self.s = min(self.VIEW_W / w, 1.0)

    def to_img(self, x, y):
        return (x / self.s + self.x0, y / self.s + self.y0)

    def to_disp(self, p):
        return (int((p[0] - self.x0) * self.s), int((p[1] - self.y0) * self.s))

    def on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.cur < 4:                         # đang click lại từ đầu (r)
                self.points[self.cur] = self.to_img(x, y); self.cur += 1
                self.modified = True; self.redraw(); return
            d = [np.hypot(*(np.subtract(self.to_disp(p), (x, y)))) if p else 1e9
                 for p in self.points]
            i = int(np.argmin(d))
            self.drag = i if d[i] < 25 else None
            if self.drag is not None:
                self.last = i
        elif event == cv2.EVENT_MOUSEMOVE and self.drag is not None:
            self.points[self.drag] = self.to_img(x, y); self.modified = True; self.redraw()
        elif event == cv2.EVENT_LBUTTONUP:
            self.drag = None

    def load(self):
        f = self.files[self.idx]
        self.img = cv2.imread(str(f))
        lab = read_yolo_label(LABEL_OUT / f"{f.stem}.txt", self.img.shape)
        self.points, self.occluded = lab if lab else ([None] * 4, [False] * 4)
        self.cur, self.drag, self.modified, self.last = 4, None, False, None
        self.set_view(); self.redraw()

    def redraw(self):
        h, w = self.img.shape[:2]
        cw, ch = int(self.VIEW_W / self.s), int(self.VIEW_H / self.s)
        crop = self.img[int(self.y0):int(self.y0) + min(ch, h), int(self.x0):int(self.x0) + min(cw, w)]
        disp = cv2.resize(crop, (int(crop.shape[1] * self.s), int(crop.shape[0] * self.s)))
        poly = [self.to_disp(p) for p in (self.points[0], self.points[2], self.points[1], self.points[3]) if p]
        if len(poly) > 2:
            cv2.polylines(disp, [np.array(poly)], True, (255, 255, 255), 1, cv2.LINE_AA)
        for i, p in enumerate(self.points):
            if p is None:
                continue
            q = self.to_disp(p)
            marker = cv2.MARKER_TRIANGLE_UP if self.occluded[i] else cv2.MARKER_CROSS
            cv2.drawMarker(disp, q, COLORS[i], marker, 26, 2)
            cv2.putText(disp, NAMES[i], (q[0] + 12, q[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
            cv2.putText(disp, NAMES[i], (q[0] + 12, q[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLORS[i], 2)
        f = self.files[self.idx]
        src = "tu dong" if f.stem in self.prog["auto"] else "gan tay"
        state = f"click {NAMES[self.cur]}" if self.cur < 4 else ("DA SUA - Enter de luu" if self.modified else "")
        top = f"[{self.idx+1}/{len(self.files)}] {f.stem} ({src})  {state}"
        help_ = ("L/R theo BUT  Enter=dung/luu  keo=sua  r=lai 4 diem  o=bi che  d=xoa  "
                 "e=khong co but  u=lui  f=zoom  q=thoat")
        for t, y, c in ((top, 28, (255, 255, 255)), (help_, disp.shape[0] - 12, (0, 255, 255))):
            sc = 0.7 if y == 28 else 0.5
            cv2.putText(disp, t, (10, y), cv2.FONT_HERSHEY_SIMPLEX, sc, (0, 0, 0), 4)
            cv2.putText(disp, t, (10, y), cv2.FONT_HERSHEY_SIMPLEX, sc, c, 1 if y != 28 else 2)
        cv2.imshow("review", disp)

    def _move(self, stem, to):
        for k in ("auto", "done", "empty", "skipped"):
            if stem in self.prog[k]:
                self.prog[k].remove(stem)
        if to:
            self.prog[to].append(stem)

    def run(self):
        rv = self.prog.setdefault("reviewed", [])
        while 0 <= self.idx < len(self.files):
            f = self.files[self.idx]
            if f.stem in rv:
                self.idx += 1; continue
            self.load()
            while True:
                key = cv2.waitKey(20) & 0xFF
                if key in (13, 10, ord('n'), ord(' ')):
                    if self.modified:
                        if not write_yolo_label(LABEL_OUT / f"{f.stem}.txt", self.points,
                                                self.occluded, self.img.shape):
                            print("Cần ít nhất 2 điểm (hoặc d để xoá, e nếu không có bút)."); continue
                        self._move(f.stem, "done")
                    rv.append(f.stem); self.idx += 1; break
                elif key == ord('r'):
                    self.points, self.occluded, self.cur = [None] * 4, [False] * 4, 0
                    self.modified = True; self.redraw()
                elif key == ord('z') and self.cur < 4 and self.cur > 0:
                    self.cur -= 1; self.points[self.cur] = None; self.redraw()
                elif key == ord('x') and self.cur < 4:
                    self.cur += 1; self.redraw()
                elif key == ord('o'):
                    # điểm vừa click (chế độ r) hoặc điểm vừa kéo
                    i = self.cur - 1 if 0 < self.cur < 4 else self.last
                    if i is None:
                        print("o: kéo (hoặc click) 1 điểm trước, rồi bấm o"); continue
                    self.occluded[i] = not self.occluded[i]; self.modified = True; self.redraw()
                elif key == ord('f'):
                    self.zoom = not self.zoom; self.set_view(); self.redraw()
                elif key == ord('d'):
                    for ext in (".txt", f.suffix):
                        (LABEL_OUT / f"{f.stem}{ext}").unlink(missing_ok=True)
                    self._move(f.stem, "skipped"); rv.append(f.stem); self.idx += 1; break
                elif key == ord('e'):
                    (LABEL_OUT / f"{f.stem}.txt").write_text("")
                    self._move(f.stem, "empty"); rv.append(f.stem); self.idx += 1; break
                elif key == ord('u') and self.idx > 0:
                    self.idx -= 1
                    while self.idx > 0 and self.files[self.idx].stem not in rv:
                        self.idx -= 1
                    if self.files[self.idx].stem in rv:
                        rv.remove(self.files[self.idx].stem)
                    break
                elif key in (27, ord('q')):
                    save_progress(self.prog); cv2.destroyAllWindows()
                    print(f"Đã duyệt {len(rv)} ảnh. Nhớ chạy lại --merge để cập nhật dataset_split.")
                    return
            save_progress(self.prog)
        cv2.destroyAllWindows()
        print(f"Duyệt xong {len(rv)} ảnh. Chạy lại --merge để cập nhật dataset_split.")


def select_files(pattern, every):
    """1 ảnh mỗi `every` ảnh của mỗi video, rồi xếp xen kẽ giữa các video."""
    per_dir = [sorted(d.glob("*.jpg"))[::every]
               for d in sorted(FRAMES_DIR.glob(pattern)) if d.is_dir()]
    out = []
    for i in range(max((len(x) for x in per_dir), default=0)):
        out += [x[i] for x in per_dir if i < len(x)]
    return out


def merge(files, val_frac=0.15):
    prog = load_progress()
    keep = set(prog["done"]) | set(prog["empty"]) | set(prog["auto"])
    by_dir = {}
    for f in files:
        if f.stem in keep:
            by_dir.setdefault(f.parent.name, []).append(f)
    # Xoá bản gộp cũ trước: ảnh đã bị xoá/đổi thành "trống" lúc duyệt không
    # được sót lại trong dataset_split.
    for old in SPLIT_DIR.glob("*/*/local_*"):
        old.unlink()
    n = {"train": 0, "val": 0}
    for d, fs in by_dir.items():
        n_val = int(round(len(fs) * val_frac))
        for i, f in enumerate(fs):
            split = "val" if i >= len(fs) - n_val else "train"
            for sub, src in (("images", LABEL_OUT / f.name), ("labels", LABEL_OUT / (f.stem + ".txt"))):
                dst = SPLIT_DIR / split / sub / ("local_" + src.name)
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(src, dst)
            n[split] += 1
    print(f"Đã gộp vào {SPLIT_DIR}: train +{n['train']}, val +{n['val']} "
          f"(tiền tố 'local_'; chạy lại lệnh này không bị trùng, chỉ ghi đè).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", default="*", help="Glob thư mục trong datasets/frames, vd 'nghieng_xa_*'")
    ap.add_argument("--every", type=int, default=1, help="Chỉ lấy 1 ảnh mỗi N ảnh")
    ap.add_argument("--merge", action="store_true", help="Gộp ảnh đã gán vào dataset_split rồi thoát")
    ap.add_argument("--review", action="store_true",
                    help="Duyệt/sửa ảnh ĐÃ gán (nhãn tự động + gán tay)")
    ap.add_argument("--only-auto", action="store_true", help="Với --review: chỉ duyệt nhãn tự động")
    ap.add_argument("--again", action="store_true", help="Với --review: duyệt lại cả ảnh đã duyệt")
    args = ap.parse_args()

    files = select_files(args.dirs, args.every)
    if args.merge:
        merge(files)
        return
    if args.review:
        prog = load_progress()
        if args.again:
            prog["reviewed"] = []
        pool = set(prog["auto"]) if args.only_auto else set(prog["auto"]) | set(prog["done"])
        todo = [f for f in files if f.stem in pool]
        left = sum(f.stem not in set(prog.get("reviewed", [])) for f in todo)
        print(f"{len(todo)} ảnh đã gán, còn {left} ảnh chưa duyệt.")
        Reviewer(todo, prog).run()
        return
    prog = load_progress()
    seen = set(prog["done"]) | set(prog["skipped"]) | set(prog["empty"]) | set(prog["auto"])
    print(f"{len(files)} ảnh trong danh sách, còn {sum(f.stem not in seen for f in files)} ảnh chưa gán.")
    Labeler(files).run()


if __name__ == "__main__":
    main()
