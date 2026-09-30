#!/usr/bin/env python3
"""Đo tài nguyên Pi 4 mà node vision (và các tiến trình đi kèm) chiếm dụng.

Chỉ dùng thư viện chuẩn (đọc /proc), không cần cài thêm gì. Chạy trong lúc
run_pi4_ros2.py đang chạy, ở một terminal SSH khác:

    python3 scripts/monitor_resources.py --seconds 60
    python3 scripts/monitor_resources.py --seconds 120 --csv res.csv

In mỗi giây 1 dòng, hết thời gian thì in bảng tổng hợp trung bình / cao nhất.

Cách đọc CPU%: tính theo 1 nhân, nên 1 tiến trình dùng trọn 2 nhân = 200%.
Pi 4 có 4 nhân, tổng tối đa của cả máy là 400%.
"""
import argparse
import csv
import os
import subprocess
import time

# Nhóm tiến trình cần theo dõi: tên hiển thị -> chuỗi cần có trong dòng lệnh.
GROUPS = {
    "vision (run_pi4_ros2.py)": "run_pi4_ros2.py",
    "web_video_server": "web_video_server",
    "ros2 CLI (topic echo/hz)": "ros2 topic",
}
TICK = os.sysconf("SC_CLK_TCK")
PAGE_MB = os.sysconf("SC_PAGE_SIZE") / 1e6


def read_cmdline(pid):
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            return f.read().replace(b"\0", b" ").decode(errors="ignore")
    except OSError:
        return ""


def proc_times(pid):
    """(utime+stime tính bằng tick, RSS MB, số thread) hoặc None nếu đã thoát."""
    try:
        with open(f"/proc/{pid}/stat") as f:
            parts = f.read().rsplit(")", 1)[1].split()
        with open(f"/proc/{pid}/statm") as f:
            rss_pages = int(f.read().split()[1])
    except (OSError, IndexError, ValueError):
        return None
    # Sau ")" các trường bắt đầu từ field 3; utime=14, stime=15, threads=20
    return int(parts[11]) + int(parts[12]), rss_pages * PAGE_MB, int(parts[17])


def find_pids():
    out = {g: [] for g in GROUPS}
    me = os.getpid()
    for d in os.listdir("/proc"):
        if not d.isdigit() or int(d) == me:
            continue
        cmd = read_cmdline(d)
        for g, pat in GROUPS.items():
            if pat in cmd and "grep" not in cmd:
                out[g].append(int(d))
    return out


def cpu_totals():
    """Danh sách (busy, total) cho toàn máy và từng nhân."""
    res = []
    with open("/proc/stat") as f:
        for line in f:
            if not line.startswith("cpu"):
                break
            v = [int(x) for x in line.split()[1:]]
            idle = v[3] + v[4]
            res.append((sum(v) - idle, sum(v)))
    return res


def mem_used_mb():
    info = {}
    with open("/proc/meminfo") as f:
        for line in f:
            k, v = line.split(":")
            info[k] = int(v.split()[0]) / 1024
    return info["MemTotal"] - info["MemAvailable"], info["MemTotal"]


def temp_c():
    try:
        with open("/sys/class/thermal/thermal_zone0/temp") as f:
            return int(f.read()) / 1000
    except OSError:
        return float("nan")


def throttled():
    """Cờ hạ xung của Pi (vcgencmd get_throttled). 0x0 = chưa từng bị hạ."""
    try:
        r = subprocess.run(["vcgencmd", "get_throttled"], capture_output=True,
                           text=True, timeout=2)
        return r.stdout.strip().split("=")[-1]
    except (OSError, subprocess.SubprocessError):
        return "n/a"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=int, default=60)
    ap.add_argument("--interval", type=float, default=1.0)
    ap.add_argument("--csv", default=None, help="Ghi từng mẫu ra file CSV")
    args = ap.parse_args()

    pids = find_pids()
    for g, ps in pids.items():
        print(f"  {g:28s}: {'PID ' + ','.join(map(str, ps)) if ps else 'KHÔNG chạy'}")
    if not pids["vision (run_pi4_ros2.py)"]:
        print("⚠️  Không thấy run_pi4_ros2.py đang chạy — hãy bật node trước.")

    prev_p = {pid: proc_times(pid) for ps in pids.values() for pid in ps}
    prev_c = cpu_totals()
    hist = {g: {"cpu": [], "rss": []} for g in GROUPS}
    hist_sys = {"cpu": [], "mem": [], "temp": []}
    writer = None
    if args.csv:
        fcsv = open(args.csv, "w", newline="")
        writer = csv.writer(fcsv)
        writer.writerow(["t"] + [f"{g} cpu%" for g in GROUPS] + [f"{g} rss_mb" for g in GROUPS]
                        + ["may cpu%", "ram_mb", "temp_c"])

    t0 = time.time()
    n_core_all = len(prev_c) - 1
    print(f"\nĐo {args.seconds}s... (CPU% tính theo 1 nhân; máy có {n_core_all} nhân "
          f"nên tối đa {n_core_all*100}%)")
    while time.time() - t0 < args.seconds:
        time.sleep(args.interval)
        cur_c = cpu_totals()
        dt_tick = cur_c[0][1] - prev_c[0][1]
        n_core = len(cur_c) - 1
        sys_cpu = (cur_c[0][0] - prev_c[0][0]) / max(dt_tick, 1) * 100 * n_core
        cores = [(c[0] - p[0]) / max(c[1] - p[1], 1) * 100 for c, p in zip(cur_c[1:], prev_c[1:])]
        prev_c = cur_c
        elapsed_ticks = dt_tick / n_core

        row_cpu, row_rss = {}, {}
        for g, ps in pids.items():
            cpu = rss = 0.0
            for pid in ps:
                now = proc_times(pid)
                if now and prev_p.get(pid):
                    cpu += (now[0] - prev_p[pid][0]) / max(elapsed_ticks, 1) * 100
                    rss += now[1]
                prev_p[pid] = now
            row_cpu[g], row_rss[g] = cpu, rss
            if ps:
                hist[g]["cpu"].append(cpu); hist[g]["rss"].append(rss)

        used, total = mem_used_mb()
        tc = temp_c()
        hist_sys["cpu"].append(sys_cpu); hist_sys["mem"].append(used); hist_sys["temp"].append(tc)
        vis = row_cpu["vision (run_pi4_ros2.py)"]
        print(f"  vision {vis:5.0f}%  web {row_cpu['web_video_server']:4.0f}%  "
              f"máy {sys_cpu:5.0f}%/{n_core*100}  nhân [{' '.join(f'{c:3.0f}' for c in cores)}]  "
              f"RAM {used:4.0f}/{total:.0f}MB  {tc:.1f}°C")
        if writer:
            writer.writerow([round(time.time() - t0, 1)] + [round(row_cpu[g], 1) for g in GROUPS]
                            + [round(row_rss[g], 1) for g in GROUPS]
                            + [round(sys_cpu, 1), round(used), round(tc, 1)])

    print("\n══════════ TỔNG HỢP ══════════")
    print(f"{'Tiến trình':30s} {'CPU TB':>8s} {'CPU max':>8s} {'% cả máy':>9s} {'RAM':>8s}")
    for g, h in hist.items():
        if not h["cpu"]:
            continue
        avg = sum(h["cpu"]) / len(h["cpu"])
        print(f"{g:30s} {avg:7.0f}% {max(h['cpu']):7.0f}% {avg/n_core_all:8.0f}% "
              f"{sum(h['rss'])/len(h['rss']):6.0f}MB")
    c = hist_sys["cpu"]
    print(f"{'Cả máy':30s} {sum(c)/len(c):7.0f}% {max(c):7.0f}% {sum(c)/len(c)/n_core_all:8.0f}% "
          f"{max(hist_sys['mem']):6.0f}MB (cao nhất)")
    t = hist_sys["temp"]
    print(f"\nNhiệt độ CPU: TB {sum(t)/len(t):.1f}°C, cao nhất {max(t):.1f}°C "
          f"(Pi 4 bắt đầu tự hạ xung ở ~80°C)")
    th = throttled()
    note = {"0x0": "chưa từng bị hạ xung/thiếu điện",
            "n/a": "không đọc được (máy này không có vcgencmd)"}.get(th, "CÓ vấn đề — xem giải thích bên dưới")
    print(f"Cờ hạ xung (vcgencmd get_throttled): {th} → {note}")
    if th not in ("0x0", "n/a"):
        print("  bit 0/16: thiếu điện áp (nguồn yếu) · bit 2/18: đang/đã bị hạ xung · "
              "bit 3/19: chạm giới hạn nhiệt")


if __name__ == "__main__":
    main()
