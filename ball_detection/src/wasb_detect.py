"""Zero-shot WASB ball detection on the tactical video, written as link.py candidates.

WASB (Tarashima et al., BMVC 2023, https://github.com/nttcom/WASB-SBDT, MIT) is an HRNet that takes
3 consecutive frames at 512x288 and returns one ball heatmap per frame. Its soccer weights were
trained on the ISSIA-CNR dataset: fixed stadium cameras, 1920x1080, 25 fps, close to our camera.

Feeding it our whole frame shrunk to 512x288 would leave a <1 px ball, so the frame is cut into
crops of (512*zoom) x (288*zoom) native pixels, each resized to 512x288. zoom=1 keeps native
resolution (ball ~4-5 px); zoom=1.5 makes the ball ~3 px, closer to ISSIA at WASB's input size.
Only crops that overlap the pitch are processed.

    python ball_detection/src/wasb_detect.py --start 11185 --end 18684 --zoom 1.5 --name wasb_z15
    python ball_detection/src/wasb_detect.py --halves --zoom 1.5 --name wasb_full            # whole match

Output: data/ball/candidates_<name>.npz, same layout as detect.py. Resumable: work is saved in
500-frame pieces under data/ball/parts_<name>/ and finished pieces are skipped on rerun.
"""

import argparse
import os
import sys
import time

import cv2
import numpy as np
import torch
import yaml

from common import BALL_DIR, PKG, IMAGE_SIZE, PITCH, graded_frames, load_hom, pitch_to_px, stream_frames

WASB_SRC = os.path.join(PKG, "external", "WASB-SBDT", "src")
sys.path.insert(0, WASB_SRC)
from models.hrnet import HRNet  # noqa: E402

# Params
CHUNK = 498            # frames per saved piece; a multiple of 3 so triplets never straddle pieces
IN_W, IN_H = 512, 288
OVERLAP = 0.15         # fraction of a crop shared with its neighbour
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
WEIGHTS_DIR = os.path.join(PKG, "weights")
DEFAULT_WEIGHTS = os.path.join(WEIGHTS_DIR, "wasb_soccer_best.pth.tar")   # weights/download_wasb.sh


class AttrDict(dict):
    """dict with attribute access, all WASB's HRNet needs from a config (no omegaconf dependency,
    so it runs on Kaggle without internet)."""

    def __init__(self, d):
        super().__init__({k: AttrDict(v) if isinstance(v, dict) else v for k, v in d.items()})

    __getattr__ = dict.__getitem__


def load_model(weights, device):
    with open(os.path.join(WASB_SRC, "configs", "model", "wasb.yaml")) as fh:
        cfg = AttrDict(yaml.safe_load(fh))
    model = HRNet(cfg)
    ck = torch.load(weights, map_location="cpu", weights_only=False)
    sd = ck.get("model_state_dict", ck.get("state_dict", ck))
    model.load_state_dict({k.replace("module.", ""): v for k, v in sd.items()}, strict=True)
    return model.to(device).eval()


def crop_origins(H, zoom, margin_m=3.0):
    """(x0, y0, cw, ch) native-pixel crops that overlap the pitch (+margin)."""
    cw, ch = int(round(IN_W * zoom)), int(round(IN_H * zoom))
    W, Hh = IMAGE_SIZE
    mx, my = margin_m / PITCH[0], margin_m / PITCH[1]
    gx, gy = np.meshgrid(np.linspace(-mx, 1 + mx, 40), np.linspace(-my, 1 + my, 26))
    u, v = pitch_to_px(H, gx.ravel(), gy.ravel())
    sx, sy = int(cw * (1 - OVERLAP)), int(ch * (1 - OVERLAP))
    xs = sorted(set(list(range(0, max(W - cw, 0), sx)) + [max(W - cw, 0)]))
    ys = sorted(set(list(range(0, max(Hh - ch, 0), sy)) + [max(Hh - ch, 0)]))
    return [(x0, y0, cw, ch) for y0 in ys for x0 in xs
            if ((u >= x0) & (u < x0 + cw) & (v >= y0) & (v < y0 + ch)).any()]


def to_input(imgs, crop):
    """3 BGR frames -> (9, 288, 512) normalised tensor for one crop."""
    x0, y0, cw, ch = crop
    chans = []
    for img in imgs:
        c = cv2.resize(img[y0:y0 + ch, x0:x0 + cw], (IN_W, IN_H), interpolation=cv2.INTER_AREA if cw > IN_W else cv2.INTER_LINEAR)
        c = (c[:, :, ::-1].astype(np.float32) / 255.0 - MEAN) / STD
        chans.append(c.transpose(2, 0, 1))
    return np.concatenate(chans, axis=0)


def peaks(hm, thresh):
    """Connected blobs above thresh -> [(x, y, peak)] in heatmap pixels, heatmap-weighted centre."""
    if hm.max() <= thresh:
        return []
    n, labels = cv2.connectedComponents((hm > thresh).astype(np.uint8))
    out = []
    for m in range(1, n):
        ys, xs = np.nonzero(labels == m)
        w = hm[ys, xs]
        out.append((float((xs * w).sum() / w.sum()), float((ys * w).sum() / w.sum()), float(w.max())))
    return out


def nms_points(dets, radius=6.0):
    dets = sorted(dets, key=lambda d: -d[2])
    out = []
    for d in dets:
        if all((d[0] - o[0]) ** 2 + (d[1] - o[1]) ** 2 > radius ** 2 for o in out):
            out.append(d)
    return out


def prefetch(start, end, depth=12):
    """stream_frames in a background thread, so video decoding overlaps with GPU work."""
    import queue
    import threading
    q = queue.Queue(maxsize=depth)

    def work():
        for item in stream_frames(start, end):
            q.put(item)
        q.put(None)

    threading.Thread(target=work, daemon=True).start()
    while (item := q.get()) is not None:
        yield item


@torch.no_grad()
def gpu_peaks(hm, thresh, k):
    """(N, h, w) sigmoid heatmaps -> per map up to k local maxima above thresh, with a 3x3
    heatmap-weighted sub-pixel centre. Returns (map_index, x, y, peak) on CPU. Replaces the CPU
    connected-components pass, so only a few numbers leave the GPU instead of every heatmap."""
    import torch.nn.functional as F
    N, h, w = hm.shape
    x = hm.unsqueeze(1)
    is_max = (F.max_pool2d(x, 5, stride=1, padding=2) == x).squeeze(1) & (hm > thresh)
    score = torch.where(is_max, hm, torch.zeros_like(hm)).view(N, -1)
    top, idx = score.topk(k, dim=1)
    keep = top > thresh
    if not keep.any():
        return []
    n_i, k_i = torch.nonzero(keep, as_tuple=True)
    flat = idx[n_i, k_i]
    py, px = flat // w, flat % w
    # 3x3 weighted centroid
    pad = F.pad(hm, (1, 1, 1, 1))
    oy, ox = torch.meshgrid(torch.arange(3, device=hm.device), torch.arange(3, device=hm.device), indexing="ij")
    wts = pad[n_i[:, None, None], (py[:, None, None] + oy), (px[:, None, None] + ox)]
    s = wts.sum((1, 2)).clamp_min(1e-6)
    cx = px + ((wts * (ox - 1)).sum((1, 2)) / s)
    cy = py + ((wts * (oy - 1)).sum((1, 2)) / s)
    return torch.stack([n_i.float(), cx, cy, top[n_i, k_i]], 1).cpu().numpy()


@torch.no_grad()
def run_gpu(model, start, end, args, H, device, rows):
    """Same output as run(), with crop/resize/normalise and peak finding on the GPU."""
    import torch.nn.functional as F
    t0, n = time.time(), 0
    mean = torch.tensor(MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(STD, device=device).view(1, 3, 1, 1)
    buf = []
    crops, crops_key = None, None
    for f, img in prefetch(start, end):
        buf.append((f, img))
        if len(buf) < 3:
            continue
        frames = [b[0] for b in buf]
        # (3, 3, H, W) RGB float in [0, 1], normalised once per frame
        t = torch.from_numpy(np.stack([b[1] for b in buf])).to(device, non_blocking=True)
        t = t.flip(-1).permute(0, 3, 1, 2).float().div_(255.0)
        t = (t - mean) / std
        buf = []
        key = frames[0] // 250
        if key != crops_key:
            crops, crops_key = crop_origins(H[frames[0]], args.zoom), key
        # all crops share one size, so slice + resize as one batch: (n_crops*3, 3, 288, 512)
        cw, ch = crops[0][2], crops[0][3]
        sl = torch.stack([t[:, :, y0:y0 + ch, x0:x0 + cw] for x0, y0, _, _ in crops])  # (C, 3, 3, ch, cw)
        # "area" = pixel averaging, the same as cv2.INTER_AREA used on the CPU path
        sl = F.interpolate(sl.reshape(-1, 3, ch, cw), size=(IN_H, IN_W),
                           mode="area" if cw > IN_W else "bilinear")
        inp = sl.reshape(len(crops), 9, IN_H, IN_W)  # frame-major channels, as to_input()
        hms = torch.cat([model(inp[i:i + args.batch])[0].sigmoid() for i in range(0, len(inp), args.batch)])
        pk = gpu_peaks(hms.reshape(-1, IN_H, IN_W), args.thresh, args.topk)  # map index = crop*3 + frame
        per_frame = [[] for _ in frames]
        for m, x, y, s in pk:
            c, j = divmod(int(m), 3)
            x0, y0, cw_, ch_ = crops[c]
            per_frame[j].append((x0 + x * cw_ / IN_W, y0 + y * ch_ / IN_H, float(s)))
        for j, f in enumerate(frames):
            for u, v, s in nms_points(per_frame[j])[:args.topk]:
                rows.append((f, u, v, 5.0, 5.0, s))
        n += 3
    print(f"  frames {start}..{end}  {n / (time.time() - t0):.1f} fps  {len(rows)} candidates", flush=True)


@torch.no_grad()
def run(model, start, end, args, H, device, rows):
    if device == "cuda" and not args.cpu_prep:
        return run_gpu(model, start, end, args, H, device, rows)
    t0, n = time.time(), 0
    buf = []
    crops, crops_key = None, None
    for f, img in stream_frames(start, end):
        buf.append((f, img))
        if len(buf) < 3:
            continue
        frames, imgs = [b[0] for b in buf], [b[1] for b in buf]
        buf = []
        key = frames[0] // 250
        if key != crops_key:
            crops, crops_key = crop_origins(H[frames[0]], args.zoom), key
        batch = np.stack([to_input(imgs, c) for c in crops])
        hms = []
        for i in range(0, len(batch), args.batch):
            out = model(torch.from_numpy(batch[i:i + args.batch]).to(device))[0]
            hms.append(out.sigmoid().float().cpu().numpy())
        hms = np.concatenate(hms)  # (n_crops, 3, 288, 512)
        for j, f in enumerate(frames):
            dets = []
            for c, (x0, y0, cw, ch) in enumerate(crops):
                for x, y, s in peaks(hms[c, j], args.thresh):
                    dets.append((x0 + x * cw / IN_W, y0 + y * ch / IN_H, s))
            for u, v, s in nms_points(dets)[:args.topk]:
                rows.append((f, u, v, 5.0, 5.0, s))
        n += 3
    print(f"  frames {start}..{end}  {n / (time.time() - t0):.1f} fps  {len(rows)} candidates", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--weights", default=DEFAULT_WEIGHTS)
    p.add_argument("--start", type=int)
    p.add_argument("--end", type=int)
    p.add_argument("--halves", action="store_true", help="both graded halves")
    p.add_argument("--zoom", type=float, default=1.5, help="native pixels per WASB input pixel")
    p.add_argument("--thresh", type=float, default=0.3, help="heatmap threshold for a candidate (WASB uses 0.5)")
    p.add_argument("--topk", type=int, default=8)
    p.add_argument("--batch", type=int, default=9)
    p.add_argument("--cpu-prep", action="store_true", help="old path: crop/resize/peaks on the CPU")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(args.weights, device)
    H = load_hom()
    ranges = graded_frames() if args.halves else [(args.start, args.end)]
    os.makedirs(BALL_DIR, exist_ok=True)
    parts_dir = os.path.join(BALL_DIR, f"parts_{args.name}")
    os.makedirs(parts_dir, exist_ok=True)
    wanted = [(c0, min(c0 + CHUNK - 1, b)) for a, b in ranges for c0 in range(a, b + 1, CHUNK)]
    for c0, c1 in wanted:
        part = os.path.join(parts_dir, f"{c0:06d}_{c1:06d}.npy")
        if os.path.isfile(part):
            continue
        rows = []
        run(model, c0, c1, args, H, device, rows)
        np.save(part + ".tmp.npy", np.array(rows, dtype=np.float64).reshape(-1, 6))
        os.replace(part + ".tmp.npy", part)
    arr = np.concatenate([np.load(os.path.join(parts_dir, f"{c0:06d}_{c1:06d}.npy")) for c0, c1 in wanted])
    out = os.path.join(BALL_DIR, f"candidates_{args.name}.npz")
    np.savez_compressed(out, cand=arr, ranges=np.array(ranges), step=1)
    print(f"wrote {out}: {len(arr)} candidates over {len(np.unique(arr[:, 0]))} frames")
