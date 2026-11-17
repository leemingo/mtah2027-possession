"""Data explorer for the More than a Hack 2027 match bundle, in one file.

    uv run streamlit run dashboard.py

Tracking playback (both recordings), Smart Tagging on the timeline, the public ground truth and
the baselines as label lanes, the homography as a camera-view polygon and as a projected pitch,
and the ATD metadata.

The video plays in sync with the tracking: it is served by a small Range-capable HTTP server on
127.0.0.1 (the files are 5-14 GB, so the browser has to seek in them rather than download them),
and each presented video frame drives the tracking frame. The tracking is drawn over the video
through the homography. That server is local only, so open the dashboard on the same machine.

The raw ATD is 140-200 MB of text, so the first load of a source takes a while; after that it is a
memory-mapped .npy under data/cache/ and opens instantly.
"""

import base64
import io
import json
import os
import re
import socket
import sys
import threading
import xml.etree.ElementTree as ET
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # repository root
sys.path.insert(0, os.path.join(ROOT, "starter"))
from games import DATA_DIR, GAME, GAMES, asset_paths, half_video_blocks  # noqa: E402
import load_homography  # noqa: E402
import load_smart_tagging  # noqa: E402

# Params
CACHE_DIR = os.path.join(DATA_DIR, "cache")
# Both videos are 1920x1080; the corners of that image are what the camera view polygon unprojects.
IMAGE_SIZE = (1920, 1080)
GT_PATH = os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")
RESULTS_DIR = os.path.join(ROOT, "evaluation", "results")
FPS = GAMES[GAME]["fps"]
PITCH = GAMES[GAME]["pitch"]
WINDOWS = {"30 s": 30, "1 min": 60, "2 min": 120, "5 min": 300, "10 min": 600}
PITCH_HEIGHT_PX = 520
KICKOFFS = {1: 11185, 2: 113742}
COLORS = {"H": "#3b82f6", "A": "#ef4444", "N": "#9ca3af", "D": "#4b5563", "X": "#1f2937"}


# ------------------------------------------------------------------------------------- data layer
def _cache_path(source, name):
    return os.path.join(CACHE_DIR, "{}_{}.npy".format(source, name))


def load_tracking(source):
    """(n_frames, n_slots + 1, 3) float32, indexed by video frame. The last object is the ball.

    For players the three values are x, y (normalised 0-1) and speed in m/s; for the ball they are
    x, y and the third raw field. Missing observations are NaN.
    """
    path = _cache_path(source, "atd")
    if not os.path.isfile(path):
        raw = asset_paths(source=source)["atd_raw"]
        with open(raw, "rb") as handle:
            text = handle.read().replace(b":", b",").replace(b";", b",")
        table = pd.read_csv(io.BytesIO(text), header=None, dtype=np.float32, engine="c")
        del text
        values = table.values
        frame_ids = values[:, 0].astype(np.int64)
        objects = values[:, 1:].reshape(len(values), -1, 3)
        out = np.full((frame_ids.max(),) + objects.shape[1:], np.nan, dtype=np.float32)
        out[frame_ids - 1] = objects  # frame_id counts from 1, video frames from 0
        os.makedirs(CACHE_DIR, exist_ok=True)
        np.save(path, out)
    return np.load(path, mmap_mode="r")


def load_metadata(source):
    """The ATD metadata XML as plain dicts: global config, session, teams, track slots."""
    root = ET.parse(asset_paths(source=source)["atd_meta"]).getroot()

    def params(node):
        return {p.findtext("Name"): p.findtext("Value") for p in node.iter("ProviderParameter")}

    config = root.find(".//GlobalConfig")
    session = root.find(".//Session")
    teams = [{"id": team.get("id"), "name": team.findtext("Name"), **params(team)}
             for team in root.iter("Team")]
    home_id = teams[0]["id"]
    slots = pd.DataFrame([{
        "slot": int(player.get("id")),
        "name": player.findtext("Name"),
        "team": next(t["name"] for t in teams if t["id"] == player.get("teamId")),
        "side": "H" if player.get("teamId") == home_id else "A",
        **params(player),
    } for player in root.iter("Player")])
    return {
        "global": {child.tag: child.text for child in config if child.tag != "ProviderGlobalParameters"},
        "global_params": params(config),
        "session": {
            "type": session.findtext("SessionType"),
            "start": session.findtext("Start"),
            "field": (session.findtext("FieldSize/Width"), session.findtext("FieldSize/Height")),
        },
        "teams": teams,
        "slots": slots,
    }


def load_homography_arrays(source):
    """(matrices (n,3,3), reprojection_error (n,), valid (n,)), cached as .npy."""
    paths = [_cache_path(source, name) for name in ("hom", "hom_err", "hom_valid")]
    if not all(os.path.isfile(p) for p in paths):
        os.makedirs(CACHE_DIR, exist_ok=True)
        for p, array in zip(paths, load_homography.load(source=source)):
            np.save(p, array)
    return tuple(np.load(p, mmap_mode="r") for p in paths)


def camera_view(matrices, valid):
    """(n, 4, 2) pitch coordinates of the image corners, NaN where they do not land on the ground.

    The homography maps pitch -> pixels, so its inverse takes the image corners back onto the pitch.
    A corner above the horizon has a non-positive homogeneous w and no ground position.
    """
    out = np.full((len(matrices), 4, 2), np.nan, dtype=np.float32)
    if not valid.any():
        return out
    w, h = IMAGE_SIZE
    corners = np.array([[0, 0, 1], [w, 0, 1], [w, h, 1], [0, h, 1]], dtype=np.float64).T
    inverse = np.linalg.inv(np.asarray(matrices)[valid])
    projected = inverse @ corners  # (k, 3, 4)
    scale = projected[:, 2, :]
    pts = projected[:, :2, :] / scale[:, None, :]
    pts[np.repeat((scale <= 0)[:, None, :], 2, axis=1)] = np.nan
    out[valid] = np.transpose(pts, (0, 2, 1))
    return out


def load_tags():
    """Smart Tagging as a DataFrame with video-frame intervals and one column per tag group."""
    events = load_smart_tagging.load()
    frame = pd.DataFrame([{k: v for k, v in e.items() if k != "tags"} for e in events])
    tags = pd.DataFrame([e["tags"] for e in events])
    frame = pd.concat([frame, tags], axis=1)
    frame["team_side"] = frame["Team"].map(
        lambda t: "H" if t == "Barcelona" else "A" if t == "Atletic Club" else "N")
    return frame


def load_label_tracks():
    """{name: array of H/A/D/X per tactical video frame} for the public GT and every result CSV."""
    out = {}
    if os.path.isfile(GT_PATH):
        out["GT (public 5 min)"] = pd.read_csv(GT_PATH)["label"].values.astype("U1")
    if os.path.isdir(RESULTS_DIR):
        for name in sorted(os.listdir(RESULTS_DIR)):
            if name.endswith(".csv"):
                table = pd.read_csv(os.path.join(RESULTS_DIR, name))
                out[name[:-4]] = table["label"].values.astype("U1")
    return out


def half_blocks():
    return half_video_blocks(GAME)


# ------------------------------------------------------------------------------------- video server
class VideoHandler(BaseHTTPRequestHandler):
    """Serves /<source>.mp4 with HTTP Range support, which is what lets a <video> element seek."""

    files = {}  # "/tactical.mp4" -> absolute path; nothing else is reachable

    def do_HEAD(self):
        self._serve(body=False)

    def do_GET(self):
        self._serve(body=True)

    def _serve(self, body):
        path = self.files.get(self.path.split("?")[0])
        if not path or not os.path.isfile(path):
            self.send_error(404)
            return
        size = os.path.getsize(path)
        first, last = 0, size - 1
        match = re.match(r"bytes=(\d*)-(\d*)", self.headers.get("Range", ""))
        if match:
            if match.group(1):
                first = int(match.group(1))
                last = int(match.group(2)) if match.group(2) else last
            else:  # suffix range: the last N bytes
                first = max(0, size - int(match.group(2)))
            last = min(last, size - 1)
            if first > last:
                self.send_response(416)
                self.send_header("Content-Range", "bytes */{}".format(size))
                self.end_headers()
                return
            self.send_response(206)
            self.send_header("Content-Range", "bytes {}-{}/{}".format(first, last, size))
        else:
            self.send_response(200)
        self.send_header("Content-Type", "video/mp4")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(last - first + 1))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        if not body:
            return
        with open(path, "rb") as handle:
            handle.seek(first)
            remaining = last - first + 1
            try:
                while remaining > 0:
                    chunk = handle.read(min(1 << 20, remaining))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass  # the browser cancels range requests all the time when seeking

    def log_message(self, *args):
        pass


def start_video_server():
    """Serve both videos on a free local port. Returns the base URL."""
    VideoHandler.files = {"/{}.mp4".format(source): asset_paths(source=source)["video"]
                          for source in ("tactical", "broadcast")}
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    server = ThreadingHTTPServer(("127.0.0.1", port), VideoHandler)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return "http://127.0.0.1:{}".format(port)


# ------------------------------------------------------------------------------------- player (HTML/JS)
PLAYER_HTML = r"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<style>
  :root { --bg:#0e1117; --panel:#161a23; --text:#e6e6e6; --muted:#8b93a7; --line:#2a3040;
          --home:#3b82f6; --away:#ef4444; --neutral:#9ca3af; --dead:#4b5563; --ball:#facc15; }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--text); font:13px/1.4 system-ui,sans-serif; }
  #wrap { display:flex; gap:12px; }
  #left { flex:1; min-width:0; }
  #side { width:270px; flex:none; background:var(--panel); border:1px solid var(--line);
          border-radius:8px; padding:10px; font-variant-numeric:tabular-nums; }
  #side h4 { margin:10px 0 4px; font-size:11px; text-transform:uppercase; letter-spacing:.06em; color:var(--muted); }
  #side h4:first-child { margin-top:0; }
  .kv { display:flex; justify-content:space-between; gap:8px; }
  .kv span:first-child { color:var(--muted); }
  #matrix { font-family:ui-monospace,Consolas,monospace; font-size:11px; white-space:pre; }
  canvas { display:block; width:100%; border-radius:6px; }
  #pitch { background:#1f5130; flex:none; }
  #stage { display:flex; gap:8px; align-items:flex-start; }
  #vbox { position:relative; flex:none; background:#000; border-radius:6px; overflow:hidden; display:none; }
  #vid, #ov { position:absolute; left:0; top:0; width:100%; height:100%; }
  #vmsg { position:absolute; left:8px; top:6px; color:#facc15; font-size:12px; text-shadow:0 0 3px #000; }
  #controls { display:flex; align-items:center; gap:6px; margin:8px 0; flex-wrap:wrap; }
  button, select { background:var(--panel); color:var(--text); border:1px solid var(--line);
                   border-radius:6px; padding:4px 10px; font:inherit; cursor:pointer; }
  button:hover { border-color:var(--muted); }
  #scrub { flex:1; min-width:120px; }
  #clock { font-variant-numeric:tabular-nums; min-width:150px; text-align:right; }
  label { color:var(--muted); display:flex; align-items:center; gap:4px; }
  #tip { position:fixed; pointer-events:none; background:#000d; border:1px solid var(--line);
         border-radius:6px; padding:6px 8px; font-size:12px; display:none; max-width:320px; z-index:9; }
  .legend { display:flex; gap:12px; color:var(--muted); font-size:12px; margin-top:4px; flex-wrap:wrap; }
  .sw { display:inline-block; width:10px; height:10px; border-radius:2px; margin-right:4px; vertical-align:-1px; }
</style>
</head>
<body>
<div id="wrap">
  <div id="left">
    <div id="stage">
      <div id="vbox">
        <video id="vid" muted playsinline preload="auto"></video>
        <canvas id="ov"></canvas>
        <span id="vmsg"></span>
      </div>
      <canvas id="pitch"></canvas>
    </div>
    <div id="controls">
      <button id="back">⏮ −5s</button>
      <button id="prev">◀ 1f</button>
      <button id="play">▶ Play</button>
      <button id="next">1f ▶</button>
      <button id="fwd">+5s ⏭</button>
      <select id="rate">
        <option value="0.25">0.25×</option><option value="0.5">0.5×</option>
        <option value="1" selected>1×</option><option value="2">2×</option>
        <option value="4">4×</option><option value="8">8×</option>
      </select>
      <input id="scrub" type="range" min="0" value="0">
      <span id="clock"></span>
    </div>
    <div id="controls2" class="legend">
      <label><input type="checkbox" id="optIds"> slot ids</label>
      <label><input type="checkbox" id="optTrail" checked> ball trail</label>
      <label><input type="checkbox" id="optCam" checked> camera view (homography)</label>
      <label><input type="checkbox" id="optTails"> player tails</label>
      <label class="vopt"><input type="checkbox" id="optOverlay" checked> tracking on video</label>
      <label class="vopt"><input type="checkbox" id="optSound"> sound</label>
      <span><span class="sw" style="background:var(--home)"></span>H</span>
      <span><span class="sw" style="background:var(--away)"></span>A</span>
      <span><span class="sw" style="background:var(--neutral)"></span>N/A</span>
      <span><span class="sw" style="background:var(--dead)"></span>D</span>
      <span>space = play/pause · ←/→ = frame · shift+←/→ = 5s · click/drag timeline to seek</span>
    </div>
    <canvas id="timeline"></canvas>
  </div>
  <div id="side">
    <h4>Frame</h4>
    <div class="kv"><span>video frame</span><span id="iFrame"></span></div>
    <div class="kv"><span>ATD frame_id</span><span id="iFid"></span></div>
    <div class="kv"><span>video time</span><span id="iTime"></span></div>
    <div class="kv"><span>match clock</span><span id="iMatch"></span></div>
    <h4>Tracking</h4>
    <div class="kv"><span>tracked H / A</span><span id="iCount"></span></div>
    <div class="kv"><span>ball (x, y)</span><span id="iBall"></span></div>
    <div class="kv"><span>ball 3rd field</span><span id="iBall3"></span></div>
    <h4>Labels at frame</h4>
    <div id="iLabels"></div>
    <h4>Active Smart Tags</h4>
    <div id="iTags" style="font-size:12px"></div>
    <h4>Homography</h4>
    <div class="kv"><span>valid</span><span id="iValid"></span></div>
    <div class="kv"><span>reproj. error</span><span id="iErr"></span></div>
    <div id="matrix"></div>
  </div>
</div>
<div id="tip"></div>
<script>
const D = __DATA__;

function b64(s, Type) {
  const bin = atob(s), buf = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) buf[i] = bin.charCodeAt(i);
  return new Type(buf.buffer);
}
const POS = b64(D.pos, Int16Array);        // (n, nobj, 2) * 1e4, -32768 = missing
const SPD = b64(D.spd, Uint16Array);       // (n, nobj-1) * 100, 65535 = missing
const CAM = b64(D.cam, Int16Array);        // (n, 4, 2) * 1e4
const HOM = b64(D.hom, Float32Array);      // (n, 9)
const ERR = b64(D.err, Float32Array);      // (n)
const VAL = b64(D.valid, Uint8Array);      // (n)
const B3  = b64(D.ball3, Float32Array);    // (n)
const N = D.n, NOBJ = D.nobj, MISS = -32768;
const COL = { H: '#3b82f6', A: '#ef4444', N: '#9ca3af', D: '#4b5563', X: null };

let cur = D.initial - D.start, playing = false, rate = 1, lastT = null, acc = 0;

// The video, when there is one, is the clock: video frame F = floor(mediaTime * fps), and tracking
// frame F is shown with it. Seeking aims at the middle of a frame so rounding cannot land a frame early.
const VIDEO = __VIDEO__;
const V = VIDEO ? document.getElementById('vid') : null;
const IMG = [1920, 1080];
const frameTime = F => (F + 0.5) / D.fps;
if (V) {
  document.getElementById('vbox').style.display = 'block';
  V.src = VIDEO.url;
  V.addEventListener('loadedmetadata', () => { V.currentTime = frameTime(D.start + cur); });
  V.addEventListener('waiting', () => { document.getElementById('vmsg').textContent = 'buffering…'; });
  V.addEventListener('playing', () => { document.getElementById('vmsg').textContent = ''; });
  V.addEventListener('seeked', () => { document.getElementById('vmsg').textContent = ''; });
  V.addEventListener('error', () => {
    document.getElementById('vmsg').textContent = 'video failed to load: ' + VIDEO.url;
  });
} else {
  document.querySelectorAll('.vopt').forEach(el => { el.style.display = 'none'; });
}

// ---------------------------------------------------------------- pitch
const pc = document.getElementById('pitch'), px = pc.getContext('2d');
const MARGIN = 0.06; // normalised margin around the pitch
const [PL, PW] = D.pitch;
let pW = 0, pH = 0, dpr = window.devicePixelRatio || 1;

const oc = document.getElementById('ov'), ox = oc.getContext('2d');
let vW = 0, vH = 0;

function sizePitch() {
  const aspect = (PW * (1 + 2 * MARGIN)) / (PL * (1 + 2 * MARGIN));
  const avail = document.getElementById('left').clientWidth;
  if (V) {
    // video and pitch side by side at the same height
    pW = Math.min((avail - 8) / (1 + aspect * IMG[0] / IMG[1]), D.pitchH / aspect);
    pH = pW * aspect;
    vH = pH; vW = vH * IMG[0] / IMG[1];
    const vb = document.getElementById('vbox');
    vb.style.width = vW + 'px'; vb.style.height = vH + 'px';
    oc.width = vW * dpr; oc.height = vH * dpr;
    ox.setTransform(dpr, 0, 0, dpr, 0, 0);
  } else {
    pW = Math.min(avail, D.pitchH / aspect);
    pH = pW * aspect;
  }
  pc.style.width = pW + 'px';
  pc.width = pW * dpr; pc.height = pH * dpr; pc.style.height = pH + 'px';
  px.setTransform(dpr, 0, 0, dpr, 0, 0);
}

// Tracking projected into the video through this frame's homography (normalised pitch -> pixels).
function drawOverlay() {
  if (!V) return;
  ox.clearRect(0, 0, vW, vH);
  const f = cur;
  if (!document.getElementById('optOverlay').checked || !VAL[f]) return;
  const h = HOM.subarray(f * 9, f * 9 + 9), k = vW / IMG[0];
  const proj = (x, y) => {
    const w = h[6] * x + h[7] * y + h[8];
    return w > 0 ? [(h[0] * x + h[1] * y + h[2]) / w * k, (h[3] * x + h[4] * y + h[5]) / w * k] : null;
  };
  const r = Math.max(4, vW / 160);
  for (let o = 0; o < NOBJ - 1; o++) {
    const p = pos(f, o); if (!p) continue;
    const q = proj(p[0], p[1]); if (!q) continue;
    ox.beginPath(); ox.ellipse(q[0], q[1], r * 1.4, r * 0.6, 0, 0, 2 * Math.PI);
    ox.strokeStyle = COL[D.sides[o]]; ox.lineWidth = 2.5; ox.stroke();
    if (document.getElementById('optIds').checked) {
      ox.fillStyle = '#fff'; ox.font = `bold ${Math.round(r * 1.6)}px system-ui`; ox.textAlign = 'center';
      ox.textBaseline = 'top'; ox.fillText(D.slotIds[o], q[0], q[1] + r * 0.8);
    }
  }
  const b = pos(f, NOBJ - 1), q = b && proj(b[0], b[1]);
  if (q) {
    ox.beginPath(); ox.arc(q[0], q[1], r * 0.9, 0, 2 * Math.PI);
    ox.strokeStyle = '#facc15'; ox.lineWidth = 2.5; ox.stroke();
  }
}
const sx = x => (x + MARGIN) / (1 + 2 * MARGIN) * pW;
const sy = y => (y + MARGIN) / (1 + 2 * MARGIN) * pH;
const mx = m => sx(m / PL) - sx(0), my = m => sy(m / PW) - sy(0);

function drawMarkings() {
  px.fillStyle = '#1f5130'; px.fillRect(0, 0, pW, pH);
  // mowing stripes
  for (let i = 0; i < 12; i++) if (i % 2) {
    px.fillStyle = '#21583a'; px.fillRect(sx(i / 12), sy(0), sx(1 / 12) - sx(0), sy(1) - sy(0));
  }
  px.strokeStyle = '#ffffffb0'; px.lineWidth = 1.5;
  const rect = (x0, y0, x1, y1) => px.strokeRect(sx(x0), sy(y0), sx(x1) - sx(x0), sy(y1) - sy(y0));
  rect(0, 0, 1, 1);
  px.beginPath(); px.moveTo(sx(.5), sy(0)); px.lineTo(sx(.5), sy(1)); px.stroke();
  px.beginPath(); px.ellipse(sx(.5), sy(.5), mx(9.15), my(9.15), 0, 0, 2 * Math.PI); px.stroke();
  const bw = 40.32 / PW, bd = 16.5 / PL, gw = 18.32 / PW, gd = 5.5 / PL;
  rect(0, .5 - bw / 2, bd, .5 + bw / 2); rect(1 - bd, .5 - bw / 2, 1, .5 + bw / 2);
  rect(0, .5 - gw / 2, gd, .5 + gw / 2); rect(1 - gd, .5 - gw / 2, 1, .5 + gw / 2);
  const goal = 7.32 / PW;
  px.lineWidth = 3;
  px.beginPath(); px.moveTo(sx(0), sy(.5 - goal / 2)); px.lineTo(sx(0), sy(.5 + goal / 2)); px.stroke();
  px.beginPath(); px.moveTo(sx(1), sy(.5 - goal / 2)); px.lineTo(sx(1), sy(.5 + goal / 2)); px.stroke();
  px.fillStyle = '#ffffffb0';
  for (const x of [11 / PL, 1 - 11 / PL, .5]) { px.beginPath(); px.arc(sx(x), sy(.5), 2, 0, 7); px.fill(); }
}

const pos = (f, o) => {
  const i = (f * NOBJ + o) * 2, x = POS[i];
  return x === MISS ? null : [x / 1e4, POS[i + 1] / 1e4];
};

function drawFrame() {
  drawMarkings();
  const f = cur;
  if (document.getElementById('optCam').checked && VAL[f]) {
    const c = [];
    for (let k = 0; k < 4; k++) {
      const x = CAM[(f * 4 + k) * 2];
      if (x !== MISS) c.push([x / 1e4, CAM[(f * 4 + k) * 2 + 1] / 1e4]);
    }
    if (c.length >= 3) {
      px.save();
      px.beginPath(); c.forEach(([x, y], i) => i ? px.lineTo(sx(x), sy(y)) : px.moveTo(sx(x), sy(y)));
      px.closePath();
      px.fillStyle = '#ffffff14'; px.fill();
      px.setLineDash([6, 4]); px.strokeStyle = '#fde68a'; px.lineWidth = 1.5; px.stroke();
      px.restore();
    }
  }
  const ids = document.getElementById('optIds').checked;
  const tails = document.getElementById('optTails').checked;
  const r = Math.max(4, pW / 140);
  for (let o = 0; o < NOBJ - 1; o++) {
    const p = pos(f, o); if (!p) continue;
    const col = COL[D.sides[o]];
    if (tails) {
      px.strokeStyle = col + '80'; px.lineWidth = 1.5; px.beginPath();
      let started = false;
      for (let g = Math.max(0, f - 50); g <= f; g++) {
        const q = pos(g, o); if (!q) { started = false; continue; }
        started ? px.lineTo(sx(q[0]), sy(q[1])) : px.moveTo(sx(q[0]), sy(q[1])); started = true;
      }
      px.stroke();
    }
    px.beginPath(); px.arc(sx(p[0]), sy(p[1]), r, 0, 2 * Math.PI);
    px.fillStyle = col; px.fill(); px.strokeStyle = '#000a'; px.lineWidth = 1; px.stroke();
    if (ids) {
      px.fillStyle = '#fff'; px.font = `${Math.round(r * 1.1)}px system-ui`; px.textAlign = 'center';
      px.textBaseline = 'middle'; px.fillText(D.slotIds[o], sx(p[0]), sy(p[1]) + 0.5);
    }
  }
  const B = NOBJ - 1;
  if (document.getElementById('optTrail').checked) {
    for (let g = Math.max(0, f - 25); g < f; g++) {
      const q = pos(g, B); if (!q) continue;
      px.fillStyle = `rgba(250,204,21,${0.15 + 0.6 * (g - f + 25) / 25})`;
      px.beginPath(); px.arc(sx(q[0]), sy(q[1]), r * 0.35, 0, 7); px.fill();
    }
  }
  const b = pos(f, B);
  if (b) {
    px.beginPath(); px.arc(sx(b[0]), sy(b[1]), r * 0.7, 0, 2 * Math.PI);
    px.fillStyle = '#facc15'; px.fill(); px.strokeStyle = '#000'; px.lineWidth = 1.2; px.stroke();
  } else {
    px.fillStyle = '#facc15'; px.font = '12px system-ui'; px.textAlign = 'left'; px.textBaseline = 'top';
    px.fillText('ball not tracked', 8, 6);
  }
  drawOverlay();
}

// ---------------------------------------------------------------- timeline
const tc = document.getElementById('timeline'), tx = tc.getContext('2d');
const LANE = 16, GAP = 3, LABELW = 170, AXIS = 18;
const lanes = []; // {name, kind, items:[{s,e,color,label,ev}]}
(function buildLanes() {
  lanes.push({ name: 'ball tracked', items: D.ballRuns.map(([s, e]) => ({ s, e, color: '#facc15' })) });
  lanes.push({ name: 'homography valid', items: D.validRuns.map(([s, e]) => ({ s, e, color: '#a3e635' })) });
  lanes.push({ name: 'reproj. error', kind: 'err' });
  for (const t of D.tracks) lanes.push({ name: t.name, items: t.runs.map(([s, e, l]) => ({ s, e, color: COL[l], label: l })) });
  const codes = [...new Set(D.events.map(e => e.code))].sort();
  for (const c of codes) lanes.push({
    name: c.toLowerCase(),
    items: D.events.filter(e => e.code === c).map(e => ({ s: e.s, e: e.e, color: COL[e.side], ev: e })),
  });
  if (!D.events.length && D.hasTags) lanes.push({ name: '(no smart tags in window)', items: [] });
})();
let tW = 0, tH = 0;
function sizeTimeline() {
  tW = tc.clientWidth; tH = AXIS + lanes.length * (LANE + GAP) + 4;
  tc.width = tW * dpr; tc.height = tH * dpr; tc.style.height = tH + 'px';
  tx.setTransform(dpr, 0, 0, dpr, 0, 0);
}
const tX = f => LABELW + (f - D.start) / N * (tW - LABELW - 4);
const fOfX = x => Math.round(D.start + (x - LABELW) / (tW - LABELW - 4) * N);
const errMax = D.errMax;

function drawTimeline() {
  tx.fillStyle = '#0e1117'; tx.fillRect(0, 0, tW, tH);
  // axis
  tx.fillStyle = '#8b93a7'; tx.font = '11px system-ui'; tx.textBaseline = 'top'; tx.textAlign = 'center';
  const secs = N / D.fps, step = [5, 10, 15, 30, 60, 120, 300].find(s => secs / s <= 12) || 600;
  const firstTick = Math.ceil(D.start / D.fps / step) * step;
  for (let s = firstTick; s * D.fps < D.start + N; s += step) {
    const x = tX(s * D.fps);
    tx.fillRect(x, AXIS - 4, 1, 4); tx.fillText(fmt(s * D.fps), x, 1);
  }
  lanes.forEach((ln, i) => {
    const y = AXIS + i * (LANE + GAP);
    tx.fillStyle = '#161a23'; tx.fillRect(LABELW, y, tW - LABELW - 4, LANE);
    tx.fillStyle = '#c7cbd6'; tx.textAlign = 'right'; tx.textBaseline = 'middle';
    tx.fillText(ln.name, LABELW - 6, y + LANE / 2);
    if (ln.kind === 'err') {
      tx.strokeStyle = '#fb923c'; tx.lineWidth = 1; tx.beginPath();
      const stepF = Math.max(1, Math.floor(N / (tW - LABELW)));
      for (let f = 0; f < N; f += stepF) {
        let m = 0; for (let g = f; g < Math.min(N, f + stepF); g++) if (ERR[g] > m) m = ERR[g];
        const yy = y + LANE - Math.min(1, m / errMax) * LANE;
        f ? tx.lineTo(tX(D.start + f), yy) : tx.moveTo(tX(D.start + f), yy);
      }
      tx.stroke();
      return;
    }
    for (const it of ln.items) {
      if (!it.color) continue;
      const x0 = Math.max(LABELW, tX(it.s)), x1 = Math.min(tW - 4, tX(it.e));
      if (x1 <= x0) continue;
      tx.fillStyle = it.color; tx.globalAlpha = it.ev ? 0.85 : 0.9;
      tx.fillRect(x0, y + 1, Math.max(1, x1 - x0), LANE - 2); tx.globalAlpha = 1;
      if (it.ev && x1 - x0 > 40) {
        tx.fillStyle = '#fff'; tx.textAlign = 'left'; tx.font = '10px system-ui';
        const t = it.ev.tags.Type || it.ev.tags.Team || '';
        tx.save(); tx.beginPath(); tx.rect(x0, y, x1 - x0 - 2, LANE); tx.clip();
        tx.fillText(t, x0 + 3, y + LANE / 2); tx.restore(); tx.font = '11px system-ui';
      }
    }
  });
  // playhead
  const x = tX(D.start + cur);
  tx.fillStyle = '#fff'; tx.fillRect(x - 0.5, 0, 1.5, tH);
}

// ---------------------------------------------------------------- side panel
const $ = id => document.getElementById(id);
function fmt(frame) {
  const s = frame / D.fps, m = Math.floor(s / 60);
  return `${String(m).padStart(2, '0')}:${String(Math.floor(s % 60)).padStart(2, '0')}`;
}
function fmtLong(frame) {
  return fmt(frame) + '.' + String(frame % D.fps).padStart(2, '0');
}
function matchClock(frame) {
  for (const h of D.halves) {
    if (frame >= h.kickoff && frame <= h.end) {
      const s = (frame - h.kickoff) / D.fps + h.offsetMin * 60;
      return `${h.half === 1 ? '1st' : '2nd'} half ${String(Math.floor(s / 60)).padStart(2, '0')}:${String(Math.floor(s % 60)).padStart(2, '0')}`;
    }
  }
  return '—';
}
function updateSide() {
  const f = cur, F = D.start + f;
  $('iFrame').textContent = F; $('iFid').textContent = F + 1;
  $('iTime').textContent = fmtLong(F); $('iMatch').textContent = D.hasClock ? matchClock(F) : 'n/a (broadcast)';
  let h = 0, a = 0;
  for (let o = 0; o < NOBJ - 1; o++) if (pos(f, o)) D.sides[o] === 'H' ? h++ : a++;
  $('iCount').textContent = `${h} / ${a}`;
  const b = pos(f, NOBJ - 1);
  $('iBall').textContent = b ? `${b[0].toFixed(3)}, ${b[1].toFixed(3)}` : '—';
  $('iBall3').textContent = Number.isFinite(B3[f]) ? B3[f].toFixed(3) : '—';
  $('iLabels').innerHTML = D.tracks.map(t => {
    const l = t.labels[f];
    return `<div class="kv"><span>${t.name}</span><span style="color:${COL[l] || '#8b93a7'}">${l}</span></div>`;
  }).join('') || '<span style="color:#8b93a7">none for this source</span>';
  const active = D.events.filter(e => e.s <= F && F < e.e);
  $('iTags').innerHTML = active.length ? active.map(e =>
    `<div><span class="sw" style="background:${COL[e.side]}"></span>${e.code.toLowerCase()}` +
    `${e.tags.Type ? ' · ' + e.tags.Type : ''}</div>`).join('')
    : `<span style="color:#8b93a7">${D.hasTags ? 'none' : 'no Smart Tagging for this source'}</span>`;
  $('iValid').textContent = VAL[f] ? 'yes' : 'no';
  $('iErr').textContent = Number.isFinite(ERR[f]) ? ERR[f].toFixed(3) : '—';
  if (VAL[f]) {
    const m = Array.from(HOM.subarray(f * 9, f * 9 + 9)).map(v => v.toPrecision(5).padStart(11));
    $('matrix').textContent = `${m[0]}${m[1]}${m[2]}\n${m[3]}${m[4]}${m[5]}\n${m[6]}${m[7]}${m[8]}`;
  } else $('matrix').textContent = '(no matrix)';
  $('clock').textContent = `${fmtLong(F)}  /  frame ${F}`;
  $('scrub').value = f;
}

function render() { drawFrame(); drawTimeline(); updateSide(); }
function seek(f) {
  cur = Math.max(0, Math.min(N - 1, f));
  if (V && V.readyState >= 1) V.currentTime = frameTime(D.start + cur);
  render();
}

// ---------------------------------------------------------------- playback
// With a video, every presented video frame sets the tracking frame; without one, a wall clock does.
function onVideoFrame(now, meta) {
  if (playing) {
    const f = Math.floor(meta.mediaTime * D.fps + 1e-3) - D.start;
    if (f >= N - 1) { cur = N - 1; setPlaying(false); render(); }
    else if (f >= 0 && f !== cur) { cur = f; render(); }
  }
  V.requestVideoFrameCallback(onVideoFrame);
}
function pollVideo() {  // fallback for browsers without requestVideoFrameCallback
  if (!playing) return;
  onVideoFrame(0, { mediaTime: V.currentTime });
  requestAnimationFrame(pollVideo);
}
if (V && 'requestVideoFrameCallback' in V) V.requestVideoFrameCallback(onVideoFrame);

function tick(t) {
  if (!playing) return;
  if (lastT !== null) {
    acc += (t - lastT) / 1000 * D.fps * rate;
    const adv = Math.floor(acc);
    if (adv > 0) { acc -= adv; cur += adv; if (cur >= N - 1) { cur = N - 1; setPlaying(false); } render(); }
  }
  lastT = t; requestAnimationFrame(tick);
}
function setPlaying(p) {
  playing = p; lastT = null; acc = 0;
  $('play').textContent = p ? '⏸ Pause' : '▶ Play';
  if (p && cur >= N - 1) seek(0);
  if (V) {
    if (p) {
      V.playbackRate = rate;
      V.play().catch(() => setPlaying(false));
      if (!('requestVideoFrameCallback' in V)) requestAnimationFrame(pollVideo);
    } else V.pause();
  } else if (p) requestAnimationFrame(tick);
}
$('play').onclick = () => setPlaying(!playing);
$('prev').onclick = () => seek(cur - 1);
$('next').onclick = () => seek(cur + 1);
$('back').onclick = () => seek(cur - 5 * D.fps);
$('fwd').onclick = () => seek(cur + 5 * D.fps);
$('rate').onchange = e => { rate = +e.target.value; if (V) V.playbackRate = rate; };
$('optOverlay').onchange = render;
$('optSound').onchange = e => { if (V) V.muted = !e.target.checked; };
$('scrub').max = N - 1;
$('scrub').oninput = e => seek(+e.target.value);
for (const id of ['optIds', 'optTrail', 'optCam', 'optTails']) $(id).onchange = render;
document.addEventListener('keydown', e => {
  if (e.code === 'Space') { e.preventDefault(); setPlaying(!playing); }
  else if (e.code === 'ArrowLeft') { e.preventDefault(); seek(cur - (e.shiftKey ? 5 * D.fps : 1)); }
  else if (e.code === 'ArrowRight') { e.preventDefault(); seek(cur + (e.shiftKey ? 5 * D.fps : 1)); }
});

// timeline seek + tooltips
const tip = $('tip');
let dragging = false;
function timelineSeek(e) {
  const r = tc.getBoundingClientRect(), x = e.clientX - r.left;
  if (x >= LABELW) seek(fOfX(x) - D.start);
}
tc.addEventListener('mousedown', e => { dragging = true; timelineSeek(e); });
window.addEventListener('mouseup', () => { dragging = false; });
tc.addEventListener('mousemove', e => {
  if (dragging) timelineSeek(e);
  const r = tc.getBoundingClientRect(), x = e.clientX - r.left, y = e.clientY - r.top;
  const li = Math.floor((y - AXIS) / (LANE + GAP)), ln = lanes[li];
  const F = fOfX(x);
  let hit = null;
  if (ln && ln.items && x >= LABELW) hit = ln.items.find(it => it.s <= F && F < it.e);
  if (hit && hit.ev) {
    const ev = hit.ev;
    tip.innerHTML = `<b>${ev.code}</b><br>${fmtLong(ev.s)} → ${fmtLong(ev.e)} (frames ${ev.s}–${ev.e})<br>` +
      Object.entries(ev.tags).map(([k, v]) => `${k}: ${v}`).join('<br>');
  } else if (hit) {
    tip.innerHTML = `<b>${ln.name}</b>${hit.label ? ': ' + hit.label : ''}<br>frames ${hit.s}–${hit.e - 1} (${((hit.e - hit.s) / D.fps).toFixed(1)} s)`;
  } else if (x >= LABELW && y > AXIS) {
    tip.innerHTML = `frame ${F} · ${fmtLong(F)}`;
  } else { tip.style.display = 'none'; return; }
  tip.style.display = 'block'; tip.style.left = (e.clientX + 12) + 'px'; tip.style.top = (e.clientY + 12) + 'px';
});
tc.addEventListener('mouseleave', () => { tip.style.display = 'none'; });

// pitch hover: nearest object
pc.addEventListener('mousemove', e => {
  const r = pc.getBoundingClientRect(), x = e.clientX - r.left, y = e.clientY - r.top;
  let best = null, bd = 144;
  for (let o = 0; o < NOBJ; o++) {
    const p = pos(cur, o); if (!p) continue;
    const d = (sx(p[0]) - x) ** 2 + (sy(p[1]) - y) ** 2;
    if (d < bd) { bd = d; best = [o, p]; }
  }
  if (!best) { tip.style.display = 'none'; return; }
  const [o, p] = best;
  if (o === NOBJ - 1) tip.innerHTML = `<b>ball</b><br>x ${p[0].toFixed(3)} · y ${p[1].toFixed(3)}<br>${(p[0] * PL).toFixed(1)} m, ${(p[1] * PW).toFixed(1)} m`;
  else {
    const s = SPD[cur * (NOBJ - 1) + o];
    tip.innerHTML = `<b>Track ${D.slotIds[o]}</b> (${D.teams[D.sides[o]]})<br>x ${p[0].toFixed(3)} · y ${p[1].toFixed(3)}` +
      `<br>${(p[0] * PL).toFixed(1)} m, ${(p[1] * PW).toFixed(1)} m<br>speed ${s === 65535 ? '—' : (s / 100).toFixed(2) + ' m/s'}`;
  }
  tip.style.display = 'block'; tip.style.left = (e.clientX + 12) + 'px'; tip.style.top = (e.clientY + 12) + 'px';
});
pc.addEventListener('mouseleave', () => { tip.style.display = 'none'; });

function resize() { dpr = window.devicePixelRatio || 1; sizePitch(); sizeTimeline(); render(); }
window.addEventListener('resize', resize);
resize();
if (D.autoplay) setPlaying(true);
</script>
</body>
</html>
"""

st.set_page_config(page_title="MTAH 2027 data explorer", layout="wide", page_icon="⚽")


# ------------------------------------------------------------------------------------- loading
@st.cache_resource(show_spinner="Parsing tracking data (first run only, then cached on disk)...")
def tracking(source):
    return load_tracking(source)


@st.cache_resource
def video_server():
    return start_video_server()


@st.cache_resource(show_spinner="Loading homography...")
def homography(source):
    matrices, err, valid = load_homography_arrays(source)
    return np.asarray(matrices), np.asarray(err), np.asarray(valid), camera_view(matrices, valid)


@st.cache_resource
def metadata(source):
    return load_metadata(source)


@st.cache_resource
def smart_tagging():
    return load_tags()


@st.cache_resource
def label_tracks():
    return load_label_tracks()


@st.cache_data
def per_second(source):
    """Per-second summaries over the whole recording, for the overview charts."""
    trk = tracking(source)
    n = (len(trk) // FPS) * FPS
    ball = np.isfinite(trk[:n, -1, 0]).reshape(-1, FPS).mean(1)
    objects = np.isfinite(trk[:n, :-1, 0]).sum(1).reshape(-1, FPS).mean(1)
    _, err, valid, _ = homography(source)
    err_max = np.nan_to_num(err[:n], nan=0).reshape(-1, FPS).max(1)
    valid_frac = valid[:n].reshape(-1, FPS).mean(1)
    return pd.DataFrame({"t": np.arange(len(ball)), "ball": ball, "objects": objects,
                         "err": err_max, "valid": valid_frac})


@st.cache_data
def slot_coverage(source):
    return np.isfinite(tracking(source)[:, :-1, 0]).mean(0)


# ------------------------------------------------------------------------------------- helpers
def fmt(frame):
    s = int(frame) // FPS
    return "{:02d}:{:02d}".format(s // 60, s % 60)


def b64(array):
    return base64.b64encode(np.ascontiguousarray(array).tobytes()).decode("ascii")


def quantise(values, scale, missing, dtype, lo, hi):
    out = np.full(values.shape, missing, dtype=dtype)
    ok = np.isfinite(values)
    out[ok] = np.clip(np.round(values[ok] * scale), lo, hi)
    return out


def runs(mask, offset):
    """[[start, end)] absolute frames of the True stretches of mask."""
    padded = np.concatenate([[False], mask, [False]]).astype(np.int8)
    edges = np.flatnonzero(np.diff(padded))
    return (edges.reshape(-1, 2) + offset).tolist()


def label_runs(labels, offset):
    change = np.flatnonzero(labels[1:] != labels[:-1]) + 1
    starts = np.concatenate([[0], change])
    ends = np.concatenate([change, [len(labels)]])
    return [[int(s + offset), int(e + offset), str(labels[s])] for s, e in zip(starts, ends)]


def tags_of(row):
    keep = ["Team", "Type", "Half", "Side", "Direction of ball entry", "Max Players in the box"]
    return {k: row[k] for k in keep if k in row and isinstance(row[k], str)}


@st.cache_data(show_spinner="Preparing window...", max_entries=8)
def payload(source, start, length, initial):
    trk = tracking(source)
    end = min(len(trk), start + length)
    window = np.asarray(trk[start:end])
    n, nobj = window.shape[:2]
    matrices, err, valid, cam = homography(source)
    meta = metadata(source)

    tracks = []
    events = []
    if source == "tactical":
        for name, labels in label_tracks().items():
            part = labels[start:end]
            if (part != "X").any():
                tracks.append({"name": name, "runs": label_runs(part, start), "labels": "".join(part)})
        tags = smart_tagging()
        hit = tags[(tags.end_frame > start) & (tags.start_frame < end)]
        events = [{"code": r["code"], "s": int(r["start_frame"]), "e": int(r["end_frame"]), "side": r["team_side"],
                   "tags": tags_of(r)} for r in hit.to_dict("records")]

    finite_err = err[valid]
    out = {
        "start": int(start), "n": int(n), "nobj": int(nobj), "fps": FPS,
        "initial": int(min(max(initial, start), end - 1)),
        "pitch": list(PITCH), "pitchH": PITCH_HEIGHT_PX,
        "sides": meta["slots"]["side"].tolist(),
        "slotIds": meta["slots"]["slot"].tolist(),
        "teams": {"H": meta["teams"][0]["name"], "A": meta["teams"][1]["name"]},
        "pos": b64(quantise(window[:, :, :2], 1e4, -32768, "<i2", -32767, 32767)),
        "spd": b64(quantise(window[:, :-1, 2], 100, 65535, "<u2", 0, 65534)),
        "ball3": b64(window[:, -1, 2].astype("<f4")),
        "cam": b64(quantise(cam[start:end], 1e4, -32768, "<i2", -32767, 32767)),
        "hom": b64(matrices[start:end].reshape(-1, 9).astype("<f4")),
        "err": b64(err[start:end].astype("<f4")),
        "valid": b64(valid[start:end].astype("u1")),
        "errMax": float(np.percentile(finite_err, 99)) if len(finite_err) else 1.0,
        "ballRuns": runs(np.isfinite(window[:, -1, 0]), start),
        "validRuns": runs(valid[start:end], start),
        "tracks": tracks,
        "events": events,
        "hasTags": source == "tactical",
        "hasClock": source == "tactical",
        "halves": [{"half": h, "kickoff": KICKOFFS[h], "end": last, "offsetMin": 0 if h == 1 else 45}
                   for h, _, last in half_blocks()],
    }
    lanes = 3 + len(tracks) + max(1, len({e["code"] for e in events}))
    return json.dumps(out), lanes


# ------------------------------------------------------------------------------------- state
def state_key(source):
    return "start_" + source


def jump(source, frame, lead_s=3):
    """Put frame near the left of the current window and start the playhead on it."""
    st.session_state[state_key(source)] = max(0, int(frame) - lead_s * FPS) // FPS
    st.session_state["initial_" + source] = int(frame)


def request_jump(source, frame, lead_s=3):
    """jump() for code running after the window slider exists: applied at the top of the next run."""
    st.session_state["pending_jump"] = (source, int(frame), lead_s)
    st.rerun()


if "pending_jump" in st.session_state:
    jump(*st.session_state.pop("pending_jump"))


# ------------------------------------------------------------------------------------- sidebar
st.sidebar.title("⚽ MTAH 2027 explorer")
source = st.sidebar.radio("Recording", ["tactical", "broadcast"], horizontal=True,
                          help="Broadcast is NOT frame-aligned to tactical; it has its own timeline, "
                               "no Smart Tagging and no ground truth.")
trk = tracking(source)
n_frames = len(trk)
key = state_key(source)
if key not in st.session_state:
    st.session_state[key] = max(0, (KICKOFFS[1] if source == "tactical" else 0) // FPS - 2)
    st.session_state["initial_" + source] = st.session_state[key] * FPS

video_file = asset_paths(source=source)["video"]
has_video = os.path.isfile(video_file)
show_video = st.sidebar.toggle("Show video", value=has_video, disabled=not has_video,
                               help="Synced to the tracking frame by frame; tracking is drawn over it "
                                    "through the homography." if has_video else
                                    "No video found at {}".format(video_file))
win_label = st.sidebar.select_slider("Window length", list(WINDOWS), value="2 min")
win_s = WINDOWS[win_label]
max_s = max(0, n_frames // FPS - 1)


def shift(delta):
    st.session_state[key] = int(np.clip(st.session_state[key] + delta, 0, max_s))
    st.session_state["initial_" + source] = st.session_state[key] * FPS


c1, c2 = st.sidebar.columns(2)
c1.button("◀ prev window", on_click=shift, args=(-win_s,), width="stretch")
c2.button("next window ▶", on_click=shift, args=(win_s,), width="stretch")
st.sidebar.slider("Window start (video seconds)", 0, max_s, key=key,
                  on_change=lambda: st.session_state.__setitem__("initial_" + source,
                                                                 st.session_state[key] * FPS))
start = st.session_state[key] * FPS
st.sidebar.caption("Window: **{} → {}** · frames {}–{}".format(
    fmt(start), fmt(min(n_frames, start + win_s * FPS)), start, min(n_frames, start + win_s * FPS) - 1))

with st.sidebar.form("goto"):
    st.number_input("Go to video frame", 0, n_frames - 1, value=start, step=1, key="goto_" + source)
    st.form_submit_button("Go", on_click=lambda: jump(source, st.session_state["goto_" + source]))

if source == "tactical":
    st.sidebar.markdown("**Quick jumps**")
    q1, q2, q3 = st.sidebar.columns(3)
    q1.button("1st KO", on_click=jump, args=(source, KICKOFFS[1]), width="stretch")
    q2.button("2nd KO", on_click=jump, args=(source, KICKOFFS[2]), width="stretch")
    q3.button("GT end", on_click=jump, args=(source, 18684 - 10 * FPS), width="stretch",
              help="last 10 s of the public 5-minute ground truth")

st.sidebar.divider()
st.sidebar.caption("{} frames @ {} fps · {} · {} track slots".format(
    n_frames, FPS, fmt(n_frames), trk.shape[1] - 1))


# ------------------------------------------------------------------------------------- tabs
tab_play, tab_tags, tab_hom, tab_meta = st.tabs(["▶ Player", "🏷 Smart Tagging", "📐 Homography", "🗂 Metadata"])

with tab_play:
    summary = per_second(source)
    fig = go.Figure()
    fig.add_bar(x=summary.t / 60, y=summary.ball, marker_color="#facc15", marker_line_width=0,
                name="ball tracked", hovertemplate="%{x:.1f} min · ball %{y:.0%}<extra></extra>")
    fig.add_scatter(x=summary.t / 60, y=summary.valid, mode="lines", line=dict(color="#a3e635", width=1),
                    name="homography valid", hovertemplate="%{x:.1f} min · valid %{y:.0%}<extra></extra>")
    if source == "tactical":
        for h, first, last in half_blocks():
            fig.add_vrect(x0=first / FPS / 60, x1=last / FPS / 60, fillcolor="#3b82f6", opacity=0.07,
                          line_width=0, annotation_text="{} half".format("1st" if h == 1 else "2nd"),
                          annotation_position="top left")
    fig.add_vrect(x0=start / FPS / 60, x1=(start + win_s * FPS) / FPS / 60, fillcolor="#ffffff",
                  opacity=0.25, line=dict(color="#ffffff", width=1))
    fig.update_layout(height=170, margin=dict(l=10, r=10, t=10, b=30), bargap=0, showlegend=True,
                      legend=dict(orientation="h", y=1.15, x=1, xanchor="right"),
                      xaxis_title="video minutes (click to jump)", yaxis=dict(range=[0, 1], tickformat=".0%"))
    picked = st.plotly_chart(fig, width="stretch", on_select="rerun", selection_mode="points",
                             key="overview_" + source)
    points = picked.selection.points if picked and picked.selection else []
    sel_id = json.dumps(points[:1], default=str)
    if points and st.session_state.get("overview_last_" + source) != sel_id:
        st.session_state["overview_last_" + source] = sel_id
        request_jump(source, int(points[0]["x"] * 60 * FPS), lead_s=0)

    blob, n_lanes = payload(source, start, win_s * FPS, st.session_state.get("initial_" + source, start))
    video = {"url": "{}/{}.mp4".format(video_server(), source)} if show_video else None
    html = PLAYER_HTML.replace("__DATA__", blob).replace("__VIDEO__", json.dumps(video))
    st.iframe(html, height=PITCH_HEIGHT_PX + 90 + 18 + n_lanes * 19 + 20)

    if source == "tactical":
        st.subheader("Smart Tags — select a row to jump")
        tags = smart_tagging()
        f1, f2, f3 = st.columns([2, 1, 1])
        codes = f1.multiselect("Code", sorted(tags.code.unique()), placeholder="all codes")
        teams = f2.multiselect("Team", sorted(tags.Team.dropna().unique()), placeholder="all teams")
        only_window = f3.toggle("Only current window", value=False)
        view = tags
        if codes:
            view = view[view.code.isin(codes)]
        if teams:
            view = view[view.Team.isin(teams)]
        if only_window:
            view = view[(view.end_frame > start) & (view.start_frame < start + win_s * FPS)]
        shown = view.assign(start=view.start_frame.map(fmt), duration_s=(view.end_s - view.start_s).round(1))[
            ["start", "code", "Team", "Type", "Half", "Side", "Direction of ball entry",
             "Max Players in the box", "duration_s", "start_frame", "end_frame"]]
        table = st.dataframe(shown, hide_index=True, width="stretch", height=280,
                             on_select="rerun", selection_mode="single-row", key="tag_table")
        rows = table.selection.rows if table and table.selection else []
        if rows:
            frame = int(shown.iloc[rows[0]].start_frame)
            if st.session_state.get("tag_last") != frame:
                st.session_state["tag_last"] = frame
                request_jump(source, frame)
    else:
        st.info("The broadcast recording has no Smart Tagging and no ground truth — both live on the "
                "tactical timeline, and the two recordings are not frame-aligned.")

with tab_tags:
    if source != "tactical":
        st.info("Smart Tagging exists only for the tactical recording. Switch the recording in the sidebar.")
    else:
        tags = smart_tagging()
        a, b, c = st.columns(3)
        a.metric("instances", len(tags))
        b.metric("codes", tags.code.nunique())
        c.metric("Team = N/A", int((tags.team_side == "N").sum()))

        counts = tags.groupby(["code", "Team"], dropna=False).size().reset_index(name="n")
        fig = go.Figure()
        for team, color in (("Barcelona", COLORS["H"]), ("Atletic Club", COLORS["A"]), ("N/A", COLORS["N"])):
            part = counts[counts.Team == team]
            fig.add_bar(y=part.code, x=part.n, orientation="h", name=team, marker_color=color)
        fig.update_layout(barmode="stack", height=420, margin=dict(l=10, r=10, t=30, b=10),
                          yaxis=dict(categoryorder="total ascending"), title="Instances per code and team")
        st.plotly_chart(fig, width="stretch")

        # Gantt-style overview of the whole match
        order = sorted(tags.code.unique())
        fig = go.Figure()
        for side, name in (("H", "Barcelona"), ("A", "Atletic Club"), ("N", "N/A")):
            part = tags[tags.team_side == side]
            fig.add_bar(y=part.code, x=(part.end_s - part.start_s) / 60, base=part.start_s / 60,
                        orientation="h", name=name, marker_color=COLORS[side], marker_line_width=0,
                        customdata=np.stack([part.start_frame.map(fmt), part.Type.fillna("")], axis=1),
                        hovertemplate="%{y}<br>%{customdata[0]} %{customdata[1]}<extra></extra>")
        for h, first, last in half_blocks():
            fig.add_vrect(x0=first / FPS / 60, x1=last / FPS / 60, fillcolor="#ffffff", opacity=0.04, line_width=0)
        fig.update_layout(barmode="overlay", height=480, margin=dict(l=10, r=10, t=30, b=30),
                          yaxis=dict(categoryorder="array", categoryarray=order[::-1]),
                          xaxis_title="video minutes", title="All tags over the match")
        st.plotly_chart(fig, width="stretch")

        groups = ["Type", "Side", "Direction of ball entry", "Max Players in the box", "Half"]
        cols = st.columns(len(groups))
        for col, group in zip(cols, groups):
            col.markdown("**{}**".format(group))
            col.dataframe(tags[group].value_counts(dropna=True).rename("n"), width="stretch")

with tab_hom:
    matrices, err, valid, cam = homography(source)
    a, b, c, d = st.columns(4)
    a.metric("frames", len(valid))
    b.metric("valid", "{:.1%}".format(valid.mean()))
    c.metric("median reproj. error", "{:.3f}".format(np.nanmedian(err[valid])) if valid.any() else "—")
    d.metric("p99 / max error", "{:.2f} / {:.1f}".format(np.nanpercentile(err[valid], 99), np.nanmax(err[valid]))
             if valid.any() else "—")

    summary = per_second(source)
    fig = go.Figure()
    fig.add_scatter(x=summary.t / 60, y=summary.err.clip(lower=1e-3), mode="lines", line=dict(width=1, color="#fb923c"),
                    name="max reproj. error per second")
    fig.add_scatter(x=summary.t / 60, y=summary.valid, mode="lines", line=dict(width=1, color="#a3e635"),
                    name="valid fraction", yaxis="y2")
    fig.update_layout(height=260, margin=dict(l=10, r=10, t=30, b=30), xaxis_title="video minutes",
                      yaxis=dict(type="log", title="error (log)"),
                      yaxis2=dict(overlaying="y", side="right", range=[0, 1.05], tickformat=".0%"),
                      legend=dict(orientation="h", y=1.15))
    st.plotly_chart(fig, width="stretch")

    frame = st.number_input("Inspect frame", 0, len(valid) - 1,
                            value=int(st.session_state.get("initial_" + source, start)), step=1,
                            help="Defaults to the playhead start of the current window.")
    left, right = st.columns([1, 2])
    with left:
        st.markdown("**valid:** {} · **reprojection error:** {:.3f}".format(bool(valid[frame]), err[frame]))
        if valid[frame]:
            st.markdown("Matrix (normalised pitch → pixels)")
            st.dataframe(pd.DataFrame(matrices[frame], columns=["c0", "c1", "c2"]), width="stretch")
            corners = pd.DataFrame(cam[frame], columns=["pitch x", "pitch y"],
                                   index=["image top-left", "top-right", "bottom-right", "bottom-left"])
            st.markdown("Image corners unprojected onto the pitch (NaN = above the horizon)")
            st.dataframe(corners, width="stretch")
        else:
            st.warning("No homography for this frame (replay, close-up or crowd shot).")

    with right:
        if valid[frame]:
            H = matrices[frame]

            def proj(x, y):
                x, y = np.asarray(x, float), np.asarray(y, float)
                w = H[2, 0] * x + H[2, 1] * y + H[2, 2]
                u = (H[0, 0] * x + H[0, 1] * y + H[0, 2]) / w
                v = (H[1, 0] * x + H[1, 1] * y + H[1, 2]) / w
                return np.where(w > 0, u, np.nan), np.where(w > 0, v, np.nan)

            L, W = PITCH
            segs = []

            def poly(xs, ys, dense=40):
                xs, ys = np.asarray(xs, float), np.asarray(ys, float)
                t = np.linspace(0, 1, dense)
                px = np.concatenate([a + (b - a) * t for a, b in zip(xs[:-1], xs[1:])])
                py = np.concatenate([a + (b - a) * t for a, b in zip(ys[:-1], ys[1:])])
                segs.append((px, py))

            poly([0, 1, 1, 0, 0], [0, 0, 1, 1, 0])
            poly([.5, .5], [0, 1])
            bw, bd, gw, gd = 40.32 / W, 16.5 / L, 18.32 / W, 5.5 / L
            for x0, x1 in ((0, bd), (1 - bd, 1)):
                poly([x0, x1, x1, x0, x0], [.5 - bw / 2, .5 - bw / 2, .5 + bw / 2, .5 + bw / 2, .5 - bw / 2])
            for x0, x1 in ((0, gd), (1 - gd, 1)):
                poly([x0, x1, x1, x0, x0], [.5 - gw / 2, .5 - gw / 2, .5 + gw / 2, .5 + gw / 2, .5 - gw / 2])
            ang = np.linspace(0, 2 * np.pi, 120)
            segs.append((.5 + 9.15 / L * np.cos(ang), .5 + 9.15 / W * np.sin(ang)))

            fig = go.Figure()
            for xs, ys in segs:
                u, v = proj(xs, ys)
                fig.add_scatter(x=u, y=v, mode="lines", line=dict(color="white", width=1.5),
                                hoverinfo="skip", showlegend=False)
            meta = metadata(source)
            objs = np.asarray(trk[frame])
            sides = np.array(meta["slots"]["side"].tolist())
            for side in ("H", "A"):
                sel = sides == side
                u, v = proj(objs[:-1, 0][sel], objs[:-1, 1][sel])
                fig.add_scatter(x=u, y=v, mode="markers", name=meta["teams"][0 if side == "H" else 1]["name"],
                                marker=dict(color=COLORS[side], size=10, line=dict(color="black", width=1)),
                                text=["Track {}".format(s) for s in meta["slots"]["slot"][sel]],
                                hovertemplate="%{text}<br>%{x:.0f}, %{y:.0f} px<extra></extra>")
            u, v = proj(objs[-1, 0], objs[-1, 1])
            fig.add_scatter(x=[u], y=[v], mode="markers", name="ball",
                            marker=dict(color="#facc15", size=9, line=dict(color="black", width=1)))
            iw, ih = IMAGE_SIZE
            fig.add_shape(type="rect", x0=0, y0=0, x1=iw, y1=ih, line=dict(color="#fde68a", dash="dash"))
            fig.update_layout(height=470, margin=dict(l=10, r=10, t=30, b=10), plot_bgcolor="#1f5130",
                              title="Pitch and tracking projected into the {}×{} image".format(iw, ih),
                              xaxis=dict(range=[-200, iw + 200], showgrid=False, zeroline=False),
                              yaxis=dict(range=[ih + 200, -200], showgrid=False, zeroline=False,
                                         scaleanchor="x"),
                              legend=dict(orientation="h", y=-0.05))
            st.plotly_chart(fig, width="stretch")

with tab_meta:
    meta = metadata(source)
    st.subheader("ATD metadata — {}".format(source))
    a, b, c = st.columns(3)
    with a:
        st.markdown("**Global config**")
        st.json(meta["global"], expanded=True)
        st.markdown("**Provider parameters**")
        st.json(meta["global_params"], expanded=False)
    with b:
        st.markdown("**Session**")
        st.json(meta["session"], expanded=True)
        st.markdown("**Teams**")
        st.json(meta["teams"], expanded=True)
    with c:
        st.markdown("**games.py**")
        st.json({
            "game": GAME,
            "fps": FPS,
            "pitch_m": list(PITCH),
            "n_video_frames (this source)": n_frames,
            "graded halves (video frames)": [list(h) for h in half_blocks()],
            "kickoffs (video frames)": KICKOFFS,
            "ATD frame_id": "video_frame + 1",
            "Smart Tagging": "video_frame = round(seconds * 25), tactical only",
        }, expanded=True)

    st.markdown("**Track slots** (slot ids are not players or shirt numbers)")
    slots = meta["slots"].copy()
    slots["coverage"] = slot_coverage(source)
    st.dataframe(slots, hide_index=True, width="stretch", height=300,
                 column_config={"coverage": st.column_config.ProgressColumn(
                     "frames present", format="%.1f%%", min_value=0, max_value=1)})

    summary = per_second(source)
    fig = go.Figure()
    fig.add_scatter(x=summary.t / 60, y=summary.objects, mode="lines", line=dict(width=1, color="#93c5fd"),
                    name="tracked players")
    fig.add_hline(y=22, line=dict(color="#ffffff", dash="dot", width=1))
    fig.update_layout(height=230, margin=dict(l=10, r=10, t=30, b=30), xaxis_title="video minutes",
                      title="Tracked players per frame (mean per second)")
    st.plotly_chart(fig, width="stretch")

    if source == "tactical":
        st.markdown("**Label files** (public GT and `evaluation/results/*.csv`)")
        rows = []
        for name, labels in label_tracks().items():
            values, counts = np.unique(labels, return_counts=True)
            rows.append({"file": name, **dict(zip(values, counts.tolist()))})
        st.dataframe(pd.DataFrame(rows).fillna(0), hide_index=True, width="stretch")
