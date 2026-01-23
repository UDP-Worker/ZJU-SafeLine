import argparse
import asyncio
import logging
import mimetypes
import os
import socket
import sys
import threading
import time
from collections import deque
from contextlib import asynccontextmanager, suppress
from urllib import parse as urlparse

import cv2
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from torch import nn

BASE_DIR = os.path.dirname(__file__)
TEMPLATE_PATH = os.path.join(BASE_DIR, "templates", "index.html")
STATIC_DIR = os.path.join(BASE_DIR, "static")


class SimpleCNN(nn.Module):
    def __init__(self, in_channels=4):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(128, 1)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x).squeeze(1)


def load_torch_checkpoint(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_model(model_path, device):
    obj = load_torch_checkpoint(model_path)
    model = SimpleCNN(in_channels=4)
    if isinstance(obj, dict) and "model_state" in obj:
        state_dict = obj["model_state"]
    elif isinstance(obj, dict):
        state_dict = obj
    else:
        raise ValueError("unsupported model checkpoint format")
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def pad_frame(gray, target_h, target_w):
    height, width = gray.shape[:2]
    if height > target_h or width > target_w:
        return cv2.resize(gray, (target_w, target_h))
    pad_bottom = target_h - height
    pad_right = target_w - width
    if pad_bottom == 0 and pad_right == 0:
        return gray
    return cv2.copyMakeBorder(gray, 0, pad_bottom, 0, pad_right, cv2.BORDER_CONSTANT, value=0)


def diff_features(frames):
    diffs = []
    prev = frames[0].astype(np.int16)
    for img in frames[1:]:
        current = img.astype(np.int16)
        diff = current - prev
        diff = np.clip((diff + 255) / 2.0, 0, 255).astype(np.uint8)
        diffs.append(diff)
        prev = current
    return np.stack(diffs, axis=0)


class AppState:
    def __init__(self):
        self.lock = threading.Lock()
        self.source = None
        self.source_name = "none"
        self.playback_url = ""
        self.source_version = 0
        self.last_label = "idle"
        self.last_prob = 0.0
        self.last_ts = 0.0
        self.playback_time = None
        self.playback_paused = True
        self.playback_seq = 0

    def set_source(self, source, name, playback_url=None):
        with self.lock:
            self.source = source
            self.source_name = name
            if playback_url is not None:
                self.playback_url = playback_url
            self.source_version += 1
            self.last_label = "warming up"
            self.last_prob = 0.0
            self.last_ts = time.time()
            self.playback_time = None
            self.playback_paused = True
            self.playback_seq = 0

    def get_source(self):
        with self.lock:
            return self.source, self.source_name, self.source_version

    def update_result(self, label, prob):
        with self.lock:
            self.last_label = label
            self.last_prob = prob
            self.last_ts = time.time()

    def update_playback(self, time_sec, paused=False):
        with self.lock:
            self.playback_time = max(0.0, float(time_sec))
            self.playback_paused = bool(paused)
            self.playback_seq += 1

    def get_playback(self):
        with self.lock:
            return self.playback_time, self.playback_paused, self.playback_seq

    def get_status(self):
        with self.lock:
            return {
                "source": self.source_name,
                "playback_url": self.playback_url,
                "label": self.last_label,
                "prob": self.last_prob,
                "timestamp": self.last_ts,
            }


def cleanup_uploads(uploads_dir, max_age_minutes, max_total_mb, state):
    now = time.time()
    max_age_seconds = max_age_minutes * 60.0
    max_total_bytes = max_total_mb * 1024 * 1024
    current_source, _, _ = state.get_source()
    current_source = os.path.abspath(current_source) if isinstance(current_source, str) else None

    files = []
    total_size = 0
    for name in os.listdir(uploads_dir):
        path = os.path.join(uploads_dir, name)
        if not os.path.isfile(path):
            continue
        try:
            stat = os.stat(path)
        except OSError:
            continue
        size = stat.st_size
        mtime = stat.st_mtime
        files.append((path, mtime, size))
        total_size += size

    files.sort(key=lambda item: item[1])
    for path, mtime, size in files[:]:
        if current_source and os.path.abspath(path) == current_source:
            continue
        if max_age_minutes > 0 and (now - mtime) >= max_age_seconds:
            try:
                os.remove(path)
                total_size -= size
            except OSError:
                continue

    if max_total_mb > 0 and total_size > max_total_bytes:
        for path, _, size in files:
            if current_source and os.path.abspath(path) == current_source:
                continue
            if not os.path.isfile(path):
                continue
            try:
                os.remove(path)
                total_size -= size
            except OSError:
                continue
            if total_size <= max_total_bytes:
                break


def cleanup_loop(uploads_dir, max_age_minutes, max_total_mb, interval, state):
    while True:
        try:
            cleanup_uploads(uploads_dir, max_age_minutes, max_total_mb, state)
        except OSError:
            pass
        time.sleep(interval)


def parse_args():
    parser = argparse.ArgumentParser(description="Web UI for real-time video inference.")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind.")
    parser.add_argument("--model", default="final_model.pt", help="Path to model checkpoint.")
    parser.add_argument("--height", type=int, default=810, help="Target height for padding.")
    parser.add_argument("--width", type=int, default=892, help="Target width for padding.")
    parser.add_argument("--interval", type=float, default=0.25, help="Frame sampling interval.")
    parser.add_argument("--threshold", type=float, default=0.7, help="Decision threshold.")
    parser.add_argument("--loop", action="store_true", help="Loop video files when finished.")
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Device for inference (default: auto).",
    )
    parser.add_argument("--uploads-dir", default="uploads", help="Upload storage directory.")
    parser.add_argument(
        "--cleanup-minutes",
        type=int,
        default=60,
        help="Delete uploads older than N minutes (default: 60).",
    )
    parser.add_argument(
        "--cleanup-max-mb",
        type=int,
        default=2048,
        help="Keep uploads under this size in MB (default: 2048).",
    )
    parser.add_argument(
        "--cleanup-interval",
        type=int,
        default=300,
        help="Cleanup loop interval in seconds (default: 300).",
    )
    return parser.parse_args()


class ConnectionManager:
    def __init__(self):
        self.active = set()

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active.add(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active.discard(websocket)

    async def broadcast(self, payload):
        to_remove = []
        for ws in list(self.active):
            try:
                await ws.send_json(payload)
            except Exception:
                to_remove.append(ws)
        for ws in to_remove:
            self.active.discard(ws)


class Broadcaster:
    def __init__(self, manager, max_queue=100):
        self.manager = manager
        self.queue = asyncio.Queue(maxsize=max_queue)
        self.loop = None

    def set_loop(self, loop):
        self.loop = loop

    def publish(self, payload):
        if not self.loop:
            return

        def _offer():
            if self.queue.full():
                try:
                    self.queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            try:
                self.queue.put_nowait(payload)
            except asyncio.QueueFull:
                pass

        self.loop.call_soon_threadsafe(_offer)

    async def run(self):
        while True:
            payload = await self.queue.get()
            await self.manager.broadcast(payload)


class InferenceWorker(threading.Thread):
    def __init__(self, state, broadcaster, model, device, args):
        super().__init__(daemon=True)
        self.state = state
        self.broadcaster = broadcaster
        self.model = model
        self.device = device
        self.args = args

    def _open_capture(self, source):
        if isinstance(source, int):
            cap = cv2.VideoCapture(source, cv2.CAP_V4L2)
            if cap.isOpened():
                return cap
        return cv2.VideoCapture(source)

    def _read_frame_at(self, cap, time_sec):
        cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, time_sec) * 1000.0)
        ok, frame = cap.read()
        if not ok:
            return None
        return frame

    def _read_window(self, cap, target_time, interval):
        start_time = target_time - interval * 4
        frames = []
        for i in range(5):
            time_sec = start_time + i * interval
            frame = self._read_frame_at(cap, time_sec)
            if frame is None:
                return None
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = pad_frame(gray, self.args.height, self.args.width)
            frames.append(gray)
        return frames

    def run(self):
        sample_interval = max(0.05, float(self.args.interval))
        frame_buffer = deque(maxlen=5)
        cap = None
        current_version = -1
        last_sample = 0.0
        last_target_time = None

        while True:
            source, _, version = self.state.get_source()
            playback_time, _, _ = self.state.get_playback()
            if source is None:
                if cap:
                    cap.release()
                    cap = None
                time.sleep(0.2)
                continue

            if version != current_version or cap is None:
                if cap:
                    cap.release()
                cap = self._open_capture(source)
                current_version = version
                frame_buffer.clear()
                last_sample = 0.0
                last_target_time = None
                if not cap.isOpened():
                    cap.release()
                    cap = None
                    self.state.update_result("source unavailable", 0.0)
                    self.broadcaster.publish(self.state.get_status())
                    time.sleep(1.0)
                    continue

            is_file = isinstance(source, str) and os.path.isfile(source)
            if is_file and playback_time is not None:
                target_time = max(0.0, float(playback_time))
                if last_target_time is not None and abs(target_time - last_target_time) < 1e-3:
                    time.sleep(0.05)
                    continue

                delta = None if last_target_time is None else target_time - last_target_time
                if last_target_time is None or abs(delta) > sample_interval * 1.25:
                    frames = self._read_window(cap, target_time, sample_interval)
                    if frames is None:
                        self.state.update_result("source ended", 0.0)
                        self.broadcaster.publish(self.state.get_status())
                        time.sleep(0.2)
                        continue
                    frame_buffer.clear()
                    frame_buffer.extend(frames)
                    last_target_time = target_time
                else:
                    if delta < sample_interval * 0.75:
                        time.sleep(0.02)
                        continue
                    next_time = last_target_time + sample_interval
                    frame = self._read_frame_at(cap, next_time)
                    if frame is None:
                        self.state.update_result("source ended", 0.0)
                        self.broadcaster.publish(self.state.get_status())
                        time.sleep(0.2)
                        continue
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    gray = pad_frame(gray, self.args.height, self.args.width)
                    frame_buffer.append(gray)
                    last_target_time = next_time
                if len(frame_buffer) != 5:
                    continue

                diffs = diff_features(list(frame_buffer))
                x = diffs.astype(np.float32)
                x = (x - 127.5) / 127.5
                x = torch.from_numpy(x).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    logits = self.model(x)
                    prob = torch.sigmoid(logits).item()
                label = "abnormal" if prob >= self.args.threshold else "normal"
                self.state.update_result(label, prob)
                self.broadcaster.publish(self.state.get_status())
                continue

            now = time.time()
            wait = sample_interval - (now - last_sample)
            if wait > 0:
                time.sleep(min(wait, 0.05))
                continue

            ok, frame = cap.read()
            if not ok:
                if self.args.loop and isinstance(source, str) and os.path.isfile(source):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                cap.release()
                cap = None
                self.state.update_result("source ended", 0.0)
                self.broadcaster.publish(self.state.get_status())
                time.sleep(0.5)
                continue

            last_sample = time.time()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = pad_frame(gray, self.args.height, self.args.width)
            frame_buffer.append(gray)

            if len(frame_buffer) != 5:
                continue

            diffs = diff_features(list(frame_buffer))
            x = diffs.astype(np.float32)
            x = (x - 127.5) / 127.5
            x = torch.from_numpy(x).unsqueeze(0).to(self.device)
            with torch.no_grad():
                logits = self.model(x)
                prob = torch.sigmoid(logits).item()
            label = "abnormal" if prob >= self.args.threshold else "normal"
            self.state.update_result(label, prob)
            self.broadcaster.publish(self.state.get_status())


class PlaybackPayload(BaseModel):
    time: float
    paused: bool = False


def create_app(args, model, device):
    state = AppState()
    manager = ConnectionManager()
    broadcaster = Broadcaster(manager)
    uploads_dir = os.path.abspath(args.uploads_dir)
    os.makedirs(uploads_dir, exist_ok=True)

    @asynccontextmanager
    async def lifespan(app):
        broadcaster.set_loop(asyncio.get_running_loop())
        app.state.broadcast_task = asyncio.create_task(broadcaster.run())
        if args.cleanup_minutes > 0 or args.cleanup_max_mb > 0:
            cleanup_thread = threading.Thread(
                target=cleanup_loop,
                args=(
                    uploads_dir,
                    args.cleanup_minutes,
                    args.cleanup_max_mb,
                    max(30, args.cleanup_interval),
                    state,
                ),
                daemon=True,
            )
            cleanup_thread.start()
            app.state.cleanup_thread = cleanup_thread
        worker = InferenceWorker(state, broadcaster, model, device, args)
        worker.start()
        app.state.worker = worker
        yield
        task = getattr(app.state, "broadcast_task", None)
        if task:
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task

    app = FastAPI(lifespan=lifespan)
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/", response_class=HTMLResponse)
    @app.get("/index", response_class=HTMLResponse)
    @app.get("/index.html", response_class=HTMLResponse)
    async def index():
        with open(TEMPLATE_PATH, "r", encoding="utf-8") as handle:
            return HTMLResponse(handle.read())

    @app.get("/status")
    async def status():
        return state.get_status()

    @app.get("/uploads/{name}")
    async def uploads(name: str):
        decoded = urlparse.unquote(name)
        safe_name = os.path.basename(decoded)
        if safe_name != decoded:
            raise HTTPException(status_code=404, detail="Not Found")
        path = os.path.join(uploads_dir, safe_name)
        if not os.path.isfile(path):
            raise HTTPException(status_code=404, detail="Not Found")
        media_type = mimetypes.guess_type(path)[0] or "application/octet-stream"
        return FileResponse(path, media_type=media_type)

    @app.post("/upload")
    async def upload(video: UploadFile = File(...)):
        if not video.filename:
            raise HTTPException(status_code=400, detail="Empty filename")
        safe_name = os.path.basename(video.filename)
        if not safe_name:
            raise HTTPException(status_code=400, detail="Empty filename")
        out_path = os.path.join(uploads_dir, safe_name)
        with open(out_path, "wb") as handle:
            while True:
                chunk = await video.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
        await video.close()
        playback_url = f"/uploads/{urlparse.quote(safe_name)}"
        state.set_source(out_path, safe_name, playback_url=playback_url)
        broadcaster.publish(state.get_status())
        return RedirectResponse(url="/", status_code=303)

    @app.post("/set_stream")
    async def set_stream(stream_url: str = Form(...), play_url: str = Form("")):
        stream_url = stream_url.strip()
        play_url = play_url.strip()
        if not stream_url:
            raise HTTPException(status_code=400, detail="Empty stream url")
        if stream_url.isdigit():
            source = int(stream_url)
        else:
            source = stream_url
        playback_url = play_url
        if not playback_url and stream_url.startswith(("http://", "https://")):
            playback_url = stream_url
        state.set_source(source, str(stream_url), playback_url=playback_url)
        broadcaster.publish(state.get_status())
        return RedirectResponse(url="/", status_code=303)

    @app.post("/playback")
    async def playback(payload: PlaybackPayload):
        state.update_playback(payload.time, payload.paused)
        return Response(status_code=204)

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        await manager.connect(websocket)
        try:
            await websocket.send_json(state.get_status())
            while True:
                await websocket.receive_text()
        except WebSocketDisconnect:
            manager.disconnect(websocket)
        except Exception:
            manager.disconnect(websocket)
            raise

    return app


class PlaybackAccessFilter(logging.Filter):
    def filter(self, record):
        return "/playback" not in record.getMessage()


def configure_access_log():
    logging.getLogger("uvicorn.access").addFilter(PlaybackAccessFilter())


def main():
    args = parse_args()
    if not os.path.isfile(args.model):
        print(f"error: model not found: {args.model}", file=sys.stderr)
        return 1
    if args.device == "cpu":
        device = torch.device("cpu")
    elif args.device == "cuda":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            print("warn: cuda not available, falling back to cpu.", file=sys.stderr)
            device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.model, device)
    app = create_app(args, model, device)
    configure_access_log()
    print(f"inference device: {device}")
    print(f"local: http://127.0.0.1:{args.port}")
    lan_ip = None
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.connect(("8.8.8.8", 80))
        lan_ip = sock.getsockname()[0]
        sock.close()
    except OSError:
        try:
            lan_ip = socket.gethostbyname(socket.gethostname())
        except OSError:
            lan_ip = None
    if lan_ip and lan_ip != "127.0.0.1":
        print(f"lan: http://{lan_ip}:{args.port}")
    uvicorn.run(app, host="0.0.0.0", port=args.port, workers=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
