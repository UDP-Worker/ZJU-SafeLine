import argparse
import base64
import cgi
import hashlib
import json
import mimetypes
import os
import socket
import struct
import sys
import threading
import time
from collections import deque
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib import parse as urlparse

import cv2
import numpy as np
import torch
from torch import nn


TEMPLATE_PATH = os.path.join(os.path.dirname(__file__), "templates", "index.html")


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
        files.append((path, stat.st_mtime, stat.st_size))
        total_size += stat.st_size

    files.sort(key=lambda item: item[1])

    for path, mtime, size in files:
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


WS_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"


def build_ws_frame(payload, opcode=0x1):
    length = len(payload)
    if length < 126:
        header = struct.pack("!BB", 0x80 | opcode, length)
    elif length < (1 << 16):
        header = struct.pack("!BBH", 0x80 | opcode, 126, length)
    else:
        header = struct.pack("!BBQ", 0x80 | opcode, 127, length)
    return header + payload


def recv_exact(sock, size):
    data = b""
    while len(data) < size:
        chunk = sock.recv(size - len(data))
        if not chunk:
            return None
        data += chunk
    return data


class WebSocketClient:
    def __init__(self, sock, addr):
        self.sock = sock
        self.addr = addr
        self.lock = threading.Lock()

    def send_frame(self, frame):
        with self.lock:
            self.sock.sendall(frame)


class WebSocketHub:
    def __init__(self):
        self.lock = threading.Lock()
        self.clients = set()

    def add(self, client):
        with self.lock:
            self.clients.add(client)

    def remove(self, client):
        with self.lock:
            self.clients.discard(client)

    def broadcast_json(self, payload):
        data = json.dumps(payload).encode("utf-8")
        frame = build_ws_frame(data, opcode=0x1)
        with self.lock:
            clients = list(self.clients)
        for client in clients:
            try:
                client.send_frame(frame)
            except OSError:
                self.remove(client)

    def send_json(self, client, payload):
        data = json.dumps(payload).encode("utf-8")
        frame = build_ws_frame(data, opcode=0x1)
        client.send_frame(frame)


class InferenceWorker(threading.Thread):
    def __init__(self, state, hub, model, device, args):
        super().__init__(daemon=True)
        self.state = state
        self.hub = hub
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
                    self.hub.broadcast_json(self.state.get_status())
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
                        self.hub.broadcast_json(self.state.get_status())
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
                        self.hub.broadcast_json(self.state.get_status())
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
                self.hub.broadcast_json(self.state.get_status())
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
                self.hub.broadcast_json(self.state.get_status())
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
            self.hub.broadcast_json(self.state.get_status())


def make_handler(state, hub, args):
    class Handler(BaseHTTPRequestHandler):
        def _write_html(self, html, status=HTTPStatus.OK):
            data = html.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _write_json(self, payload, status=HTTPStatus.OK):
            data = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == "/" or self.path.startswith("/index"):
                with open(TEMPLATE_PATH, "r", encoding="utf-8") as handle:
                    return self._write_html(handle.read())
            if self.path.startswith("/status"):
                return self._write_json(state.get_status())
            if self.path.startswith("/ws"):
                return self._handle_websocket()
            if self.path.startswith("/static/"):
                return self._serve_static(self.path[len("/static/") :])
            if self.path.startswith("/uploads/"):
                return self._serve_upload(self.path[len("/uploads/") :])
            self.send_error(HTTPStatus.NOT_FOUND, "Not Found")

        def do_POST(self):
            if self.path == "/upload":
                return self._handle_upload()
            if self.path == "/set_stream":
                return self._handle_stream()
            if self.path == "/playback":
                return self._handle_playback()
            self.send_error(HTTPStatus.NOT_FOUND, "Not Found")

        def _handle_upload(self):
            form = cgi.FieldStorage(fp=self.rfile, headers=self.headers, environ={"REQUEST_METHOD": "POST"})
            if "video" not in form:
                return self._write_html("No file uploaded", status=HTTPStatus.BAD_REQUEST)
            file_item = form["video"]
            if not file_item.filename:
                return self._write_html("Empty filename", status=HTTPStatus.BAD_REQUEST)
            uploads_dir = os.path.abspath(args.uploads_dir)
            os.makedirs(uploads_dir, exist_ok=True)
            safe_name = os.path.basename(file_item.filename)
            out_path = os.path.join(uploads_dir, safe_name)
            with open(out_path, "wb") as handle:
                handle.write(file_item.file.read())
            playback_url = f"/uploads/{urlparse.quote(safe_name)}"
            state.set_source(out_path, safe_name, playback_url=playback_url)
            hub.broadcast_json(state.get_status())
            self.send_response(HTTPStatus.SEE_OTHER)
            self.send_header("Location", "/")
            self.end_headers()

        def _handle_stream(self):
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length).decode("utf-8")
            params = urlparse.parse_qs(body, keep_blank_values=True)
            stream_url = params.get("stream_url", [""])[0].strip()
            play_url = params.get("play_url", [""])[0].strip()
            if not stream_url:
                return self._write_html("Empty stream url", status=HTTPStatus.BAD_REQUEST)
            if stream_url.isdigit():
                source = int(stream_url)
            else:
                source = stream_url
            playback_url = play_url
            if not playback_url and stream_url.startswith(("http://", "https://")):
                playback_url = stream_url
            state.set_source(source, str(stream_url), playback_url=playback_url)
            hub.broadcast_json(state.get_status())
            self.send_response(HTTPStatus.SEE_OTHER)
            self.send_header("Location", "/")
            self.end_headers()

        def _handle_playback(self):
            length = int(self.headers.get("Content-Length", 0))
            if length <= 0:
                self.send_error(HTTPStatus.BAD_REQUEST, "Missing body")
                return
            body = self.rfile.read(length).decode("utf-8")
            try:
                payload = json.loads(body)
            except json.JSONDecodeError:
                self.send_error(HTTPStatus.BAD_REQUEST, "Invalid JSON")
                return
            if "time" not in payload:
                self.send_error(HTTPStatus.BAD_REQUEST, "Missing time")
                return
            try:
                time_value = float(payload["time"])
            except (TypeError, ValueError):
                self.send_error(HTTPStatus.BAD_REQUEST, "Invalid time")
                return
            paused = bool(payload.get("paused", False))
            state.update_playback(time_value, paused)
            self.send_response(HTTPStatus.NO_CONTENT)
            self.end_headers()

        def _serve_upload(self, name):
            decoded = urlparse.unquote(name)
            safe_name = os.path.basename(decoded)
            if safe_name != decoded:
                self.send_error(HTTPStatus.NOT_FOUND, "Not Found")
                return
            uploads_dir = os.path.abspath(args.uploads_dir)
            path = os.path.join(uploads_dir, safe_name)
            if not os.path.isfile(path):
                self.send_error(HTTPStatus.NOT_FOUND, "Not Found")
                return
            self._send_file(path)

        def _serve_static(self, name):
            decoded = urlparse.unquote(name)
            safe_name = os.path.basename(decoded)
            if safe_name != decoded:
                self.send_error(HTTPStatus.NOT_FOUND, "Not Found")
                return
            static_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "static"))
            path = os.path.join(static_dir, safe_name)
            if not os.path.isfile(path):
                self.send_error(HTTPStatus.NOT_FOUND, "Not Found")
                return
            self._send_file(path)

        def _send_file(self, path):
            size = os.path.getsize(path)
            content_type = mimetypes.guess_type(path)[0] or "application/octet-stream"
            range_header = self.headers.get("Range")
            start = 0
            end = size - 1
            status = HTTPStatus.OK

            if range_header and range_header.startswith("bytes="):
                try:
                    range_spec = range_header.replace("bytes=", "", 1).strip()
                    if "," not in range_spec:
                        start_text, end_text = range_spec.split("-", 1)
                        if start_text:
                            start = int(start_text)
                            if end_text:
                                end = int(end_text)
                        else:
                            suffix = int(end_text)
                            start = max(0, size - suffix)
                        end = min(end, size - 1)
                        if start <= end:
                            status = HTTPStatus.PARTIAL_CONTENT
                        else:
                            self.send_error(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                            return
                except ValueError:
                    self.send_error(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                    return

            length = end - start + 1
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Accept-Ranges", "bytes")
            if status == HTTPStatus.PARTIAL_CONTENT:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.send_header("Content-Length", str(length))
            self.end_headers()

            with open(path, "rb") as handle:
                if start:
                    handle.seek(start)
                remaining = length
                while remaining > 0:
                    chunk = handle.read(min(65536, remaining))
                    if not chunk:
                        break
                    try:
                        self.wfile.write(chunk)
                    except (BrokenPipeError, ConnectionResetError):
                        break
                    remaining -= len(chunk)

        def _handle_websocket(self):
            key = self.headers.get("Sec-WebSocket-Key")
            if not key:
                self.send_error(HTTPStatus.BAD_REQUEST, "Missing WebSocket key")
                return
            accept_src = (key + WS_GUID).encode("utf-8")
            accept = base64.b64encode(hashlib.sha1(accept_src).digest()).decode("ascii")
            self.send_response(HTTPStatus.SWITCHING_PROTOCOLS)
            self.send_header("Upgrade", "websocket")
            self.send_header("Connection", "Upgrade")
            self.send_header("Sec-WebSocket-Accept", accept)
            self.end_headers()

            client = WebSocketClient(self.request, self.client_address)
            hub.add(client)
            try:
                hub.send_json(client, state.get_status())
                self._ws_read_loop(client)
            finally:
                hub.remove(client)

        def _ws_read_loop(self, client):
            sock = client.sock
            while True:
                header = recv_exact(sock, 2)
                if header is None:
                    break
                b1, b2 = header
                opcode = b1 & 0x0F
                masked = b2 & 0x80
                length = b2 & 0x7F
                if length == 126:
                    ext = recv_exact(sock, 2)
                    if ext is None:
                        break
                    length = struct.unpack("!H", ext)[0]
                elif length == 127:
                    ext = recv_exact(sock, 8)
                    if ext is None:
                        break
                    length = struct.unpack("!Q", ext)[0]
                mask = recv_exact(sock, 4) if masked else None
                payload = recv_exact(sock, length) if length else b""
                if payload is None:
                    break
                if mask:
                    payload = bytes(b ^ mask[i % 4] for i, b in enumerate(payload))
                if opcode == 0x8:
                    break
                if opcode == 0x9:
                    pong = build_ws_frame(payload, opcode=0xA)
                    try:
                        client.send_frame(pong)
                    except OSError:
                        break

        def log_message(self, format, *args):
            return

    return Handler


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
    state = AppState()
    hub = WebSocketHub()
    uploads_dir = os.path.abspath(args.uploads_dir)
    os.makedirs(uploads_dir, exist_ok=True)
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
    worker = InferenceWorker(state, hub, model, device, args)
    worker.start()
    handler = make_handler(state, hub, args)
    server = ThreadingHTTPServer(("0.0.0.0", args.port), handler)
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
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
