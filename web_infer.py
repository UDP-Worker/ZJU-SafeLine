import argparse
import cgi
import io
import os
import threading
import time
from collections import deque
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np
import torch
from torch import nn


HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>视频异常监测</title>
  <style>
    :root {
      --ink: #0c131d;
      --muted: #5b6675;
      --accent: #1f7a5e;
      --accent-2: #e2b451;
      --card: #ffffff;
      --stroke: #d9dde3;
      --shadow: 0 18px 40px rgba(15, 20, 30, 0.12);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: "Space Grotesk", "Avenir Next", "Gill Sans", sans-serif;
      color: var(--ink);
      background:
        radial-gradient(circle at 10% 20%, #f9f1d8 0%, rgba(249, 241, 216, 0) 50%),
        radial-gradient(circle at 80% 0%, #d7efe3 0%, rgba(215, 239, 227, 0) 45%),
        linear-gradient(135deg, #f7f8fb 0%, #eef1f6 100%);
      min-height: 100vh;
      padding: 32px 24px 60px;
    }
    header {
      max-width: 1100px;
      margin: 0 auto 24px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 24px;
      flex-wrap: wrap;
    }
    .title {
      font-size: 28px;
      letter-spacing: -0.5px;
      margin: 0;
    }
    .subtitle {
      color: var(--muted);
      margin-top: 6px;
      font-size: 15px;
    }
    .pill {
      background: rgba(31, 122, 94, 0.12);
      color: var(--accent);
      border-radius: 999px;
      padding: 6px 14px;
      font-size: 12px;
      font-weight: 600;
    }
    .layout {
      max-width: 1100px;
      margin: 0 auto;
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
      gap: 20px;
    }
    .card {
      background: var(--card);
      border: 1px solid var(--stroke);
      border-radius: 16px;
      padding: 18px;
      box-shadow: var(--shadow);
      animation: rise 0.5s ease-out;
    }
    .card h3 {
      margin: 0 0 12px;
      font-size: 18px;
    }
    .card p {
      margin: 0 0 12px;
      color: var(--muted);
      font-size: 13px;
      line-height: 1.5;
    }
    input[type="file"], input[type="text"] {
      width: 100%;
      padding: 10px 12px;
      border-radius: 10px;
      border: 1px solid var(--stroke);
      font-size: 14px;
    }
    button {
      margin-top: 12px;
      padding: 10px 16px;
      border-radius: 999px;
      border: none;
      background: linear-gradient(135deg, var(--accent) 0%, #1d5f86 100%);
      color: white;
      font-weight: 600;
      cursor: pointer;
      transition: transform 0.15s ease;
    }
    .file-row {
      display: flex;
      align-items: center;
      gap: 10px;
      flex-wrap: wrap;
    }
    .file-input {
      display: none;
    }
    .file-button {
      display: inline-block;
      padding: 9px 14px;
      border-radius: 999px;
      background: linear-gradient(135deg, #27354a 0%, #1f7a5e 100%);
      color: #fff;
      font-size: 13px;
      cursor: pointer;
      text-decoration: none;
    }
    .file-name {
      font-size: 13px;
      color: var(--muted);
      max-width: 160px;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    button:hover { transform: translateY(-1px); }
    .viewer {
      max-width: 1100px;
      margin: 28px auto 0;
      background: var(--card);
      border-radius: 20px;
      border: 1px solid var(--stroke);
      box-shadow: var(--shadow);
      padding: 18px;
      animation: fade 0.7s ease-out;
    }
    .viewer h3 {
      margin: 0 0 12px;
      font-size: 18px;
      display: flex;
      align-items: center;
      gap: 10px;
    }
    .viewer h3 span {
      color: var(--accent-2);
      font-size: 12px;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 1px;
    }
    .viewer img {
      width: 100%;
      border-radius: 14px;
      border: 1px solid #e4e6ea;
      background: #f2f4f7;
    }
    @keyframes rise { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: translateY(0); } }
    @keyframes fade { from { opacity: 0; } to { opacity: 1; } }
  </style>
</head>
<body>
  <header>
    <div>
      <h2 class="title">视频异常监测</h2>
      <div class="subtitle">《机器视觉与图像处理》课程大作业  王弘昊</div>
    </div>
    <div class="pill">实时分析</div>
  </header>
  <section class="layout">
    <div class="card">
      <h3>上传视频</h3>
      <p>选择本地视频，上传到服务器并使用我们训练的模型进行推理。</p>
      <form action="/upload" method="post" enctype="multipart/form-data">
        <div class="file-row">
          <input class="file-input" type="file" id="videoInput" name="video" accept="video/*" required />
          <label for="videoInput" class="file-button">选择文件</label>
          <span class="file-name" id="fileName">未选择任何文件</span>
        </div>
        <button type="submit">上传并播放</button>
      </form>
    </div>
    <div class="card">
      <h3>视频流地址</h3>
      <p>输入 RTSP/HTTP 地址，利用我们的模型对实时视频流进行推理。</p>
      <form action="/set_stream" method="post">
        <input type="text" name="stream_url" placeholder="rtsp://... 或 http://... 或 0" />
        <button type="submit">开始播放</button>
      </form>
    </div>
    <div class="card">
      <h3>工作方式</h3>
      <p>我们以固定间隔采样 5 帧，构建 4 张差分图并利用机器视觉算法分类正常/异常。由于采样需要一定时间，所以推理结果可能略微滞后，但不会超过2s。</p>
      <p>推理结果会叠加显示在视频画面上。</p>
    </div>
  </section>
  <section class="viewer">
    <h3><span>实时</span> 推理画面</h3>
    <img src="/video_feed" width="800" />
  </section>
  <script>
    const input = document.getElementById("videoInput");
    const nameBox = document.getElementById("fileName");
    if (input && nameBox) {
      input.addEventListener("change", () => {
        if (input.files && input.files.length > 0) {
          nameBox.textContent = input.files[0].name;
        } else {
          nameBox.textContent = "未选择任何文件";
        }
      });
    }
  </script>
</body>
</html>
"""


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

    def set_source(self, source, name):
        with self.lock:
            self.source = source
            self.source_name = name

    def get_source(self):
        with self.lock:
            return self.source, self.source_name


def parse_args():
    parser = argparse.ArgumentParser(description="Web UI for real-time video inference.")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind.")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind.")
    parser.add_argument("--model", default="final_model.pt", help="Path to model checkpoint.")
    parser.add_argument("--height", type=int, default=810, help="Target height for padding.")
    parser.add_argument("--width", type=int, default=892, help="Target width for padding.")
    parser.add_argument("--interval", type=float, default=0.25, help="Frame sampling interval.")
    parser.add_argument("--threshold", type=float, default=0.7, help="Decision threshold.")
    parser.add_argument("--loop", action="store_true", help="Loop video files when finished.")
    return parser.parse_args()


def make_handler(state, model, device, args):
    class Handler(BaseHTTPRequestHandler):
        def _write_html(self, html, status=HTTPStatus.OK):
            data = html.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == "/" or self.path.startswith("/index"):
                return self._write_html(HTML_PAGE)
            if self.path.startswith("/video_feed"):
                return self._stream_video()
            self.send_error(HTTPStatus.NOT_FOUND, "Not Found")

        def do_POST(self):
            if self.path == "/upload":
                return self._handle_upload()
            if self.path == "/set_stream":
                return self._handle_stream()
            self.send_error(HTTPStatus.NOT_FOUND, "Not Found")

        def _handle_upload(self):
            form = cgi.FieldStorage(fp=self.rfile, headers=self.headers, environ={"REQUEST_METHOD": "POST"})
            if "video" not in form:
                return self._write_html("No file uploaded", status=HTTPStatus.BAD_REQUEST)
            file_item = form["video"]
            if not file_item.filename:
                return self._write_html("Empty filename", status=HTTPStatus.BAD_REQUEST)
            uploads_dir = os.path.join(os.getcwd(), "uploads")
            os.makedirs(uploads_dir, exist_ok=True)
            safe_name = os.path.basename(file_item.filename)
            out_path = os.path.join(uploads_dir, safe_name)
            with open(out_path, "wb") as handle:
                handle.write(file_item.file.read())
            state.set_source(out_path, safe_name)
            self.send_response(HTTPStatus.SEE_OTHER)
            self.send_header("Location", "/")
            self.end_headers()

        def _handle_stream(self):
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length).decode("utf-8")
            params = {}
            for part in body.split("&"):
                if "=" in part:
                    key, value = part.split("=", 1)
                    params[key] = value.replace("+", " ")
            stream_url = params.get("stream_url", "").strip()
            if not stream_url:
                return self._write_html("Empty stream url", status=HTTPStatus.BAD_REQUEST)
            if stream_url.isdigit():
                source = int(stream_url)
            else:
                source = stream_url
            state.set_source(source, str(stream_url))
            self.send_response(HTTPStatus.SEE_OTHER)
            self.send_header("Location", "/")
            self.end_headers()

        def _stream_video(self):
            source, _ = state.get_source()
            if source is None:
                return self._write_html("No source selected.", status=HTTPStatus.OK)

            cap = cv2.VideoCapture(source)
            if not cap.isOpened():
                return self._write_html("Failed to open source.", status=HTTPStatus.BAD_REQUEST)

            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()

            fps = cap.get(cv2.CAP_PROP_FPS)
            if not fps or fps <= 1e-3 or np.isnan(fps):
                fps = 30.0
            frame_interval = 1.0 / float(fps)
            sample_interval = max(0.01, float(args.interval))
            frame_buffer = deque(maxlen=5)
            last_sample = 0.0
            last_label = "warming up"
            last_prob = 0.0

            while True:
                loop_start = time.time()
                ok, frame = cap.read()
                if not ok:
                    if args.loop and isinstance(source, str) and os.path.isfile(source):
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    print("stream ended or frame read failed.")
                    break

                now = time.time()
                if now - last_sample >= sample_interval:
                    last_sample = now
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    gray = pad_frame(gray, args.height, args.width)
                    frame_buffer.append(gray)
                    if len(frame_buffer) == 5:
                        diffs = diff_features(list(frame_buffer))
                        x = diffs.astype(np.float32)
                        x = (x - 127.5) / 127.5
                        x = torch.from_numpy(x).unsqueeze(0).to(device)
                        with torch.no_grad():
                            logits = model(x)
                            prob = torch.sigmoid(logits).item()
                        last_prob = prob
                        last_label = "abnormal" if prob >= args.threshold else "normal"

                label_text = f"{last_label} ({last_prob:.2f})"
                cv2.putText(
                    frame,
                    label_text,
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 0, 255) if last_label == "abnormal" else (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
                ok, encoded = cv2.imencode(".jpg", frame)
                if not ok:
                    continue
                payload = encoded.tobytes()
                try:
                    self.wfile.write(b"--frame\r\n")
                    self.wfile.write(b"Content-Type: image/jpeg\r\n\r\n")
                    self.wfile.write(payload)
                    self.wfile.write(b"\r\n")
                except BrokenPipeError:
                    break
                elapsed = time.time() - loop_start
                if elapsed < frame_interval:
                    time.sleep(frame_interval - elapsed)
            cap.release()

        def log_message(self, format, *args):
            return

    return Handler


def main():
    args = parse_args()
    if not os.path.isfile(args.model):
        print(f"error: model not found: {args.model}", file=sys.stderr)
        return 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.model, device)
    state = AppState()
    handler = make_handler(state, model, device, args)
    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(f"serving on http://{args.host}:{args.port}")
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
