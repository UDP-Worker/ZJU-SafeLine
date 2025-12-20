import argparse
import csv
import os
import re
import sys
import time

import cv2
import numpy as np


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def _natural_key(text):
    parts = re.split(r"(\d+)", text)
    key = []
    for part in parts:
        if part.isdigit():
            key.append(int(part))
        else:
            key.append(part.lower())
    return key


def collect_sequences(images_dir):
    sequences = []
    for root, _, files in os.walk(images_dir):
        image_files = []
        for name in files:
            _, ext = os.path.splitext(name)
            if ext.lower() in IMAGE_EXTS:
                image_files.append(name)
        if not image_files:
            continue
        image_files.sort(key=_natural_key)
        abs_files = [os.path.join(root, name) for name in image_files]
        rel_dir = os.path.relpath(root, images_dir)
        sequences.append((rel_dir, abs_files))
    sequences.sort(key=lambda item: _natural_key(item[0]))
    return sequences


def load_labels(path):
    labels = {}
    if not os.path.exists(path):
        return labels
    with open(path, "r", newline="") as handle:
        reader = csv.reader(handle)
        rows = list(reader)
    if not rows:
        return labels
    start = 0
    if rows[0] and rows[0][0].strip().lower() == "path":
        start = 1
    for row in rows[start:]:
        if len(row) < 2:
            continue
        labels[row[0]] = row[1]
    return labels


def save_labels(labels, path):
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["path", "label"])
        for rel_path in sorted(labels.keys(), key=_natural_key):
            writer.writerow([rel_path, labels[rel_path]])
    os.replace(tmp_path, path)


def rel_image_path(abs_path, images_dir):
    return os.path.relpath(abs_path, images_dir)


def find_start(sequences, labels, images_dir):
    for seq_idx, (_, files) in enumerate(sequences):
        for frame_idx, path in enumerate(files):
            rel_path = rel_image_path(path, images_dir)
            if rel_path not in labels:
                return seq_idx, frame_idx
    return len(sequences), 0


def build_panel(image_paths, thumb_height):
    thumbs = []
    for path in image_paths:
        img = cv2.imread(path)
        if img is None:
            thumb = np.zeros((thumb_height, thumb_height, 3), dtype=np.uint8)
            cv2.putText(
                thumb,
                "missing",
                (10, thumb_height // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
        else:
            height, width = img.shape[:2]
            scale = thumb_height / float(height)
            new_width = max(1, int(width * scale))
            thumb = cv2.resize(img, (new_width, thumb_height))
        label = os.path.basename(path)
        cv2.rectangle(
            thumb,
            (0, thumb_height - 26),
            (thumb.shape[1], thumb_height),
            (0, 0, 0),
            -1,
        )
        cv2.putText(
            thumb,
            label,
            (6, thumb_height - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        thumbs.append(thumb)
    if not thumbs:
        return None
    return np.concatenate(thumbs, axis=1)


def render_screen(panel, info_lines):
    width = panel.shape[1]
    info_height = 26 * len(info_lines) + 10
    info = np.zeros((info_height, width, 3), dtype=np.uint8)
    y = 26
    for line in info_lines:
        cv2.putText(
            info,
            line,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        y += 26
    return np.vstack([info, panel])


def parse_args():
    parser = argparse.ArgumentParser(description="Label images with a 5-frame-gap preview.")
    parser.add_argument(
        "--images-dir",
        default="images",
        help="Root directory containing image folders (default: images).",
    )
    parser.add_argument(
        "--labels-file",
        default="labels.csv",
        help="CSV file to store labels (default: labels.csv).",
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=5,
        help="Number of preview images shown at a time (default: 5).",
    )
    parser.add_argument(
        "--gap",
        type=int,
        default=5,
        help="Frame gap between preview images (default: 5).",
    )
    parser.add_argument(
        "--thumb-height",
        type=int,
        default=220,
        help="Preview thumbnail height in pixels (default: 220).",
    )
    parser.add_argument(
        "--auto-interval",
        type=float,
        default=0.2,
        help="Interval (seconds) for auto-normal labeling (default: 0.2).",
    )
    parser.add_argument(
        "--hold-threshold",
        type=float,
        default=0.35,
        help="Seconds to hold left mouse before auto-normal (default: 0.35).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    images_dir = os.path.abspath(args.images_dir)
    if not os.path.isdir(images_dir):
        print(f"error: images dir not found: {images_dir}", file=sys.stderr)
        return 1

    sequences = collect_sequences(images_dir)
    if not sequences:
        print("error: no images found under images dir", file=sys.stderr)
        return 1

    labels = load_labels(args.labels_file)
    total_images = sum(len(files) for _, files in sequences)

    gap = max(1, args.gap)
    num_images = max(1, args.num_images)
    segment_len = gap * (num_images - 1) + 1

    seq_idx, frame_idx = find_start(sequences, labels, images_dir)
    if seq_idx >= len(sequences):
        print("all images already labeled.")
        return 0

    window_name = "Labeler"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    state = {
        "mouse_down": False,
        "mouse_down_time": 0.0,
        "mouse_auto": False,
        "mouse_click": False,
    }

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            state["mouse_down"] = True
            state["mouse_down_time"] = time.time()
            state["mouse_auto"] = False
            state["mouse_click"] = False
        elif event == cv2.EVENT_LBUTTONUP:
            if state["mouse_down"] and not state["mouse_auto"]:
                state["mouse_click"] = True
            state["mouse_down"] = False
            state["mouse_down_time"] = 0.0
            state["mouse_auto"] = False

    cv2.setMouseCallback(window_name, mouse_callback)

    auto_key = False
    last_auto = 0.0

    while True:
        if seq_idx >= len(sequences):
            done = np.zeros((200, 600, 3), dtype=np.uint8)
            cv2.putText(
                done,
                "All images labeled.",
                (30, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow(window_name, done)
            cv2.waitKey(0)
            break

        seq_name, files = sequences[seq_idx]
        segment_end = min(frame_idx + segment_len, len(files))
        segment_files = files[frame_idx:segment_end]

        preview_files = []
        for i in range(num_images):
            idx = frame_idx + i * gap
            if idx < len(files):
                preview_files.append(files[idx])
            else:
                break

        panel = build_panel(preview_files, args.thumb_height)
        if panel is None:
            frame_idx = segment_end
            continue

        labeled_count = len(labels)
        progress = 100.0 * labeled_count / float(total_images)
        auto_state = "ON" if auto_key or state["mouse_auto"] else "OFF"
        info_lines = [
            f"Seq: {seq_name} | Segment: {frame_idx + 1}-{segment_end} ({len(segment_files)} frames)",
            f"Progress: {labeled_count}/{total_images} ({progress:.1f}%) | Auto-normal: {auto_state}",
            "Keys: [N/Space/Enter]=normal  [A]=abnormal  [F]=toggle auto  [Q/Esc]=quit",
            "Mouse: click=normal  hold=auto-normal",
        ]
        screen = render_screen(panel, info_lines)
        cv2.imshow(window_name, screen)

        key = cv2.waitKey(30) & 0xFF

        if state["mouse_down"] and not state["mouse_auto"]:
            if time.time() - state["mouse_down_time"] >= args.hold_threshold:
                state["mouse_auto"] = True
                last_auto = 0.0

        action = None
        if state["mouse_click"]:
            action = "normal"
            state["mouse_click"] = False
        elif key in (ord("q"), ord("Q"), 27):
            break
        elif key in (ord("f"), ord("F")):
            auto_key = not auto_key
            if auto_key:
                last_auto = 0.0
        elif key in (ord("n"), ord("N"), ord(" "), 13, 10):
            action = "normal"
        elif key in (ord("a"), ord("A"), ord("x"), ord("X")):
            action = "abnormal"
        else:
            auto_mode = auto_key or state["mouse_auto"]
            if auto_mode and (time.time() - last_auto) >= args.auto_interval:
                action = "normal"

        if action is not None:
            label_value = "normal" if action == "normal" else "abnormal"
            for path in segment_files:
                rel_path = rel_image_path(path, images_dir)
                labels[rel_path] = label_value
            save_labels(labels, args.labels_file)
            last_auto = time.time()
            frame_idx = segment_end
            if frame_idx >= len(files):
                seq_idx += 1
                frame_idx = 0

    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
