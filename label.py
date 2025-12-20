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
    header = [cell.strip().lower() for cell in rows[0]]
    if header and header[0] in {"group_id", "path_1", "path"}:
        start = 1
        if header[0] == "path" and "group_id" not in header and "path_1" not in header:
            print(
                "warn: labels file looks like per-image format; "
                "please re-label with group labels.",
                file=sys.stderr,
            )
            return labels
    for row in rows[start:]:
        if len(row) < 2:
            continue
        group_id = row[0]
        label = row[1]
        paths = []
        if len(row) >= 7:
            paths = [cell for cell in row[2:7] if cell]
        labels[group_id] = {"label": label, "paths": paths}
    return labels


def save_labels(labels, path, order=None, group_index=None):
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["group_id", "label", "path_1", "path_2", "path_3", "path_4", "path_5"])
        if order:
            group_ids = [group_id for group_id in order if group_id in labels]
            extra = [group_id for group_id in labels.keys() if group_id not in order]
            group_ids.extend(sorted(extra, key=_natural_key))
        else:
            group_ids = sorted(labels.keys(), key=_natural_key)
        for group_id in group_ids:
            entry = labels[group_id]
            row = [group_id, entry["label"]]
            paths = list(entry.get("paths") or [])
            if group_index and not paths:
                paths = list(group_index.get(group_id, []))
            if len(paths) < 5:
                paths.extend([""] * (5 - len(paths)))
            row.extend(paths[:5])
            writer.writerow(row)
    os.replace(tmp_path, path)


def rel_image_path(abs_path, images_dir):
    return os.path.relpath(abs_path, images_dir)


def build_group(files, start_idx, gap, num_images):
    max_offset = gap * (num_images - 1)
    if start_idx + max_offset >= len(files):
        return None
    group = []
    for i in range(num_images):
        idx = start_idx + i * gap
        group.append(files[idx])
    return group


def build_groups(sequences, images_dir, gap, num_images, step):
    groups = []
    group_index = {}
    for seq_idx, (seq_name, files) in enumerate(sequences):
        max_offset = gap * (num_images - 1)
        max_start = len(files) - max_offset
        for start_idx in range(0, max_start, step):
            group_files = build_group(files, start_idx, gap, num_images)
            if not group_files:
                break
            rel_paths = [rel_image_path(path, images_dir) for path in group_files]
            group_id = rel_paths[0]
            group_index[group_id] = rel_paths
            groups.append(
                {
                    "seq_idx": seq_idx,
                    "seq_name": seq_name,
                    "start_idx": start_idx,
                    "files": group_files,
                    "group_id": group_id,
                    "rel_paths": rel_paths,
                }
            )
    return groups, group_index


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
        "--step",
        type=int,
        default=1,
        help="Step between group starts (default: 1).",
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

    gap = max(1, args.gap)
    num_images = max(1, args.num_images)
    step = max(1, args.step)

    groups, group_index = build_groups(sequences, images_dir, gap, num_images, step)
    group_order = [group["group_id"] for group in groups]
    total_groups = len(groups)
    if total_groups == 0:
        print("error: no complete groups found for labeling.", file=sys.stderr)
        return 1

    labels = load_labels(args.labels_file)
    if labels:
        kept = {group_id: entry for group_id, entry in labels.items() if group_id in group_index}
        if len(kept) != len(labels):
            print("warn: dropped labels that do not match current grouping.", file=sys.stderr)
        labels = kept

    first_unlabeled = next(
        (idx for idx, group in enumerate(groups) if group["group_id"] not in labels),
        None,
    )
    group_pos = first_unlabeled if first_unlabeled is not None else 0

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
        if group_pos < 0:
            group_pos = 0
        if group_pos >= total_groups:
            group_pos = total_groups - 1

        group = groups[group_pos]
        seq_name = group["seq_name"]
        group_files = group["files"]
        group_id = group["group_id"]
        preview_files = list(group_files)

        panel = build_panel(preview_files, args.thumb_height)
        if panel is None:
            group_pos += 1
            continue

        labeled_count = len(labels)
        progress = 100.0 * labeled_count / float(total_groups)
        auto_state = "ON" if auto_key or state["mouse_auto"] else "OFF"
        frame_indices = [group["start_idx"] + i * gap + 1 for i in range(len(group_files))]
        index_text = ",".join(str(idx) for idx in frame_indices)
        current_label = labels.get(group_id, {}).get("label", "none")
        all_labeled = "YES" if labeled_count == total_groups else "NO"
        info_lines = [
            f"Seq: {seq_name} | Group start: {group['start_idx'] + 1} | Frames: {index_text}",
            f"Progress: {labeled_count}/{total_groups} ({progress:.1f}%) | All labeled: {all_labeled}",
            f"Label: {current_label} | Auto-normal: {auto_state}",
            "Keys: [N/Space/Enter]=normal  [A]=abnormal  [F]=toggle auto  [Q/Esc]=quit",
            "Nav: [B/Left]=prev  [V/Right]=next | Mouse: click=normal  hold=auto-normal",
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
        elif key in (ord("b"), ord("B"), ord("["), ord(","), 81):
            group_pos -= 1
        elif key in (ord("v"), ord("V"), ord("]"), ord("."), 83):
            group_pos += 1
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
            labels[group_id] = {"label": label_value, "paths": group["rel_paths"]}
            save_labels(labels, args.labels_file, order=group_order, group_index=group_index)
            last_auto = time.time()
            if group_pos < total_groups - 1:
                group_pos += 1
            else:
                auto_key = False
                state["mouse_auto"] = False

    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
