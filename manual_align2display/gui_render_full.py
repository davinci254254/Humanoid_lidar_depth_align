#!/usr/bin/env python3
"""Interactive GUI for the full-width LiDAR projection over a depth canvas."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np

from function_lidar_gen_chair import _collect_livox_points, _points_to_camera
from render_full_lidar import (
    FULL_HEIGHT,
    FULL_PADDING,
    FULL_WIDTH,
    _load_state,
    _project_points,
    _spatial_filter_points,
)

WINDOW_NAME = "Full LiDAR Renderer"
DEFAULT_DEPTH_IMAGE = Path("depth_sobel_top.png")
DEFAULT_CANVAS_WIDTH = 1920
DEFAULT_CANVAS_HEIGHT = 1080
MIN_LIDAR_WIDTH = 320
MAX_LIDAR_WIDTH = 3840
MIN_LIDAR_HEIGHT = 240
MAX_LIDAR_HEIGHT = 2000

TRACKBARS: Dict[str, Tuple[int, int, int]] = {
    "Yaw (deg)": (-360, 360, 0),
    "Pitch (deg)": (-360, 360, 0),
    "Roll (deg)": (-360, 360, 0),
    "LiDAR rotation (deg)": (-180, 180, 0),
    "LiDAR X shift": (-960, 960, 0),
    "LiDAR Y shift": (-540, 540, 0),
    "LiDAR width (px)": (MIN_LIDAR_WIDTH, MAX_LIDAR_WIDTH, FULL_WIDTH),
    "LiDAR height (px)": (MIN_LIDAR_HEIGHT, MAX_LIDAR_HEIGHT, FULL_HEIGHT),
    "Depth X shift": (-960, 960, 0),
    "Depth Y shift": (-540, 540, 0),
    "Depth rotation (deg)": (-180, 180, 0),
    "Depth zoom (%)": (25, 300, 100),
    "Depth alpha": (0, 100, 60),
    "LiDAR alpha": (0, 100, 70),
}


def _load_points(points_path: Path | None) -> np.ndarray:
    if points_path and points_path.exists():
        cloud = np.load(points_path).astype(np.float32)
        print(f"[gui_render_full] Loaded cached points from {points_path} -> {cloud.shape}")
        return cloud

    print("[gui_render_full] Capturing ~10 seconds of Livox data...")
    cloud = _collect_livox_points().astype(np.float32)
    print(f"[gui_render_full] Captured {cloud.shape[0]} raw points")
    if points_path:
        try:
            points_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(points_path, cloud)
            print(f"[gui_render_full] Saved captured points to {points_path}")
        except Exception as exc:  # pragma: no cover - cache optional
            print(f"[gui_render_full] Warning: failed to cache points: {exc}")
    return cloud


def _load_depth_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Depth image not found at {path}")
    return image.astype(np.float32)


def _apply_zoom(image: np.ndarray, zoom_percent: float) -> np.ndarray:
    scale = max(zoom_percent / 100.0, 0.01)
    if np.isclose(scale, 1.0):
        return image
    h = max(1, int(round(image.shape[0] * scale)))
    w = max(1, int(round(image.shape[1] * scale)))
    return cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)


def _rotate_image(image: np.ndarray, angle_deg: float) -> np.ndarray:
    if np.isclose(angle_deg, 0.0):
        return image
    h, w = image.shape[:2]
    center = (w / 2.0, h / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    cos_val = abs(matrix[0, 0])
    sin_val = abs(matrix[0, 1])
    new_w = int(round((h * sin_val) + (w * cos_val)))
    new_h = int(round((h * cos_val) + (w * sin_val)))
    matrix[0, 2] += (new_w / 2.0) - center[0]
    matrix[1, 2] += (new_h / 2.0) - center[1]
    border_value = 0 if image.ndim == 2 else (0, 0, 0)
    return cv2.warpAffine(
        image,
        matrix,
        (max(1, new_w), max(1, new_h)),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_value,
    )


def _place_layer_on_canvas(
    image: np.ndarray,
    canvas_shape: Tuple[int, int],
    shift_x: float = 0.0,
    shift_y: float = 0.0,
) -> np.ndarray:
    canvas_h, canvas_w = canvas_shape
    if image.ndim == 2:
        canvas = np.zeros((canvas_h, canvas_w), dtype=image.dtype)
    else:
        canvas = np.zeros((canvas_h, canvas_w, image.shape[2]), dtype=image.dtype)

    h, w = image.shape[:2]
    center_y = canvas_h // 2 + int(round(shift_y))
    center_x = canvas_w // 2 + int(round(shift_x))
    y0 = center_y - h // 2
    x0 = center_x - w // 2
    y1 = y0 + h
    x1 = x0 + w

    src_y0 = max(0, -y0)
    src_x0 = max(0, -x0)
    src_y1 = min(h, canvas_h - y0)
    src_x1 = min(w, canvas_w - x0)
    if src_y0 >= src_y1 or src_x0 >= src_x1:
        return canvas

    dst_y0 = max(y0, 0)
    dst_x0 = max(x0, 0)
    dst_y1 = dst_y0 + (src_y1 - src_y0)
    dst_x1 = dst_x0 + (src_x1 - src_x0)

    canvas_slice = (slice(dst_y0, dst_y1), slice(dst_x0, dst_x1))
    image_slice = (slice(src_y0, src_y1), slice(src_x0, src_x1))
    canvas[canvas_slice] = image[image_slice]
    return canvas


def _prepare_depth_layer(
    depth_img: np.ndarray,
    canvas_shape: Tuple[int, int],
    params: Dict[str, float],
) -> np.ndarray:
    zoomed = _apply_zoom(depth_img, params["depth_zoom"])
    rotated = _rotate_image(zoomed, params["depth_rotation"])
    return _place_layer_on_canvas(
        rotated,
        canvas_shape,
        params["depth_shift_x"],
        params["depth_shift_y"],
    )


def _render_lidar_layer(
    points_world: np.ndarray,
    params: Dict[str, float],
    lidar_width: int,
    lidar_height: int,
    padding: int,
    canvas_shape: Tuple[int, int],
    shift_x: float,
    shift_y: float,
) -> tuple[np.ndarray, bool]:
    if points_world.size == 0:
        canvas = np.zeros((canvas_shape[0], canvas_shape[1], 3), dtype=np.uint8)
        return canvas, False

    camera_points = _points_to_camera(
        points_world,
        params["yaw"],
        params["pitch"],
        params["roll"],
    )
    if camera_points.size == 0:
        canvas = np.zeros((canvas_shape[0], canvas_shape[1], 3), dtype=np.uint8)
        return canvas, False

    camera_points = camera_points.copy()
    camera_points[:, 0] *= -1.0

    image, _, _ = _project_points(
        camera_points.astype(np.float32, copy=False),
        lidar_width,
        lidar_height,
        padding,
    )
    rotated = _rotate_image(image, params["lidar_rotation"])
    placed = _place_layer_on_canvas(rotated, canvas_shape, shift_x, shift_y)
    return placed.astype(np.uint8), True


def _create_trackbars() -> Dict[str, int]:
    offsets: Dict[str, int] = {}
    for name, (min_v, max_v, init) in TRACKBARS.items():
        if max_v <= min_v:
            raise ValueError(f"Trackbar '{name}' has invalid range")
        max_raw = int(max_v - min_v)
        start = int(np.clip(init, min_v, max_v) - min_v)
        cv2.createTrackbar(name, WINDOW_NAME, start, max_raw, lambda _value: None)
        offsets[name] = min_v
    return offsets


def _apply_state_to_trackbars(state: Dict[str, float], offsets: Dict[str, int]) -> None:
    mapping = {
        "Yaw (deg)": state.get("yaw_deg"),
        "Pitch (deg)": state.get("pitch_deg"),
        "Roll (deg)": state.get("roll_deg"),
        "LiDAR rotation (deg)": state.get("lidar_rotation_deg"),
        "LiDAR X shift": state.get("lidar_x_shift_px"),
        "LiDAR Y shift": state.get("lidar_y_shift_px"),
        "LiDAR width (px)": state.get("lidar_width_px"),
        "LiDAR height (px)": state.get("lidar_height_px"),
        "Depth X shift": state.get("depth_x_shift_px", state.get("x_shift_px")),
        "Depth Y shift": state.get("depth_y_shift_px", state.get("y_shift_px")),
        "Depth rotation (deg)": state.get("depth_rotation_deg", state.get("roll_deg_depth")),
        "Depth zoom (%)": state.get("depth_zoom_percent", state.get("lidar_zoom_percent")),
        "Depth alpha": state.get("depth_alpha"),
        "LiDAR alpha": state.get("lidar_alpha"),
    }
    for track, value in mapping.items():
        if value is None or track not in TRACKBARS:
            continue
        min_v, max_v, _ = TRACKBARS[track]
        clamped = float(np.clip(value, min_v, max_v))
        offset = offsets[track]
        max_raw = int(max_v - min_v)
        position = int(np.clip(round(clamped - offset), 0, max_raw))
        cv2.setTrackbarPos(track, WINDOW_NAME, position)


def _read_params(offsets: Dict[str, int]) -> Dict[str, float]:
    raw_values: Dict[str, float] = {}
    for name in TRACKBARS:
        raw = cv2.getTrackbarPos(name, WINDOW_NAME)
        raw_values[name] = float(raw + offsets[name])
    return {
        "yaw": raw_values["Yaw (deg)"],
        "pitch": raw_values["Pitch (deg)"],
        "roll": raw_values["Roll (deg)"],
        "lidar_rotation": raw_values["LiDAR rotation (deg)"],
        "lidar_shift_x": raw_values["LiDAR X shift"],
        "lidar_shift_y": raw_values["LiDAR Y shift"],
        "lidar_width": raw_values["LiDAR width (px)"],
        "lidar_height": raw_values["LiDAR height (px)"],
        "depth_shift_x": raw_values["Depth X shift"],
        "depth_shift_y": raw_values["Depth Y shift"],
        "depth_rotation": raw_values["Depth rotation (deg)"],
        "depth_zoom": raw_values["Depth zoom (%)"],
        "depth_alpha": raw_values["Depth alpha"],
        "lidar_alpha": raw_values["LiDAR alpha"],
    }


def _save_snapshot(frame: np.ndarray, output_dir: Path, timestamp: str | None = None) -> tuple[Path, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = timestamp or time.strftime("%Y%m%d_%H%M%S")
    path = output_dir / f"full_lidar_{ts}.png"
    cv2.imwrite(str(path), frame)
    print(f"[gui_render_full] Saved snapshot to {path}")
    return path, ts


def _save_state(params: Dict[str, float], state_dir: Path, timestamp: str | None = None) -> Path:
    state_dir.mkdir(parents=True, exist_ok=True)
    ts = timestamp or time.strftime("%Y%m%d_%H%M%S")
    payload = {
        "timestamp": ts,
        "yaw_deg": params["yaw"],
        "pitch_deg": params["pitch"],
        "roll_deg": params["roll"],
        "lidar_rotation_deg": params["lidar_rotation"],
        "lidar_x_shift_px": params["lidar_shift_x"],
        "lidar_y_shift_px": params["lidar_shift_y"],
        "lidar_width_px": params["lidar_width"],
        "lidar_height_px": params["lidar_height"],
        "depth_x_shift_px": params["depth_shift_x"],
        "depth_y_shift_px": params["depth_shift_y"],
        "depth_rotation_deg": params["depth_rotation"],
        "depth_zoom_percent": params["depth_zoom"],
        "depth_alpha": params["depth_alpha"],
        "lidar_alpha": params["lidar_alpha"],
    }
    path = state_dir / f"full_lidar_state_{ts}.json"
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(f"[gui_render_full] Saved slider state to {path}")
    return path


def _maybe_load_state(path: Path) -> Dict[str, float] | None:
    if not path:
        return None
    if not path.exists():
        print(f"[gui_render_full] State file not found at {path}, using defaults.")
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    try:
        (
            yaw_deg,
            pitch_deg,
            roll_deg,
            lidar_rot,
            lidar_width,
            lidar_height,
        ) = _load_state(path)
        state = {
            "yaw_deg": yaw_deg,
            "pitch_deg": pitch_deg,
            "roll_deg": roll_deg,
            "lidar_rotation_deg": lidar_rot,
        }
        if lidar_width is not None:
            state["lidar_width_px"] = lidar_width
        if lidar_height is not None:
            state["lidar_height_px"] = lidar_height
        if "lidar_x_shift_px" in data:
            state["lidar_x_shift_px"] = data["lidar_x_shift_px"]
        if "lidar_y_shift_px" in data:
            state["lidar_y_shift_px"] = data["lidar_y_shift_px"]
        if lidar_width is not None:
            state["lidar_width_px"] = lidar_width
        if lidar_height is not None:
            state["lidar_height_px"] = lidar_height
        return state
    except Exception as exc:
        print(f"[gui_render_full] Failed to parse state file {path}: {exc}")
    return None


def _compose_frame(
    depth_img: np.ndarray,
    points_world: np.ndarray,
    params: Dict[str, float],
    canvas_width: int,
    canvas_height: int,
    padding: int,
) -> np.ndarray:
    canvas_shape = (canvas_height, canvas_width)
    depth_layer = _prepare_depth_layer(depth_img, canvas_shape, params)
    lidar_layer, has_points = _render_lidar_layer(
        points_world,
        params,
        int(max(1, round(params["lidar_width"]))),
        int(max(1, round(params["lidar_height"]))),
        padding,
        canvas_shape,
        params["lidar_shift_x"],
        params["lidar_shift_y"],
    )

    depth_u8 = np.clip(depth_layer, 0, 255).astype(np.uint8)
    lidar_u8 = np.clip(lidar_layer, 0, 255).astype(np.uint8)
    depth_bgr = cv2.cvtColor(depth_u8, cv2.COLOR_GRAY2BGR)

    depth_weight = params["depth_alpha"]
    lidar_weight = params["lidar_alpha"]
    if depth_weight == 0 and lidar_weight == 0:
        depth_weight = 1.0
    weight_sum = depth_weight + lidar_weight
    depth_scale = depth_weight / weight_sum
    lidar_scale = lidar_weight / weight_sum
    composite = cv2.addWeighted(
        depth_bgr.astype(np.float32),
        depth_scale,
        lidar_u8.astype(np.float32),
        lidar_scale,
        0.0,
    )
    composite = np.clip(composite, 0, 255).astype(np.uint8)

    info = (
        f"Yaw {params['yaw']:+.1f}  Pitch {params['pitch']:+.1f}  Roll {params['roll']:+.1f}  "
        f"Depth X {params['depth_shift_x']:+.0f}px  Depth Y {params['depth_shift_y']:+.0f}px  "
        f"LiDAR X {params['lidar_shift_x']:+.0f}px  LiDAR Y {params['lidar_shift_y']:+.0f}px"
    )
    cv2.putText(
        composite,
        info,
        (30, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    footer = "Press 's' to save PNG+JSON, 'q'/Esc to exit"
    cv2.putText(
        composite,
        footer,
        (30, canvas_height - 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (200, 200, 200),
        2,
        cv2.LINE_AA,
    )
    if not has_points:
        cv2.putText(
            composite,
            "No LiDAR points after projection",
            (30, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
    return composite


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interactive viewer for full-width LiDAR overlaid on depth"
    )
    parser.add_argument("--depth-path", type=Path, default=DEFAULT_DEPTH_IMAGE)
    parser.add_argument("--state-path", type=Path, default=Path("alignment_states") / "screenshot_state.json")
    parser.add_argument("--state-dir", type=Path, default=Path("alignment_states"))
    parser.add_argument("--output-dir", type=Path, default=Path("alignment_views"))
    parser.add_argument("--points-path", type=Path, default=None, help="Optional .npy cache for world points")
    parser.add_argument("--canvas-width", type=int, default=DEFAULT_CANVAS_WIDTH)
    parser.add_argument("--canvas-height", type=int, default=DEFAULT_CANVAS_HEIGHT)
    parser.add_argument("--lidar-width", type=int, default=FULL_WIDTH)
    parser.add_argument("--lidar-height", type=int, default=FULL_HEIGHT)
    parser.add_argument("--padding", type=int, default=FULL_PADDING)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    depth_img = _load_depth_image(args.depth_path)

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, args.canvas_width, args.canvas_height)
    offsets = _create_trackbars()

    def _set_initial_trackbar(name: str, value: float) -> None:
        if name not in TRACKBARS:
            return
        min_v, max_v, _ = TRACKBARS[name]
        clamped = float(np.clip(value, min_v, max_v))
        offset = offsets[name]
        max_raw = int(max_v - min_v)
        position = int(np.clip(round(clamped - offset), 0, max_raw))
        cv2.setTrackbarPos(name, WINDOW_NAME, position)

    _set_initial_trackbar("LiDAR width (px)", args.lidar_width)
    _set_initial_trackbar("LiDAR height (px)", args.lidar_height)

    state_values = _maybe_load_state(args.state_path)
    if state_values:
        _apply_state_to_trackbars(state_values, offsets)

    world_points = _spatial_filter_points(_load_points(args.points_path))
    if world_points.size == 0:
        raise RuntimeError("All collected points were filtered out; nothing to display.")

    print("[gui_render_full] Controls: ESC/q exit, 's' save PNG+JSON.")
    last_params: Dict[str, float] | None = None
    while True:
        params = _read_params(offsets)
        last_params = params
        frame = _compose_frame(
            depth_img,
            world_points,
            params,
            args.canvas_width,
            args.canvas_height,
            args.padding,
        )
        cv2.imshow(WINDOW_NAME, frame)
        key = cv2.waitKey(16) & 0xFF
        if key in (27, ord("q")):
            break
        if key == ord("s") and last_params is not None:
            _, timestamp = _save_snapshot(frame, args.output_dir)
            _save_state(last_params, args.state_dir, timestamp)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
