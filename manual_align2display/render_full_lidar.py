#!/usr/bin/env python3
"""Render a full-width LiDAR snapshot and highlight the <=3 m mini-snapshot region."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np

from function_lidar_gen_chair import (
    _collect_livox_points,
    _points_to_camera,
)

DEFAULT_STATE_PATH = Path("alignment_states") / "screenshot_state.json"
DEFAULT_OUTPUT = Path("full_lidar_snapshot.png")
FULL_WIDTH = 1280
FULL_HEIGHT = 480
FULL_PADDING = 40
CROP_RANGE_METERS = 3.0


def _load_state(path: Path) -> Tuple[float, float, float, float, int | None, int | None]:
    if not path.exists():
        raise FileNotFoundError(f"State JSON not found at {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    try:
        yaw = float(data["yaw_deg"])
        pitch = float(data["pitch_deg"])
        roll = float(data["roll_deg"])
        lidar_rot = float(data.get("lidar_rotation_deg", 0.0))
        width = data.get("lidar_width_px")
        height = data.get("lidar_height_px")
        width_val = int(width) if width is not None else None
        height_val = int(height) if height is not None else None
        return yaw, pitch, roll, lidar_rot, width_val, height_val
    except KeyError as exc:
        raise KeyError(f"JSON file {path} missing key {exc.args[0]}") from exc


def _spatial_filter_points(points_world: np.ndarray) -> np.ndarray:
    if points_world.size == 0:
        return points_world

    y = points_world[:, 1]
    x = points_world[:, 0]
    mask = ((x + y) >= 0.0) & ((x - y) >= 0.6)
    return points_world[mask]


def _project_points(
    points: np.ndarray,
    width: int,
    height: int,
    padding: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if points.size == 0:
        blank = np.zeros((height, width, 3), dtype=np.uint8)
        return blank, np.zeros((0, 2), dtype=np.int32), np.zeros((0,), dtype=np.float32)

    projected = points[:, [0, 2]]
    min_x, max_x = projected[:, 0].min(), projected[:, 0].max()
    min_z, max_z = projected[:, 1].min(), projected[:, 1].max()
    if np.isclose(max_x - min_x, 0.0):
        max_x = min_x + 1.0
    if np.isclose(max_z - min_z, 0.0):
        max_z = min_z + 1.0
    scale_x = (width - 2 * padding) / (max_x - min_x)
    scale_z = (height - 2 * padding) / (max_z - min_z)

    u = ((projected[:, 0] - min_x) * scale_x + padding).astype(np.int32)
    v = (height - ((projected[:, 1] - min_z) * scale_z + padding)).astype(np.int32)
    u = np.clip(u, 0, width - 1)
    v = np.clip(v, 0, height - 1)

    distances = np.linalg.norm(points, axis=1)
    dist_min, dist_max = distances.min(), distances.max()
    if np.isclose(dist_max - dist_min, 0.0):
        intensity = np.zeros_like(distances)
    else:
        intensity = (distances - dist_min) / (dist_max - dist_min)

    image = np.zeros((height, width), dtype=np.uint8)
    depth_map = np.full((height, width), np.inf, dtype=np.float32)
    for idx in range(points.shape[0]):
        vx = v[idx]
        ux = u[idx]
        d = distances[idx]
        if d < depth_map[vx, ux]:
            depth_map[vx, ux] = d
            image[vx, ux] = int((1.0 - intensity[idx]) * 255)

    image_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    pixel_coords = np.stack([u, v], axis=1)
    return image_bgr, pixel_coords, distances


def _draw_crop_rectangle(
    image: np.ndarray,
    pixel_coords: np.ndarray,
    distances: np.ndarray,
    max_range: float,
) -> None:
    if pixel_coords.size == 0:
        return
    mask = distances <= max_range
    if not np.any(mask):
        return
    crop_pixels = pixel_coords[mask]
    x0 = int(crop_pixels[:, 0].min())
    y0 = int(crop_pixels[:, 1].min())
    x1 = int(crop_pixels[:, 0].max())
    y1 = int(crop_pixels[:, 1].max())
    cv2.rectangle(image, (x0, y0), (x1, y1), (0, 255, 0), 2)
    cv2.putText(
        image,
        "<= 3 m snapshot region",
        (x0 + 10, max(y0 - 10, 20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect ~10s of LiDAR data, render full projection, and highlight mini snapshot"
    )
    parser.add_argument("--state-path", type=Path, default=DEFAULT_STATE_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--width", type=int, default=None, help="LiDAR image width in pixels")
    parser.add_argument("--height", type=int, default=None, help="LiDAR image height in pixels")
    parser.add_argument("--padding", type=int, default=FULL_PADDING)
    parser.add_argument("--crop-range", type=float, default=CROP_RANGE_METERS)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    (
        yaw_deg,
        pitch_deg,
        roll_deg,
        lidar_rotation,
        state_width,
        state_height,
    ) = _load_state(args.state_path)
    width = int(args.width) if args.width is not None else (state_width or FULL_WIDTH)
    height = int(args.height) if args.height is not None else (state_height or FULL_HEIGHT)
    print("[render_full_lidar] Capturing ~10 seconds of Livox data...")
    points_world = _collect_livox_points().astype(np.float32)
    print(f"[render_full_lidar] Captured {points_world.shape[0]} points")
    filtered_points = _spatial_filter_points(points_world)
    if filtered_points.size == 0:
        raise RuntimeError("All collected points were filtered out; nothing to render.")

    camera_points = _points_to_camera(
        filtered_points.astype(np.float32, copy=False),
        yaw_deg,
        pitch_deg,
        roll_deg,
    )
    if camera_points.size == 0:
        raise RuntimeError("Projection to camera frame produced no points.")

    image, pixel_coords, distances = _project_points(
        camera_points,
        width,
        height,
        args.padding,
    )
    _draw_crop_rectangle(image, pixel_coords, distances, args.crop_range)

    if not np.isclose(lidar_rotation, 0.0):
        center = (width / 2.0, height / 2.0)
        matrix = cv2.getRotationMatrix2D(center, lidar_rotation, 1.0)
        image = cv2.warpAffine(
            image,
            matrix,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )

    cv2.putText(
        image,
        f"Yaw {yaw_deg:+.1f}  Pitch {pitch_deg:+.1f}  Roll {roll_deg:+.1f}  LiDAR Θ {lidar_rotation:+.1f}",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.imwrite(str(args.output), image)
    print(f"[render_full_lidar] Saved {args.output}")


if __name__ == "__main__":
    main()
