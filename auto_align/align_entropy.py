#!/usr/bin/env python3
"""Iterative LiDAR-camera alignment via helper utilities."""

from __future__ import annotations

import argparse
import random
import shutil
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np

try:  # prefer canonical dimensions but fall back if unavailable
    from frame_segment_customMsg import SNAPSHOT_HEIGHT, SNAPSHOT_WIDTH
except ImportError:  # pragma: no cover
    SNAPSHOT_WIDTH = 640
    SNAPSHOT_HEIGHT = 480

try:
    from depth_edge_generate import depth_gen
    _DEPTH_ARGV_STUB = ["depth_edge_generate.py"]
except ImportError:
    from depth_edge_generate import depth_gen  # type: ignore
    _DEPTH_ARGV_STUB = ["depth_edge_generate.py"]

from function_lidar_gen_chair import (
    CAMERA_PITCH_DEG,
    CAMERA_ROLL_DEG,
    CAMERA_YAW_DEG,
    function_lidar_gen_chair,
)

DEPTH_IMAGE_PATH = Path("depth_sobel_top.png")
LIDAR_EDGE_PATH = Path("scharr_regions_scharr.png")
CANVAS_SHAPE = (SNAPSHOT_HEIGHT, SNAPSHOT_WIDTH)
MIN_OVERLAP_PIXELS = 200


def _load_grayscale(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Expected grayscale image at {path}")
    return image.astype(np.float32)


def _center_on_canvas(image: np.ndarray, canvas_shape: Tuple[int, int]) -> np.ndarray:
    canvas_h, canvas_w = canvas_shape
    img_h, img_w = image.shape[:2]
    copy_h = min(img_h, canvas_h)
    copy_w = min(img_w, canvas_w)

    src_y0 = max(0, (img_h - copy_h) // 2)
    src_x0 = max(0, (img_w - copy_w) // 2)
    dst_y0 = max(0, (canvas_h - copy_h) // 2)
    dst_x0 = max(0, (canvas_w - copy_w) // 2)

    canvas = np.zeros((canvas_h, canvas_w), dtype=np.float32)
    canvas[dst_y0 : dst_y0 + copy_h, dst_x0 : dst_x0 + copy_w] = image[
        src_y0 : src_y0 + copy_h, src_x0 : src_x0 + copy_w
    ]
    return canvas


def _entropy_loss(camera_canvas: np.ndarray, lidar_canvas: np.ndarray) -> float:
    mask = (camera_canvas > 0) & (lidar_canvas > 0)
    overlap = int(mask.sum())
    if overlap < MIN_OVERLAP_PIXELS:
        return float("inf")

    cam_vals = camera_canvas[mask] / 255.0
    lid_vals = lidar_canvas[mask] / 255.0

    cam_norm = (cam_vals - cam_vals.min()) / (cam_vals.ptp() + 1e-6)
    lid_norm = (lid_vals - lid_vals.min()) / (lid_vals.ptp() + 1e-6)

    mse = float(np.mean((cam_norm - lid_norm) ** 2))
    if cam_norm.size < 2:
        corr = 0.0
    else:
        corr_mat = np.corrcoef(cam_norm, lid_norm)
        corr = float(corr_mat[0, 1]) if np.isfinite(corr_mat[0, 1]) else 0.0
    return mse - corr


@contextmanager
def _temporary_argv(fake_argv: list[str]):
    original = list(sys.argv)
    sys.argv = list(fake_argv)
    try:
        yield
    finally:
        sys.argv = original


def _capture_camera_canvas(verbose: bool) -> np.ndarray:
    with _temporary_argv(_DEPTH_ARGV_STUB):
        depth_gen()

    if not DEPTH_IMAGE_PATH.exists():
        raise FileNotFoundError(
            f"depth_gen did not create {DEPTH_IMAGE_PATH}. Check depth helper configuration."
        )

    depth_image = _load_grayscale(DEPTH_IMAGE_PATH)
    canvas = _center_on_canvas(depth_image, CANVAS_SHAPE)
    if verbose:
        cv2.imwrite("camera_canvas.png", np.clip(canvas, 0, 255).astype(np.uint8))
    return canvas


def _acquire_lidar_canvas(
    yaw_deg: float,
    pitch_deg: float,
    roll_deg: float,
    iteration: int,
) -> np.ndarray:
    function_lidar_gen_chair(yaw_deg=yaw_deg, pitch_deg=pitch_deg, roll_deg=roll_deg)

    if not LIDAR_EDGE_PATH.exists():
        raise FileNotFoundError(
            f"function_lidar_gen_chair did not create {LIDAR_EDGE_PATH}."
        )

    lidar_image = _load_grayscale(LIDAR_EDGE_PATH)
    iter_path = Path(f"scharr_regions_scharr_iter_{iteration:03d}.png")
    shutil.copy(LIDAR_EDGE_PATH, iter_path)
    print(f"[align_entropy] iteration {iteration:03d} saved {iter_path}")
    return _center_on_canvas(lidar_image, CANVAS_SHAPE)


def _evaluate_pose(
    camera_canvas: np.ndarray,
    yaw_deg: float,
    pitch_deg: float,
    roll_deg: float,
    iteration: int,
) -> float:
    lidar_canvas = _acquire_lidar_canvas(yaw_deg, pitch_deg, roll_deg, iteration)
    return _entropy_loss(camera_canvas, lidar_canvas)


def _stochastic_alignment(args: argparse.Namespace) -> tuple[float, float, float, float]:
    print("[align_entropy] capturing camera canvas...")
    camera_canvas = _capture_camera_canvas(verbose=not args.quiet)

    best_pose = (args.yaw_init, args.pitch_init, args.roll_init)
    best_loss = _evaluate_pose(camera_canvas, *best_pose, iteration=0)
    step = args.step_init
    no_improve = 0

    print(
        f"[align_entropy] iter 000 loss={best_loss:.6f}, pose=(yaw={best_pose[0]:.2f}, "
        f"pitch={best_pose[1]:.2f}, roll={best_pose[2]:.2f}), step={step:.3f}"
    )
    print("[align_entropy] starting stochastic alignment...")
    for iteration in range(1, args.max_iters + 1):
        random.seed(args.random_seed + iteration)
        step = max(args.min_step, min(args.max_step, step / (1.0 + args.step_decay)))

        candidate = tuple(
            best_pose[idx] + random.uniform(-1.0, 1.0) * step for idx in range(3)
        )

        loss = _evaluate_pose(camera_canvas, *candidate, iteration=iteration)
        improved = loss + args.tolerance < best_loss

        if improved:
            best_pose = candidate
            best_loss = loss
            no_improve = 0
            step = min(args.max_step, step * args.step_increase)
        else:
            no_improve += 1
            step = max(args.min_step, step * args.step_decrease)

        print(
            f"[align_entropy] iter {iteration:03d} loss={loss:.6f} (best={best_loss:.6f}), "
            f"pose=(yaw={candidate[0]:.2f}, pitch={candidate[1]:.2f}, roll={candidate[2]:.2f}), "
            f"step={step:.3f}, improved={improved}"
        )

        if no_improve >= args.patience:
            print(
                f"[align_entropy] early stop after {iteration} iterations "
                f"(no improvement for {args.patience})."
            )
            break

    print(
        f"[align_entropy] best pose yaw={best_pose[0]:.2f}, pitch={best_pose[1]:.2f}, "
        f"roll={best_pose[2]:.2f}, loss={best_loss:.6f}"
    )
    return (*best_pose, best_loss)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Align LiDAR to camera edges via helper generators.",
    )
    parser.add_argument("--yaw-init", type=float, default=CAMERA_YAW_DEG)
    parser.add_argument("--pitch-init", type=float, default=CAMERA_PITCH_DEG)
    parser.add_argument("--roll-init", type=float, default=CAMERA_ROLL_DEG)
    parser.add_argument("--max-iters", type=int, default=200)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--tolerance", type=float, default=1e-3)
    parser.add_argument("--step-init", type=float, default=20)
    parser.add_argument("--min-step", type=float, default=1.0)
    parser.add_argument("--max-step", type=float, default=100)
    parser.add_argument("--step-decay", type=float, default=0)
    parser.add_argument("--step-increase", type=float, default=1.2)
    parser.add_argument("--step-decrease", type=float, default=0.9)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--quiet", action="store_true", help="Disable camera canvas dump")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    _stochastic_alignment(args)


if __name__ == "__main__":
    main()
