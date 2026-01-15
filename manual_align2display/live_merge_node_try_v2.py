#!/usr/bin/env python3
"""ROS 2 node that fuses depth images with LiDAR projections in real time."""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, PointCloud2

try:  # Livox publishes livox_ros_driver2/CustomMsg by default
    from livox_ros_driver2.msg import CustomMsg  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    CustomMsg = None

from function_lidar_gen_chair import (
    _points_to_camera,
    livox_custom_msg_to_pointcloud2,
    pc2_read_points,
)
from render_full_lidar import (
    FULL_PADDING,
    FULL_WIDTH,
    FULL_HEIGHT,
    _project_points,
    _spatial_filter_points,
)

WINDOW_WIDTH = 1920
WINDOW_HEIGHT = 1080
MIN_LIDAR_WIDTH = 320
MAX_LIDAR_WIDTH = 3840
MIN_LIDAR_HEIGHT = 240
MAX_LIDAR_HEIGHT = 2000


def _depth_to_uint8_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 3 and image.shape[2] > 1:
        gray = cv2.cvtColor(image.astype(np.float32, copy=False), cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    orig_dtype = gray.dtype
    gray = gray.astype(np.float32, copy=False)
    if np.issubdtype(orig_dtype, np.floating):
        mask = np.isfinite(gray)
    else:
        mask = gray > 0

    # Match depth_edge_generate.py: crop off unreliable top 5% rows and left 5% cols.
    h, w = gray.shape[:2]
    cut_top = int(0.05 * h)
    cut_left = int(0.05 * w)
    if cut_top > 0 or cut_left > 0:
        gray = gray[cut_top:, cut_left:]
        mask = mask[cut_top:, cut_left:]

    normalized = np.zeros_like(gray, dtype=np.float32)
    if np.any(mask):
        valid = gray[mask]
        vmin = float(np.min(valid))
        vmax = float(np.max(valid))
        if np.isclose(vmax - vmin, 0.0):
            normalized[mask] = 0.0
        else:
            normalized[mask] = (gray[mask] - vmin) / (vmax - vmin)

    return np.clip(normalized * 255.0, 0, 255).astype(np.uint8)


def _load_alignment_state(path: Path) -> Dict[str, float]:
    if not path.exists():
        raise FileNotFoundError(f"State JSON not found at {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"State file {path} does not contain a JSON object")
    return data


def _state_to_params(state: Dict[str, float]) -> Dict[str, float]:
    return {
        "yaw": float(state.get("yaw_deg", 0.0)),
        "pitch": float(state.get("pitch_deg", 0.0)),
        "roll": float(state.get("roll_deg", 0.0)),
        "lidar_rotation": float(state.get("lidar_rotation_deg", 0.0)),
        "lidar_shift_x": float(state.get("lidar_x_shift_px", 0.0)),
        "lidar_shift_y": float(state.get("lidar_y_shift_px", 0.0)),
        "depth_shift_x": float(state.get("depth_x_shift_px", state.get("x_shift_px", 0.0))),
        "depth_shift_y": float(state.get("depth_y_shift_px", state.get("y_shift_px", 0.0))),
        "depth_rotation": float(state.get("depth_rotation_deg", state.get("roll_deg_depth", 0.0))),
        "depth_zoom": float(state.get("depth_zoom_percent", state.get("lidar_zoom_percent", 100.0))),
    }


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


class DepthLidarFusionNode(Node):
    def __init__(self) -> None:
        super().__init__("depth_lidar_fusion")
        self.declare_parameter("depth_topic", "/camera/depth/image_rect_raw")
        self.declare_parameter("lidar_topic", "/livox/points")
        self.declare_parameter("lidar_msg_type", "auto")  # auto | pointcloud2 | custom
        self.declare_parameter("state_path", str(Path("alignment_states") / "screenshot_state.json"))
        self.declare_parameter("output_topic", "/merged/depth_image")
        self.declare_parameter("lidar_padding", FULL_PADDING)
        self.declare_parameter("collect_duration_sec", 0.5)
        self.declare_parameter("hold_duration_sec", 2.0)
        self.declare_parameter("lidar_snapshot_dir", str(Path("lidar_snapshots")))

        depth_topic = self.get_parameter("depth_topic").value
        lidar_topic = self.get_parameter("lidar_topic").value
        state_path = Path(self.get_parameter("state_path").value)
        output_topic = self.get_parameter("output_topic").value
        self.padding = int(self.get_parameter("lidar_padding").value)
        self.collect_duration = max(float(self.get_parameter("collect_duration_sec").value), 0.05)
        self.hold_duration = max(float(self.get_parameter("hold_duration_sec").value), 0.0)
        snapshot_dir = Path(self.get_parameter("lidar_snapshot_dir").value)
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        self.snapshot_dir = snapshot_dir

        state = _load_alignment_state(state_path)
        self.params = _state_to_params(state)
        self.canvas_shape = (WINDOW_HEIGHT, WINDOW_WIDTH)
        lidar_width = int(state.get("lidar_width_px", FULL_WIDTH))
        lidar_height = int(state.get("lidar_height_px", FULL_HEIGHT))
        self.lidar_width = int(np.clip(lidar_width, MIN_LIDAR_WIDTH, MAX_LIDAR_WIDTH))
        self.lidar_height = int(np.clip(lidar_height, MIN_LIDAR_HEIGHT, MAX_LIDAR_HEIGHT))

        self.bridge = CvBridge()
        self.publisher = self.create_publisher(Image, output_topic, 10)
        self.depth_sub = self.create_subscription(
            Image, depth_topic, self._depth_callback, qos_profile_sensor_data
        )

        lidar_msg_type = str(self.get_parameter("lidar_msg_type").value).strip().lower()
        if lidar_msg_type not in {"auto", "pointcloud2", "custom"}:
            self.get_logger().warning(
                f"Invalid lidar_msg_type='{lidar_msg_type}', using 'auto'"
            )
            lidar_msg_type = "auto"

        if lidar_msg_type == "auto":
            if CustomMsg is not None and (lidar_topic.endswith("/lidar") or "livox/lidar" in lidar_topic):
                lidar_msg_type = "custom"
            else:
                lidar_msg_type = "pointcloud2"

        if lidar_msg_type == "custom":
            if CustomMsg is None:
                self.get_logger().error(
                    "lidar_msg_type=custom requested but livox_ros_driver2 is not available; falling back to PointCloud2"
                )
                lidar_msg_type = "pointcloud2"
            else:
                self.lidar_sub = self.create_subscription(
                    CustomMsg, lidar_topic, self._lidar_custom_callback, qos_profile_sensor_data
                )
                self.get_logger().info(f"Subscribed LiDAR as CustomMsg on {lidar_topic}")

        if lidar_msg_type == "pointcloud2":
            self.lidar_sub = self.create_subscription(
                PointCloud2, lidar_topic, self._lidar_pc2_callback, qos_profile_sensor_data
            )
            self.get_logger().info(f"Subscribed LiDAR as PointCloud2 on {lidar_topic}")

        self.last_depth_image: np.ndarray | None = None
        self.last_depth_header = None
        self._last_cropped_depth_height: int = 0
        self._lidar_accumulator: list[np.ndarray] = []
        self._lidar_layer = np.zeros((WINDOW_HEIGHT, WINDOW_WIDTH, 3), dtype=np.uint8)
        self._lidar_has_points = False

        self._cycle_state = "collect"  # 'collect' | 'hold'
        self._cycle_deadline = time.monotonic() + self.collect_duration
        self._cycle_timer = self.create_timer(0.05, self._cycle_tick)

        self.get_logger().info(
            f"Depth-LiDAR fusion node is running. Publishing merged frames to {output_topic}"
        )
        self.get_logger().info(
            f"LiDAR cycle: collect {self.collect_duration:.2f}s, hold {self.hold_duration:.2f}s"
        )

    def _depth_callback(self, depth_msg: Image) -> None:
        try:
            depth_np = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
        except Exception as exc:  # pragma: no cover - conversion errors
            self.get_logger().error(f"Failed to convert depth image: {exc}")
            return

        self.last_depth_image = _depth_to_uint8_gray(depth_np)
        self._last_cropped_depth_height = int(self.last_depth_image.shape[0])
        self.last_depth_header = depth_msg.header
        self._publish_frame()

    def _lidar_pc2_callback(self, lidar_msg: PointCloud2) -> None:
        if self._cycle_state != "collect":
            return
        points = self._pointcloud_to_numpy(lidar_msg)
        if points.size == 0:
            return
        self._lidar_accumulator.append(points)

    def _lidar_custom_callback(self, lidar_msg) -> None:
        if self._cycle_state != "collect":
            return
        try:
            cloud_msg = livox_custom_msg_to_pointcloud2(lidar_msg)
        except Exception as exc:  # pragma: no cover - conversion errors
            self.get_logger().warning(f"Failed to convert CustomMsg -> PointCloud2: {exc}")
            return
        points = self._pointcloud_to_numpy(cloud_msg)
        if points.size == 0:
            return
        self._lidar_accumulator.append(points)

    def _cycle_tick(self) -> None:
        now = time.monotonic()
        if now < self._cycle_deadline:
            return

        if self._cycle_state == "collect":
            self._process_lidar_batch()
            self._cycle_state = "hold"
            self._cycle_deadline = now + self.hold_duration
            return

        self._lidar_accumulator = []
        self._cycle_state = "collect"
        self._cycle_deadline = now + self.collect_duration

    def _process_lidar_batch(self) -> None:
        if not self._lidar_accumulator:
            self._lidar_layer = np.zeros_like(self._lidar_layer)
            self._lidar_has_points = False
            self.get_logger().warning("LiDAR cycle: collected 0 points")
            return

        self.get_logger().info(
            f"_last_cropped_depth_height={self._last_cropped_depth_height} depth_shift_y={self.params['depth_shift_y']}"
        )

        batch = self._lidar_accumulator
        self._lidar_accumulator = []
        if len(batch) == 1:
            lidar_points = batch[0]
        else:
            lidar_points = np.vstack(batch)

        filtered = _spatial_filter_points(lidar_points)
        if filtered.size == 0:
            self._lidar_layer = np.zeros_like(self._lidar_layer)
            self._lidar_has_points = False
            return

        lidar_layer, has_points = _render_lidar_layer(
            filtered,
            self.params,
            self.lidar_width,
            self.lidar_height,
            self.padding,
            self.canvas_shape,
            self.params["lidar_shift_x"],
            self.params["lidar_shift_y"]
            - (
                float(self._last_cropped_depth_height) * 3 / 4
                + (
                    float(self.params["depth_shift_y"])
                    # + 0.5 * (WINDOW_HEIGHT - float(self._last_cropped_depth_height))
                )
                - (WINDOW_HEIGHT / 2.0)
            ),
            
        )
        if has_points:
            self._lidar_layer = lidar_layer
            self._lidar_has_points = True
            # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            # snapshot_path = self.snapshot_dir / f"lidar_snapshot_{timestamp}.png"
            # try:
            #     cv2.imwrite(str(snapshot_path), self._lidar_layer)
            #     self.get_logger().info(f"Saved LiDAR snapshot to {snapshot_path}")
            # except Exception as exc:
            #     self.get_logger().warning(f"Failed to save LiDAR snapshot: {exc}")
        else:
            self._lidar_layer = np.zeros_like(self._lidar_layer)
            self._lidar_has_points = False

    def _publish_frame(self) -> None:
        if self.last_depth_image is None:
            return

        effective_params = dict(self.params)
        effective_params["depth_shift_y"] = 0.5 * (
            WINDOW_HEIGHT - float(self._last_cropped_depth_height)
        )
        depth_layer = _prepare_depth_layer(self.last_depth_image, self.canvas_shape, effective_params)
        depth_u8 = np.clip(depth_layer, 0, 255).astype(np.uint8)
        depth_bgr = cv2.cvtColor(depth_u8, cv2.COLOR_GRAY2BGR)

        composite = depth_bgr.copy()
        if self._lidar_has_points:
            lidar_layer = self._lidar_layer
            lidar_gray = cv2.cvtColor(lidar_layer, cv2.COLOR_BGR2GRAY)
            depth_gray = cv2.cvtColor(depth_bgr, cv2.COLOR_BGR2GRAY)
            lidar_mask = lidar_gray > 0
            depth_mask = depth_gray > 0
            only_lidar = lidar_mask & (~depth_mask)
            overlap = lidar_mask & depth_mask
            composite[only_lidar] = lidar_layer[only_lidar]
            if np.any(overlap):
                blended = (
                    0.5 * depth_bgr[overlap].astype(np.float32)
                    + 0.5 * lidar_layer[overlap].astype(np.float32)
                )
                composite[overlap] = np.clip(blended, 0, 255).astype(np.uint8)

        header = self.last_depth_header
        try:
            msg = self.bridge.cv2_to_imgmsg(composite, encoding="bgr8")
            if header is not None:
                msg.header = header
            else:
                msg.header.stamp = self.get_clock().now().to_msg()
                msg.header.frame_id = "depth_lidar_fusion"
            self.publisher.publish(msg)
        except Exception as exc:  # pragma: no cover - conversion errors
            self.get_logger().error(f"Failed to publish shifted depth image: {exc}")

    @staticmethod
    def _pointcloud_to_numpy(cloud: PointCloud2) -> np.ndarray:
        points = [
            (x, y, z)
            for x, y, z in pc2_read_points(cloud, field_names=("x", "y", "z"), skip_nans=True)
        ]
        if not points:
            return np.empty((0, 3), dtype=np.float32)
        return np.asarray(points, dtype=np.float32)


def main() -> None:
    rclpy.init()
    node = DepthLidarFusionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:  # pragma: no cover - interactive use
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
