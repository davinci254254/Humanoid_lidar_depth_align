#!/usr/bin/env python3
"""Capture a depth image from a ROS 2 topic and output Sobel edges.

This node subscribes to a 16UC1 depth image topic for about 1 second,
collects incoming frames, averages them, converts the result to a
"readable" OpenCV image, and computes Sobel edges on the depth map.

- Input topic: configurable via --topic (default: /camera/depth/image_rect_raw)
- Duration: configurable via --duration (seconds, default: 1.0)
- Final output: top-half Sobel edges of the cropped depth map
    (default filename: depth_sobel_top.png).

Depth encoding assumptions:
- ROS encoding: 16UC1
- Units: millimeters (depth[m] = depth_raw / 1000.0)

If ``--verbose`` is set, additional intermediate images are written:
- Colored depth image: depth_colored.png
- Full Sobel edge map: depth_sobel.png
"""

import argparse
import pathlib
import time

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


class DepthSnapshotNode(Node):
    """Subscribe to a depth topic, collect ~1s of frames, and save a colored image."""

    def __init__(self, topic: str, output_path: pathlib.Path, duration: float = 1.0, verbose: bool = False) -> None:
        super().__init__("depth_snapshot_node")
        self._topic = topic
        self._output_path = output_path
        self._duration = float(duration)
        self._verbose = bool(verbose)

        self._bridge = CvBridge()
        self._frames = []  # list of np.ndarray (H, W) uint16
        self._start_wall = time.time()
        self._done = False

        self._sub = self.create_subscription(
            Image,
            self._topic,
            self._image_callback,
            10,
        )
        self.get_logger().info(
            f"Subscribing to depth topic '{self._topic}' for {self._duration:.2f}s",
        )

    @property
    def done(self) -> bool:
        return self._done

    def _image_callback(self, msg: Image) -> None:
        if self._done:
            return

        now = time.time()
        elapsed = now - self._start_wall

        try:
            depth_u16 = self._bridge.imgmsg_to_cv2(msg, desired_encoding="16UC1")
        except Exception as exc:  # pragma: no cover - runtime safety
            self.get_logger().error(f"Failed to convert Image to cv2: {exc}")
            return

        if elapsed <= self._duration:
            # Still within collection window: store frame
            if depth_u16.ndim != 2:
                self.get_logger().warn("Received depth image with unexpected shape; skipping")
                return
            self._frames.append(depth_u16.copy())
            return

        # Past collection window: if we haven't processed yet, do it now.
        if not self._frames:
            self.get_logger().warn("No depth frames collected within duration; exiting")
            self._done = True
            return

        self._done = True
        depth_stack = np.stack(self._frames, axis=0).astype(np.float32)  # (N, H, W)
        depth_avg_u16 = depth_stack.mean(axis=0).astype(np.uint16)

        self._save_colored_depth(depth_avg_u16)
        self.get_logger().info(f"Saved colored depth image to: {self._output_path}")
        return

    def _save_colored_depth(self, depth_u16: np.ndarray) -> None:
        """Convert uint16 depth (mm) to a colored image and save it."""
        depth = depth_u16.astype(np.float32)

        # Treat zeros as invalid
        valid = depth > 0

        # Erase unreliable regions: top 5% rows and left 5% columns
        h, w = depth.shape
        cut_top = int(0.05 * h)
        cut_left = int(0.05 * w)
        if cut_top > 0:
            valid[:cut_top, :] = False
        if cut_left > 0:
            valid[:, :cut_left] = False

        if not np.any(valid):
            self.get_logger().warn("No valid depth values found; saving empty image")
            img = np.zeros((*depth.shape, 3), dtype=np.uint8)
            cv2.imwrite(str(self._output_path), img)
            return

        # Convert to meters for readability (optional)
        depth_m = depth / 1000.0

        # Use only valid pixels to compute normalization range
        d_min = float(depth_m[valid].min())
        d_max = float(depth_m[valid].max())
        if d_max - d_min < 1e-6:
            # Nearly constant depth; just make a mid-gray image
            depth_norm = np.zeros_like(depth_m, dtype=np.uint8)
            depth_norm[valid] = 128
            depth_norm[~valid] = 255  # treat invalid pixels as max depth
        else:
            depth_clipped = np.clip(depth_m, d_min, d_max)
            depth_norm = (255.0 * (depth_clipped - d_min) / (d_max - d_min)).astype(np.uint8)
            depth_norm[~valid] = 255  # treat invalid/cropped pixels as max depth

        # Apply a color map to visualize depth (far = red or blue depending on COLORMAP)
        depth_color = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)

        # Optionally, mask invalid pixels to black completely
        depth_color[~valid] = (0, 0, 0)

        # Denoise the colored depth image before saving (non-local means)
        depth_color_denoised = cv2.fastNlMeansDenoisingColored(
            depth_color,
            None,
            10,
            10,
            7,
            21,
        )

        # Save colored depth only in verbose mode
        if self._verbose:
            cv2.imwrite("depth_colored.png", depth_color_denoised)

        # Also compute and save standard edge detections on the depth map itself.
        # Use the normalized 8-bit depth image as input for edge operators,
        # but crop off the unreliable top/left margins so there is no black edge.
        gray_full = depth_norm

        # Reuse the previously computed crop sizes (top 5%, left 5%).
        h_full, w_full = gray_full.shape
        cut_top = int(0.05 * h_full)
        cut_left = int(0.05 * w_full)
        gray = gray_full[cut_top:, cut_left:]

        # Report the cropped resolution (width x height)
        h_crop, w_crop = gray.shape
        self.get_logger().info(f"Cropped depth image for edges: {w_crop}x{h_crop} (WxH)")

        # Optional slight blur to reduce noise before edge detection
        blurred = cv2.GaussianBlur(gray, (3, 3), 0.0)

        # Sobel magnitude (the only edge method we keep)
        sobel_x = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)
        sobel_mag = cv2.magnitude(sobel_x, sobel_y)
        sobel_edges = cv2.convertScaleAbs(sobel_mag)
        sobel_edges_denoised = cv2.fastNlMeansDenoising(sobel_edges, None, 10, 7, 21)

        # Save full Sobel edge map only if verbose
        if self._verbose:
            cv2.imwrite("depth_sobel.png", sobel_edges_denoised)

        # Top-half Sobel: this is the final required output
        h_sb, w_sb = sobel_edges_denoised.shape
        sobel_top = sobel_edges_denoised[: h_sb // 2, :]
        cv2.imwrite(str(self._output_path), sobel_top)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture 1s of depth from a ROS 2 topic and save a colored image",
    )
    parser.add_argument(
        "--topic",
        default="/camera/depth/image_rect_raw",
        help="Depth image topic (sensor_msgs/Image, encoding 16UC1)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=1.0,
        help="Duration in seconds to collect depth frames (default: 1.0)",
    )
    parser.add_argument(
        "--output",
        default="depth_sobel_top.png",
        help="Final Sobel top-half image path (default: depth_sobel_top.png)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="If set, save intermediate images (colored depth, full Sobel)",
    )
    return parser.parse_args()


def depth_gen() -> None:
    args = _parse_args()
    rclpy.init(args=None)

    output_path = pathlib.Path(args.output)
    node = DepthSnapshotNode(
        topic=args.topic,
        output_path=output_path,
        duration=args.duration,
        verbose=args.verbose,
    )

    try:
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()



if __name__ == "__main__":
    depth_gen()
