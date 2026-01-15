"""Capture 10 s of Livox data, render a 1280x480 image, and run Scharr pipeline."""

from __future__ import annotations

import threading
from dataclasses import dataclass
import struct
from pathlib import Path

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
try:
	from sensor_msgs_py import point_cloud2 as pointcloud2_helper  # type: ignore
except ImportError:  # pragma: no cover - helper not available
	pointcloud2_helper = None
from livox_ros_driver2.msg import CustomMsg


from frame_segment_customMsg import (
	PointCloudMerger,
	livox_custom_msg_to_pointcloud2,
	pc2_read_points,
	SNAPSHOT_WIDTH,
	SNAPSHOT_HEIGHT,
)

# Camera pose and snapshot geometry
# manually tuned, used as initial parameters
CAMERA_POSITION = np.array([0.0, 0.0, 0.0], dtype=float)
CAMERA_YAW_DEG = 180.0
CAMERA_PITCH_DEG = -12.2
CAMERA_ROLL_DEG = 96.2

SNAPSHOT_WIDTH = 640
SNAPSHOT_HEIGHT = 480
SNAPSHOT_PADDING = 30

# LiDAR range filtering
MAX_RANGE_METERS = 2.0


DEFAULT_LIVOX_TOPIC = "/livox/lidar"
COLLECTION_DURATION_SEC = 10.0
FILTERED_POINTS_CACHE = Path("filtered_livox_points.npy")


_LIVOX_STRUCT = struct.Struct("<IffffBB")


def pc2_read_points(cloud: PointCloud2, field_names=None, skip_nans=False):
	"""Read points from a PointCloud2 message (simplified helper)."""

	if not field_names:
		field_names = ("x", "y", "z")

	field_offsets = {f.name: f.offset for f in cloud.fields}
	offsets = [field_offsets[name] for name in field_names]
	count = cloud.width * cloud.height
	step = cloud.point_step

	for i in range(count):
		base = i * step
		values = []
		valid = True
		for off in offsets:
			value = struct.unpack_from("<f", cloud.data, base + off)[0]
			if skip_nans and (value != value):
				valid = False
				break
			values.append(value)
		if skip_nans and not valid:
			continue
		yield tuple(values)


def _make_livox_point_fields():
	return [
		PointField(name="offset_time", offset=0, datatype=PointField.UINT32, count=1),
		PointField(name="x", offset=4, datatype=PointField.FLOAT32, count=1),
		PointField(name="y", offset=8, datatype=PointField.FLOAT32, count=1),
		PointField(name="z", offset=12, datatype=PointField.FLOAT32, count=1),
		PointField(name="intensity", offset=16, datatype=PointField.FLOAT32, count=1),
		PointField(name="tag", offset=20, datatype=PointField.UINT8, count=1),
		PointField(name="line", offset=21, datatype=PointField.UINT8, count=1),
	]


def _create_pointcloud2(header, fields, points):
	if pointcloud2_helper is not None:
		cloud = pointcloud2_helper.create_cloud(header, fields, points)
		cloud.is_dense = bool(points)
		return cloud

	cloud = PointCloud2()
	cloud.header = header
	cloud.height = 1
	cloud.width = len(points)
	cloud.fields = list(fields)
	cloud.is_bigendian = False
	cloud.point_step = _LIVOX_STRUCT.size
	cloud.row_step = cloud.point_step * cloud.width
	cloud.is_dense = bool(points)

	if points:
		buffer = bytearray(cloud.row_step)
		offset = 0
		for point in points:
			buffer[offset : offset + cloud.point_step] = _LIVOX_STRUCT.pack(*point)
			offset += cloud.point_step
		cloud.data = bytes(buffer)
	else:
		cloud.data = b""

	return cloud


def livox_custom_msg_to_pointcloud2(livox_msg: CustomMsg) -> PointCloud2:
	fields = _make_livox_point_fields()
	points = [
		(
			point.offset_time,
			point.x,
			point.y,
			point.z,
			float(point.reflectivity),
			point.tag,
			point.line,
		)
		for point in livox_msg.points
	]
	return _create_pointcloud2(livox_msg.header, fields, points)


def save_colored_snapshot(
	points_camera: np.ndarray,
	save_path: str = "lidar_snapshot.png",
	width: int = SNAPSHOT_WIDTH,
	height: int = SNAPSHOT_HEIGHT,
) -> None:
	"""Render a colored 2D image from camera-frame points."""

	if points_camera.shape[0] == 0:
		blank = np.zeros((height, width, 3), dtype=np.uint8)
		cv2.imwrite(save_path, blank)
		print(f"[function_lidar_gen_chair] Saved blank snapshot to {save_path}")
		return

	projected = points_camera[:, [0, 2]]
	min_x, max_x = projected[:, 0].min(), projected[:, 0].max()
	min_z, max_z = projected[:, 1].min(), projected[:, 1].max()

	if max_x - min_x < 1e-6:
		max_x = min_x + 1.0
	if max_z - min_z < 1e-6:
		max_z = min_z + 1.0

	scale_x = (width - 2 * SNAPSHOT_PADDING) / (max_x - min_x)
	scale_z = (height - 2 * SNAPSHOT_PADDING) / (max_z - min_z)

	u = ((projected[:, 0] - min_x) * scale_x + SNAPSHOT_PADDING).astype(int)
	v = (height - ((projected[:, 1] - min_z) * scale_z + SNAPSHOT_PADDING)).astype(int)

	u = (width - 1) - u
	v = (height - 1) - v

	distances = np.linalg.norm(points_camera, axis=1)
	dist_min, dist_max = distances.min(), distances.max()
	if dist_max - dist_min < 1e-6:
		normalized = np.zeros_like(distances)
	else:
		normalized = (distances - dist_min) / (dist_max - dist_min)

	color_values = (normalized * 255).astype(np.uint8)
	colors = cv2.applyColorMap(color_values.reshape(-1, 1), cv2.COLORMAP_JET).reshape(-1, 3)

	image = np.zeros((height, width, 3), dtype=np.uint8)
	depth_map = np.full((height, width), np.inf, dtype=np.float32)

	for idx in range(points_camera.shape[0]):
		uu = u[idx]
		vv = v[idx]
		if 0 <= uu < width and 0 <= vv < height:
			if distances[idx] < depth_map[vv, uu]:
				depth_map[vv, uu] = distances[idx]
				image[vv, uu] = colors[idx]

	image = cv2.flip(image, 1)
	image = cv2.rotate(image, cv2.ROTATE_180)
	cv2.imwrite(save_path, image)
	print(f"[function_lidar_gen_chair] Saved colored snapshot to {save_path}")


def euler_to_R_cw(yaw_deg: float, pitch_deg: float, roll_deg: float) -> np.ndarray:
	yaw = np.deg2rad(yaw_deg)
	pitch = np.deg2rad(pitch_deg)
	roll = np.deg2rad(roll_deg)
	Ry = np.array(
		[
			[np.cos(yaw), 0, np.sin(yaw)],
			[0, 1, 0],
			[-np.sin(yaw), 0, np.cos(yaw)],
		]
	)
	Rx = np.array(
		[
			[1, 0, 0],
			[0, np.cos(pitch), -np.sin(pitch)],
			[0, np.sin(pitch), np.cos(pitch)],
		]
	)
	Rz = np.array(
		[
			[np.cos(roll), -np.sin(roll), 0],
			[np.sin(roll), np.cos(roll), 0],
			[0, 0, 1],
		]
	)
	return Rz @ Rx @ Ry


def world_to_camera_two_mult(
	points_world: np.ndarray,
	cam_pos: np.ndarray,
	yaw_deg: float,
	pitch_deg: float,
	roll_deg: float,
) -> np.ndarray:
	T4 = np.eye(4)
	T4[:3, 3] = -np.asarray(cam_pos, dtype=float)

	R_cw = euler_to_R_cw(yaw_deg, pitch_deg, roll_deg)
	R_wc = R_cw.T
	R4 = np.eye(4)
	R4[:3, :3] = R_wc

	Pw_h = np.hstack([points_world, np.ones((points_world.shape[0], 1))])
	cam_h = (R4 @ (T4 @ Pw_h.T)).T
	return cam_h[:, :3]


def project_depth(
	points: np.ndarray,
	width: int = SNAPSHOT_WIDTH,
	height: int = SNAPSHOT_HEIGHT,
	padding: int = SNAPSHOT_PADDING,
) -> tuple[np.ndarray, np.ndarray]:
	"""Rasterize 3D points into a 2D depth image."""

	if points.size == 0:
		return np.zeros((height, width), dtype=np.float32), np.zeros((height, width), dtype=bool)

	distances = np.linalg.norm(points, axis=1)
	in_range = distances < MAX_RANGE_METERS
	if not np.any(in_range):
		return np.zeros((height, width), dtype=np.float32), np.zeros((height, width), dtype=bool)

	points = points[in_range]
	distances = distances[in_range]
	projected = points[:, [0, 2]]
	min_x, max_x = projected[:, 0].min(), projected[:, 0].max()
	min_z, max_z = projected[:, 1].min(), projected[:, 1].max()

	if max_x - min_x < 1e-6:
		max_x = min_x + 1.0
	if max_z - min_z < 1e-6:
		max_z = min_z + 1.0

	scale_x = (width - 2 * padding) / (max_x - min_x)
	scale_z = (height - 2 * padding) / (max_z - min_z)

	u = ((projected[:, 0] - min_x) * scale_x + padding).astype(int)
	v = (height - ((projected[:, 1] - min_z) * scale_z + padding)).astype(int)

	u = (width - 1) - u
	v = (height - 1) - v

	depth_image = np.full((height, width), np.inf, dtype=np.float32)
	mask = np.zeros((height, width), dtype=bool)

	for idx in range(points.shape[0]):
		uu = u[idx]
		vv = v[idx]
		if 0 <= uu < width and 0 <= vv < height:
			depth = distances[idx]
			if depth < depth_image[vv, uu]:
				depth_image[vv, uu] = depth
			mask[vv, uu] = True

	depth_image[~mask] = np.nan
	depth_image = cv2.flip(depth_image, 1)
	mask = np.flip(mask, 1)
	depth_image = np.rot90(depth_image, k=2)
	mask = np.rot90(mask, k=2)

	return depth_image, mask


def normalize_depth(depth_image: np.ndarray, mask: np.ndarray) -> np.ndarray:
	normalized = np.zeros_like(depth_image, dtype=np.float32)
	if not np.any(mask):
		return normalized.astype(np.uint8)

	valid_depths = depth_image[mask]
	d_min, d_max = valid_depths.min(), valid_depths.max()
	if d_max - d_min < 1e-6:
		normalized[mask] = 0.0
	else:
		normalized[mask] = (valid_depths - d_min) / (d_max - d_min)
	return (normalized * 255.0).astype(np.uint8)


def scharr_edges_from_gray(gray: np.ndarray, blur_ksize: int = 5, blur_sigma: float = 1.0) -> np.ndarray:
	if blur_ksize > 1:
		blurred = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), blur_sigma)
	else:
		blurred = gray

	scharr_x = cv2.Scharr(blurred, cv2.CV_32F, 1, 0)
	scharr_y = cv2.Scharr(blurred, cv2.CV_32F, 0, 1)
	scharr_mag = cv2.magnitude(scharr_x, scharr_y)
	return cv2.convertScaleAbs(scharr_mag)


def region_growing_from_scharr(scharr_img: np.ndarray) -> np.ndarray:
	img = cv2.normalize(scharr_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
	img_blur = cv2.GaussianBlur(img, (5, 5), 1.0)
	_, edge_mask = cv2.threshold(img_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
	interior_mask = edge_mask == 0
	interior_uint8 = np.zeros_like(img, dtype=np.uint8)
	interior_uint8[interior_mask] = 255
	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
	interior_uint8 = cv2.morphologyEx(interior_uint8, cv2.MORPH_OPEN, kernel)
	_, labels = cv2.connectedComponents(interior_uint8)
	region_mask = np.zeros_like(img, dtype=np.uint8)
	region_mask[labels > 0] = 255
	return region_mask


@dataclass
class _CollectedData:
	frames: list[np.ndarray]

	def merged(self) -> np.ndarray:
		if not self.frames:
			return np.empty((0, 3), dtype=np.float32)
		return np.vstack(self.frames)


class _LivoxCollector(Node):
	def __init__(self, topic: str, duration_sec: float) -> None:
		super().__init__("livox_snapshot_collector")
		self._duration_sec = float(duration_sec)
		self._data = _CollectedData([])
		self._start_time = None
		self._done_event = threading.Event()
		self._last_progress_tick = -1  # helps emit progress updates once per second
		self._sub = self.create_subscription(CustomMsg, topic, self._lidar_callback, 10)
		self._timer = self.create_timer(0.1, self._check_timeout)
		self.get_logger().info(
			f"Collecting Livox data from '{topic}' for {duration_sec:.1f} s...",
		)

	def _lidar_callback(self, livox_msg: CustomMsg) -> None:
		if self._done_event.is_set():
			return
		if self._start_time is None:
			self._start_time = self.get_clock().now().nanoseconds / 1e9
		cloud_msg = livox_custom_msg_to_pointcloud2(livox_msg)
		if cloud_msg.width == 0:
			return
		points = []
		for point in pc2_read_points(cloud_msg, field_names=("x", "y", "z"), skip_nans=True):
			points.append([point[0], point[1], point[2]])
		if points:
			self._data.frames.append(np.array(points, dtype=np.float32))

	def _check_timeout(self) -> None:
		if self._done_event.is_set() or self._start_time is None:
			return
		elapsed = self.get_clock().now().nanoseconds / 1e9 - self._start_time
		progress_tick = int(elapsed)
		if progress_tick != self._last_progress_tick and not self._done_event.is_set():
			remaining = max(self._duration_sec - elapsed, 0.0)
			self.get_logger().info(
				f"Collecting... {elapsed:.1f}/{self._duration_sec:.1f} s elapsed, "
				f"~{remaining:.1f} s remaining",
			)
			self._last_progress_tick = progress_tick
		if elapsed >= self._duration_sec:
			self._done_event.set()
			self.get_logger().info("Collection window elapsed; finishing up.")
			self._timer.cancel()

	def spin_until_done(self) -> None:
		while rclpy.ok() and not self._done_event.is_set():
			rclpy.spin_once(self, timeout_sec=0.1)

	@property
	def frames(self) -> list[np.ndarray]:
		return self._data.frames


def _collect_livox_points() -> np.ndarray:
	"""Capture ~10 seconds of Livox data and return merged point cloud."""

	if not rclpy.ok():
		rclpy.init(args=None)

	node = _LivoxCollector(DEFAULT_LIVOX_TOPIC, COLLECTION_DURATION_SEC)
	try:
		node.spin_until_done()
	finally:
		frames = node.frames.copy()
		node.destroy_node()
		if rclpy.ok():
			rclpy.shutdown()

	if not frames:
		raise RuntimeError("No Livox frames were collected in the allotted time.")

	merged = np.vstack(frames)
	return merged


def _filter_points(points_world: np.ndarray) -> np.ndarray:
	"""Apply spatial and distance filters identical to the GUI workflow."""

	if points_world.size == 0:
		return points_world

	y = points_world[:, 1]
	x = points_world[:, 0]
	mask = ((x + y) >= 0.0) & ((x - y) >= 0.6)
	points_world = points_world[mask]

	if points_world.size == 0:
		return points_world

	distances = np.linalg.norm(points_world, axis=1)
	range_mask = distances < MAX_RANGE_METERS
	return points_world[range_mask]


def _points_to_camera(
	points_world: np.ndarray,
	yaw_deg: float,
	pitch_deg: float,
	roll_deg: float,
) -> np.ndarray:
	"""Transform filtered world points into the camera frame."""

	if points_world.size == 0:
		return points_world.reshape(0, 3)

	camera_points = PointCloudMerger.world_to_camera_two_mult(
		points_world,
		CAMERA_POSITION,
		yaw_deg,
		pitch_deg,
		roll_deg,
	)
	return camera_points.astype(np.float32)


def _save_snapshot_image(camera_points: np.ndarray) -> None:
	"""Render a 640x480 colored snapshot image from camera-frame points."""

	if camera_points.size == 0:
		blank = np.zeros((SNAPSHOT_HEIGHT, SNAPSHOT_WIDTH, 3), dtype=np.uint8)
		cv2.imwrite("lidar_snapshot.png", blank)
		print("[function_lidar_gen_chair] Saved blank lidar_snapshot.png (no points)")
		return

	rotated = camera_points.copy()  # rotation already handled via camera pose
	save_colored_snapshot(
		rotated,
		save_path="lidar_snapshot.png",
		width=SNAPSHOT_WIDTH,
		height=SNAPSHOT_HEIGHT,
	)


def _run_scharr_pipeline(camera_points: np.ndarray) -> None:
	"""Execute Scharr → region-growing → Scharr on rotated camera points."""

	if camera_points.size == 0:
		raise RuntimeError("No camera-frame points available for Scharr pipeline.")

	depth_image, mask = project_depth(
		camera_points,
		width=SNAPSHOT_WIDTH,
		height=SNAPSHOT_HEIGHT,
		padding=SNAPSHOT_PADDING,
	)
	depth_uint8 = normalize_depth(depth_image, mask)
	scharr_img = scharr_edges_from_gray(depth_uint8)
	cv2.imwrite("scharr.png", scharr_img)
	region_mask = region_growing_from_scharr(scharr_img)
	cv2.imwrite("scharr_regions.png", region_mask)
	regions_scharr = scharr_edges_from_gray(region_mask)
	cv2.imwrite("scharr_regions_scharr.png", regions_scharr)
	print("[function_lidar_gen_chair] Saved Scharr pipeline outputs.")



def function_lidar_gen_chair(yaw_deg: float, pitch_deg: float, roll_deg: float) -> None:
	"""Entry-point: capture LiDAR, render snapshot, and run edge pipeline."""

	first_run_capture = True
	if FILTERED_POINTS_CACHE.exists():
		filtered = np.load(FILTERED_POINTS_CACHE).astype(np.float32)
		if filtered.ndim != 2 or filtered.shape[1] != 3:
			print(
				"[function_lidar_gen_chair] Cached filtered points malformed; regenerating cache."
			)
			filtered = np.empty((0, 3), dtype=np.float32)
	else:
		filtered = np.empty((0, 3), dtype=np.float32)

	if filtered.size == 0:
		points_world = _collect_livox_points()
		filtered = _filter_points(points_world)
		if filtered.size == 0:
			raise RuntimeError("All collected points were filtered out; nothing to process.")
		try:
			FILTERED_POINTS_CACHE.parent.mkdir(parents=True, exist_ok=True)
		except Exception:
			pass
		np.save(FILTERED_POINTS_CACHE, filtered)
		print(
			f"[function_lidar_gen_chair] Cached filtered points -> {FILTERED_POINTS_CACHE}"
		)
		first_run_capture = True
	else:
		print(
			f"[function_lidar_gen_chair] Loaded cached filtered points from {FILTERED_POINTS_CACHE}"
		)

	camera_points = _points_to_camera(filtered, yaw_deg, pitch_deg, roll_deg)
	if camera_points.size == 0:
		raise RuntimeError("Projection to camera frame produced no points.")

	_save_snapshot_image(camera_points)
	if first_run_capture:
		save_colored_snapshot(
			camera_points,
			save_path="first_time_cam.png",
			width=640,
			height=480,
		)
	_run_scharr_pipeline(camera_points)
	print("[function_lidar_gen_chair] Completed snapshot and edge generation.")


if __name__ == "__main__":
	function_lidar_gen_chair(
		yaw_deg=CAMERA_YAW_DEG,
		pitch_deg=CAMERA_PITCH_DEG,
		roll_deg=CAMERA_ROLL_DEG,
	)
