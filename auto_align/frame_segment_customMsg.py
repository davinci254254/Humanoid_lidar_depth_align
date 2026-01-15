import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
try:
    from sensor_msgs_py import point_cloud2 as pointcloud2_helper  # type: ignore
except ImportError:  # pragma: no cover - fallback when helper is unavailable
    pointcloud2_helper = None
from livox_ros_driver2.msg import CustomMsg  # type: ignore
import numpy as np
import open3d as o3d
import struct
import threading
import csv
import cv2

def pc2_read_points(cloud, field_names=None, skip_nans=False):
    fmt = 'fff'
    width = cloud.width
    height = cloud.height
    point_step = cloud.point_step
    row_step = cloud.row_step
    data = cloud.data
    fields = [f.name for f in cloud.fields]
    field_offsets = {f.name: f.offset for f in cloud.fields}
    count = width * height

    if not field_names:
        field_names = ('x', 'y', 'z')

    offsets = [field_offsets[name] for name in field_names]
    data_fmt = '<' + 'f' * len(field_names)
    step = point_step

    for i in range(count):
        base = i * step
        values = []
        nan_in_value = False
        for off in offsets:
            v = struct.unpack_from('<f', data, base + off)[0]
            if skip_nans and (v != v):
                nan_in_value = True
                break
            values.append(v)
        if skip_nans and nan_in_value:
            continue
        yield tuple(values)


_LIVOX_STRUCT = struct.Struct('<IffffBB')


CAMERA_POSITION = np.array([0.0, 0.0, 0.0], dtype=float)
# CAMERA_YAW_DEG = -6.3
# CAMERA_PITCH_DEG = 13.9
# CAMERA_ROLL_DEG = -84.6

# CAMERA_YAW_DEG = 35.8

CAMERA_YAW_DEG = 180
CAMERA_PITCH_DEG = -12.2
CAMERA_ROLL_DEG = 96.2

# CAMERA_YAW_DEG = 0
# CAMERA_PITCH_DEG = 0
# CAMERA_ROLL_DEG = 0

SNAPSHOT_WIDTH = 1280
SNAPSHOT_HEIGHT = 480
SNAPSHOT_PADDING = 30


def _make_livox_point_fields():
    return [
        PointField(name='offset_time', offset=0, datatype=PointField.UINT32, count=1),
        PointField(name='x', offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name='y', offset=8, datatype=PointField.FLOAT32, count=1),
        PointField(name='z', offset=12, datatype=PointField.FLOAT32, count=1),
        PointField(name='intensity', offset=16, datatype=PointField.FLOAT32, count=1),
        PointField(name='tag', offset=20, datatype=PointField.UINT8, count=1),
        PointField(name='line', offset=21, datatype=PointField.UINT8, count=1),
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
            buffer[offset:offset + cloud.point_step] = _LIVOX_STRUCT.pack(*point)
            offset += cloud.point_step
        cloud.data = bytes(buffer)
    else:
        cloud.data = b''

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

    cloud_msg = _create_pointcloud2(livox_msg.header, fields, points)
    return cloud_msg


def save_colored_snapshot(points_camera: np.ndarray, save_path: str = "snapshot_colored.png",
                          width: int = SNAPSHOT_WIDTH, height: int = SNAPSHOT_HEIGHT,
                          return_maps: bool = False, points_original: np.ndarray = None):
    if points_camera.shape[0] == 0:
        blank = np.ones((height, width, 3), dtype=np.uint8) * 255
        cv2.imwrite(save_path, blank)
        print(f"Saved blank snapshot to {save_path} (no points available)")
        if return_maps:
            xyz_transformed = np.zeros((height, width, 3), dtype=np.float32)
            xyz_original = np.zeros((height, width, 3), dtype=np.float32)
            counts = np.zeros((height, width), dtype=np.int32)
            return blank, xyz_transformed, xyz_original, counts
        return

    # Use original points for distance calculation if provided
    if points_original is None:
        points_original = points_camera

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

    # Calculate distances from ORIGINAL (un-transformed) points
    distances = np.linalg.norm(points_original, axis=1)
    dist_min, dist_max = distances.min(), distances.max()
    if dist_max - dist_min < 1e-6:
        normalized = np.zeros_like(distances)
    else:
        normalized = (distances - dist_min) / (dist_max - dist_min)

    color_values = (normalized * 255).astype(np.uint8)
    colors = cv2.applyColorMap(color_values.reshape(-1, 1), cv2.COLORMAP_JET).reshape(-1, 3)

    image = np.ones((height, width, 3), dtype=np.uint8) * 255
    depth_map = np.full((height, width), np.inf, dtype=np.float32)
    xyz_transformed_accum = np.zeros((height, width, 3), dtype=np.float64)
    xyz_original_accum = np.zeros((height, width, 3), dtype=np.float64)
    counts = np.zeros((height, width), dtype=np.int32)

    for idx in range(points_camera.shape[0]):
        uu = u[idx]
        vv = v[idx]
        if 0 <= uu < width and 0 <= vv < height:
            counts[vv, uu] += 1
            xyz_transformed_accum[vv, uu] += points_camera[idx]  # Store TRANSFORMED coordinates
            xyz_original_accum[vv, uu] += points_original[idx]   # Store ORIGINAL coordinates
            if distances[idx] < depth_map[vv, uu]:
                depth_map[vv, uu] = distances[idx]
                color = colors[idx]
                image[vv, uu] = (int(color[0]), int(color[1]), int(color[2]))

    # Mirror image left-to-right (flip horizontally)
    image = cv2.flip(image, 1)
    # Also flip the coordinate accumulation arrays to match
    xyz_transformed_accum = np.flip(xyz_transformed_accum, axis=1)
    xyz_original_accum = np.flip(xyz_original_accum, axis=1)
    counts = np.flip(counts, axis=1)
    
    # Then rotate by 180 degrees
    image = cv2.rotate(image, cv2.ROTATE_180)
    xyz_transformed_accum = np.rot90(xyz_transformed_accum, k=2)
    xyz_original_accum = np.rot90(xyz_original_accum, k=2)
    counts = np.rot90(counts, k=2)
    
    cv2.imwrite(save_path, image)
    print(f"Saved colored snapshot to {save_path}")

    if return_maps:
        return image, xyz_transformed_accum, xyz_original_accum, counts


class PointCloudMerger(Node):
    def __init__(self, duration_sec=10):
        super().__init__('pointcloud_merger')
        self.subscription = self.create_subscription(
            CustomMsg,
            '/livox/lidar',
            self.custom_listener_callback,
            10)
        self.all_points = []
        self.frame_count = 0
        self.duration_sec = duration_sec
        self.start_time = None
        self.done = False
        self.timer = self.create_timer(0.1, self.check_timeout)
        self.visualization_done = threading.Event()

    def listener_callback(self, msg):
        if self.done:
            return

        if self.start_time is None:
            self.start_time = self.get_clock().now().nanoseconds / 1e9
            self.get_logger().info("Started collecting point clouds...")

        points = []
        for p in pc2_read_points(msg, field_names=('x', 'y', 'z'), skip_nans=True):
            points.append([p[0], p[1], p[2]])
        if points:
            self.all_points.append(np.array(points))
            self.frame_count += 1
            self.get_logger().info(f"Frame {self.frame_count}: total points so far: {sum(len(a) for a in self.all_points)}")

    def check_timeout(self):
        if self.done or self.start_time is None:
            return
        elapsed = self.get_clock().now().nanoseconds / 1e9 - self.start_time
        if elapsed >= self.duration_sec:
            self.get_logger().info(f"Reached {self.duration_sec} seconds. Merging and visualizing point cloud.")
            self.done = True
            self.timer.cancel()
            threading.Thread(target=self.merge_and_visualize_and_shutdown, daemon=True).start()

    def custom_listener_callback(self, livox_msg: CustomMsg):
        cloud_msg = livox_custom_msg_to_pointcloud2(livox_msg)
        if cloud_msg.width == 0:
            self.get_logger().debug("Received Livox message with no points; skipping frame.")
            return
        self.listener_callback(cloud_msg)
    
    @staticmethod
    def euler_to_R_cw(yaw_deg, pitch_deg, roll_deg):
        # Build rotation that maps camera axes to world axes (R_cw)
        yaw = np.deg2rad(yaw_deg)    # about Y
        pitch = np.deg2rad(pitch_deg) # about X
        roll = np.deg2rad(roll_deg)   # about Z
        Ry = np.array([[np.cos(yaw), 0, np.sin(yaw)],
                       [0,           1, 0],
                       [-np.sin(yaw),0, np.cos(yaw)]])
        Rx = np.array([[1, 0, 0],
                       [0, np.cos(pitch), -np.sin(pitch)],
                       [0, np.sin(pitch),  np.cos(pitch)]])
        Rz = np.array([[np.cos(roll), -np.sin(roll), 0],
                       [np.sin(roll),  np.cos(roll), 0],
                       [0,             0,            1]])
        return Rz @ Rx @ Ry  # camera->world
    
    @staticmethod
    def world_to_camera_two_mult(points_world, cam_pos, yaw_deg, pitch_deg, roll_deg):
        # 1) Build 4x4 translation by -C (move world so camera is at origin)
        T4 = np.eye(4)
        T4[:3, 3] = -np.asarray(cam_pos, dtype=float)
    
        # 2) Build 4x4 rotation R_wc (world->camera) from camera pose
        R_cw = PointCloudMerger.euler_to_R_cw(yaw_deg, pitch_deg, roll_deg)
        R_wc = R_cw.T
        R4 = np.eye(4)
        R4[:3, :3] = R_wc
    
        # Homogenize and do two matrix multiplications: Pc_h = R4 @ (T4 @ Pw_h)
        Pw_h = np.hstack([points_world, np.ones((points_world.shape[0], 1))])
        cam_h = (R4 @ (T4 @ Pw_h.T)).T
        return cam_h[:, :3]
    
    def merge_and_visualize(self):
        if not self.all_points:
            print("No point cloud frames received!")
            return
        merged_points = np.vstack(self.all_points)
        # Filter points by y and x+y
        y = merged_points[:, 1]
        x = merged_points[:, 0]
        # mask = (y >= -3.5) & (y <= 3) & ((x + y) >= 2)
        mask = ((x + y) >= 0) & ((x - y) >= 0.6)
        filtered_points = merged_points[mask]
        # filtered_points = merged_points

        # Rotate points around the Y axis so downstream views match legacy snapshots
        theta = np.deg2rad(0.0)
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        Ry = np.array([
            [cos_t, 0.0, sin_t],
            [0.0,   1.0, 0.0 ],
            [-sin_t,0.0, cos_t]
        ])
        filtered_points = filtered_points @ Ry.T

        if filtered_points.shape[0] == 0:
            print("No points in specified range!")
            return

        camera_points = PointCloudMerger.world_to_camera_two_mult(
            filtered_points,
            CAMERA_POSITION,
            CAMERA_YAW_DEG,
            CAMERA_PITCH_DEG,
            CAMERA_ROLL_DEG,
        )

        # camera_points = camera_points[camera_points[:, 2] > 0]
        if camera_points.shape[0] == 0:
            print("No points remain after applying camera pose filter!")
            return

        # No additional rotation - camera pose is already perfect
        camera_points_rotated = camera_points.copy()

        # Pass both rotated points (for plotting) and original points (for distance calculation)
        snapshot_image, xyz_transformed_accum, xyz_original_accum, counts = save_colored_snapshot(
            camera_points_rotated,
            save_path="snapshot_colored.png",
            return_maps=True,
            points_original=camera_points,
        )

        averaged_xyz_transformed = np.zeros_like(xyz_transformed_accum, dtype=np.float32)
        averaged_xyz_original = np.zeros_like(xyz_original_accum, dtype=np.float32)
        valid_mask = counts > 0
        averaged_xyz_transformed[valid_mask] = (xyz_transformed_accum[valid_mask] / counts[valid_mask, None]).astype(np.float32)
        averaged_xyz_original[valid_mask] = (xyz_original_accum[valid_mask] / counts[valid_mask, None]).astype(np.float32)

        csv_path = "snapshot_points.csv"
        with open(csv_path, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["row", "col", "x_trans", "y_trans", "z_trans", "x_orig", "y_orig", "z_orig"])
            for row in range(SNAPSHOT_HEIGHT):
                for col in range(SNAPSHOT_WIDTH):
                    if counts[row, col] > 0:
                        x_trans, y_trans, z_trans = averaged_xyz_transformed[row, col]
                        x_orig, y_orig, z_orig = averaged_xyz_original[row, col]
                    else:
                        x_trans = y_trans = z_trans = float('nan')
                        x_orig = y_orig = z_orig = float('nan')
                    writer.writerow([row, col, x_trans, y_trans, z_trans, x_orig, y_orig, z_orig])

        print(f"Saved averaged point grid to {csv_path}")

        # Save points in camera frame for GUI-based tuning workflows
        np.save("filtered_points_rotated.npy", camera_points_rotated)
        np.save("filtered_points_original.npy", camera_points)
        print("Saved filtered_points_rotated.npy (plotting coordinates)")
        print("Saved filtered_points_original.npy (distance calculation coordinates)")

        print(f"Merged {self.frame_count} frames, total points after filtering: {camera_points_rotated.shape[0]}")
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(camera_points_rotated)
        geoms = self.add_reference_lines_to_pcd(pcd) # this does not work. 
        print("Opening Open3D visualization window with point picking support...")
        self.display_with_point_picking(pcd, geoms[1:])
        print("Open3D window closed.")

        # if picked_indices:
        #     picked_points = filtered_points[picked_indices]
        #     print("Picked point coordinates:\n", picked_points)

    
    def merge_and_visualize_and_shutdown(self):
        self.merge_and_visualize()
        self.visualization_done.set()
        rclpy.shutdown()


    def create_axis_line(self, p1, p2, color):
        # p1, p2: endpoints as (x, y, z)
        points = [p1, p2]
        lines = [[0, 1]]
        colors = [color]
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(points)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector(colors)
        return line_set

    def add_reference_lines_to_pcd(self, pcd, x_range=[0, 10], y_range=[0, 10], z_range=[0, 10]):
        # y=0 (XZ plane): draw a line from (x_min,0,z_min) to (x_max,0,z_max)
        y0_p1 = np.array([x_range[0], 0, z_range[0]])
        y0_p2 = np.array([x_range[1], 0, z_range[1]])
        y0_line = self.create_axis_line(y0_p1, y0_p2, [1, 0, 0])  # Red

        # z=0 (XY plane): draw a line from (x_min,y_min,0) to (x_max,y_max,0)
        z0_p1 = np.array([x_range[0], y_range[0], 0])
        z0_p2 = np.array([x_range[1], y_range[1], 0])
        z0_line = self.create_axis_line(z0_p1, z0_p2, [0, 1, 0])  # Green

        # x=0 (YZ plane): draw a line from (0,y_min,z_min) to (0,y_max,z_max)
        x0_p1 = np.array([0, y_range[0], z_range[0]])
        x0_p2 = np.array([0, y_range[1], z_range[1]])
        x0_line = self.create_axis_line(x0_p1, x0_p2, [0, 0, 1])  # Blue

        x_range = [-10, 10]
        y_range = [-10, 10]
        z_range = [-10, 10]

        # Endpoints from your function
        y0_p1 = np.array([x_range[0], 0, z_range[0]])
        y0_p2 = np.array([x_range[1], 0, z_range[1]])

        z0_p1 = np.array([x_range[0], y_range[0], 0])
        z0_p2 = np.array([x_range[1], y_range[1], 0])

        x0_p1 = np.array([0, y_range[0], z_range[0]])
        x0_p2 = np.array([0, y_range[1], z_range[1]])

        # Direction vectors
        vx = y0_p2 - y0_p1
        vy = z0_p2 - z0_p1
        vz = x0_p2 - x0_p1

        # Angle function
        def angle_between(v1, v2):
            v1u = v1 / np.linalg.norm(v1)
            v2u = v2 / np.linalg.norm(v2)
            angle_rad = np.arccos(np.clip(np.dot(v1u, v2u), -1.0, 1.0))
            return np.degrees(angle_rad)

        # Compute angles
        angle_xy = angle_between(vx, vy)
        angle_xz = angle_between(vx, vz)
        angle_yz = angle_between(vy, vz)

        print(f"Angle between XZ and XY lines: {angle_xy:.2f} degrees")
        print(f"Angle between XZ and YZ lines: {angle_xz:.2f} degrees")
        print(f"Angle between XY and YZ lines: {angle_yz:.2f} degrees")

        return [pcd, y0_line, z0_line, x0_line]



    def display_with_point_picking(self, pcd: o3d.geometry.PointCloud, extra_geometries):
        print("Hold SHIFT and left-click on points to select them. Press 'q' to exit the window.")
        try:
            vis = o3d.visualization.VisualizerWithEditing()
            vis.create_window(window_name="Livox PointCloud (Camera Pose)", width=SNAPSHOT_WIDTH, height=SNAPSHOT_HEIGHT)
        except Exception as exc:  # pragma: no cover - Open3D backend failure
            print(f"VisualizerWithEditing unavailable ({exc}); falling back to basic viewer without picking.")
            o3d.visualization.draw_geometries([pcd, *extra_geometries], window_name="Livox PointCloud (Camera Pose)", width=SNAPSHOT_WIDTH, height=SNAPSHOT_HEIGHT)
            return

        vis.add_geometry(pcd)
        for geom in extra_geometries:
            vis.add_geometry(geom)

        vis.run()
        picked_indices = vis.get_picked_points()
        vis.destroy_window()

        if not picked_indices:
            print("No points picked during the session.")
            return

        pts_np = np.asarray(pcd.points)
        print("Picked point details:")
        for idx in picked_indices:
            if 0 <= idx < pts_np.shape[0]:
                pt = pts_np[idx]
                dist = np.linalg.norm(pt)
                print(f"  Index {idx}: ({pt[0]:.3f}, {pt[1]:.3f}, {pt[2]:.3f}), distance={dist:.3f} m")
            else:
                print(f"  Index {idx}: out of bounds for current point cloud")

def main():
    rclpy.init()
    node = PointCloudMerger(duration_sec=10)
    print("Listening for point cloud frames for 10 seconds...")
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.visualization_done.wait()
    node.destroy_node()

if __name__ == '__main__':
    main()
