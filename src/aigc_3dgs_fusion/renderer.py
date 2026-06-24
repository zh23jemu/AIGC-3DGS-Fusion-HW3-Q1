from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from .config import ProjectConfig
from .geometry import PointCloud, apply_transform, read_obj_as_points, read_ply


def load_scene_clouds(config: ProjectConfig) -> list[PointCloud]:
    """读取配置中声明的背景和 A/B/C 资产，并转换到统一世界坐标系。"""

    clouds: list[PointCloud] = []
    for name, spec in config.data["assets"].items():
        path = config.resolve(spec["path"])
        if spec["type"] == "point_cloud":
            cloud = read_ply(path, name=name)
        elif spec["type"] == "mesh":
            cloud = read_obj_as_points(path, name=name)
        else:
            raise ValueError(f"不支持的资产类型: {spec['type']}")
        clouds.append(apply_transform(cloud, spec.get("transform", {})))
    return clouds


def _camera_basis(eye: np.ndarray, look_at: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """根据相机位置和观察点计算右、上、前三个正交方向。"""

    forward = look_at - eye
    forward = forward / np.linalg.norm(forward)
    world_up = np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
    right = np.cross(forward, world_up)
    right = right / np.linalg.norm(right)
    up = np.cross(right, forward)
    up = up / np.linalg.norm(up)
    return right, up, forward


def _render_points(
    cloud: PointCloud,
    eye: np.ndarray,
    look_at: np.ndarray,
    width: int,
    height: int,
    point_size: int,
    background_color: list[int],
) -> Image.Image:
    """使用简单 z-buffer 将点云投影到图像平面。

    这个渲染器不是为了替代 3DGS 的高质量可微渲染，而是提供一个无 GPU、无外部
    工具时也能验证场景融合和多视角漫游的轻量实现。
    """

    right, up, forward = _camera_basis(eye, look_at)
    relative = cloud.points - eye[None, :]
    x_cam = relative @ right
    y_cam = relative @ up
    z_cam = relative @ forward
    valid = z_cam > 0.05

    focal = 0.85 * width
    px = (width / 2 + focal * x_cam[valid] / z_cam[valid]).astype(np.int32)
    py = (height / 2 - focal * y_cam[valid] / z_cam[valid]).astype(np.int32)
    depth = z_cam[valid]
    colors = cloud.colors[valid]

    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:, :] = np.asarray(background_color, dtype=np.uint8)
    z_buffer = np.full((height, width), np.inf, dtype=np.float32)

    order = np.argsort(depth)[::-1]
    radius = max(1, int(point_size))
    for idx in order:
        x = px[idx]
        y = py[idx]
        if x < 0 or x >= width or y < 0 or y >= height:
            continue
        z = depth[idx]
        x0, x1 = max(0, x - radius), min(width, x + radius + 1)
        y0, y1 = max(0, y - radius), min(height, y + radius + 1)
        patch = z_buffer[y0:y1, x0:x1]
        mask = z < patch
        if np.any(mask):
            image[y0:y1, x0:x1][mask] = colors[idx]
            patch[mask] = z

    return Image.fromarray(image, mode="RGB")


def render_sequence(config: ProjectConfig) -> list[Path]:
    """渲染多视角漫游帧，并额外导出 GIF 方便报告快速预览。"""

    render_cfg = config.data["render"]
    width = int(render_cfg["width"])
    height = int(render_cfg["height"])
    frames = int(render_cfg["frames"])
    radius = float(render_cfg["camera_radius"])
    camera_height = float(render_cfg["camera_height"])
    look_at = np.asarray(render_cfg["look_at"], dtype=np.float32)

    clouds = load_scene_clouds(config)
    fused = PointCloud(
        points=np.concatenate([cloud.points for cloud in clouds], axis=0),
        colors=np.concatenate([cloud.colors for cloud in clouds], axis=0),
        name="fused_scene",
    )

    render_dir = config.output_dir / "renders"
    render_dir.mkdir(parents=True, exist_ok=True)
    frame_paths: list[Path] = []
    images: list[Image.Image] = []
    for frame in range(frames):
        angle = 2 * np.pi * frame / frames
        eye = np.asarray([radius * np.cos(angle), radius * np.sin(angle), camera_height], dtype=np.float32)
        image = _render_points(
            fused,
            eye=eye,
            look_at=look_at,
            width=width,
            height=height,
            point_size=int(render_cfg["point_size"]),
            background_color=render_cfg["background_color"],
        )
        frame_path = render_dir / f"frame_{frame:03d}.png"
        image.save(frame_path)
        frame_paths.append(frame_path)
        images.append(image)

    gif_path = render_dir / "fusion_turntable.gif"
    images[0].save(gif_path, save_all=True, append_images=images[1:], duration=90, loop=0)
    frame_paths.append(gif_path)
    return frame_paths

