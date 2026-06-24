from __future__ import annotations

from pathlib import Path

import numpy as np

from .config import ProjectConfig
from .geometry import PointCloud, write_ply


def _grid_background() -> PointCloud:
    """生成一个带地面、后墙和侧墙的简化背景点云。

    这个背景不是替代真实 3DGS，而是用于验证“背景 + 三个资产 + 多视角漫游”的
    工程闭环。正式实验时可把真实 3DGS 采样点云或 Blender 导出的背景点云放到
    配置指定路径。
    """

    xs = np.linspace(-2.4, 2.4, 90)
    ys = np.linspace(-1.8, 1.8, 72)
    floor = np.array([[x, y, 0.0] for x in xs for y in ys], dtype=np.float32)
    floor_color = np.tile(np.array([[185, 188, 170]], dtype=np.uint8), (len(floor), 1))

    wall_xs = np.linspace(-2.4, 2.4, 90)
    wall_zs = np.linspace(0.0, 2.4, 60)
    back_wall = np.array([[x, 1.85, z] for x in wall_xs for z in wall_zs], dtype=np.float32)
    back_color = np.tile(np.array([[205, 214, 223]], dtype=np.uint8), (len(back_wall), 1))

    side_ys = np.linspace(-1.8, 1.8, 72)
    side_wall = np.array([[-2.45, y, z] for y in side_ys for z in wall_zs], dtype=np.float32)
    side_color = np.tile(np.array([[202, 196, 184]], dtype=np.uint8), (len(side_wall), 1))

    points = np.concatenate([floor, back_wall, side_wall], axis=0)
    colors = np.concatenate([floor_color, back_color, side_color], axis=0)
    return PointCloud(points=points, colors=colors, name="background_scene")


def _colored_sphere(radius: float = 0.45, rings: int = 28, segments: int = 56) -> PointCloud:
    """生成近似真实多视角重建物体 A 的彩色球体点云。"""

    points: list[list[float]] = []
    colors: list[list[int]] = []
    for i in range(rings):
        theta = np.pi * (i + 0.5) / rings
        for j in range(segments):
            phi = 2 * np.pi * j / segments
            x = radius * np.sin(theta) * np.cos(phi)
            y = radius * np.sin(theta) * np.sin(phi)
            z = radius * np.cos(theta) + radius
            points.append([x, y, z])
            colors.append([70 + int(120 * i / rings), 116, 200 - int(80 * j / segments)])
    return PointCloud(np.asarray(points, dtype=np.float32), np.asarray(colors, dtype=np.uint8), "object_a")


def _single_image_cone(height: float = 0.9, radius: float = 0.38) -> PointCloud:
    """生成近似单图到 3D 物体 C 的锥体点云。

    单图生成通常背面纹理与几何不稳定，锥体点云有明显前后视角差异，适合在报告中
    解释 single-view prior 的局限。
    """

    points: list[list[float]] = []
    colors: list[list[int]] = []
    for level in np.linspace(0.0, 1.0, 40):
        current_radius = radius * (1.0 - level)
        z = height * level
        for j in range(48):
            phi = 2 * np.pi * j / 48
            points.append([current_radius * np.cos(phi), current_radius * np.sin(phi), z])
            colors.append([215, 80 + int(90 * level), 88])
    return PointCloud(np.asarray(points, dtype=np.float32), np.asarray(colors, dtype=np.uint8), "object_c")


def _write_robot_obj(path: Path) -> None:
    """写出一个简化机器人 OBJ，模拟 threestudio 文本到 3D 的 Mesh 产物。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    vertices = [
        (-0.35, -0.22, 0.0), (0.35, -0.22, 0.0), (0.35, 0.22, 0.0), (-0.35, 0.22, 0.0),
        (-0.35, -0.22, 0.65), (0.35, -0.22, 0.65), (0.35, 0.22, 0.65), (-0.35, 0.22, 0.65),
        (-0.22, -0.15, 0.65), (0.22, -0.15, 0.65), (0.22, 0.15, 0.65), (-0.22, 0.15, 0.65),
        (-0.22, -0.15, 1.05), (0.22, -0.15, 1.05), (0.22, 0.15, 1.05), (-0.22, 0.15, 1.05),
    ]
    faces = [
        (1, 2, 3), (1, 3, 4), (5, 8, 7), (5, 7, 6), (1, 5, 6), (1, 6, 2),
        (2, 6, 7), (2, 7, 3), (3, 7, 8), (3, 8, 4), (4, 8, 5), (4, 5, 1),
        (9, 10, 11), (9, 11, 12), (13, 16, 15), (13, 15, 14), (9, 13, 14),
        (9, 14, 10), (10, 14, 15), (10, 15, 11), (11, 15, 16), (11, 16, 12),
        (12, 16, 13), (12, 13, 9),
    ]
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# demo text-to-3D robot mesh\n")
        for vertex in vertices:
            handle.write(f"v {vertex[0]} {vertex[1]} {vertex[2]}\n")
        for face in faces:
            handle.write(f"f {face[0]} {face[1]} {face[2]}\n")


def create_demo_assets(config: ProjectConfig) -> list[Path]:
    """按配置路径生成轻量 demo 资产。

    返回:
        list[Path]: 实际创建的资产路径，便于 CLI 打印和后续摘要。
    """

    assets = config.data["assets"]
    created: list[Path] = []

    background_path = config.resolve(assets["background"]["path"])
    write_ply(background_path, _grid_background())
    created.append(background_path)

    object_a_path = config.resolve(assets["object_a"]["path"])
    write_ply(object_a_path, _colored_sphere())
    created.append(object_a_path)

    object_b_path = config.resolve(assets["object_b"]["path"])
    _write_robot_obj(object_b_path)
    created.append(object_b_path)

    object_c_path = config.resolve(assets["object_c"]["path"])
    write_ply(object_c_path, _single_image_cone())
    created.append(object_c_path)

    return created

