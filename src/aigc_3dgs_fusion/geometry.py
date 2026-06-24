from __future__ import annotations

from dataclasses import dataclass
from math import cos, radians, sin
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass
class PointCloud:
    """统一点云表示。

    points 使用 `(N, 3)` 浮点数组保存 XYZ 坐标，colors 使用 `(N, 3)` uint8 数组
    保存 RGB。真实 3DGS、AIGC Mesh 和 demo 资产都会先转换到这个表示，便于做
    空间变换、融合和轻量渲染。
    """

    points: np.ndarray
    colors: np.ndarray
    name: str


def rotation_matrix_xyz(degrees: Iterable[float]) -> np.ndarray:
    """根据 XYZ 欧拉角生成旋转矩阵。

    参数:
        degrees: 三个角度值，单位为度，依次表示绕 X、Y、Z 轴旋转。
    """

    rx, ry, rz = [radians(float(v)) for v in degrees]
    mx = np.array([[1, 0, 0], [0, cos(rx), -sin(rx)], [0, sin(rx), cos(rx)]])
    my = np.array([[cos(ry), 0, sin(ry)], [0, 1, 0], [-sin(ry), 0, cos(ry)]])
    mz = np.array([[cos(rz), -sin(rz), 0], [sin(rz), cos(rz), 0], [0, 0, 1]])
    return mz @ my @ mx


def apply_transform(cloud: PointCloud, transform: dict) -> PointCloud:
    """对点云应用缩放、旋转和平移。

    配置中的变换用于把不同来源的 A/B/C 物体放入同一个背景坐标系中，是题目一
    “合理比例、空间位置插入”的核心可控参数。
    """

    scale = float(transform.get("scale", 1.0))
    rotation = rotation_matrix_xyz(transform.get("rotate_degrees", [0.0, 0.0, 0.0]))
    translate = np.asarray(transform.get("translate", [0.0, 0.0, 0.0]), dtype=np.float32)
    points = (cloud.points * scale) @ rotation.T + translate
    return PointCloud(points=points, colors=cloud.colors.copy(), name=cloud.name)


def write_ply(path: Path, cloud: PointCloud) -> None:
    """写出 ASCII PLY 点云，便于被 MeshLab、Blender 或 Open3D 检查。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("ply\nformat ascii 1.0\n")
        handle.write(f"element vertex {len(cloud.points)}\n")
        handle.write("property float x\nproperty float y\nproperty float z\n")
        handle.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        handle.write("end_header\n")
        for point, color in zip(cloud.points, cloud.colors):
            handle.write(
                f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f} "
                f"{int(color[0])} {int(color[1])} {int(color[2])}\n"
            )


def read_ply(path: Path, name: str | None = None) -> PointCloud:
    """读取本项目使用的 ASCII PLY 点云。

    真实工具若导出二进制 PLY，建议先用 Blender/Open3D 转成 ASCII，或后续扩展此
    函数的二进制解析分支。
    """

    with path.open("r", encoding="utf-8") as handle:
        lines = handle.readlines()

    vertex_count = 0
    header_end = 0
    for index, line in enumerate(lines):
        if line.startswith("element vertex"):
            vertex_count = int(line.split()[-1])
        if line.strip() == "end_header":
            header_end = index + 1
            break

    rows = [line.split() for line in lines[header_end : header_end + vertex_count]]
    points = np.asarray([[float(v[0]), float(v[1]), float(v[2])] for v in rows], dtype=np.float32)
    colors = np.asarray([[int(v[3]), int(v[4]), int(v[5])] for v in rows], dtype=np.uint8)
    return PointCloud(points=points, colors=colors, name=name or path.stem)


def read_obj_as_points(path: Path, name: str | None = None, samples_per_face: int = 3) -> PointCloud:
    """将 OBJ Mesh 采样为点云。

    为了让文本到 3D 生成的 Mesh 可以与 3DGS 背景统一融合，这里读取 OBJ 顶点和
    面，并在三角面内做确定性采样。颜色暂用暖色系，正式实验可从纹理贴图采样。
    """

    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("v "):
                _, x, y, z = line.split()[:4]
                vertices.append([float(x), float(y), float(z)])
            elif line.startswith("f "):
                face = []
                for token in line.split()[1:]:
                    face.append(int(token.split("/")[0]) - 1)
                if len(face) >= 3:
                    faces.append(face[:3])

    vertex_array = np.asarray(vertices, dtype=np.float32)
    points: list[np.ndarray] = []
    for face in faces:
        tri = vertex_array[np.asarray(face)]
        points.extend(tri)
        for step in range(samples_per_face):
            a = (step + 1) / (samples_per_face + 2)
            b = 1.0 / (samples_per_face + 2)
            c = 1.0 - a - b
            points.append(a * tri[0] + b * tri[1] + c * tri[2])

    point_array = np.asarray(points, dtype=np.float32)
    colors = np.tile(np.asarray([[236, 133, 65]], dtype=np.uint8), (len(point_array), 1))
    return PointCloud(points=point_array, colors=colors, name=name or path.stem)

