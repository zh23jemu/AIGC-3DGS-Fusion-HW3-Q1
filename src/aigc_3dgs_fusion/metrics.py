from __future__ import annotations

from pathlib import Path

import numpy as np

from .config import ProjectConfig
from .renderer import load_scene_clouds


def write_summary(config: ProjectConfig) -> Path:
    """生成用于报告的轻量质量摘要。

    摘要聚焦几何规模、空间范围、渲染产物和各资产来源，帮助报告中解释三类生成
    路线的差异。真实实验的 PSNR、LPIPS、训练耗时等指标可在此文件后继续追加。
    """

    clouds = load_scene_clouds(config)
    render_dir = config.output_dir / "renders"
    frames = sorted(render_dir.glob("frame_*.png")) if render_dir.exists() else []

    lines = ["# HW3 题目一质量摘要", ""]
    lines.append("## 资产统计")
    for cloud in clouds:
        mins = cloud.points.min(axis=0)
        maxs = cloud.points.max(axis=0)
        extent = maxs - mins
        lines.append(
            f"- `{cloud.name}`: 点数 {len(cloud.points)}, "
            f"范围 x/y/z=({extent[0]:.2f}, {extent[1]:.2f}, {extent[2]:.2f})"
        )

    fused_points = np.concatenate([cloud.points for cloud in clouds], axis=0)
    fused_min = fused_points.min(axis=0)
    fused_max = fused_points.max(axis=0)
    lines.extend(
        [
            "",
            "## 融合场景",
            f"- 总点数: {len(fused_points)}",
            f"- 世界坐标最小值: [{fused_min[0]:.2f}, {fused_min[1]:.2f}, {fused_min[2]:.2f}]",
            f"- 世界坐标最大值: [{fused_max[0]:.2f}, {fused_max[1]:.2f}, {fused_max[2]:.2f}]",
            f"- 渲染帧数: {len(frames)}",
            "",
            "## 报告分析提示",
            "- 多视角重建通常几何更稳定，但采集和 COLMAP 位姿估计成本更高。",
            "- 文本到 3D 便于快速生成虚拟资产，但几何尺度和纹理细节需要人工校准。",
            "- 单图到 3D 输入成本最低，但背面几何和遮挡区域更依赖先验，跨视角一致性较弱。",
            "- 当前轻量渲染器使用统一点云表示；正式结果可替换为 Blender 或 3DGS 原生混合渲染。",
        ]
    )

    output_path = config.output_dir / "metrics_summary.md"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path

