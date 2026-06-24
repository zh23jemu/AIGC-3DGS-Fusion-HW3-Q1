from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ProjectConfig:
    """项目配置的轻量封装。

    题目一会同时涉及真实训练工具和本仓库内置的轻量验证链路，因此这里不把
    YAML 强行展开成大量固定字段，而是保留原始字典，并提供最常用路径的便捷
    属性。这样后续替换真实 3DGS、threestudio 或 Zero123 产物时，不需要改代码。
    """

    path: Path
    data: dict[str, Any]

    @property
    def root(self) -> Path:
        """返回配置文件所在仓库根目录。

        当前配置位于 `configs/` 下，因此根目录是配置文件父目录的上一级。
        如果未来把配置放到其他位置，也可以在调用端传入绝对路径来保持稳定。
        """

        return self.path.resolve().parent.parent

    @property
    def output_dir(self) -> Path:
        """返回输出目录的绝对路径，并兼容配置里的相对路径。"""

        output = Path(self.data["project"]["output_dir"])
        return output if output.is_absolute() else self.root / output

    def resolve(self, value: str | Path) -> Path:
        """将配置中的相对路径解析到仓库根目录下。"""

        path = Path(value)
        return path if path.is_absolute() else self.root / path


def load_config(config_path: str | Path) -> ProjectConfig:
    """读取 YAML 配置并执行必要的结构校验。

    参数:
        config_path: 配置文件路径，可以是相对路径或绝对路径。

    返回:
        ProjectConfig: 带有原始配置和路径解析能力的配置对象。
    """

    path = Path(config_path).resolve()
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)

    required_sections = ["project", "inputs", "prompts", "external_tools", "assets", "render"]
    missing = [section for section in required_sections if section not in data]
    if missing:
        raise ValueError(f"配置缺少必要章节: {', '.join(missing)}")

    return ProjectConfig(path=path, data=data)

