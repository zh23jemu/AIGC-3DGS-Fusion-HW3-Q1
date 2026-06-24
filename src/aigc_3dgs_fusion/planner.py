from __future__ import annotations

from pathlib import Path

from .config import ProjectConfig


def build_command_plan(config: ProjectConfig) -> str:
    """生成真实训练链路的命令计划。

    这些命令不会在本地直接执行，因为外部工具路径、GPU 环境和数据集规模通常因
    机器而异。计划文件的作用是把题目一要求的四条生成路线明确串起来，方便复制
    到本地服务器、AutoDL、Colab 或 Slurm 脚本中。
    """

    tools = config.data["external_tools"]
    inputs = config.data["inputs"]
    prompt = config.data["prompts"]["object_b"]
    output_dir = config.output_dir

    object_a_images = config.resolve(inputs["object_a_images"])
    object_c_image = config.resolve(inputs["object_c_image"])
    background_images = config.resolve(inputs["background_images"])
    gs_root = config.resolve(tools["gaussian_splatting_root"])
    threestudio_root = config.resolve(tools["threestudio_root"])
    zero123_root = config.resolve(tools["zero123_root"])

    return f"""# HW3 题目一真实训练命令计划

## 1. 物体 A：多视角图片 -> COLMAP -> 3DGS

```bash
mkdir -p {output_dir / "object_a_3dgs"}
{tools["colmap"]} automatic_reconstructor \\
  --workspace_path {output_dir / "object_a_colmap"} \\
  --image_path {object_a_images} \\
  --dense 0

cd {gs_root}
python convert.py -s {output_dir / "object_a_colmap"}
python train.py -s {output_dir / "object_a_colmap"} -m {output_dir / "object_a_3dgs"}
python render.py -m {output_dir / "object_a_3dgs"}
```

## 2. 物体 B：文本 Prompt -> threestudio SDS -> Mesh/点云

```bash
cd {threestudio_root}
python launch.py --config configs/prolificdreamer.yaml --train \\
  system.prompt_processor.prompt="{prompt}" \\
  trial_name=hw3_object_b_text_to_3d
```

建议训练完成后导出带纹理 Mesh，并转换为 `outputs/object_b_text_to_3d.obj` 或配置中指定的路径。

## 3. 物体 C：单张图片 -> Zero123 -> 3D

```bash
cd {zero123_root}
python run_zero123.py \\
  --input {object_c_image} \\
  --output_dir {output_dir / "object_c_zero123"}
```

如果使用 Zero123++、Wonder3D 或其他实现，请保持输入单图和输出 Mesh/点云路径与配置一致。

## 4. 背景：开源场景多视角图片 -> 3DGS

```bash
mkdir -p {output_dir / "background_3dgs"}
{tools["colmap"]} automatic_reconstructor \\
  --workspace_path {output_dir / "background_colmap"} \\
  --image_path {background_images} \\
  --dense 0

cd {gs_root}
python convert.py -s {output_dir / "background_colmap"}
python train.py -s {output_dir / "background_colmap"} -m {output_dir / "background_3dgs"}
python render.py -m {output_dir / "background_3dgs"}
```

## 5. 统一融合渲染

```bash
python -m aigc_3dgs_fusion render --config {config.path}
python -m aigc_3dgs_fusion summarize --config {config.path}
```

说明：本仓库轻量渲染器以 PLY/OBJ 为统一表示。正式报告中可以选择：

- 将 AIGC Mesh 放入 Blender，与 3DGS 背景渲染帧合成；
- 或将 Mesh/3DGS 采样成点云，在统一坐标系下进行代码级融合；
- 或使用支持 Gaussian + Mesh 混合渲染的自定义 viewer。
"""


def write_command_plan(config: ProjectConfig) -> Path:
    """把命令计划写入输出目录。"""

    output_path = config.output_dir / "command_plan.md"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(build_command_plan(config), encoding="utf-8")
    return output_path

