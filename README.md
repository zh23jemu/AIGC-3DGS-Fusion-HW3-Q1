# AIGC-3DGS-Fusion

本仓库用于完成《HW3_深度学习与空间智能.pdf》题目一：基于 3DGS 与 AIGC 的多源资产生成与真实场景融合。工程提供两条路径：

- 真实链路：编排 COLMAP、3D Gaussian Splatting、threestudio、Zero123 等外部工具，适合在 GPU/Slurm 环境完成正式训练。
- 轻量链路：生成 demo 3D 资产并用 Python 融合渲染器输出多视角帧，适合先验证目录、配置、空间布局和报告图表流程。

## 环境准备

```powershell
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt
& .\.venv\Scripts\python.exe -m pip install -e .
```

如果需要运行真实训练，请额外安装并配置：

- COLMAP
- 3D Gaussian Splatting 官方或兼容实现
- threestudio
- Zero123 / Zero123++ / Wonder3D 等单图到 3D 工具

## 快速运行

生成轻量 demo 资产、渲染多视角融合结果，并输出质量摘要：

```powershell
& .\.venv\Scripts\python.exe -m aigc_3dgs_fusion demo --config configs/hw3_q1.yaml
```

分步骤运行：

```powershell
& .\.venv\Scripts\python.exe -m aigc_3dgs_fusion make-demo-assets --config configs/hw3_q1.yaml
& .\.venv\Scripts\python.exe -m aigc_3dgs_fusion render --config configs/hw3_q1.yaml
& .\.venv\Scripts\python.exe -m aigc_3dgs_fusion summarize --config configs/hw3_q1.yaml
```

生成真实训练命令计划：

```powershell
& .\.venv\Scripts\python.exe -m aigc_3dgs_fusion plan --config configs/hw3_q1.yaml
```

命令计划会写入 `outputs/command_plan.md`，可直接复制到本地、AutoDL、Colab 或 Slurm 作业脚本中调整运行。

## 数据准备

建议按以下结构放置真实数据：

```text
data/
  raw/
    object_a_multiview/images/
    object_c_single_view/object_c.png
    background_scene/images/
```

配置文件 `configs/hw3_q1.yaml` 中包含以下关键字段：

- `inputs.object_a_images`：真实物体 A 的多视角图片目录。
- `inputs.object_c_image`：真实物体 C 的单张输入图片。
- `inputs.background_images`：背景场景的多视角图片目录。
- `prompts.object_b`：物体 B 文本到 3D 的 Prompt。
- `assets.*.transform`：A、B、C 三个物体插入背景时的位置、旋转和缩放。
- `render`：输出帧数、分辨率、相机半径和俯仰角。

## Slurm 运行

本仓库提供以下脚本：

- `slurm/train_object_a_3dgs.sbatch`：真实物体 A 的 COLMAP + 3DGS 重建。
- `slurm/train_background_3dgs.sbatch`：背景场景的 COLMAP + 3DGS 重建。
- `slurm/generate_object_b_threestudio.sbatch`：物体 B 文本到 3D。
- `slurm/generate_object_c_zero123.sbatch`：物体 C 单图到 3D。
- `slurm/render_fusion_demo.sbatch`：轻量融合渲染和摘要生成。

提交前请按实际集群环境修改脚本中的项目路径和外部工具路径。默认使用 `gpu` 分区、`gpo-ifv7xx` 账号和 `normal` QOS。

## 输出说明

```text
outputs/
  demo_assets/          # 轻量 demo 点云与 OBJ 资产
  renders/              # 多视角融合渲染帧与 GIF
  command_plan.md       # 真实训练命令计划
  metrics_summary.md    # 几何规模、覆盖范围和渲染结果摘要
```

正式报告建议包含：

- 三类资产生成方式对比：多视角重建、文本到 3D、单图到 3D。
- 背景 3DGS 重建与统一表示方式。
- A/B/C 三个物体插入背景的空间变换、比例和遮挡处理。
- 多视角渲染结果截图或视频。
- 训练耗时、质量指标、失败案例和改进方向。
