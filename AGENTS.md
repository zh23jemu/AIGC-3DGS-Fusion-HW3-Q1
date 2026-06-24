# AIGC-3DGS-Fusion 项目上下文

## 项目目标

完成《HW3_深度学习与空间智能.pdf》中题目一：基于 3DGS 与 AIGC 的多源资产生成与真实场景融合。项目需要覆盖真实多视角重建、文本到 3D、单图到 3D、背景 3DGS 重建、统一场景融合、多视角渲染和报告材料整理。

## 技术栈

- Python 3，本地虚拟环境 `.venv`
- 轻量验证链路：`numpy`、`Pillow`、`PyYAML`、`typer`、`rich`
- 真实训练链路：COLMAP、3D Gaussian Splatting、threestudio、Zero123 或等价单图到 3D 工具
- 集群训练：Slurm，默认 `gpu` 分区、`gpo-ifv7xx` 账号、`normal` QOS

## 当前架构

- `configs/hw3_q1.yaml`：题目一主配置，描述真实数据路径、AIGC 提示词、资产变换、渲染参数和外部工具命令。
- `src/aigc_3dgs_fusion/`：核心 Python 包，包含配置读取、轻量 demo 资产生成、统一点云/OBJ 读取、场景融合渲染、命令计划生成和指标摘要。
- `slurm/`：集群训练与渲染脚本，用于 3DGS、threestudio、Zero123 和融合渲染。
- `outputs/`：默认输出目录，保存 demo 资产、渲染帧、命令计划和摘要结果。
- Slurm 远端同步目录默认使用 `/mnt/users/xj62kv/AIGC-3DGS-Fusion`，避免与其他项目目录混用。

## 开发规范

- 修改代码前先读取相关文件，保持最小修改。
- 新增 Python 代码使用较详细中文注释，解释用途、关键参数、重要分支和输出。
- 所有 Python 命令必须使用项目 `.venv`。
- 不直接删除文件；如需清理，仅给出建议命令或移动到临时目录。
- 不把真实数据、大模型权重、checkpoint、缓存、密钥和本地环境文件提交到仓库。

## Current Status

已完成题目一的可运行工程实现：真实训练链路以命令编排和 Slurm 脚本承接，轻量验证链路以内置 demo 资产与 Python 融合渲染器保证无外部数据时可跑通。已在本地 `.venv` 中安装依赖并运行 `python -m aigc_3dgs_fusion demo --config configs/hw3_q1.yaml`，生成 36 帧融合渲染、GIF、命令计划和质量摘要。

## Recent Changes

- 新建项目级 `AGENTS.md`，记录项目目标、架构、规范和长期状态。
- 规划题目一工程结构：配置驱动、外部 3D 工具命令计划、轻量融合渲染、Slurm 集群脚本。
- 新增 `README.md`、`requirements.txt`、`environment.yml`、`pyproject.toml` 和 `.gitignore`。
- 新增 `configs/hw3_q1.yaml`，集中配置输入路径、AIGC Prompt、资产变换、渲染参数和外部工具路径。
- 新增 `src/aigc_3dgs_fusion/` Python 包，包含 demo 资产生成、PLY/OBJ 解析、统一坐标变换、CPU 点云投影渲染、命令计划生成和质量摘要。
- 新增 `slurm/` 下 5 个脚本，覆盖物体 A 3DGS、背景 3DGS、物体 B threestudio、物体 C Zero123 和轻量融合渲染。
- 本地验证产物已输出到 `outputs/demo_assets/`、`outputs/renders/`、`outputs/command_plan.md` 和 `outputs/metrics_summary.md`。
- 将 Slurm 脚本默认项目目录统一为 `/mnt/users/xj62kv/AIGC-3DGS-Fusion`。

## Next TODO

- 根据真实采集数据更新 `configs/hw3_q1.yaml` 中的输入路径与 Prompt。
- 在真实 GPU 环境中运行 COLMAP/3DGS/threestudio/Zero123，并把产物路径填入配置。
- 将生成帧、命令计划、质量摘要和报告图表整理进最终 PDF。
- 如正式报告需要更高质量视觉结果，可将当前轻量融合替换为 Blender 合成或支持 Mesh/Gaussian 混合的 viewer。

## Open Issues

- 当前仓库未包含真实多视角图片、单图输入、开源场景数据和训练权重；真实重建需要用户补充数据或在集群/云端下载数据集。
- threestudio 与 Zero123 依赖较重，Slurm 脚本提供入口，但具体环境路径需按实际安装位置调整。
- Windows PowerShell 默认编码会导致 Typer/Rich 的中文控制台输出显示乱码，但生成文件内容为 UTF-8，流程执行成功。

## Architecture Decisions

- 采用“真实训练命令编排 + 轻量可验证渲染器”的双路径设计，避免在没有大模型和数据时无法验证工程完整性。
- 统一中间表示优先使用 PLY 点云和 OBJ Mesh；真实 3DGS 背景可通过导出采样点或渲染帧进入融合链路。
- 融合渲染默认使用 CPU 端点云投影，便于快速生成多视角漫游结果；正式报告可替换为 Blender 或 3DGS 原生渲染。
- Slurm 脚本默认遵循当前集群约定：GPU 任务使用 `gpu` 分区，CPU 渲染使用 `defq` 分区，账号为 `gpo-ifv7xx`，QOS 为 `normal`。
