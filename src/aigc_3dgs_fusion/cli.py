from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from .config import load_config
from .demo_assets import create_demo_assets
from .metrics import write_summary
from .planner import write_command_plan
from .renderer import render_sequence

app = typer.Typer(help="HW3 题目一：3DGS 与 AIGC 多源资产融合流水线")
console = Console()


@app.command()
def make_demo_assets(config: Path = typer.Option(Path("configs/hw3_q1.yaml"), "--config", "-c")) -> None:
    """生成轻量 demo 资产，用于无真实数据时验证完整流程。"""

    cfg = load_config(config)
    created = create_demo_assets(cfg)
    for path in created:
        console.print(f"[green]已生成[/green] {path}")


@app.command()
def plan(config: Path = typer.Option(Path("configs/hw3_q1.yaml"), "--config", "-c")) -> None:
    """输出 COLMAP/3DGS/threestudio/Zero123 的真实训练命令计划。"""

    cfg = load_config(config)
    path = write_command_plan(cfg)
    console.print(f"[green]命令计划已写入[/green] {path}")


@app.command()
def render(config: Path = typer.Option(Path("configs/hw3_q1.yaml"), "--config", "-c")) -> None:
    """读取配置中的资产并渲染多视角融合序列。"""

    cfg = load_config(config)
    paths = render_sequence(cfg)
    console.print(f"[green]已渲染[/green] {len(paths) - 1} 帧")
    console.print(f"[green]GIF[/green] {paths[-1]}")


@app.command()
def summarize(config: Path = typer.Option(Path("configs/hw3_q1.yaml"), "--config", "-c")) -> None:
    """生成报告可用的轻量质量摘要。"""

    cfg = load_config(config)
    path = write_summary(cfg)
    console.print(f"[green]摘要已写入[/green] {path}")


@app.command()
def demo(config: Path = typer.Option(Path("configs/hw3_q1.yaml"), "--config", "-c")) -> None:
    """一键跑通 demo 资产、融合渲染、命令计划和质量摘要。"""

    cfg = load_config(config)
    create_demo_assets(cfg)
    render_sequence(cfg)
    plan_path = write_command_plan(cfg)
    summary_path = write_summary(cfg)
    console.print("[green]题目一轻量闭环已完成[/green]")
    console.print(f"命令计划: {plan_path}")
    console.print(f"质量摘要: {summary_path}")

