"""Generate or continuously refresh the bilingual project tree and content index.

Run once:  python code/tools/update_project_map.py
Watch:     python code/tools/update_project_map.py --watch --interval 3
The watcher must remain running to provide continuous updates.
"""

from __future__ import annotations

import argparse
import ast
import csv
import io
import json
import os
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
from zipfile import ZipFile


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "PROJECT_MAP.md"
TEMP = ROOT / ".PROJECT_MAP.md.tmp"

# These labels describe file roles; extracted headings/definitions below track
# changing content without repeating another Markdown document verbatim.
ROLES = {
    ".gitignore": ("Git rules for local artifacts", "本地运行产物的 Git 规则"),
    "README.md": ("Entry point or directory guide", "项目入口或目录说明"),
    "patient_split.json": ("Frozen patient groups and expected counts", "固定患者分组与预期数量"),
    "STUDY_PROTOCOL.md": ("Research hypotheses, endpoints, and freeze rules", "研究假设、终点指标与冻结规则"),
    "SOURCE_DECISIONS.md": ("Paper sources and implementation deviations", "论文来源与实现差异"),
    "EXECUTION.md": ("Evidence and execution gates", "证据与执行关口"),
    "RANGPUR_COMMANDS.md": ("Separate Windows and Rangpur commands", "分开的 Windows 与 Rangpur 命令"),
    "requirements.txt": ("Python dependency ranges", "Python 依赖版本范围"),
    "__init__.py": ("Six-class package constants", "六类别包常量"),
    "audit.py": ("Train/Validation-only real NIfTI preflight", "仅针对训练与验证数据的真实 NIfTI 预检"),
    "audit_train_raw.py": ("Train-only raw MRI/mask integrity audit", "仅对训练集原始 MRI/Mask 做完整性审计"),
    "boundaries.py": ("Physical base/apex, slice and surface error analysis", "物理位置的基底/尖端、切片和表面误差分析"),
    "data.py": ("Paired loading, common-grid downsampling and mask encoding", "配对读取、共用网格降采样及标签编码"),
    "inference.py": ("2D/3D predictions restored to the input XYZ grid", "将 2D/3D 预测还原至输入 XYZ 网格"),
    "losses.py": ("Six-class cross-entropy and soft-Dice objectives", "六类别交叉熵与 Soft Dice 目标函数"),
    "metrics.py": ("Confusion counts, Dice, IoU, precision and recall", "混淆计数、Dice、IoU、精确率与召回率"),
    "arm.json": ("Reserved model-arm identity and training state", "预留的模型分组标识及训练状态"),
    "common.py": ("Shared standard U-Net convolutional blocks", "标准 U-Net 共用卷积模块"),
    "baseline_2d.py": ("Slice-wise 2D U-Net baseline", "逐切片 2D U-Net 基线"),
    "baseline_3d.py": ("Standard volumetric 3D U-Net baseline", "标准体数据 3D U-Net 基线"),
    "improved_3d.py": ("Residual 3D U-Net with deep supervision", "含残差结构及深监督的 3D U-Net"),
    "profile.py": ("Train-only GPU forward/backward capacity probe", "仅使用训练数据的 GPU 前反向容量检查"),
    "splits.py": ("Frozen patient identity and MRI/mask file pairing", "固定患者身份及 MRI/mask 文件配对"),
    "test_boundaries.py": ("Physical direction and empty-surface tests", "物理方向及空表面测试"),
    "test_data.py": ("NIfTI alignment, downsampling and label tests", "NIfTI 对齐、降采样与标签测试"),
    "test_metrics.py": ("Class-confusion and overlap metric tests", "类别混淆与重叠指标测试"),
    "test_raw_audit.py": ("Train inventory and file-hash audit tests", "训练集文件盘点与文件哈希审计测试"),
    "test_models_and_losses.py": ("Shape, gradient, supervision and inference tests", "形状、梯度、深监督与推理测试"),
    "test_splits.py": ("Patient isolation and file pairing tests", "患者隔离与文件配对测试"),
    "package_transfer.py": ("Build and verify a source-only transfer ZIP", "创建并校验仅含源码的传输包"),
    "review_g1_overlays.py": ("Render raw Train MRI and ground-truth overlays for G1 review", "绘制训练集原始 MRI 与真实标签叠加图供 G1 审查"),
    "review_g1_bladder_islands.py": ("Locate and visualize two flagged Train Bladder components", "定位并可视化两例训练集膀胱小连通区域"),
    "review_g1_train_contact_sheets.py": ("Render two-slice Train patient sheets for G1 alignment screening", "绘制训练集患者双切面图供 G1 对齐筛查"),
    "summarize_g2_train_audit.py": ("Summarize verified raw Train audit into G2 tables and plots", "将已核验的训练集原始审计汇总为 G2 表格与图"),
    "audit_g2_train_slices.py": ("Measure Train axial label coverage and MRI intensity bins on Rangpur CPU", "在 Rangpur CPU 测量训练集逐切片标签覆盖与 MRI 强度分箱"),
    "audit_g3_existing_pipeline.py": ("Audit current Train downsampling, normalization and whole-volume arrays", "审计当前训练集降采样、归一化与整卷张量"),
    "audit_g3_candidate_sampling.py": ("Compare fixed Train-only uniform and organ-aware candidate crops", "比较固定的训练集均匀与器官感知候选裁剪"),
    "summarize_g3_candidate_sampling.py": ("Verify full candidate patch records and summarize exposure", "核验完整候选 Patch 记录并汇总曝光"),
    "g3_patch_audit.json": ("Frozen candidate-crop audit design, not a training configuration", "固定的候选裁剪审计设计，非训练配置"),
    "test_sampling_audit.py": ("Candidate crop bounds and repeatability checks", "候选裁剪边界与可重复性检查"),
    "summarize_g2_slice_detail.py": ("Reconcile G2 slice and intensity CSVs with G1 and draw local plots", "将 G2 切片及强度 CSV 与 G1 核对并绘制本地图"),
    "update_project_map.py": ("Generate and watch this bilingual inventory", "生成并持续更新本双语目录"),
    "20261017_source.zip": ("Generated source-only transfer package", "生成的仅含源码传输包"),
    "PROJECT_MAP.md": ("Generated bilingual tree and file content index", "自动生成的双语文件树与内容索引"),
}


def _files_and_dirs() -> list[Path]:
    paths = [path for path in ROOT.rglob("*") if path != TEMP]
    if OUTPUT not in paths:
        paths.append(OUTPUT)
    return sorted(paths, key=lambda path: (tuple(part.casefold() for part in path.relative_to(ROOT).parts),
                                           0 if path.is_dir() else 1))


def _tree(paths: list[Path]) -> str:
    children: dict[Path, list[Path]] = {}
    for path in paths:
        children.setdefault(path.parent, []).append(path)
    for items in children.values():
        items.sort(key=lambda item: (not item.is_dir(), item.name.casefold()))
    lines = ["20261017/"]

    def descend(parent: Path, prefix: str) -> None:
        siblings = children.get(parent, [])
        for index, path in enumerate(siblings):
            last = index == len(siblings) - 1
            lines.append(prefix + ("└── " if last else "├── ") + path.name + ("/" if path.is_dir() else ""))
            if path.is_dir() and not path.is_symlink():
                descend(path, prefix + ("    " if last else "│   "))

    descend(ROOT, "")
    return "\n".join(lines)


def _read_text(path: Path, maximum: int = 100_000) -> str | None:
    if not path.is_file() or path.stat().st_size > maximum:
        return None
    try:
        return path.read_text(encoding="utf-8")
    except (UnicodeError, OSError):
        return None


def _content(path: Path) -> tuple[str, str]:
    if path == OUTPUT:
        return ("Self-entry; regenerated from current tree and file structure",
                "本文件条目；根据当前文件树和文件结构重新生成")
    if path.is_symlink():
        return (f"Symbolic link to {path.resolve()}", f"符号链接，目标为 {path.resolve()}")
    if path.is_dir():
        count = sum(1 for _ in path.iterdir())
        return (f"Directory with {count} direct entries", f"目录，含 {count} 个直接条目")
    size = path.stat().st_size
    text = _read_text(path)
    if path.suffix == ".md" and text is not None:
        headings = [line.lstrip("# ").strip() for line in text.splitlines()
                    if line.startswith("# ") or line.startswith("## ")]
        sample = "; ".join(headings[:5]) or "no headings"
        return (f"Sections: {sample}", f"章节标题：{sample}")
    if path.suffix == ".py" and text is not None:
        try:
            module = ast.parse(text)
            names = [node.name for node in module.body if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))]
            sample = ", ".join(names[:12]) or "module-level code"
            return (f"Definitions: {sample}", f"定义：{sample}")
        except SyntaxError:
            return ("Python file with syntax error", "存在语法错误的 Python 文件")
    if path.suffix == ".json" and text is not None:
        try:
            record = json.loads(text)
            if isinstance(record, dict):
                keys = ", ".join(str(key) for key in list(record)[:12])
                return (f"JSON fields: {keys}", f"JSON 字段：{keys}")
            return (f"JSON list with {len(record)} entries", f"JSON 列表，含 {len(record)} 项")
        except (ValueError, TypeError):
            return ("Invalid JSON; inspect source file", "JSON 无效；请检查源文件")
    if path.suffix in {".csv", ".tsv"} and text is not None:
        rows = list(csv.reader(io.StringIO(text), delimiter="\t" if path.suffix == ".tsv" else ","))
        columns = ", ".join(rows[0][:12]) if rows else "none"
        return (f"Table with {max(len(rows) - 1, 0)} data rows; columns: {columns}",
                f"表格含 {max(len(rows) - 1, 0)} 条数据行；列：{columns}")
    if path.suffix == ".zip":
        try:
            with ZipFile(path) as archive:
                count = len(archive.infolist())
            return (f"ZIP archive with {count} members; {size:,} bytes",
                    f"ZIP 压缩包，含 {count} 个成员；{size:,} 字节")
        except (OSError, ValueError):
            return (f"ZIP archive; {size:,} bytes; unreadable index",
                    f"ZIP 压缩包；{size:,} 字节；目录不可读取")
    if text is not None:
        lines = len(text.splitlines())
        return (f"UTF-8 text with {lines} lines; {size:,} bytes",
                f"UTF-8 文本，{lines} 行；{size:,} 字节")
    return (f"Binary or large file; {size:,} bytes", f"二进制或大文件；{size:,} 字节")


def _escape(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def render() -> str:
    paths = _files_and_dirs()
    files = [path for path in paths if not path.is_dir()]
    now = datetime.now(ZoneInfo("Australia/Brisbane")).strftime("%Y-%m-%d %H:%M:%S AEST")
    lines = [
        "# Project File Map / 项目文件总览", "",
        f"Updated / 更新时间: **{now}**", "",
        "**English:** This generated index lists every directory and file under `20261017`, including local logs and generated artifacts. The content column summarizes structure (headings, code definitions, fields or file type) rather than copying another document's prose. Run `python code/tools/update_project_map.py --watch` to keep it current while the watcher is active.", "",
        "**中文：** 本自动目录列出 `20261017` 下的全部目录和文件，包括本地日志及生成产物。内容栏概述章节、代码定义、字段或文件类型，不重复抄录其他文档。运行 `python code/tools/update_project_map.py --watch`，并保持进程运行，即可持续更新。", "",
        f"Directories / 目录: **{sum(path.is_dir() for path in paths)}** · Files / 文件: **{len(files)}**", "",
        "## Directory Tree / 目录树", "", "```text", _tree(paths), "```", "",
        "## File Contents / 文件内容概览", "",
        "| File / 文件 | English content overview | 中文内容概览 |", "|---|---|---|",
    ]
    for path in files:
        relative = path.relative_to(ROOT).as_posix()
        role_en, role_zh = ROLES.get(path.name, ("", ""))
        detail_en, detail_zh = _content(path)
        english = "; ".join(part for part in (role_en, detail_en) if part)
        chinese = "；".join(part for part in (role_zh, detail_zh) if part)
        lines.append(f"| `{_escape(relative)}` | {_escape(english)} | {_escape(chinese)} |")
    lines.extend(["", "Generated by `code/tools/update_project_map.py`; edit the source files, then regenerate this index. / 由 `code/tools/update_project_map.py` 生成；修改源文件后重新生成本目录。", ""])
    return "\n".join(lines)


def refresh() -> None:
    content = render()
    if OUTPUT.exists() and OUTPUT.read_text(encoding="utf-8") == content:
        return
    TEMP.write_text(content, encoding="utf-8")
    os.replace(TEMP, OUTPUT)


def _fingerprint() -> tuple:
    records = []
    for path in _files_and_dirs():
        if path == OUTPUT:
            continue
        stat = path.stat()
        records.append((path.relative_to(ROOT).as_posix(), path.is_dir(), stat.st_size, stat.st_mtime_ns))
    return tuple(records)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--watch", action="store_true", help="Refresh after changes while this process runs")
    parser.add_argument("--interval", type=float, default=3, help="Watcher poll interval in seconds")
    args = parser.parse_args()
    if args.interval <= 0:
        parser.error("interval must be positive")
    previous = _fingerprint()
    refresh()
    if args.watch:
        while True:
            time.sleep(args.interval)
            current = _fingerprint()
            if current != previous:
                refresh()
                previous = current
    else:
        print(OUTPUT)


if __name__ == "__main__":
    main()
