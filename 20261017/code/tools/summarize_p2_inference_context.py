"""Verify downloaded P2 records and produce an exploratory six-class review."""
from __future__ import annotations
import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "code"))
from hipmri_rewrite.metrics import overlap
from tools.evaluate_g6_test import sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=PROJECT /
                        "logs/validation/inference_context/p2_four_cases_20261020/manifest.json")
    args = parser.parse_args()
    manifest = args.manifest.resolve()
    data = json.loads(manifest.read_text(encoding="utf-8"))
    checks = json.loads((manifest.parent / "checksums.json").read_text(encoding="utf-8"))
    if checks.get("manifest.json") != sha256(manifest):
        raise ValueError("Manifest checksum differs from its transfer receipt")
    for name, digest in checks.items():
        if Path(name).name != name or sha256(manifest.parent / name) != digest:
            raise ValueError(f"Unsafe name or altered measurement: {name}")
    expected = {(arm, case) for arm in ("standard_3d", "improved_3d")
                for case in ("K018_Week4", "B040_Week4", "J026_Week4", "K008_Week0")}
    actual = {(r["model"], r["case"]) for r in data["cases"]}
    if (not data["complete"] or data["split"] != "validation" or actual != expected
            or len(data["cases"]) != 8 or not all(r["full_confusion_reproduced"] for r in data["cases"])):
        raise ValueError("P2 paired cohort is incomplete or full inference was not reproduced")
    if (sha256(PROJECT / "code/config/p2_inference_context.json") != data["config_sha256"]
            or any(sha256(PROJECT / name) != digest for name, digest in data["sources"].items())):
        raise ValueError("P2 local sources differ from the measured protocol")
    # Keep the immutable measured run ID in logs; presentation uses a topic name.
    output = PROJECT / "results/diagnostics/inference_context/context_sensitivity_20261020"
    output.mkdir(parents=True, exist_ok=True)
    if (output / "CONTEXT_SENSITIVITY_REVIEW_ZH.md").exists():
        raise FileExistsError("P2 summary already exists; inspect before regenerating")
    fields = ("model", "case", "protocol", "class", "dice", "iou", "precision", "recall")
    with (output / "six_class_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        for row in data["cases"]:
            for protocol, measurement in row["protocols"].items():
                for c, label in enumerate(data["classes"]):
                    writer.writerow({"model": row["model"], "case": row["case"],
                                     "protocol": protocol, "class": label,
                                     **{key: measurement["overlap"][key][c]
                                        for key in ("dice", "iou", "precision", "recall")}})
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), layout="constrained")
    for ax, arm in zip(axes, ("standard_3d", "improved_3d")):
        for offset, protocol, color in ((-0.18, "full", "#246391"), (0.18, "sliding", "#c36b29")):
            pooled = sum(np.asarray(r["protocols"][protocol]["confusion"], dtype=np.int64)
                         for r in data["cases"] if r["model"] == arm)
            dice = overlap(pooled)["dice"]
            ax.bar(np.arange(6)+offset, [np.nan if v is None else v for v in dice],
                   width=0.36, label=protocol, color=color)
        ax.set_xticks(np.arange(6), data["classes"], rotation=35, ha="right")
        ax.set_ylim(0, 1); ax.set_ylabel("Pooled Dice across four selected cases")
        ax.set_title(arm); ax.legend()
    fig.suptitle("P2 exploratory Validation diagnostic — unchanged weights; not model selection")
    for suffix in ("png", "svg"):
        fig.savefig(output / f"six_class_context_comparison.{suffix}", dpi=170)
    plt.close(fig)
    lines = ["# P2 固定权重推理上下文诊断", "",
             "四个预先指定的 Validation 病例、两个未改动的第 50 轮模型。所有整卷混淆矩阵先复现 G5；没有训练或读取 Test 数组。",
             "这四例是机制探索，不代表完整验证集，不能据此选最终模型。", "",
             "## Body 误检与小器官变化", "",
             "| 模型 / 病例 | Body HD95：整卷 → 滑窗（mm） | 远处背景误判 Body：整卷 → 滑窗 | 三小器官 Dice 变化（Bladder / Rectum / Prostate） |",
             "|---|---|---|---|"]
    def number(v):
        return "NA" if v is None else f"{v:.3f}"
    for row in data["cases"]:
        full, slide = (row["protocols"][p] for p in ("full", "sliding"))
        changes = [None if full["overlap"]["dice"][c] is None or slide["overlap"]["dice"][c] is None
                   else slide["overlap"]["dice"][c]-full["overlap"]["dice"][c] for c in (3, 4, 5)]
        lines.append(f"| {row['model']} / {row['case']} | "
                     f"{number(full['body_surface']['hd95_mm'])} → {number(slide['body_surface']['hd95_mm'])} | "
                     f"{full['background_to_body_at_least_20mm']} → {slide['background_to_body_at_least_20mm']} | "
                     + " / ".join(number(v) for v in changes) + " |")
    lines += ["", "## 如何解释", "",
              "- 六类完整 Dice/IoU/Precision/Recall 见同目录 CSV；图中的 pooled Dice 不是患者均衡选模分数。",
              "- 原始逐病例 JSON 同时保存远处误检、体积边缘物理距离、六邻域连通域及同一组体素上的 head margin。",
              "- margin 图为 Body−Background 原始 logit，不是概率；类别竞争还须查看 Body−最大其他类的 margin。",
              "- 若 Improved 远处误检下降且小器官无明显损伤，而 Standard 变化较小，可支持扩大到全部 38 例的必要性；还不能把作用单独归因于 InstanceNorm。",
              "- 若改善很弱或小器官受损，按 P1 转向已有审计的 G7 单因素训练；不继续扫描滑窗参数。",
              "- 需要逐例检查对照病例和失败病例，不用四例总体均值掩盖损伤。", "",
              f"来源：`{manifest.relative_to(PROJECT)}`；SHA-256 `{sha256(manifest)}`。"]
    (output / "CONTEXT_SENSITIVITY_REVIEW_ZH.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print(output / "CONTEXT_SENSITIVITY_REVIEW_ZH.md")


if __name__ == "__main__":
    main()
