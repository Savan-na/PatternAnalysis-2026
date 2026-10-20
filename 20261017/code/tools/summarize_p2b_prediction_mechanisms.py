"""Generate P2 mechanism evidence from existing records or completed P2B.

No MRI/checkpoint/Test access. Original records and figures are never changed.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT = Path(__file__).resolve().parents[2]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checked(path):
    checks = json.loads((path.parent / "checksums.json").read_text(encoding="utf-8"))
    for name, digest in checks.items():
        if Path(name).name != name or sha(path.parent / name) != digest:
            raise ValueError(f"Changed measurement: {name}")
    data = json.loads(path.read_text(encoding="utf-8"))
    expected = {(a, c) for a in ("standard_3d", "improved_3d") for c in (
        "K018_Week4", "B040_Week4", "J026_Week4", "K008_Week0")}
    if (not data["complete"] or data["split"] != "validation" or len(data["cases"]) != 8
            or {(r["model"], r["case"]) for r in data["cases"]} != expected):
        raise ValueError("Incomplete paired Validation diagnostic")
    return data


def existing():
    source = PROJECT / "logs/validation/inference_context/p2_four_cases_20261020/manifest.json"
    if sha(source) != "a21a38f6bd2c312768c923138da25190ce4346b202dea9a1a61563c6d7acad68":
        raise ValueError("Existing P2 manifest differs from remote receipt")
    data = checked(source)
    output = PROJECT / "results/diagnostics/inference_context/prediction_mechanisms_20261020"
    if output.exists():
        raise FileExistsError("Evidence folder exists; inspect before regenerating")
    output.mkdir(parents=True)
    records = []
    for row in data["cases"]:
        ref = row["same_voxel_margins"]["reference_improved_full_distant_background_to_body"]
        for mode in ("full", "sliding"):
            v = row["protocols"][mode]
            m = ref["protocols"][mode]
            record = {"model": row["model"], "case": row["case"], "protocol": mode,
                      "distant_background_to_body": v["background_to_body_at_least_20mm"],
                      "body_hd95_mm": v["body_surface"]["hd95_mm"],
                      "all_body_fp_edge_distance_median_mm": v["false_body_distance_to_volume_edge_mm"]["median"],
                      "all_body_fp_to_gt_distance_p95_mm": v["false_body_distance_to_gt_body_mm"]["p95"],
                      "pred_body_component_count_6": v["body_components"]["count"],
                      "pred_body_components_without_gt_overlap": v["body_components"]["components_without_gt_body_overlap"],
                      "pred_body_voxels_in_components_without_gt_overlap": v["body_components"]["voxels_in_components_without_gt_body_overlap"],
                      "reference_voxels": ref["voxels"],
                      **{"margin_"+key: m["body_minus_background"]["combined"][key] for key in ("p05", "median", "p95")},
                      "winner_margin_p95_on_reference": m["body_minus_best_other_class"]["p95"]}
            for h in ("main", "half", "quarter"):
                record[h+"_margin_median"] = m["body_minus_background"].get(h, {}).get("median")
            records.append(record)
    with (output / "spatial_logit_evidence.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=records[0].keys()); writer.writeheader(); writer.writerows(records)
    cases = data["config"]["cases"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), layout="constrained")
    styles = (("standard_3d", "full", "#246391", "o", -.21),
              ("standard_3d", "sliding", "#246391", "s", -.07),
              ("improved_3d", "full", "#c36b29", "o", .07),
              ("improved_3d", "sliding", "#c36b29", "s", .21))
    for arm, mode, color, marker, offset in styles:
        rows = [next(r for r in records if r["case"] == c and r["model"] == arm and r["protocol"] == mode) for c in cases]
        y = np.arange(4)+offset
        med = np.array([r["margin_median"] for r in rows])
        errors = np.array([[r["margin_median"]-r["margin_p05"] for r in rows],
                           [r["margin_p95"]-r["margin_median"] for r in rows]])
        label = f"{arm}: {mode}"
        axes[0].errorbar(med, y, xerr=errors, color=color, marker=marker, linestyle="none", capsize=2, label=label)
        axes[1].plot([r["distant_background_to_body"] for r in rows], y,
                     marker=marker, color=color, linestyle="none", label=label)
    for ax in axes:
        ax.set_yticks(np.arange(4), cases); ax.invert_yaxis(); ax.grid(axis="x", alpha=.2)
    axes[0].axvline(0, color="gray", lw=.7)
    axes[0].set_xlabel("Body minus Background logit: median, 5th–95th percentile")
    axes[0].set_title("Same Improved-full distant-error voxels in both models")
    axes[1].set_xscale("symlog", linthresh=10)
    axes[1].set_xlim(0, 100000)
    axes[1].set_xticks((0, 10, 100, 1000, 10000, 100000), ("0", "10", "100", "1,000", "10,000", "100,000"))
    axes[1].set_xlabel("Own distant Background→Body voxels (symlog; zero shown)")
    axes[1].set_title("Errors ≥20 mm from GT Body; descriptive four-case sample")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=9, loc="lower center", bbox_to_anchor=(.5, -.07), ncol=4)
    fig.suptitle("P2 existing evidence — fixed weights; margins are not calibrated confidence")
    for suffix in ("png", "svg"):
        path = output / f"same_voxel_logit_evidence.{suffix}"
        fig.savefig(path, dpi=170, bbox_inches="tight")
        if suffix == "svg":
            path.write_bytes(("\n".join(line.rstrip() for line in path.read_text(encoding="utf-8").splitlines())+"\n").encode())
    plt.close(fig)
    lines = ["# P2：远处误检与预测机制——现有证据", "",
             "本页仅复算作业 677414 的四例固定权重记录，没有新增 GPU 测量或训练。P2B 统计回放、六分类置信度与固定连通域过滤尚未运行。", "",
             "## A. 推理等价性", "",
             "原 G5 两模型共用配对降采样、整卷非零 Z-score、XYZ→DHW、高端补零、argmax 和反向裁剪；原流程没有滑窗/overlap/blending。两模型输出均与输入 padded grid 同大小，Improved 先将辅助头上采样再相加。P2 八组原整卷混淆矩阵已复现。没有找到已证实的共同错轴、错网格或漏裁剪缺陷；相同适配器不排除网络对上下文和 padding 的不同响应。", "",
             "## B–D. 同体素 logits、几何与辅助头", "",
             "以下 margin 在同一组 Improved 原整卷远处误检体素上计算，不能把图中的百分位范围当作置信区间。", "",
             "| 病例 | Improved margin 中位数：整卷→滑窗 | Standard：整卷→滑窗 | Improved 整卷 winner margin 的 P95 |",
             "|---|---|---|---|"]
    for c in cases:
        values = {(r["model"], r["protocol"]): r for r in records if r["case"] == c}
        i, s = "improved_3d", "standard_3d"
        lines.append(f"| {c} | {values[i,'full']['margin_median']:.3f}→{values[i,'sliding']['margin_median']:.3f} | "
                     f"{values[s,'full']['margin_median']:.3f}→{values[s,'sliding']['margin_median']:.3f} | {values[i,'full']['winner_margin_p95_on_reference']:.3f} |")
    lines += ["", "错误中存在明显大于微小决策边界偏差的 margin，不能把全部误检解释成接近零的数值抖动。但现有记录没有全部六类概率或逐体素 logits，不能据此报告 softmax≥0.9 的比例或校准置信度。", "",
              "| 模型 / 病例 | 整卷全部 Body FP 到体积边缘的中位距离（mm） | 整卷不接触 GT Body 的预测 Body 连通域数 |",
              "|---|---:|---:|"]
    for r in records:
        if r["protocol"] == "full":
            lines.append(f"| {r['model']} / {r['case']} | {r['all_body_fp_edge_distance_median_mm']:.3f} | {r['pred_body_components_without_gt_overlap']} |")
    lines += ["", "此距离来自全部 Body FP，不限于 Background 或≥20 mm 子集；旧指标以外层体素中心为边缘，不能和新 P2B 的体素面距离直接混合。GT 不接触连通域是评估描述，不能用来定义推理时过滤规则。", "",
              "三个 head 只对 logits 总和计算 loss；梯度传到所有 head 不代表每个 head 已独立学成分割器。quarter 在不同病例既有抵消也有助长 Body 偏差的表现。主头单独推理不是重训消融，禁止据此直接删除辅助监督。", "",
              "## 根因证据排序", "",
              "1. **高把握的现象**：当前权重与推理上下文存在交互；Improved 远处误检降低，但部分器官及 K018 HD95 退步，不采用滑窗替换。",
              "2. **机制有证据，归因未闭合**：多头共适应与归一化范围均可能参与。现有整卷/窗口对照同时改变统计、感受野和边缘条件，不能单独证明 InstanceNorm 是原因。",
              "3. **候选相关性**：G7 增加有内容角落曝光，尚无分割收益或新增曝光与真实错误的直接覆盖证据；loss 取舍和单 seed 也未排除。", "",
              "## E. 下一项与后处理边界", "",
              "已准备一次固定四例 P2B：相同窗口输入下仅回放该病例整卷 InstanceNorm 统计；先检查整卷自回放数值等价，保留参数/缓冲区哈希。并测六分类 softmax、物理空间分布和连通域范围。", 
              "另做一次无 GT 过滤：仅删除体积<100 mm³且小于最大预测 Body 连通域1%的六邻域分量，改为最优非 Body 类。它仍可能删除真组织，仅是预先固定的诊断，不是临床安全策略或默认模型。", 
              "当前不把 G7 视为最佳下一项训练的已定结论。先用统计回放区分 H4；按 P1 预设机制判据决定是否支持归一化因素，再决定单因素训练，禁止扫描窗口或过滤阈值。", "",
              "## 查看与复现", "",
              "[同体素 margin 图](same_voxel_logit_evidence.png) · [空间/logit 明细](spatial_logit_evidence.csv)",
              "代表性 MRI：[K018 Body](../context_sensitivity_20261020/K018_Week4_improved_3d_body.png)、[B040 Body](../context_sensitivity_20261020/B040_Week4_improved_3d_body.png)、[J026 head](../context_sensitivity_20261020/J026_Week4_head_margins.png)。原图只展示已记录的病例，不从截图估算指标。", "",
              "复现：`python -B code/tools/summarize_p2b_prediction_mechanisms.py --phase existing`；为防覆盖，已有输出目录时拒绝重建。",
              f"原始 manifest SHA-256：`{sha(source)}`。"]
    (output / "README.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print(output / "README.md")


def replay():
    source = PROJECT / "logs/validation/inference_context/p2b_prediction_mechanisms_20261020/manifest.json"
    data = checked(source)
    if (data["kind"] != "P2B_prediction_mechanisms"
            or sha(PROJECT / "code/config/p2b_prediction_mechanisms.json") != data["config_sha256"]
            or not all(v["state_dict_unchanged"] for v in data["models"].values())):
        raise ValueError("P2B protocol or model-state integrity differs")
    for name, digest in data["sources"].items():
        if sha(PROJECT / name) != digest:
            raise ValueError(f"P2B local diagnostic source changed: {name}")
    output = PROJECT / "results/diagnostics/inference_context/p2b_prediction_mechanisms_20261020"
    output.mkdir(parents=True, exist_ok=True)
    report = output / "P2B_review_ZH.md"
    if report.exists():
        raise FileExistsError("P2B review already exists; inspect it first")
    lines = ["# P2B：统计回放、置信度与空间诊断", "",
             "真实 GPU 测量，八组模型/病例记录完整；整卷及原生滑窗复现 P2。softmax 未做校准，固定统计回放和连通域过滤都不作为默认协议。", "",
             "| 模型 / 病例 | 原整卷远处 FP | 原生滑窗远处 FP | 回放远处 FP | R | 过滤删除真 Body 体素 |",
             "|---|---:|---:|---:|---:|---:|"]
    records = []
    for r in data["cases"]:
        ps = r["protocols"]
        replay_count = ps.get("replay_sliding", {}).get("background_to_body_at_least_20mm")
        ratio = r.get("mechanism_gate", {}).get("restored_far_fp_fraction_R")
        lines.append(f"| {r['model']} / {r['case']} | {ps['full']['background_to_body_at_least_20mm']} | "
                     f"{ps['native_sliding']['background_to_body_at_least_20mm']} | {replay_count if replay_count is not None else 'NA'} | "
                     f"{f'{ratio:.3f}' if ratio is not None else 'NA'} | {r['postprocessing_diagnostic']['removed_true_body_voxels']} |")
        for mode, m in ps.items():
            for region, conf in m["confidence"].items():
                records.append({"model": r["model"], "case": r["case"], "protocol": mode, "region": region,
                    "voxels": conf["voxels"], "body_probability_median": conf["body_probability"]["median"],
                    "body_probability_ge_0p9_voxels": conf["body_probability_ge_0p9"],
                    "winner_margin_median": conf["body_minus_best_other"]["median"],
                    "winner_margin_small_voxels": conf["body_winner_margin_0_to_0p25"],
                    "winner_margin_gt2_voxels": conf["body_winner_margin_gt_2"]})
    with (output / "confidence_regions.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=records[0]); writer.writeheader(); writer.writerows(records)
    gate = data["mechanism_gate"]
    lines += ["", f"预设机制关口：{gate['supporting_cases']}/4 例支持，要求至少3例；结论为 **"
              + ("支持统计来源对 P2 改善有实质贡献" if gate['supports_statistic_source_contribution'] else
                 "未达到统计来源解释大部分 P2 改善的预设条件") + "**。这不是总体泛化、临床安全或 InstanceNorm 固有优劣证明。", "",
              "[置信度明细](confidence_regions.csv)；完整六类、空间分布、统计向量及过滤影响见原始逐病例 JSON。过滤改善或损伤须独立解释，不用于重选历史模型。",
              f"来源：`{source.relative_to(PROJECT)}`；SHA-256 `{sha(source)}`。"]
    report.write_text("\n".join(lines)+"\n", encoding="utf-8")
    print(report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("existing", "replay"), default="existing")
    args = parser.parse_args()
    existing() if args.phase == "existing" else replay()
