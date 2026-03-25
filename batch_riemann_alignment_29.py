import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pyriemann.estimation import Covariances
from pyriemann.utils.mean import mean_covariance

from strict_riemann_alignment import (
    align_to_identity,
    compute_shared_space,
    extract_epochs_eeg,
    extract_epochs_nirs,
    load_cells,
    paired_by_label_order,
)


def process_one_subject(root: Path, sid: int):
    subject_name = f"subject {sid:02d}"
    eeg_root = root / "EEG-fNIRs异构数据集" / "EEG_01-29" / subject_name / "with occular artifact"
    nirs_root = root / "EEG-fNIRs异构数据集" / "NIRS_01-29" / subject_name

    eeg_cnt = eeg_root / "cnt.mat"
    eeg_mrk = eeg_root / "mrk.mat"
    nirs_cnt = nirs_root / "cnt.mat"
    nirs_mrk = nirs_root / "mrk.mat"

    for p in [eeg_cnt, eeg_mrk, nirs_cnt, nirs_mrk]:
        if not p.exists():
            return {
                "subject": sid,
                "status": "missing_file",
                "reason": str(p),
            }

    cnt_eeg = load_cells(str(eeg_cnt), "cnt")
    mrk_eeg = load_cells(str(eeg_mrk), "mrk")
    cnt_nirs = load_cells(str(nirs_cnt), "cnt")
    mrk_nirs = load_cells(str(nirs_mrk), "mrk")

    sessions_mi = [0, 2, 4]
    eeg_map = {16: 0, 32: 1}
    nirs_map = {1: 0, 2: 1}

    eeg_epochs, eeg_labels = extract_epochs_eeg(
        cnt_eeg, mrk_eeg, sessions=sessions_mi, label_map=eeg_map, window_s=(0.0, 10.0)
    )
    nirs_epochs, nirs_labels = extract_epochs_nirs(
        cnt_nirs, mrk_nirs, sessions=sessions_mi, label_map=nirs_map, window_s=(2.0, 12.0), baseline_s=(-2.0, 0.0)
    )

    eeg_epochs, nirs_epochs, labels = paired_by_label_order(eeg_epochs, eeg_labels, nirs_epochs, nirs_labels)
    if len(labels) < 20:
        return {
            "subject": sid,
            "status": "too_few_trials",
            "paired_trials": int(len(labels)),
        }

    cov = Covariances(estimator="oas")
    cov_eeg = cov.fit_transform(eeg_epochs)
    cov_nirs = cov.fit_transform(nirs_epochs)

    mean_eeg = mean_covariance(cov_eeg, metric="riemann")
    mean_nirs = mean_covariance(cov_nirs, metric="riemann")
    aligned_eeg = align_to_identity(cov_eeg, mean_eeg)
    aligned_nirs = align_to_identity(cov_nirs, mean_nirs)

    z_eeg_pre, z_nirs_pre, z_eeg, z_nirs, corrs = compute_shared_space(aligned_eeg, aligned_nirs)

    pair_dist_pre = np.linalg.norm(z_eeg_pre - z_nirs_pre, axis=1)
    pair_dist_post = np.linalg.norm(z_eeg - z_nirs, axis=1)

    row = {
        "subject": sid,
        "status": "ok",
        "paired_trials": int(len(labels)),
        "class_0": int(np.sum(labels == 0)),
        "class_1": int(np.sum(labels == 1)),
        "mean_paired_distance_pre": float(np.mean(pair_dist_pre)),
        "mean_paired_distance_post": float(np.mean(pair_dist_post)),
        "distance_reduction_ratio": float(np.mean(pair_dist_post) / np.mean(pair_dist_pre)),
        "mean_canonical_correlation": float(np.mean(corrs)),
    }
    for i, c in enumerate(corrs, start=1):
        row[f"can_corr_{i}"] = float(c)
    return row


def mean_std(values):
    arr = np.asarray(values, dtype=np.float64)
    return float(np.nanmean(arr)), float(np.nanstd(arr, ddof=1))


def write_subject_csv(rows, csv_path: Path):
    can_cols = [f"can_corr_{i}" for i in range(1, 11)]
    fieldnames = [
        "subject",
        "status",
        "paired_trials",
        "class_0",
        "class_1",
        "mean_paired_distance_pre",
        "mean_paired_distance_post",
        "distance_reduction_ratio",
        "mean_canonical_correlation",
    ] + can_cols + ["reason"]

    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            out = {k: "" for k in fieldnames}
            out.update(r)
            w.writerow(out)


def write_summary_csv(summary_items, csv_path: Path):
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["metric", "mean", "std"])
        for name, m, s in summary_items:
            w.writerow([name, f"{m:.6f}", f"{s:.6f}"])


def write_appendix_md(ok_rows, fail_rows, summary_items, md_path: Path):
    lines = []
    lines.append("# 29名被试批处理结果（可用于附录）")
    lines.append("")
    lines.append("## 总体统计（均值 ± 标准差）")
    lines.append("")
    lines.append("| 指标 | 均值 | 标准差 |")
    lines.append("|---|---:|---:|")
    for name, m, s in summary_items:
        lines.append(f"| {name} | {m:.6f} | {s:.6f} |")

    lines.append("")
    lines.append("## 分被试结果")
    lines.append("")
    header = (
        "| 被试 | 配对试次 | 类0 | 类1 | 对齐前距离 | 对齐后距离 | 距离比(post/pre) | 平均典型相关 |"
    )
    sep = "|---:|---:|---:|---:|---:|---:|---:|---:|"
    lines.append(header)
    lines.append(sep)
    for r in ok_rows:
        lines.append(
            f"| {r['subject']:02d} | {r['paired_trials']} | {r['class_0']} | {r['class_1']} | "
            f"{r['mean_paired_distance_pre']:.4f} | {r['mean_paired_distance_post']:.4f} | "
            f"{r['distance_reduction_ratio']:.4f} | {r['mean_canonical_correlation']:.4f} |"
        )

    if fail_rows:
        lines.append("")
        lines.append("## 未成功被试")
        lines.append("")
        lines.append("| 被试 | 状态 | 说明 |")
        lines.append("|---:|---|---|")
        for r in fail_rows:
            lines.append(f"| {r['subject']:02d} | {r.get('status','')} | {r.get('reason','')} |")

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def plot_summary(ok_rows, summary_items, fig_path: Path):
    pre = np.array([r["mean_paired_distance_pre"] for r in ok_rows], dtype=np.float64)
    post = np.array([r["mean_paired_distance_post"] for r in ok_rows], dtype=np.float64)
    ratio = np.array([r["distance_reduction_ratio"] for r in ok_rows], dtype=np.float64)
    mcc = np.array([r["mean_canonical_correlation"] for r in ok_rows], dtype=np.float64)

    can_mat = np.array([[r.get(f"can_corr_{i}", np.nan) for i in range(1, 11)] for r in ok_rows], dtype=np.float64)
    can_mean = np.nanmean(can_mat, axis=0)
    can_std = np.nanstd(can_mat, axis=0, ddof=1)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    pm, ps = mean_std(pre)
    qpm, qps = mean_std(post)
    axes[0, 0].bar(["pre", "post"], [pm, qpm], yerr=[ps, qps], capsize=6, color=["#4c78a8", "#f58518"])
    axes[0, 0].set_title("跨模态配对距离（均值±标准差）")
    axes[0, 0].set_ylabel("Euclidean distance")

    axes[0, 1].errorbar(np.arange(1, 11), can_mean, yerr=can_std, marker="o", capsize=4, color="#2ca02c")
    axes[0, 1].set_title("典型相关（各分量均值±标准差）")
    axes[0, 1].set_xlabel("Component")
    axes[0, 1].set_ylabel("Correlation")
    axes[0, 1].set_ylim(0, 1.0)

    axes[1, 0].hist(ratio, bins=10, color="#e45756", alpha=0.85)
    axes[1, 0].set_title("距离比 post/pre 分布")
    axes[1, 0].set_xlabel("distance_reduction_ratio")
    axes[1, 0].set_ylabel("被试数")

    axes[1, 1].boxplot([mcc], tick_labels=["mean canonical corr"])
    axes[1, 1].set_title("平均典型相关分布")
    axes[1, 1].set_ylabel("Correlation")

    n_ok = len(ok_rows)
    fig.suptitle(f"29名被试黎曼对齐批处理总体结果 (成功被试数={n_ok})")
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    fig.savefig(fig_path, dpi=200)
    plt.close(fig)


def main():
    root = Path(__file__).resolve().parent
    out_dir = root / "outputs" / "batch_29"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for sid in range(1, 30):
        try:
            r = process_one_subject(root, sid)
        except Exception as e:
            r = {"subject": sid, "status": "error", "reason": str(e)}
        rows.append(r)
        print(f"subject {sid:02d}: {r.get('status', 'unknown')}")

    ok_rows = [r for r in rows if r.get("status") == "ok"]
    fail_rows = [r for r in rows if r.get("status") != "ok"]
    if not ok_rows:
        raise RuntimeError("No valid subjects were processed successfully.")

    paired_trials_mean, paired_trials_std = mean_std([r["paired_trials"] for r in ok_rows])
    pre_mean, pre_std = mean_std([r["mean_paired_distance_pre"] for r in ok_rows])
    post_mean, post_std = mean_std([r["mean_paired_distance_post"] for r in ok_rows])
    ratio_mean, ratio_std = mean_std([r["distance_reduction_ratio"] for r in ok_rows])
    mcc_mean, mcc_std = mean_std([r["mean_canonical_correlation"] for r in ok_rows])

    summary_items = [
        ("paired_trials", paired_trials_mean, paired_trials_std),
        ("mean_paired_distance_pre", pre_mean, pre_std),
        ("mean_paired_distance_post", post_mean, post_std),
        ("distance_reduction_ratio", ratio_mean, ratio_std),
        ("mean_canonical_correlation", mcc_mean, mcc_std),
    ]

    write_subject_csv(rows, out_dir / "subject_metrics_29.csv")
    write_summary_csv(summary_items, out_dir / "summary_mean_std.csv")
    write_appendix_md(ok_rows, fail_rows, summary_items, out_dir / "appendix_table_29subjects.md")
    plot_summary(ok_rows, summary_items, out_dir / "summary_plot_mean_std.png")

    payload = {
        "n_subjects_total": 29,
        "n_subjects_ok": len(ok_rows),
        "n_subjects_failed": len(fail_rows),
        "summary": {k: {"mean": m, "std": s} for k, m, s in summary_items},
    }
    with open(out_dir / "summary_payload.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print("\n=== Batch done ===")
    print("Outputs:")
    print(out_dir / "subject_metrics_29.csv")
    print(out_dir / "summary_mean_std.csv")
    print(out_dir / "appendix_table_29subjects.md")
    print(out_dir / "summary_plot_mean_std.png")
    print(out_dir / "summary_payload.json")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
