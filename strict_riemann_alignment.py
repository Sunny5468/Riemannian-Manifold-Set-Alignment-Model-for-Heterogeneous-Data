import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from pyriemann.utils.mean import mean_covariance
from scipy.linalg import fractional_matrix_power
from scipy.signal import butter, sosfiltfilt
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# ==============================
# 板块1：基础信号处理工具
# 这部分在做什么：
# 统一提供带通滤波函数，供 EEG 与 fNIRS 的预处理复用。
# ==============================
def butter_bandpass(data, fs, low, high, order=4):
    nyq = 0.5 * fs
    low_n = max(low / nyq, 1e-6)
    high_n = min(high / nyq, 0.999)
    if high_n <= low_n:
        raise ValueError(f"Invalid band: low={low}, high={high}, fs={fs}")
    sos = butter(order, [low_n, high_n], btype="band", output="sos")
    return sosfiltfilt(sos, data, axis=0)


# ==============================
# 板块2：MAT 文件读取工具
# 这部分在做什么：
# 读取 BBCI 风格 .mat 的 cell 结构，并统一返回 Python list，
# 便于后续按 session 逐段处理。
# ==============================
def load_cells(mat_path, key):
    m = sio.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    arr = m[key]
    if isinstance(arr, np.ndarray):
        return list(arr.ravel())
    return [arr]


# ==============================
# 板块3：SPD 流形中心化
# 这部分在做什么：
# 将协方差矩阵通过 C_bar^{-1/2} C C_bar^{-1/2} 对齐到单位中心，
# 用于降低模态内分布偏移。
# ==============================
def align_to_identity(cov_matrices, mean_matrix):
    mean_inv_sqrt = fractional_matrix_power(mean_matrix, -0.5)
    out = np.empty_like(cov_matrices)
    for i in range(cov_matrices.shape[0]):
        out[i] = mean_inv_sqrt @ cov_matrices[i] @ mean_inv_sqrt
    return out


# ==============================
# 板块4：EEG 试次切片与预处理
# 这部分在做什么：
# 按事件时间戳切出 EEG trial，做 1-40Hz 滤波与标准化，
# 最终输出形状为 (trials, channels, timepoints) 的 epoch 张量。
# ==============================
def extract_epochs_eeg(cnt_cells, mrk_cells, sessions, label_map, window_s=(0.0, 10.0)):
    epochs = []
    labels = []
    for s in sessions:
        cnt = cnt_cells[s]
        mrk = mrk_cells[s]

        x = np.asarray(cnt.x, dtype=np.float64)
        fs = float(cnt.fs)
        times_ms = np.asarray(mrk.time, dtype=np.float64).ravel()
        desc = np.asarray(mrk.event.desc, dtype=np.int32).ravel()

        for t_ms, d in zip(times_ms, desc):
            if d not in label_map:
                continue
            onset = int(round(t_ms * fs / 1000.0))
            start = onset + int(round(window_s[0] * fs))
            end = onset + int(round(window_s[1] * fs))
            if start < 0 or end > x.shape[0] or end <= start:
                continue

            ep = x[start:end, :]
            ep = butter_bandpass(ep, fs=fs, low=1.0, high=40.0)
            ep = (ep - ep.mean(axis=0, keepdims=True)) / (ep.std(axis=0, keepdims=True) + 1e-8)
            epochs.append(ep.T)
            labels.append(label_map[d])

    return np.asarray(epochs), np.asarray(labels, dtype=np.int32)


# ==============================
# 板块5：fNIRS 试次切片与预处理
# 这部分在做什么：
# 按事件切片后先做 baseline 参考，再将光强转为光密度(OD)，
# 随后做 0.01-0.2Hz 滤波与标准化，得到可用于协方差建模的 trial。
# ==============================
def extract_epochs_nirs(
    cnt_cells,
    mrk_cells,
    sessions,
    label_map,
    window_s=(2.0, 12.0),
    baseline_s=(-2.0, 0.0),
):
    epochs = []
    labels = []
    for s in sessions:
        cnt = cnt_cells[s]
        mrk = mrk_cells[s]

        x = np.asarray(cnt.x, dtype=np.float64)
        fs = float(cnt.fs)
        times_ms = np.asarray(mrk.time, dtype=np.float64).ravel()
        desc = np.asarray(mrk.event.desc, dtype=np.int32).ravel()

        for t_ms, d in zip(times_ms, desc):
            if d not in label_map:
                continue

            onset = int(round(t_ms * fs / 1000.0))
            start = onset + int(round(window_s[0] * fs))
            end = onset + int(round(window_s[1] * fs))
            b_start = onset + int(round(baseline_s[0] * fs))
            b_end = onset + int(round(baseline_s[1] * fs))
            if b_start < 0 or b_end > x.shape[0] or b_end <= b_start:
                continue
            if start < 0 or end > x.shape[0] or end <= start:
                continue

            baseline = x[b_start:b_end, :]
            signal = x[start:end, :]

            # Optical Density transform for fNIRS intensity.
            i0 = np.mean(np.clip(baseline, 1e-8, None), axis=0, keepdims=True)
            od = -np.log(np.clip(signal, 1e-8, None) / i0)
            od = butter_bandpass(od, fs=fs, low=0.01, high=0.2)
            od = (od - od.mean(axis=0, keepdims=True)) / (od.std(axis=0, keepdims=True) + 1e-8)

            epochs.append(od.T)
            labels.append(label_map[d])

    return np.asarray(epochs), np.asarray(labels, dtype=np.int32)


# ==============================
# 板块6：跨模态 trial 配对
# 这部分在做什么：
# 对 EEG 与 fNIRS 按顺序进行同长度截断，并检查标签一致性；
# 若存在错配，仅保留同标签配对样本。
# ==============================
def paired_by_label_order(eeg_epochs, eeg_labels, nirs_epochs, nirs_labels):
    n = min(len(eeg_labels), len(nirs_labels))
    eeg_epochs = eeg_epochs[:n]
    nirs_epochs = nirs_epochs[:n]
    eeg_labels = eeg_labels[:n]
    nirs_labels = nirs_labels[:n]

    if not np.array_equal(eeg_labels, nirs_labels):
        idx_keep = np.where(eeg_labels == nirs_labels)[0]
        eeg_epochs = eeg_epochs[idx_keep]
        nirs_epochs = nirs_epochs[idx_keep]
        labels = eeg_labels[idx_keep]
    else:
        labels = eeg_labels
    return eeg_epochs, nirs_epochs, labels


# ==============================
# 板块7：共享潜在空间构建
# 这部分在做什么：
# 先各自做切空间映射，再分别 PCA 降维，最后通过 CCA 学习共享子空间，
# 并输出每个典型分量的相关系数用于量化对齐质量。
# ==============================
def compute_shared_space(aligned_cov_eeg, aligned_cov_nirs, n_components_pre=20, n_components_cca=10):
    ts_eeg = TangentSpace(metric="riemann")
    ts_nirs = TangentSpace(metric="riemann")
    x_eeg = ts_eeg.fit_transform(aligned_cov_eeg)
    x_nirs = ts_nirs.fit_transform(aligned_cov_nirs)

    pre_dim = min(n_components_pre, x_eeg.shape[1], x_nirs.shape[1], x_eeg.shape[0] - 1)
    if pre_dim < 2:
        raise ValueError("Not enough samples/features for strict shared-space analysis.")

    z_eeg_pre = PCA(n_components=pre_dim, random_state=0).fit_transform(StandardScaler().fit_transform(x_eeg))
    z_nirs_pre = PCA(n_components=pre_dim, random_state=0).fit_transform(StandardScaler().fit_transform(x_nirs))

    cca_dim = min(n_components_cca, pre_dim, z_eeg_pre.shape[0] - 1)
    cca = CCA(n_components=cca_dim, max_iter=2000)
    z_eeg, z_nirs = cca.fit_transform(z_eeg_pre, z_nirs_pre)

    corrs = []
    for i in range(cca_dim):
        c = np.corrcoef(z_eeg[:, i], z_nirs[:, i])[0, 1]
        corrs.append(float(c))
    return z_eeg_pre, z_nirs_pre, z_eeg, z_nirs, np.asarray(corrs)


# ==============================
# 板块8：散点图绘制工具
# 这部分在做什么：
# 在二维嵌入空间中同时绘制 EEG / NIRS 及两类标签，
# 用于直观看对齐前后分布是否趋近。
# ==============================
def scatter_modalities(ax, emb_a, emb_b, labels, title):
    color_map = {0: "#1f77b4", 1: "#d62728"}
    for cls in sorted(np.unique(labels)):
        idx = np.where(labels == cls)[0]
        ax.scatter(emb_a[idx, 0], emb_a[idx, 1], c=color_map[int(cls)], marker="o", alpha=0.75, s=24, label=f"EEG class {cls}")
        ax.scatter(emb_b[idx, 0], emb_b[idx, 1], c=color_map[int(cls)], marker="^", alpha=0.75, s=24, label=f"NIRS class {cls}")
    ax.set_title(title)
    ax.set_xlabel("Dim-1")
    ax.set_ylabel("Dim-2")


# ==============================
# 板块9：主流程入口
# 这部分在做什么：
# 串联完整流程：
# 1) 读取数据
# 2) 提取 EEG/fNIRS epoch
# 3) 构建协方差并做黎曼中心化
# 4) 进入共享空间并计算指标
# 5) 保存图像与 JSON 结果
# ==============================
def main():
    # 9.1 路径配置：定位当前工程目录与 EEG/NIRS 数据目录。
    root = Path(__file__).resolve().parent
    eeg_root = root / "EEG-fNIRs异构数据集" / "EEG_01-29" / "subject 01" / "with occular artifact"
    nirs_root = root / "EEG-fNIRs异构数据集" / "NIRS_01-29" / "subject 01"

    # 9.2 数据读取：加载每个 session 的 cnt/mrk 结构。
    cnt_eeg = load_cells(str(eeg_root / "cnt.mat"), "cnt")
    mrk_eeg = load_cells(str(eeg_root / "mrk.mat"), "mrk")
    cnt_nirs = load_cells(str(nirs_root / "cnt.mat"), "cnt")
    mrk_nirs = load_cells(str(nirs_root / "mrk.mat"), "mrk")

    # Dataset A (MI) sessions are 1,3,5 in 1-based indexing.
    sessions_mi = [0, 2, 4]
    eeg_map = {16: 0, 32: 1}
    nirs_map = {1: 0, 2: 1}

    # 9.3 试次提取：EEG 使用任务期窗口，fNIRS 使用考虑时延的窗口。
    eeg_epochs, eeg_labels = extract_epochs_eeg(
        cnt_eeg, mrk_eeg, sessions=sessions_mi, label_map=eeg_map, window_s=(0.0, 10.0)
    )
    nirs_epochs, nirs_labels = extract_epochs_nirs(
        cnt_nirs, mrk_nirs, sessions=sessions_mi, label_map=nirs_map, window_s=(2.0, 12.0), baseline_s=(-2.0, 0.0)
    )

    # 9.4 跨模态配对：确保每个 EEG trial 对应一个同标签 fNIRS trial。
    eeg_epochs, nirs_epochs, labels = paired_by_label_order(eeg_epochs, eeg_labels, nirs_epochs, nirs_labels)
    if len(labels) < 20:
        raise RuntimeError(f"Too few paired trials after strict filtering: {len(labels)}")

    # 9.5 协方差与流形中心化：将两模态 trial 投到 SPD 流形并各自重心对齐。
    cov = Covariances(estimator="oas")
    cov_eeg = cov.fit_transform(eeg_epochs)
    cov_nirs = cov.fit_transform(nirs_epochs)

    mean_eeg = mean_covariance(cov_eeg, metric="riemann")
    mean_nirs = mean_covariance(cov_nirs, metric="riemann")
    aligned_eeg = align_to_identity(cov_eeg, mean_eeg)
    aligned_nirs = align_to_identity(cov_nirs, mean_nirs)

    # 9.6 共享空间学习：切空间 -> PCA -> CCA，并返回典型相关系数。
    z_eeg_pre, z_nirs_pre, z_eeg, z_nirs, corrs = compute_shared_space(aligned_eeg, aligned_nirs)

    # 9.7 量化指标：比较共享空间前后的配对距离变化。
    pair_dist_pre = np.linalg.norm(z_eeg_pre - z_nirs_pre, axis=1)
    pair_dist_post = np.linalg.norm(z_eeg - z_nirs, axis=1)

    # 9.8 输出目录准备。
    out_dir = root / "outputs"
    out_dir.mkdir(exist_ok=True)

    # 9.9 指标汇总：保存可复现实验指标 JSON。
    metrics = {
        "paired_trials": int(len(labels)),
        "class_balance": {"class_0": int(np.sum(labels == 0)), "class_1": int(np.sum(labels == 1))},
        "mean_paired_distance_pre": float(np.mean(pair_dist_pre)),
        "mean_paired_distance_post": float(np.mean(pair_dist_post)),
        "distance_reduction_ratio": float(np.mean(pair_dist_post) / np.mean(pair_dist_pre)),
        "canonical_correlations": [float(x) for x in corrs],
        "mean_canonical_correlation": float(np.mean(corrs)),
    }

    with open(out_dir / "riemann_alignment_metrics_subject01_mi.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    # 9.10 可视化输出：
    # 左上: 对齐前散点；右上: 共享空间后散点；
    # 左下: 配对距离箱线图；右下: 典型相关柱状图。
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    scatter_modalities(axes[0, 0], z_eeg_pre, z_nirs_pre, labels, "Before shared-space (PCA-pre)")
    scatter_modalities(axes[0, 1], z_eeg, z_nirs, labels, "After shared-space (CCA)")

    axes[1, 0].boxplot([pair_dist_pre, pair_dist_post], tick_labels=["pre", "post"])
    axes[1, 0].set_title("Paired cross-modal distance")
    axes[1, 0].set_ylabel("Euclidean distance")

    axes[1, 1].bar(np.arange(1, len(corrs) + 1), corrs, color="#2ca02c")
    axes[1, 1].set_ylim(0, 1.0)
    axes[1, 1].set_title("Canonical correlations")
    axes[1, 1].set_xlabel("Component")
    axes[1, 1].set_ylabel("Correlation")

    handles, labels_leg = axes[0, 1].get_legend_handles_labels()
    uniq = dict(zip(labels_leg, handles))
    axes[0, 1].legend(uniq.values(), uniq.keys(), loc="best", fontsize=8)

    fig.suptitle("Strict Riemannian Alignment on Subject 01 (Dataset A: MI)")
    fig.tight_layout(rect=[0, 0.02, 1, 0.97])
    fig.savefig(out_dir / "riemann_alignment_subject01_mi.png", dpi=180)
    plt.close(fig)

    # 9.11 控制台打印：方便命令行快速查看结果路径与核心指标。
    print("Saved figure:", out_dir / "riemann_alignment_subject01_mi.png")
    print("Saved metrics:", out_dir / "riemann_alignment_metrics_subject01_mi.json")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
