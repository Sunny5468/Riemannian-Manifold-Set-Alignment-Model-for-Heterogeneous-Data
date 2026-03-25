import argparse
import json
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.sans-serif"] = ["Times New Roman"]
plt.rcParams["axes.unicode_minus"] = False
import numpy as np
from pyriemann.utils.mean import mean_covariance
from scipy.linalg import fractional_matrix_power
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError:
    torch = None
    nn = None
    DataLoader = None
    TensorDataset = None

from strict_riemann_alignment import (
    extract_epochs_eeg,
    extract_epochs_nirs,
    load_cells,
    paired_by_label_order,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Torch pipeline: EEG=EEGNet, fNIRS=TCN, fusion=EEGNet+TCN+Attention"
    )
    parser.add_argument("--eeg", action="store_true", help="Run EEG single-modality branch (EEGNet)")
    parser.add_argument("--fnirs", action="store_true", help="Run fNIRS single-modality branch (TCN)")
    parser.add_argument("--concatenate", action="store_true", help="Run early fusion branch (EEGNet+TCN+Attention), no alignment")
    parser.add_argument("--riemann", action="store_true", help="Run fusion with Riemann alignment only")
    parser.add_argument("--fullalign", action="store_true", help="Run fusion with Riemann alignment + shared space")
    parser.add_argument("--loso", action="store_true", help="Use LOSO cross-subject evaluation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--epochs", type=int, default=40, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--progress_bar", dest="progress_bar", action="store_true", default=True, help="Enable progress bar (default: on)")
    parser.add_argument("--no_progress_bar", dest="progress_bar", action="store_false", help="Disable progress bar")
    return parser.parse_args()


def selected_modes(args) -> List[str]:
    modes = []
    if args.eeg:
        modes.append("eeg")
    if args.fnirs:
        modes.append("fnirs")
    if args.concatenate:
        modes.append("concatenate")
    if args.riemann:
        modes.append("riemann")
    if args.fullalign:
        modes.append("fullalign")
    if not modes:
        modes = ["eeg", "fnirs", "concatenate", "riemann", "fullalign"]
    return modes


def load_subject_trials(root: Path, sid: int):
    subject_name = f"subject {sid:02d}"
    eeg_root = root / "EEG-fNIRs异构数据集" / "EEG_01-29" / subject_name / "with occular artifact"
    nirs_root = root / "EEG-fNIRs异构数据集" / "NIRS_01-29" / subject_name

    eeg_cnt = eeg_root / "cnt.mat"
    eeg_mrk = eeg_root / "mrk.mat"
    nirs_cnt = nirs_root / "cnt.mat"
    nirs_mrk = nirs_root / "mrk.mat"
    for p in [eeg_cnt, eeg_mrk, nirs_cnt, nirs_mrk]:
        if not p.exists():
            return None

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
        cnt_nirs,
        mrk_nirs,
        sessions=sessions_mi,
        label_map=nirs_map,
        window_s=(2.0, 12.0),
        baseline_s=(-2.0, 0.0),
    )

    eeg_epochs, nirs_epochs, labels = paired_by_label_order(eeg_epochs, eeg_labels, nirs_epochs, nirs_labels)
    if len(labels) < 20:
        return None
    return eeg_epochs, nirs_epochs, labels


def build_dataset(root: Path, progress_bar: bool):
    eeg_all, nirs_all, y_all, subj_all = [], [], [], []
    sid_iter = range(1, 30)
    if progress_bar and tqdm is not None:
        sid_iter = tqdm(sid_iter, desc="Loading subjects", leave=True, dynamic_ncols=True)

    for sid in sid_iter:
        out = load_subject_trials(root, sid)
        if out is None:
            if progress_bar and tqdm is not None:
                tqdm.write(f"[skip] subject {sid:02d}: missing file or too few trials")
            else:
                print(f"[skip] subject {sid:02d}: missing file or too few trials")
            continue
        eeg_epochs, nirs_epochs, labels = out
        n = len(labels)
        eeg_all.append(eeg_epochs)
        nirs_all.append(nirs_epochs)
        y_all.append(labels)
        subj_all.append(np.full(n, sid, dtype=np.int32))
        if progress_bar and tqdm is not None:
            tqdm.write(f"[ok] subject {sid:02d}: paired_trials={n}")
        else:
            print(f"[ok] subject {sid:02d}: paired_trials={n}")

    if not y_all:
        raise RuntimeError("No valid subject could be loaded.")

    eeg = np.concatenate(eeg_all, axis=0)
    nirs = np.concatenate(nirs_all, axis=0)
    y = np.concatenate(y_all, axis=0)
    subjects = np.concatenate(subj_all, axis=0)
    return eeg, nirs, y, subjects


def epoch_cov(x_ct: np.ndarray) -> np.ndarray:
    c = np.cov(x_ct)
    c = c + 1e-6 * np.eye(c.shape[0])
    return c


def robust_mean_covariance(covs: np.ndarray) -> np.ndarray:
    # Prefer affine-invariant Riemann mean; fallback when iterative solver does not converge.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        mean_cov = mean_covariance(covs, metric="riemann")
    has_nonconv = any("Convergence not reached" in str(w.message) for w in caught)
    if has_nonconv:
        mean_cov = mean_covariance(covs, metric="logeuclid")
    return mean_cov


def riemann_align_epochs_by_subject(epochs: np.ndarray, subjects: np.ndarray) -> np.ndarray:
    out = np.empty_like(epochs)
    for sid in np.unique(subjects):
        idx = np.where(subjects == sid)[0]
        covs = np.array([epoch_cov(epochs[i]) for i in idx])
        mean_cov = robust_mean_covariance(covs)
        inv_sqrt = fractional_matrix_power(mean_cov, -0.5).real
        for k, i in enumerate(idx):
            out[i] = inv_sqrt @ epochs[i]
    return out


def shared_project_fit_transform(eeg_train, nirs_train, eeg_test, nirs_test, seed):
    x_eeg_tr = eeg_train.reshape(eeg_train.shape[0], -1)
    x_nir_tr = nirs_train.reshape(nirs_train.shape[0], -1)
    x_eeg_te = eeg_test.reshape(eeg_test.shape[0], -1)
    x_nir_te = nirs_test.reshape(nirs_test.shape[0], -1)

    scaler_eeg = StandardScaler()
    scaler_nir = StandardScaler()
    x_eeg_tr = scaler_eeg.fit_transform(x_eeg_tr)
    x_nir_tr = scaler_nir.fit_transform(x_nir_tr)
    x_eeg_te = scaler_eeg.transform(x_eeg_te)
    x_nir_te = scaler_nir.transform(x_nir_te)

    pca_dim = min(128, x_eeg_tr.shape[1], x_nir_tr.shape[1], x_eeg_tr.shape[0] - 1)
    pca_eeg = PCA(n_components=pca_dim, random_state=seed)
    pca_nir = PCA(n_components=pca_dim, random_state=seed)

    z_eeg_tr = pca_eeg.fit_transform(x_eeg_tr)
    z_nir_tr = pca_nir.fit_transform(x_nir_tr)
    z_eeg_te = pca_eeg.transform(x_eeg_te)
    z_nir_te = pca_nir.transform(x_nir_te)

    cca_dim = min(64, pca_dim, z_eeg_tr.shape[0] - 1)
    cca = CCA(n_components=cca_dim, max_iter=2000)
    z_eeg_tr, z_nir_tr = cca.fit_transform(z_eeg_tr, z_nir_tr)
    z_eeg_te, z_nir_te = cca.transform(z_eeg_te, z_nir_te)

    eeg_train_new = z_eeg_tr[:, np.newaxis, :]
    nirs_train_new = z_nir_tr[:, np.newaxis, :]
    eeg_test_new = z_eeg_te[:, np.newaxis, :]
    nirs_test_new = z_nir_te[:, np.newaxis, :]
    return eeg_train_new, nirs_train_new, eeg_test_new, nirs_test_new


class EEGNetEncoder(nn.Module):
    def __init__(self, chans: int, samples: int, emb_dim: int = 64):
        super().__init__()
        f1, d = 8, 2
        self.block1 = nn.Sequential(
            nn.Conv2d(1, f1, kernel_size=(1, 32), padding=(0, 16), bias=False),
            nn.BatchNorm2d(f1),
            nn.Conv2d(f1, f1 * d, kernel_size=(chans, 1), groups=f1, bias=False),
            nn.BatchNorm2d(f1 * d),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4)),
            nn.Dropout(0.25),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(f1 * d, f1 * d, kernel_size=(1, 16), padding=(0, 8), groups=f1 * d, bias=False),
            nn.Conv2d(f1 * d, 16, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 8)),
            nn.Dropout(0.25),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, 1, chans, samples)
            feat_dim = self.block2(self.block1(dummy)).flatten(1).shape[1]
        self.proj = nn.Linear(feat_dim, emb_dim)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = x.flatten(1)
        return self.proj(x)


class TCNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k=5, d=1, p_drop=0.2):
        super().__init__()
        pad = (k - 1) * d // 2
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=k, dilation=d, padding=pad),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(),
            nn.Dropout(p_drop),
        )
        self.skip = nn.Conv1d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        return self.net(x) + self.skip(x)


class TCNEncoder(nn.Module):
    def __init__(self, in_ch: int, emb_dim: int = 64):
        super().__init__()
        self.b1 = TCNBlock(in_ch, 32, d=1)
        self.b2 = TCNBlock(32, 64, d=2)
        self.b3 = TCNBlock(64, 64, d=4)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Linear(64, emb_dim)

    def forward(self, x):
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        x = self.pool(x).squeeze(-1)
        return self.proj(x)


class EEGOnlyModel(nn.Module):
    def __init__(self, eeg_ch, eeg_t):
        super().__init__()
        self.enc = EEGNetEncoder(eeg_ch, eeg_t, emb_dim=64)
        self.cls = nn.Linear(64, 2)

    def forward(self, eeg, nirs=None):
        emb = self.enc(eeg)
        return self.cls(emb)


class FNIRSOnlyModel(nn.Module):
    def __init__(self, nirs_ch):
        super().__init__()
        self.enc = TCNEncoder(nirs_ch, emb_dim=64)
        self.cls = nn.Linear(64, 2)

    def forward(self, eeg, nirs):
        emb = self.enc(nirs)
        return self.cls(emb)


class FusionAttentionModel(nn.Module):
    def __init__(self, eeg_ch, eeg_t, nirs_ch):
        super().__init__()
        self.eeg_enc = EEGNetEncoder(eeg_ch, eeg_t, emb_dim=64)
        self.nir_enc = TCNEncoder(nirs_ch, emb_dim=64)
        self.attn = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
            nn.Softmax(dim=1),
        )
        self.cls = nn.Linear(64, 2)

    def forward(self, eeg, nirs):
        e = self.eeg_enc(eeg)
        n = self.nir_enc(nirs)
        w = self.attn(torch.cat([e, n], dim=1))
        fused = w[:, 0:1] * e + w[:, 1:2] * n
        return self.cls(fused)


def make_model(mode: str, eeg_ch: int, eeg_t: int, nirs_ch: int):
    if mode == "eeg":
        return EEGOnlyModel(eeg_ch, eeg_t)
    if mode == "fnirs":
        return FNIRSOnlyModel(nirs_ch)
    return FusionAttentionModel(eeg_ch, eeg_t, nirs_ch)


def prepare_fold_data(mode, eeg_train, eeg_test, nirs_train, nirs_test, subj_train, subj_test, seed):
    # 融合分支在三种模式中统一使用 EEGNet+TCN+Attention，仅输入处理不同。
    if mode == "riemann" or mode == "fullalign":
        eeg_train = riemann_align_epochs_by_subject(eeg_train, subj_train)
        eeg_test = riemann_align_epochs_by_subject(eeg_test, subj_test)
        nirs_train = riemann_align_epochs_by_subject(nirs_train, subj_train)
        nirs_test = riemann_align_epochs_by_subject(nirs_test, subj_test)

    if mode == "fullalign":
        eeg_train, nirs_train, eeg_test, nirs_test = shared_project_fit_transform(
            eeg_train, nirs_train, eeg_test, nirs_test, seed
        )

    eeg_train_t = eeg_train.astype(np.float32)
    eeg_test_t = eeg_test.astype(np.float32)
    nirs_train_t = nirs_train.astype(np.float32)
    nirs_test_t = nirs_test.astype(np.float32)

    # EEGNet expects (B,1,C,T)
    eeg_train_t = np.expand_dims(eeg_train_t, axis=1)
    eeg_test_t = np.expand_dims(eeg_test_t, axis=1)

    return eeg_train_t, eeg_test_t, nirs_train_t, nirs_test_t


def compute_metrics(y_true, y_pred, y_prob):
    result = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }
    try:
        result["auc"] = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        result["auc"] = float("nan")
    return result


def save_binary_confusion_matrix(y_true, y_pred, save_path: Path, title: str):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred 0", "Pred 1"])
    ax.set_yticklabels(["True 0", "True 1"])
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
    fig.tight_layout()
    fig.savefig(save_path, dpi=180)
    plt.close(fig)


def save_training_curve(histories, save_path: Path, title: str):
    if not histories:
        return
    loss_mat = np.array([h["loss"] for h in histories], dtype=np.float64)
    acc_mat = np.array([h["acc"] for h in histories], dtype=np.float64)
    loss_mean = np.mean(loss_mat, axis=0)
    loss_std = np.std(loss_mat, axis=0, ddof=1) if loss_mat.shape[0] > 1 else np.zeros(loss_mat.shape[1])
    acc_mean = np.mean(acc_mat, axis=0)
    acc_std = np.std(acc_mat, axis=0, ddof=1) if acc_mat.shape[0] > 1 else np.zeros(acc_mat.shape[1])

    epochs = np.arange(1, loss_mat.shape[1] + 1)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].plot(epochs, loss_mean, color="#d62728")
    axes[0].fill_between(epochs, loss_mean - loss_std, loss_mean + loss_std, color="#d62728", alpha=0.2)
    axes[0].set_title("Train Loss vs Epoch")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")

    axes[1].plot(epochs, acc_mean, color="#1f77b4")
    axes[1].fill_between(epochs, acc_mean - acc_std, acc_mean + acc_std, color="#1f77b4", alpha=0.2)
    axes[1].set_title("Train Accuracy vs Epoch")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0.0, 1.0)

    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    fig.savefig(save_path, dpi=180)
    plt.close(fig)


def train_one_fold(mode, eeg_train, y_train, nirs_train, eeg_test, nirs_test, y_test, seed, epochs, batch_size, lr, device):
    torch.manual_seed(seed)
    model = make_model(mode, eeg_train.shape[2], eeg_train.shape[3], nirs_train.shape[1]).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    ds = TensorDataset(
        torch.from_numpy(eeg_train),
        torch.from_numpy(nirs_train),
        torch.from_numpy(y_train.astype(np.int64)),
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    history = {"loss": [], "acc": []}
    model.train()
    for _ in range(epochs):
        running_loss = 0.0
        running_correct = 0
        total = 0
        for eeg_b, nirs_b, y_b in loader:
            eeg_b = eeg_b.to(device)
            nirs_b = nirs_b.to(device)
            y_b = y_b.to(device)

            optimizer.zero_grad()
            logits = model(eeg_b, nirs_b)
            loss = criterion(logits, y_b)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * y_b.size(0)
            pred = torch.argmax(logits, dim=1)
            running_correct += (pred == y_b).sum().item()
            total += y_b.size(0)

        history["loss"].append(running_loss / max(total, 1))
        history["acc"].append(running_correct / max(total, 1))

    model.eval()
    with torch.no_grad():
        eeg_t = torch.from_numpy(eeg_test).to(device)
        nirs_t = torch.from_numpy(nirs_test).to(device)
        logits = model(eeg_t, nirs_t)
        prob = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        pred = torch.argmax(logits, dim=1).cpu().numpy()

    return pred, prob, history


def aggregate_metrics(metric_list: List[Dict[str, float]]):
    keys = list(metric_list[0].keys())
    out = {}
    for k in keys:
        vals = np.array([m[k] for m in metric_list], dtype=np.float64)
        out[k] = {
            "mean": float(np.nanmean(vals)),
            "std": float(np.nanstd(vals, ddof=1)) if len(vals) > 1 else 0.0,
        }
    return out


def evaluate_loso(mode, eeg, nirs, y, subjects, seed, epochs, batch_size, lr, progress_bar, device):
    fold_metrics, histories = [], []
    y_true_all, y_pred_all = [], []

    sid_iter = np.unique(subjects)
    if progress_bar and tqdm is not None:
        sid_iter = tqdm(sid_iter, desc=f"LOSO-{mode}", leave=True, dynamic_ncols=True)

    for sid in sid_iter:
        test_idx = np.where(subjects == sid)[0]
        train_idx = np.where(subjects != sid)[0]

        eeg_tr, eeg_te, nirs_tr, nirs_te = prepare_fold_data(
            mode,
            eeg[train_idx],
            eeg[test_idx],
            nirs[train_idx],
            nirs[test_idx],
            subjects[train_idx],
            subjects[test_idx],
            seed,
        )

        pred, prob, hist = train_one_fold(
            mode,
            eeg_tr,
            y[train_idx],
            nirs_tr,
            eeg_te,
            nirs_te,
            y[test_idx],
            seed,
            epochs,
            batch_size,
            lr,
            device,
        )

        fold_metrics.append(compute_metrics(y[test_idx], pred, prob))
        histories.append(hist)
        y_true_all.append(y[test_idx])
        y_pred_all.append(pred)

    artifacts = {
        "y_true_all": np.concatenate(y_true_all, axis=0),
        "y_pred_all": np.concatenate(y_pred_all, axis=0),
        "histories": histories,
    }
    return aggregate_metrics(fold_metrics), fold_metrics, artifacts


def evaluate_within_session(mode, eeg, nirs, y, subjects, seed, epochs, batch_size, lr, progress_bar, device):
    fold_metrics, histories = [], []
    y_true_all, y_pred_all = [], []

    sid_iter = np.unique(subjects)
    if progress_bar and tqdm is not None:
        sid_iter = tqdm(sid_iter, desc=f"Within-{mode}", leave=True, dynamic_ncols=True)

    for sid in sid_iter:
        idx = np.where(subjects == sid)[0]
        y_sid = y[idx]
        if len(np.unique(y_sid)) < 2 or len(y_sid) < 10:
            continue

        splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
        local_train, local_test = next(splitter.split(np.zeros(len(y_sid)), y_sid))
        train_idx = idx[local_train]
        test_idx = idx[local_test]

        eeg_tr, eeg_te, nirs_tr, nirs_te = prepare_fold_data(
            mode,
            eeg[train_idx],
            eeg[test_idx],
            nirs[train_idx],
            nirs[test_idx],
            subjects[train_idx],
            subjects[test_idx],
            seed,
        )

        pred, prob, hist = train_one_fold(
            mode,
            eeg_tr,
            y[train_idx],
            nirs_tr,
            eeg_te,
            nirs_te,
            y[test_idx],
            seed,
            epochs,
            batch_size,
            lr,
            device,
        )

        fold_metrics.append(compute_metrics(y[test_idx], pred, prob))
        histories.append(hist)
        y_true_all.append(y[test_idx])
        y_pred_all.append(pred)

    artifacts = {
        "y_true_all": np.concatenate(y_true_all, axis=0),
        "y_pred_all": np.concatenate(y_pred_all, axis=0),
        "histories": histories,
    }
    return aggregate_metrics(fold_metrics), fold_metrics, artifacts


def main():
    args = parse_args()
    np.random.seed(args.seed)
    train_start_dt = datetime.now()

    if torch is None:
        raise ImportError("PyTorch is required now. Please install torch in your environment.")

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    modes = selected_modes(args)
    protocol = "LOSO" if args.loso else "within_session"

    root = Path(__file__).resolve().parent
    results_root = root / "results"
    results_root.mkdir(parents=True, exist_ok=True)
    run_stamp = train_start_dt.strftime("%m%d_%H%M")

    print(f"Selected modes: {modes}")
    print(f"Protocol: {protocol}")
    print(f"Seed: {args.seed}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"LR: {args.lr}")
    print(f"Device: {device}")
    print(f"Progress bar: {args.progress_bar}")
    if args.progress_bar and tqdm is None:
        print("[warn] tqdm is not installed. Running without progress bar.")

    eeg, nirs, y, subjects = build_dataset(root, args.progress_bar)
    print(f"Total paired trials: {len(y)}")
    print(f"Subjects used: {len(np.unique(subjects))}")

    all_results = {
        "run_stamp": run_stamp,
        "train_start_time": train_start_dt.isoformat(timespec="seconds"),
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "protocol": protocol,
        "modes": modes,
        "n_trials": int(len(y)),
        "n_subjects": int(len(np.unique(subjects))),
        "results": {},
    }

    mode_iter = modes
    if args.progress_bar and tqdm is not None:
        mode_iter = tqdm(modes, desc="Running modes", leave=True, dynamic_ncols=True)

    for mode in mode_iter:
        msg = f"\n=== Running mode: {mode} ==="
        if args.progress_bar and tqdm is not None:
            tqdm.write(msg)
        else:
            print(msg)
        if args.loso:
            summary, folds, artifacts = evaluate_loso(
                mode, eeg, nirs, y, subjects, args.seed, args.epochs, args.batch_size, args.lr, args.progress_bar, device
            )
        else:
            summary, folds, artifacts = evaluate_within_session(
                mode, eeg, nirs, y, subjects, args.seed, args.epochs, args.batch_size, args.lr, args.progress_bar, device
            )

        mode_dir = results_root / f"{run_stamp}_{mode}_{protocol.lower()}_seed{args.seed}"
        fig_dir = mode_dir / "figures"
        fig_dir.mkdir(parents=True, exist_ok=True)

        cm_path = fig_dir / f"cm_{mode}.png"
        curve_path = fig_dir / f"curve_{mode}.png"
        save_binary_confusion_matrix(
            artifacts["y_true_all"],
            artifacts["y_pred_all"],
            cm_path,
            title=f"Confusion Matrix ({protocol}, {mode})",
        )
        save_training_curve(
            artifacts["histories"],
            curve_path,
            title=f"Training Curves ({protocol}, {mode})",
        )

        mode_payload = {
            "run_stamp": run_stamp,
            "mode": mode,
            "protocol": protocol,
            "seed": args.seed,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "n_trials": int(len(y)),
            "n_subjects": int(len(np.unique(subjects))),
            "n_folds": len(folds),
            "summary": summary,
            "confusion_matrix_png": str(cm_path),
            "train_curve_png": str(curve_path),
        }
        mode_json = mode_dir / f"summary_{mode}.json"
        with open(mode_json, "w", encoding="utf-8") as f:
            json.dump(mode_payload, f, ensure_ascii=False, indent=2)

        all_results["results"][mode] = {
            "summary": summary,
            "n_folds": len(folds),
            "result_dir": str(mode_dir),
            "mode_summary_json": str(mode_json),
            "confusion_matrix_png": str(cm_path),
            "train_curve_png": str(curve_path),
        }

        print(
            "accuracy={:.4f}, f1_macro={:.4f}, auc={:.4f}".format(
                summary["accuracy"]["mean"],
                summary["f1_macro"]["mean"],
                summary["auc"]["mean"],
            )
        )

    out_json = results_root / f"{run_stamp}_all_{protocol.lower()}_seed{args.seed}.json"
    train_end_dt = datetime.now()
    all_results["train_end_time"] = train_end_dt.isoformat(timespec="seconds")
    all_results["train_duration_seconds"] = float((train_end_dt - train_start_dt).total_seconds())
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print(f"\nSaved summary: {out_json}")


if __name__ == "__main__":
    main()
