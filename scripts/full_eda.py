import os
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import librosa
import librosa.display
import torchaudio
import torch
from scipy import stats
from scipy.stats import mannwhitneyu
import mlflow
from tqdm import tqdm

# --- Kaggle / Notebook Specifics ---
try:
    from IPython.display import Image, display
    IN_NOTEBOOK = True
except ImportError:
    IN_NOTEBOOK = False

label_names = {
    0: "M1 Normal", 1: "M1 Abnormal",
    2: "M2 Normal", 3: "M2 Abnormal",
    4: "M3 Normal", 5: "M3 Abnormal",
}

def infer_machine(wav_path: Path) -> int:
    path_str = str(wav_path).lower()
    if 'machine 1' in path_str or 'machine1' in path_str: return 1
    if 'machine 2' in path_str or 'machine2' in path_str: return 2
    if 'machine 3' in path_str or 'machine3' in path_str: return 3
    return 1  # Default

def infer_state(wav_path: Path) -> str:
    path_str = str(wav_path).lower()
    if 'abnormal' in path_str: return 'abnormal'
    if 'normal' in path_str: return 'normal'
    return 'normal'

def infer_label(wav_path: Path) -> int:
    m = infer_machine(wav_path)
    s = infer_state(wav_path)
    base = (m - 1) * 2
    return base + 1 if s == 'abnormal' else base

# ---------------------------------------------------------------------------
# EDA 1 — Dataset Metadata Audit
# ---------------------------------------------------------------------------
def audit_dataset(root_dir: str, cache_path: str = None) -> pd.DataFrame:
    # 1. Try to load from cache
    if cache_path and os.path.exists(cache_path):
        print(f"✅ Loading dataset audit from cache: {cache_path}")
        return pd.read_csv(cache_path)

    # 2. Otherwise Scan
    records = []
    audio_files = list(Path(root_dir).rglob("*.wav"))
    print(f"📡 Scanning {len(audio_files)} files for metadata (First run only)...")
    for i, wav_path in enumerate(audio_files):
        if i % 1000 == 0:
            print(f"  Scanned {i}/{len(audio_files)} files...")
        try:
            # Try torchaudio.info (fastest)
            if hasattr(torchaudio, 'info'):
                info = torchaudio.info(str(wav_path))
                sr, frames, channels = info.sample_rate, info.num_frames, info.num_channels
            else:
                # Fallback to librosa (metadata only mode)
                sr = librosa.get_samplerate(str(wav_path))
                duration = librosa.get_duration(path=str(wav_path))
                frames = int(duration * sr)
                channels = 1 # librosa defaults to 1 or you can use wave for channel count
            
            records.append({
                "path":         str(wav_path),
                "filename":     wav_path.name,
                "label":        infer_label(wav_path),
                "machine":      infer_machine(wav_path),
                "state":        infer_state(wav_path),
                "sample_rate":  sr,
                "num_frames":   frames,
                "num_channels": channels,
                "duration_s":   frames / sr,
                "file_size_kb": wav_path.stat().st_size / 1024,
            })
        except Exception as e:
            if i < 5: # Only print the first few errors to avoid flooding
                print(f"[!] Metadata fallback failed for {wav_path.name}: {e}")

    df = pd.DataFrame(records)
    return df

from tqdm import tqdm

# ---------------------------------------------------------------------------
# Feature helpers
# ---------------------------------------------------------------------------
def compute_heavy_features(df_subset: pd.DataFrame):
    """Computes RMS and Silence in a single load pass to save 50% I/O time."""
    print(f"🧬 Computing energy & silence on a representative sample of {len(df_subset)} files...")
    rms_vals = []
    silence_vals = []
    
    for path in tqdm(df_subset["path"], desc="Extracting features", mininterval=10):
        try:
            # Load once for both calculations
            waveform, sr = torchaudio.load(path)
            
            # 1. RMS Energy
            rms = waveform.pow(2).mean().sqrt().item()
            rms_vals.append(rms)
            
            # 2. Silence Ratio (Modified for torchaudio/tensor efficiency)
            # Threshold matches librosa top_db=20.0 approx
            is_silent = (waveform.abs() < 0.01).float().mean().item()
            silence_vals.append(is_silent)
        except Exception:
            rms_vals.append(np.nan)
            silence_vals.append(np.nan)
            
    df_subset["rms_energy"] = rms_vals
    df_subset["silence_ratio"] = silence_vals
    return df_subset

def mean_mfcc(wav_path: str, n_mfcc: int = 40) -> np.ndarray:
    """Compute time-averaged MFCC vector for a file."""
    audio, sr = librosa.load(wav_path, sr=16000, mono=True, duration=10.0)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    return mfcc.mean(axis=1)  # [n_mfcc]

# ---------------------------------------------------------------------------
# Main EDA runner
# ---------------------------------------------------------------------------
def run_eda(data_root: str, out_dir: str, sample_per_class: int = 1500, force_recompute: bool = False):
    os.makedirs(out_dir, exist_ok=True)
    
    # 1. Metadata Audit (Cached)
    audit_csv = os.path.join(out_dir, "dataset_audit.csv")
    if force_recompute and os.path.exists(audit_csv): os.remove(audit_csv)
    df_all = audit_dataset(data_root, audit_csv)
    
    if len(df_all) == 0:
        raise ValueError("No valid audio files to process.")

    # 2. Feature Extraction (Cached)
    sampled_csv = os.path.join(out_dir, "sampled_features.csv")
    if not force_recompute and os.path.exists(sampled_csv):
        print(f"✅ Loading sampled features from cache: {sampled_csv}")
        df = pd.read_csv(sampled_csv)
    else:
        print(f"🎯 Sampling {sample_per_class} files per class for heavy analysis...")
        df = df_all.groupby("label").apply(
            lambda x: x.sample(n=min(len(x), sample_per_class))
        ).reset_index(drop=True)
        
        # Combined Energy & Silence extraction
        df = compute_heavy_features(df)
        df.to_csv(sampled_csv, index=False)
        print(f"💾 Sampled features saved to: {sampled_csv}")

    with mlflow.start_run():
        audit_path = os.path.join(out_dir, "dataset_audit.csv")
        df_all.to_csv(audit_path, index=False)
        mlflow.log_artifact(audit_path)

        print("\nFull Dataset overview:")
        print(df_all[["sample_rate", "num_channels", "duration_s", "file_size_kb"]].describe())

    with mlflow.start_run():
        # Log the full audit for reference
        audit_path = os.path.join(out_dir, "dataset_audit.csv")
        df_all.to_csv(audit_path, index=False)
        mlflow.log_artifact(audit_path)

        print("\n--- Full Dataset Audit Summary ---")
        print(df_all[["sample_rate", "num_channels", "duration_s", "file_size_kb"]].describe())
        print("Unique sample rates  :", df_all["sample_rate"].unique().tolist())
        
        print(f"\n--- EDA Analysis on Sample (N={len(df)}) ---")
        print(df[["rms_energy", "silence_ratio"]].describe())

        # ---------------------------------------------------------------
        # EDA 2 — Class Balance
        # ---------------------------------------------------------------
        counts = df["label"].value_counts().sort_index()
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        axes[0].bar(
            [label_names[i] for i in counts.index],
            counts.values,
            color=["#388E3C", "#C62828", "#0288D1", "#F57C00", "#7B1FA2", "#E64A19"],
            edgecolor="black", linewidth=0.6,
        )
        axes[0].set_title("Sample Count per Class", fontsize=13, fontweight="bold")
        axes[0].set_ylabel("Number of Audio Files")
        axes[0].tick_params(axis="x", rotation=35)
        for i, v in enumerate(counts.values):
            axes[0].text(i, v + max(counts.values) * 0.01, str(v), ha="center", fontsize=9)

        state_counts = df["state"].value_counts()
        axes[1].pie(state_counts, labels=state_counts.index, autopct="%1.1f%%",
                    startangle=90, colors=["#388E3C", "#C62828"])
        axes[1].set_title("Overall Normal vs Abnormal Split", fontsize=13, fontweight="bold")

        plt.tight_layout()
        balance_path = os.path.join(out_dir, "eda_class_balance.png")
        plt.savefig(balance_path, dpi=150, bbox_inches="tight")
        mlflow.log_artifact(balance_path)
        plt.close()

        # Log imbalance ratio
        n_normal   = state_counts.get("normal",   0)
        n_abnormal = state_counts.get("abnormal", 0)
        if n_abnormal > 0:
            mlflow.log_metric("imbalance_ratio", n_normal / n_abnormal)

        # ---------------------------------------------------------------
        # EDA 3 — Duration & Sample Rate
        # ---------------------------------------------------------------
        fig, axes = plt.subplots(1, 3, figsize=(16, 4))

        axes[0].hist(df["duration_s"], bins=40, color="#1565C0", edgecolor="white", alpha=0.85)
        axes[0].axvline(10.0, color="red", linestyle="--", label="Target: 10s", linewidth=2)
        axes[0].set_title("Audio Duration Distribution")
        axes[0].set_xlabel("Duration (seconds)")
        axes[0].set_ylabel("Count")
        axes[0].legend()

        df.boxplot(column="duration_s", by="label", ax=axes[1], patch_artist=True)
        axes[1].set_title("Duration by Class")
        axes[1].set_xlabel("Class Label (0–5)")
        axes[1].set_ylabel("Duration (s)")

        sr_counts = df["sample_rate"].value_counts()
        axes[2].bar(sr_counts.index.astype(str), sr_counts.values,
                    color="#00838F", edgecolor="black")
        axes[2].set_title("Sample Rate Distribution")
        axes[2].set_xlabel("Sample Rate (Hz)")
        axes[2].set_ylabel("File Count")

        plt.suptitle("")
        plt.tight_layout()
        duration_path = os.path.join(out_dir, "eda_duration_sr.png")
        plt.savefig(duration_path, dpi=150, bbox_inches="tight")
        mlflow.log_artifact(duration_path)
        plt.close()

        # Log duration stats
        mlflow.log_metric("duration_mean_s", df["duration_s"].mean())
        mlflow.log_metric("duration_std_s",  df["duration_s"].std())
        needs_padding_pct = (df["duration_s"] < 10.0).mean() * 100
        mlflow.log_metric("pct_files_needing_padding", needs_padding_pct)

        # ---------------------------------------------------------------
        # EDA 4 — Energy & Volume
        # ---------------------------------------------------------------
        # (Already computed during the combined feature extraction pass)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        axes[0].violinplot(
            [df[df["label"] == i]["rms_energy"].values for i in range(6)],
            positions=range(6), showmedians=True,
        )
        axes[0].set_xticks(range(6))
        axes[0].set_xticklabels([label_names[i] for i in range(6)], rotation=30)
        axes[0].set_title("RMS Energy Distribution by Class")
        axes[0].set_ylabel("RMS Energy")

        for state, color, ls in [("normal", "#388E3C", "-"), ("abnormal", "#C62828", "--")]:
            subset = df[df["state"] == state]["rms_energy"]
            if len(subset) > 0:
                sns.kdeplot(subset, ax=axes[1], label=state.capitalize(),
                            color=color, linestyle=ls, linewidth=2)
        axes[1].set_title("Normal vs Abnormal: RMS Energy KDE")
        axes[1].set_xlabel("RMS Energy")
        axes[1].legend()

        plt.tight_layout()
        energy_path = os.path.join(out_dir, "eda_energy.png")
        plt.savefig(energy_path, dpi=150, bbox_inches="tight")
        mlflow.log_artifact(energy_path)
        plt.close()

        # ---------------------------------------------------------------
        # EDA 5 — Silence Analysis
        # ---------------------------------------------------------------
        # (Already computed during the combined feature extraction pass)

        fig, axes = plt.subplots(1, 2, figsize=(14, 4))

        df.boxplot(column="silence_ratio", by="state", ax=axes[0], patch_artist=True)
        axes[0].set_title("Silence Ratio: Normal vs Abnormal")
        axes[0].set_ylabel("Fraction of File that is Silent")

        df.boxplot(column="silence_ratio", by="label", ax=axes[1])
        axes[1].set_title("Silence Ratio by Class")
        axes[1].set_xlabel("Class (0–5)")

        plt.suptitle("")
        plt.tight_layout()
        silence_path = os.path.join(out_dir, "eda_silence.png")
        plt.savefig(silence_path, dpi=150, bbox_inches="tight")
        mlflow.log_artifact(silence_path)
        plt.close()

        mlflow.log_metric("silence_ratio_mean", df["silence_ratio"].mean())

        # ---------------------------------------------------------------
        # EDA 6 — Waveform Inspection
        # ---------------------------------------------------------------
        fig, axes = plt.subplots(3, 2, figsize=(14, 10))
        fig.suptitle("Waveform Comparison: One Sample per Class", fontsize=14, fontweight="bold")

        for label_id in range(6):
            row, col = label_id // 2, label_id % 2
            ax = axes[row][col]
            subset = df[df["label"] == label_id]
            if len(subset) == 0:
                ax.set_visible(False)
                continue
            waveform, sr = torchaudio.load(subset.iloc[0]["path"])
            audio = waveform[0].numpy()
            time  = np.linspace(0, len(audio) / sr, len(audio))
            color = "#388E3C" if label_id % 2 == 0 else "#C62828"
            ax.plot(time, audio, color=color, linewidth=0.4, alpha=0.8)
            ax.set_title(label_names[label_id], fontweight="bold", color=color)
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Amplitude")
            dc = audio.mean()
            ax.axhline(dc, color="navy", linestyle="--", linewidth=1,
                       label=f"DC offset: {dc:.4f}")
            ax.legend(fontsize=8)

        plt.tight_layout()
        waveform_path = os.path.join(out_dir, "eda_waveforms.png")
        plt.savefig(waveform_path, dpi=150, bbox_inches="tight")
        mlflow.log_artifact(waveform_path)
        plt.close()

        # ---------------------------------------------------------------
        # EDA 7 — Mel-Spectrogram Comparison
        # ---------------------------------------------------------------
        print("Generating spectrograms...")
        mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000, n_fft=1024, hop_length=512,
            n_mels=128, f_min=50, f_max=8000,
        )
        db_transform = torchaudio.transforms.AmplitudeToDB(top_db=80)

        fig, axes = plt.subplots(3, 2, figsize=(14, 12))
        fig.suptitle("Mel-Spectrograms: Normal (left) vs Abnormal (right)",
                     fontsize=14, fontweight="bold")

        for label_id in range(6):
            row, col = label_id // 2, label_id % 2
            ax = axes[row][col]
            subset = df[df["label"] == label_id]
            if len(subset) == 0:
                ax.set_visible(False)
                continue
            waveform, sr = torchaudio.load(subset.iloc[0]["path"])
            if sr != 16000:
                waveform = torchaudio.functional.resample(waveform, sr, 16000)
            waveform = waveform[:1, :160000]
            mel_db = db_transform(mel_transform(waveform))[0].numpy()
            img = ax.imshow(mel_db, aspect="auto", origin="lower",
                            cmap="magma", extent=[0, 10, 50, 8000])
            ax.set_title(label_names[label_id],
                         color="#388E3C" if label_id % 2 == 0 else "#C62828")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Frequency (Hz)")
            plt.colorbar(img, ax=ax, label="dB")

        plt.tight_layout()
        spec_path = os.path.join(out_dir, "eda_spectrograms.png")
        plt.savefig(spec_path, dpi=150, bbox_inches="tight")
        mlflow.log_artifact(spec_path)
        plt.close()

        # ---------------------------------------------------------------
        # EDA 8 — Spectrogram Difference Maps
        # ---------------------------------------------------------------
        def avg_spectrogram(df_sub, n_samples: int = 30) -> np.ndarray:
            """Average mel-spectrogram over up to n_samples files."""
            specs = []
            for path in df_sub["path"].head(n_samples):
                wav, sr = torchaudio.load(path)
                if sr != 16000:
                    wav = torchaudio.functional.resample(wav, sr, 16000)
                wav = wav[:1, :160000]
                if wav.shape[1] == 0:
                    continue
                mel = db_transform(mel_transform(wav))[0].numpy()
                specs.append(mel)
            if len(specs) == 0:
                return np.zeros((128, 313))
            min_w  = min(s.shape[1] for s in specs)
            specs  = [s[:, :min_w] for s in specs]
            return np.stack(specs).mean(axis=0)

        fig, axes = plt.subplots(1, 3, figsize=(16, 4))
        for i, m in enumerate([1, 2, 3]):
            n_avg  = avg_spectrogram(df[(df["machine"] == m) & (df["state"] == "normal")])
            a_avg  = avg_spectrogram(df[(df["machine"] == m) & (df["state"] == "abnormal")])
            diff   = a_avg - n_avg
            im = axes[i].imshow(diff, aspect="auto", origin="lower",
                                cmap="RdBu_r", vmin=-10, vmax=10)
            axes[i].set_title(f"Machine {m}: Abnormal − Normal", fontweight="bold")
            axes[i].set_xlabel("Time frames")
            axes[i].set_ylabel("Mel bands")
            plt.colorbar(im, ax=axes[i], label="dB difference")

        plt.suptitle("Average Spectrogram Difference Maps", fontsize=13, fontweight="bold")
        plt.tight_layout()
        diff_path = os.path.join(out_dir, "eda_diff_maps.png")
        plt.savefig(diff_path, dpi=150, bbox_inches="tight")
        mlflow.log_artifact(diff_path)
        plt.close()

        # ---------------------------------------------------------------
        # EDA 9 — Frequency Profile Analysis  (was mislabelled as EDA 9 again)
        # ---------------------------------------------------------------
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        colors_normal   = ["#1565C0", "#00838F", "#6A1B9A"]
        colors_abnormal = ["#C62828", "#E65100", "#AD1457"]

        for i, m in enumerate([1, 2, 3]):
            for state, color_list, ls in [
                ("normal",   colors_normal,   "-"),
                ("abnormal", colors_abnormal, "--"),
            ]:
                subset = df[(df["machine"] == m) & (df["state"] == state)]
                profs  = []
                for path in subset["path"].head(30):
                    wav, sr = torchaudio.load(path)
                    if sr != 16000:
                        wav = torchaudio.functional.resample(wav, sr, 16000)
                    mel = db_transform(mel_transform(wav[:1, :160000]))[0].numpy()
                    profs.append(mel.mean(axis=1))
                if profs:
                    mean_p = np.stack(profs).mean(axis=0)
                    std_p  = np.stack(profs).std(axis=0)
                    axes[i].plot(np.arange(128), mean_p,
                                 color=color_list[i], linestyle=ls,
                                 label=state.capitalize(), linewidth=2)
                    axes[i].fill_between(np.arange(128),
                                         mean_p - std_p, mean_p + std_p,
                                         alpha=0.15, color=color_list[i])
            axes[i].set_title(f"Machine {m}: Frequency Profile", fontweight="bold")
            axes[i].set_xlabel("Mel Band Index")
            axes[i].set_ylabel("Average Energy (dB)")
            axes[i].legend()

        plt.tight_layout()
        freq_path = os.path.join(out_dir, "eda_freq_profiles.png")
        plt.savefig(freq_path, dpi=150, bbox_inches="tight")
        mlflow.log_artifact(freq_path)
        plt.close()

        # ---------------------------------------------------------------
        # EDA 10 — Inter-Class Correlation  (was incorrectly labelled EDA 9)
        # ---------------------------------------------------------------
        print("Computing MFCCs for correlation matrix...")
        class_mfccs = {}
        for label_id in range(6):
            subset = df[df["label"] == label_id]
            if len(subset) == 0:
                continue
            vecs = np.stack([mean_mfcc(p) for p in subset["path"].head(50)])
            class_mfccs[label_names[label_id]] = vecs.mean(axis=0)

        if len(class_mfccs) > 1:
            keys        = list(class_mfccs.keys())
            mfcc_matrix = np.stack([class_mfccs[k] for k in keys])
            corr_matrix = np.corrcoef(mfcc_matrix)

            fig, ax = plt.subplots(figsize=(8, 7))
            im = ax.imshow(corr_matrix, cmap="coolwarm", vmin=-1, vmax=1)
            ax.set_xticks(range(len(keys)))
            ax.set_yticks(range(len(keys)))
            ax.set_xticklabels(keys, rotation=40, ha="right")
            ax.set_yticklabels(keys)
            plt.colorbar(im, ax=ax, label="Pearson Correlation")
            ax.set_title("Inter-Class MFCC Correlation Matrix",
                         fontsize=13, fontweight="bold")
            for i in range(len(keys)):
                for j in range(len(keys)):
                    ax.text(j, i, f"{corr_matrix[i, j]:.2f}",
                            ha="center", va="center", fontsize=9,
                            color="white" if abs(corr_matrix[i, j]) > 0.7 else "black")

            plt.tight_layout()
            corr_path = os.path.join(out_dir, "eda_correlation.png")
            plt.savefig(corr_path, dpi=150, bbox_inches="tight")
            mlflow.log_artifact(corr_path)
            plt.close()

            # Log the hardest class pair (highest off-diagonal correlation)
            np.fill_diagonal(corr_matrix, -np.inf)
            max_idx = np.unravel_index(np.argmax(corr_matrix), corr_matrix.shape)
            mlflow.log_metric("max_interclass_correlation",
                              float(corr_matrix[max_idx]))
            print(f"\nHardest class pair: {keys[max_idx[0]]} ↔ {keys[max_idx[1]]} "
                  f"(r = {corr_matrix[max_idx]:.3f})")

        # ---------------------------------------------------------------
        # EDA 11 — Outlier Detection  (was EDA 10)
        # ---------------------------------------------------------------
        df["rms_zscore"] = df.groupby("label")["rms_energy"].transform(
            lambda x: np.abs(stats.zscore(x))
        )
        outliers = df[df["rms_zscore"] > 3.0]

        print(f"\nOutlier files found: {len(outliers)}")
        if len(outliers) > 0:
            print(outliers[["filename", "label", "rms_energy", "rms_zscore"]].to_string())
        mlflow.log_metric("outlier_count", len(outliers))
        mlflow.log_metric("outlier_pct",   100.0 * len(outliers) / len(df))

        fig, ax = plt.subplots(figsize=(12, 5))
        for label_id in range(6):
            sub = df[df["label"] == label_id]
            ax.scatter(sub.index, sub["rms_energy"],
                       label=label_names[label_id], s=8, alpha=0.6)
        ax.scatter(outliers.index, outliers["rms_energy"],
                   color="red", s=60, marker="x", linewidths=2,
                   label="Outlier (z > 3)", zorder=5)
        ax.set_title("RMS Energy per File with Outliers Flagged", fontweight="bold")
        ax.set_xlabel("File Index")
        ax.set_ylabel("RMS Energy")
        ax.legend(ncol=4, fontsize=8)
        plt.tight_layout()
        outlier_path = os.path.join(out_dir, "eda_outliers.png")
        plt.savefig(outlier_path, dpi=150, bbox_inches="tight")
        mlflow.log_artifact(outlier_path)
        plt.close()

        # ---------------------------------------------------------------
        # EDA 12 — Statistical Separability Tests (Mann-Whitney U)
        # ---------------------------------------------------------------
        print("\n" + "=" * 60)
        print("EDA 12 — Mann-Whitney U Test: Normal vs Abnormal RMS Energy")
        print("=" * 60)

        stat_results = {}
        for m in [1, 2, 3]:
            n_rms = df[(df["machine"] == m) & (df["state"] == "normal")]["rms_energy"].values
            a_rms = df[(df["machine"] == m) & (df["state"] == "abnormal")]["rms_energy"].values
            if len(n_rms) == 0 or len(a_rms) == 0:
                print(f"Machine {m}: insufficient data — skipping.")
                continue
            stat, p = mannwhitneyu(n_rms, a_rms, alternative="two-sided")
            effect   = 1 - (2 * stat) / (len(n_rms) * len(a_rms))
            sig      = "SIGNIFICANT" if p < 0.05 else "not significant"
            mag      = "large" if abs(effect) > 0.5 else "medium" if abs(effect) > 0.3 else "small"
            print(f"\nMachine {m}:")
            print(f"  p-value    = {p:.4f} ({sig})")
            print(f"  Effect (r) = {effect:.3f}  [{mag}]")
            stat_results[m] = {"p_value": p, "effect_size": effect}
            mlflow.log_metric(f"machine{m}_mw_pvalue",      p)
            mlflow.log_metric(f"machine{m}_mw_effect_size", effect)

        # ---------------------------------------------------------------
        # EDA 13 — Summary & Findings
        # ---------------------------------------------------------------
        print("\n" + "=" * 60)
        print("EDA 13 — Summary & Findings")
        print("=" * 60)

        dur_stats = df.groupby("label")["duration_s"].agg(["mean", "std", "min", "max"])
        dur_stats.index = [label_names[i] for i in dur_stats.index]
        print("\nDuration statistics per class (seconds):")
        print(dur_stats.round(3).to_string())

        print(f"\nClass Balance:")
        for label_id, cnt in counts.items():
            print(f"  {label_names[label_id]:<15}: {cnt} samples")
        print(f"  Imbalance ratio (Normal/Abnormal): {n_normal / max(n_abnormal, 1):.2f}")

        print(f"\nDuration: mean={df['duration_s'].mean():.2f}s, "
              f"std={df['duration_s'].std():.2f}s")
        print(f"Files needing padding (<10s): {needs_padding_pct:.1f}%")

        print(f"\nSample rates found  : {df['sample_rate'].unique().tolist()}")
        print(f"Channel counts found: {df['num_channels'].unique().tolist()}")

        print(f"\nSilence ratio: mean={df['silence_ratio'].mean():.3f}, "
              f"max={df['silence_ratio'].max():.3f}")

        print(f"\nOutliers (z-score > 3): {len(outliers)} files "
              f"({100.0 * len(outliers) / len(df):.1f}%)")

        if stat_results:
            all_sig = all(v["p_value"] < 0.05 for v in stat_results.values())
            print(f"\nStatistical separability (RMS): "
                  f"{'all machines significant' if all_sig else 'mixed results'}")

        # Save summary as a text artifact
        summary_path = os.path.join(out_dir, "eda_summary.txt")
        with open(summary_path, "w") as f:
            f.write("      EDA FINAL SUMMARY\n")
            f.write("=" * 30 + "\n")
            f.write(f"Total Files Audited : {len(df_all)}\n")
            f.write(f"Sampling Used       : {len(df)} files\n")
            f.write(f"Average Duration    : {df_all['duration_s'].mean():.2f}s\n")
            f.write(f"Normal/Abnormal     : {n_normal}/{n_abnormal}\n")
            f.write(f"Imbalance Ratio     : {n_normal/max(1, n_abnormal):.2f}\n")
            f.write(f"Silence Ratio Mean  : {df['silence_ratio'].mean():.3f}\n")
            f.write(f"Outliers Found      : {len(outliers)}\n")

        # ---------------------------------------------------------------
        # FINAL LOUD SUMMARY (So you see the numbers immediately!)
        # ---------------------------------------------------------------
        print("\n" + "⭐" * 30)
        print("       EDA FINAL SUMMARY")
        print("⭐" * 30)
        print(f"Total Files Audited : {len(df_all)}")
        print(f"Sampling Used       : {len(df)} files")
        print(f"Average Duration    : {df_all['duration_s'].mean():.2f} seconds")
        print(f"Normal/Abnormal     : {n_normal}/{n_abnormal} files")
        print(f"Imbalance Ratio     : {n_normal/max(1, n_abnormal):.2f}")
        print(f"Silence Ratio Mean  : {df['silence_ratio'].mean():.3f}")
        print(f"Outliers Found      : {len(outliers)}")
        
        # Mann-Whitney Results
        for m, res in stat_results.items():
            sig = "✅ Significant" if res['p_value'] < 0.05 else "❌ Not Significant"
            print(f"Machine {m} Diff      : {sig} (p={res['p_value']:.4f})")
        
        print("\n🎨 ALL GRAPHS SAVED TO: " + str(out_dir))
        print("⭐" * 30)

        mlflow.log_artifact(summary_path)
        print(f"\nEDA complete! Check your DAGsHub 'Artifacts' tab for the graphs.")

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def display_results(out_dir):
    """Utility to pop up saved graphs in a Kaggle notebook cell."""
    if not IN_NOTEBOOK: return
    print("\n" + "🖼️" * 15)
    print("   DISPLAYING GENERATED REPORTS")
    print("🖼️" * 15)
    for filename in sorted(os.listdir(out_dir)):
        if filename.endswith(".png"):
            print(f"\n📊 {filename}:")
            display(Image(filename=os.path.join(out_dir, filename)))

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # 1. Smart Path Discovery for Kaggle
    KAG_PATH = "/kaggle/input/datasets/alieldinalaa/nn-cmp27-dataset"
    if not os.path.exists(KAG_PATH):
        for root, dirs, files in os.walk('/kaggle/input'):
            if 'Machine 3' in dirs or 'machine_data' in dirs:
                KAG_PATH = root
                break

    parser = argparse.ArgumentParser(description="Full EDA for Machine Fault Recognition")
    parser.add_argument("--data-root", type=str, default=KAG_PATH)
    parser.add_argument("--out-dir",   type=str, default="outputs/full_eda")
    parser.add_argument("--sample-per-class", type=int, default=1500)
    parser.add_argument("--force-recompute", action="store_true")
    args, unknown = parser.parse_known_args()

    # 2. Integrated DAGsHub Setup
    try:
        import dagshub
        from kaggle_secrets import UserSecretsClient
        secrets = UserSecretsClient()
        token = secrets.get_secret("DAGSHUB_TOKEN")
        if token:
            os.environ["DAGSHUB_TOKEN"] = token
            dagshub.init(repo_owner="MohamedAbdelaiem", 
                         repo_name="machine-sound-classification", 
                         mlflow=True)
            print("🔗 Connected to DAGsHub!")
    except Exception as e:
        print(f"💡 Info: DAGsHub init skipped or failed: {e}")

    # 3. Main Run
    if not os.path.exists(args.data_root):
        print(f"❌ Error: Data not found at {args.data_root}. Provide --data-root.")
    else:
        run_eda(args.data_root, args.out_dir, args.sample_per_class, args.force_recompute)
        display_results(args.out_dir)