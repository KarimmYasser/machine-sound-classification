import argparse
from pathlib import Path

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, RadioButtons, Slider


def find_wav_files(folder: Path) -> list[Path]:
    return sorted(folder.glob("*.wav"))


def pick_file(files: list[Path], requested: str | None) -> Path:
    if not files:
        raise FileNotFoundError("No .wav files found in the selected folder.")

    if requested is None:
        return files[0]

    requested_path = Path(requested)
    if requested_path.is_file():
        return requested_path

    candidate = next((f for f in files if f.name == requested), None)
    if candidate is None:
        raise FileNotFoundError(f"Requested file '{requested}' was not found.")
    return candidate


def load_audio(file_path: Path, sr: int | None):
    signal, sample_rate = librosa.load(file_path, sr=sr)
    return signal, sample_rate


def compute_frequency_spectrum(signal: np.ndarray, sr: int):
    magnitude = np.abs(np.fft.rfft(signal))
    freqs = np.fft.rfftfreq(signal.size, d=1.0 / sr)
    spectrum_db = 20 * np.log10(np.maximum(magnitude, 1e-12))
    return freqs, spectrum_db


def plot_waveforms(
    normal_signal: np.ndarray,
    abnormal_signal: np.ndarray,
    sr: int,
    normal_name: str,
    abnormal_name: str,
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 4), sharey=True)

    librosa.display.waveshow(normal_signal, sr=sr, ax=axes[0], color="#1f77b4")
    axes[0].set_title(f"Normal Waveform\n{normal_name}")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Amplitude")

    librosa.display.waveshow(abnormal_signal, sr=sr, ax=axes[1], color="#d62728")
    axes[1].set_title(f"Abnormal Waveform\n{abnormal_name}")
    axes[1].set_xlabel("Time (s)")

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_mel_spectrograms(
    normal_signal: np.ndarray,
    abnormal_signal: np.ndarray,
    sr: int,
    normal_name: str,
    abnormal_name: str,
    out_path: Path,
    n_mels: int,
    hop_length: int,
) -> None:
    normal_mel = librosa.feature.melspectrogram(
        y=normal_signal, sr=sr, n_mels=n_mels, hop_length=hop_length
    )
    abnormal_mel = librosa.feature.melspectrogram(
        y=abnormal_signal, sr=sr, n_mels=n_mels, hop_length=hop_length
    )

    normal_db = librosa.power_to_db(normal_mel, ref=np.max)
    abnormal_db = librosa.power_to_db(abnormal_mel, ref=np.max)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    normal_img = librosa.display.specshow(
        normal_db,
        sr=sr,
        hop_length=hop_length,
        x_axis="time",
        y_axis="mel",
        ax=axes[0],
        cmap="magma",
    )
    axes[0].set_title(f"Normal Mel-Spectrogram\n{normal_name}")

    abnormal_img = librosa.display.specshow(
        abnormal_db,
        sr=sr,
        hop_length=hop_length,
        x_axis="time",
        y_axis="mel",
        ax=axes[1],
        cmap="magma",
    )
    axes[1].set_title(f"Abnormal Mel-Spectrogram\n{abnormal_name}")

    fig.colorbar(normal_img, ax=axes[0], format="%+2.0f dB")
    fig.colorbar(abnormal_img, ax=axes[1], format="%+2.0f dB")

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


class InteractiveAudioExplorer:
    def __init__(
        self,
        data_root: Path,
        sr: int | None,
        n_mels: int,
        hop_length: int,
    ) -> None:
        self.data_root = data_root
        self.sr = sr
        self.n_mels = n_mels
        self.hop_length = hop_length

        self.machine_names = ["Machine 1", "Machine 2", "Machine 3"]
        self.state_names = ["Normal", "Abnormal"]
        self.graph_types = ["time_domain", "freq_domain", "mel_spectrogram"]

        self.current_machine = self.machine_names[0]
        self.current_state = self.state_names[0]
        self.current_graph_type = self.graph_types[0]
        self.zoom_axis_mode = "both"

        self.sample_files: list[Path] = []
        self.current_index = 0
        self.current_signal: np.ndarray | None = None
        self.current_sr: int | None = None
        self.current_mel_db: np.ndarray | None = None

        self.fig = plt.figure(figsize=(14, 8))
        self.ax_plot = self.fig.add_axes([0.07, 0.12, 0.62, 0.80])
        self.ax_colorbar = self.fig.add_axes([0.70, 0.12, 0.02, 0.80])
        self.ax_colorbar.set_visible(False)

        self.ax_machine = self.fig.add_axes([0.74, 0.71, 0.22, 0.20])
        self.ax_state = self.fig.add_axes([0.74, 0.56, 0.22, 0.12])
        self.ax_graph = self.fig.add_axes([0.74, 0.38, 0.22, 0.15])
        self.ax_zoom_mode = self.fig.add_axes([0.74, 0.24, 0.22, 0.11])
        self.ax_slider = self.fig.add_axes([0.74, 0.17, 0.22, 0.04])
        self.ax_prev = self.fig.add_axes([0.74, 0.10, 0.10, 0.05])
        self.ax_next = self.fig.add_axes([0.86, 0.10, 0.10, 0.05])

        self.machine_radio = RadioButtons(self.ax_machine, self.machine_names)
        self.state_radio = RadioButtons(self.ax_state, self.state_names)
        self.graph_radio = RadioButtons(self.ax_graph, self.graph_types)
        self.zoom_radio = RadioButtons(self.ax_zoom_mode, ["x", "y", "both"])

        self.sample_slider = Slider(
            self.ax_slider,
            "Sample",
            0,
            1,
            valinit=0,
            valstep=1,
        )

        self.colorbar = None

        self.prev_button = Button(self.ax_prev, "Prev")
        self.next_button = Button(self.ax_next, "Next")

        self.machine_radio.on_clicked(self.on_machine_change)
        self.state_radio.on_clicked(self.on_state_change)
        self.graph_radio.on_clicked(self.on_graph_change)
        self.zoom_radio.on_clicked(self.on_zoom_mode_change)
        self.sample_slider.on_changed(self.on_slider_change)
        self.prev_button.on_clicked(self.on_prev)
        self.next_button.on_clicked(self.on_next)
        self.fig.canvas.mpl_connect("scroll_event", self.on_scroll)

        self.refresh_samples(reset_index=True)
        self.update_plot()

    def get_folder(self) -> Path:
        return (
            self.data_root
            / self.current_machine
            / "machine_data"
            / self.current_state
        )

    def refresh_samples(self, reset_index: bool) -> None:
        self.sample_files = find_wav_files(self.get_folder())
        if not self.sample_files:
            self.current_index = 0
            self.current_signal = None
            self.current_sr = None
            self.current_mel_db = None
            self.sample_slider.valmin = 0
            self.sample_slider.valmax = 1
            self.sample_slider.ax.set_xlim(0, 1)
            self.sample_slider.set_active(False)
            self.sample_slider.set_val(0)
            self.sample_slider.ax.set_title("No samples found")
            return

        if reset_index:
            self.current_index = 0
        else:
            self.current_index = min(self.current_index, len(self.sample_files) - 1)

        self.sample_slider.valmin = 0
        self.sample_slider.valmax = max(1, len(self.sample_files) - 1)
        self.sample_slider.ax.set_xlim(self.sample_slider.valmin, self.sample_slider.valmax)
        self.sample_slider.set_active(len(self.sample_files) > 1)
        self.sample_slider.ax.set_title("")
        self.sample_slider.set_val(self.current_index)

    def load_current_signal(self) -> None:
        if not self.sample_files:
            self.current_signal = None
            self.current_sr = None
            self.current_mel_db = None
            return

        file_path = self.sample_files[self.current_index]
        signal, sample_rate = load_audio(file_path, self.sr)
        self.current_signal = signal
        self.current_sr = sample_rate
        mel = librosa.feature.melspectrogram(
            y=signal,
            sr=sample_rate,
            n_mels=self.n_mels,
            hop_length=self.hop_length,
        )
        self.current_mel_db = librosa.power_to_db(mel, ref=np.max)

    def update_plot(self) -> None:
        self.ax_plot.clear()
        self.load_current_signal()

        if self.current_signal is None or self.current_sr is None:
            if self.colorbar is not None:
                self.ax_colorbar.cla()
                self.ax_colorbar.set_visible(False)
                self.colorbar = None
            self.ax_plot.text(0.5, 0.5, "No samples available", ha="center", va="center")
            self.ax_plot.set_axis_off()
            self.fig.canvas.draw_idle()
            return

        self.ax_plot.set_axis_on()
        file_name = self.sample_files[self.current_index].name

        if self.current_graph_type == "time_domain":
            librosa.display.waveshow(
                self.current_signal,
                sr=self.current_sr,
                ax=self.ax_plot,
                color="#1f77b4" if self.current_state == "Normal" else "#d62728",
            )
            self.ax_plot.set_xlabel("Time (s)")
            self.ax_plot.set_ylabel("Amplitude")
            self.ax_plot.set_title(
                f"Time Domain | {self.current_machine} | {self.current_state} | {file_name}"
            )

        elif self.current_graph_type == "freq_domain":
            freqs, spectrum_db = compute_frequency_spectrum(self.current_signal, self.current_sr)
            self.ax_plot.plot(
                freqs,
                spectrum_db,
                color="#1f77b4" if self.current_state == "Normal" else "#d62728",
                linewidth=1.0,
            )
            self.ax_plot.set_xlabel("Frequency (Hz)")
            self.ax_plot.set_ylabel("Magnitude (dB)")
            self.ax_plot.set_title(
                f"Frequency Domain | {self.current_machine} | {self.current_state} | {file_name}"
            )

        else:
            mel_img = librosa.display.specshow(
                self.current_mel_db,
                sr=self.current_sr,
                hop_length=self.hop_length,
                x_axis="time",
                y_axis="mel",
                cmap="magma",
                ax=self.ax_plot,
            )
            self.ax_plot.set_title(
                f"Mel Spectrogram | {self.current_machine} | {self.current_state} | {file_name}"
            )
            self.ax_colorbar.set_visible(True)
            if self.colorbar is None:
                self.colorbar = self.fig.colorbar(
                    mel_img,
                    cax=self.ax_colorbar,
                    format="%+2.0f dB",
                )
            else:
                self.colorbar.update_normal(mel_img)

        if self.current_graph_type != "mel_spectrogram":
            if self.colorbar is not None:
                self.ax_colorbar.cla()
                self.ax_colorbar.set_visible(False)
            self.colorbar = None

        self.ax_plot.grid(alpha=0.2)
        self.fig.canvas.draw_idle()

    def on_machine_change(self, label: str) -> None:
        self.current_machine = label
        self.refresh_samples(reset_index=True)
        self.update_plot()

    def on_state_change(self, label: str) -> None:
        self.current_state = label
        self.refresh_samples(reset_index=True)
        self.update_plot()

    def on_graph_change(self, label: str) -> None:
        self.current_graph_type = label
        self.update_plot()

    def on_zoom_mode_change(self, label: str) -> None:
        self.zoom_axis_mode = label

    def on_slider_change(self, value: float) -> None:
        if not self.sample_files:
            return
        new_index = min(int(value), len(self.sample_files) - 1)
        if new_index != self.current_index:
            self.current_index = new_index
            self.update_plot()

    def on_prev(self, _event) -> None:
        if not self.sample_files:
            return
        self.current_index = max(0, self.current_index - 1)
        self.sample_slider.set_val(self.current_index)

    def on_next(self, _event) -> None:
        if not self.sample_files:
            return
        self.current_index = min(len(self.sample_files) - 1, self.current_index + 1)
        self.sample_slider.set_val(self.current_index)

    def on_scroll(self, event) -> None:
        if event.inaxes != self.ax_plot:
            return

        xlim = self.ax_plot.get_xlim()
        ylim = self.ax_plot.get_ylim()
        base_scale = 1.1
        scale_factor = 1 / base_scale if event.button == "up" else base_scale

        xdata = event.xdata if event.xdata is not None else sum(xlim) / 2
        ydata = event.ydata if event.ydata is not None else sum(ylim) / 2

        if self.zoom_axis_mode in ("x", "both"):
            new_width = (xlim[1] - xlim[0]) * scale_factor
            relx = (xlim[1] - xdata) / max(xlim[1] - xlim[0], 1e-12)
            self.ax_plot.set_xlim([xdata - new_width * (1 - relx), xdata + new_width * relx])

        if self.zoom_axis_mode in ("y", "both"):
            new_height = (ylim[1] - ylim[0]) * scale_factor
            rely = (ylim[1] - ydata) / max(ylim[1] - ylim[0], 1e-12)
            self.ax_plot.set_ylim(
                [ydata - new_height * (1 - rely), ydata + new_height * rely]
            )

        self.fig.canvas.draw_idle()

    def show(self) -> None:
        self.fig.suptitle(
            "Interactive Audio Explorer | Scroll to zoom (pick x/y/both on the right)",
            fontsize=12,
        )
        plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase-1 EDA baseline: compare Normal vs Abnormal audio for one machine."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data"),
        help="Path to dataset root containing 'Machine 1..3'.",
    )
    parser.add_argument(
        "--machine",
        type=str,
        default="Machine 1",
        help="Machine folder name (e.g., 'Machine 1').",
    )
    parser.add_argument(
        "--normal-file",
        type=str,
        default=None,
        help="Optional normal .wav filename or full path.",
    )
    parser.add_argument(
        "--abnormal-file",
        type=str,
        default=None,
        help="Optional abnormal .wav filename or full path.",
    )
    parser.add_argument(
        "--sr",
        type=int,
        default=None,
        help="Target sampling rate. Use None to keep original.",
    )
    parser.add_argument(
        "--n-mels",
        type=int,
        default=128,
        help="Number of Mel bands for Mel-spectrogram.",
    )
    parser.add_argument(
        "--hop-length",
        type=int,
        default=512,
        help="Hop length for STFT/Mel-spectrogram.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/eda"),
        help="Directory where figures are saved.",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Open interactive window to browse samples and graph types.",
    )

    args = parser.parse_args()

    if args.interactive:
        explorer = InteractiveAudioExplorer(
            data_root=args.data_root,
            sr=args.sr,
            n_mels=args.n_mels,
            hop_length=args.hop_length,
        )
        explorer.show()
        return

    machine_dir = args.data_root / args.machine / "machine_data"
    normal_dir = machine_dir / "Normal"
    abnormal_dir = machine_dir / "Abnormal"

    normal_files = find_wav_files(normal_dir)
    abnormal_files = find_wav_files(abnormal_dir)

    normal_file = pick_file(normal_files, args.normal_file)
    abnormal_file = pick_file(abnormal_files, args.abnormal_file)

    normal_signal, sr_normal = load_audio(normal_file, args.sr)
    abnormal_signal, sr_abnormal = load_audio(abnormal_file, args.sr)

    if sr_normal != sr_abnormal:
        raise ValueError(
            "Sample rates differ after loading. Use --sr to force a shared sample rate."
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)

    waveform_path = args.out_dir / f"{args.machine.replace(' ', '_').lower()}_waveforms.png"
    melspec_path = args.out_dir / f"{args.machine.replace(' ', '_').lower()}_melspec.png"

    plot_waveforms(
        normal_signal,
        abnormal_signal,
        sr_normal,
        normal_file.name,
        abnormal_file.name,
        waveform_path,
    )
    plot_mel_spectrograms(
        normal_signal,
        abnormal_signal,
        sr_normal,
        normal_file.name,
        abnormal_file.name,
        melspec_path,
        args.n_mels,
        args.hop_length,
    )

    print("EDA plots saved:")
    print(f"- {waveform_path}")
    print(f"- {melspec_path}")


if __name__ == "__main__":
    main()
