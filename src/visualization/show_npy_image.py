import argparse
from pathlib import Path

import numpy as np


def _select_2d_slice(array: np.ndarray, channel_index: int) -> np.ndarray:
    if array.ndim == 2:
        return array

    if array.ndim == 3:
        if channel_index < 0 or channel_index >= array.shape[0]:
            raise IndexError(
                f"channel_index out of range. Expected 0..{array.shape[0] - 1}, got {channel_index}."
            )
        return array[channel_index]

    raise ValueError(
        f"Unsupported array shape {array.shape}. Expected 2D (H, W) or 3D (C, H, W)."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Display an image from a .npy file (e.g., mel spectrogram)."
    )
    parser.add_argument("npy_path", type=Path, help="Path to the .npy file")
    parser.add_argument(
        "--channel",
        type=int,
        default=0,
        help="Channel index for 3D arrays (C, H, W). Ignored for 2D arrays.",
    )
    parser.add_argument(
        "--cmap",
        default="magma",
        help="Matplotlib colormap (default: magma)",
    )
    parser.add_argument(
        "--origin",
        choices=["lower", "upper"],
        default="lower",
        help="Image origin (default: lower). Use upper if your reference plot is flipped vertically.",
    )
    parser.add_argument(
        "--transpose",
        action="store_true",
        help="Transpose the image before plotting. Useful when time/mel axes appear swapped.",
    )
    parser.add_argument(
        "--scale",
        choices=["auto", "linear", "db"],
        default="auto",
        help="Display scale. 'auto' uses dB for non-negative arrays with large dynamic range.",
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=1e-10,
        help="Small floor used when dB scale is applied to avoid log(0).",
    )
    parser.add_argument(
        "--vmin",
        type=float,
        default=None,
        help="Manual lower color limit. If omitted, percentile limits are used.",
    )
    parser.add_argument(
        "--vmax",
        type=float,
        default=None,
        help="Manual upper color limit. If omitted, percentile limits are used.",
    )
    parser.add_argument(
        "--pmin",
        type=float,
        default=1.0,
        help="Lower percentile for auto color scaling (default: 1).",
    )
    parser.add_argument(
        "--pmax",
        type=float,
        default=99.5,
        help="Upper percentile for auto color scaling (default: 99.5).",
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Optional plot title",
    )
    parser.add_argument(
        "--save-path",
        type=Path,
        default=None,
        help="Optional output image path. Defaults to <npy_stem>.png next to the input file.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Save the image without opening a display window.",
    )
    args = parser.parse_args()

    if not args.npy_path.exists():
        raise FileNotFoundError(f"File not found: {args.npy_path}")

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required to display images. Install it with: pip install matplotlib"
        ) from exc

    arr = np.load(args.npy_path)
    image = _select_2d_slice(arr, args.channel)

    if args.transpose:
        image = image.T

    apply_db = False
    if args.scale == "db":
        apply_db = True
    elif args.scale == "auto":
        # Typical mel-power arrays are non-negative with high dynamic range.
        apply_db = bool(np.min(image) >= 0 and np.max(image) > 20)

    if apply_db:
        image = 10.0 * np.log10(np.maximum(image, args.eps))

    if args.vmin is not None and args.vmax is not None:
        vmin, vmax = args.vmin, args.vmax
    else:
        vmin = float(np.percentile(image, args.pmin))
        vmax = float(np.percentile(image, args.pmax))
        if vmax <= vmin:
            vmin = float(np.min(image))
            vmax = float(np.max(image))

    print(f"Array shape: {arr.shape}, plotted shape: {image.shape}, dtype: {arr.dtype}")
    print(f"Value range shown: min={float(np.min(image)):.6f}, max={float(np.max(image)):.6f}")
    print(f"Display scale: {'dB' if apply_db else 'linear'} | color limits: vmin={vmin:.6f}, vmax={vmax:.6f}")

    plt.figure(figsize=(10, 4))
    plt.imshow(
        image,
        aspect="auto",
        origin=args.origin,
        cmap=args.cmap,
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
    )
    plt.colorbar()
    plt.xlabel("Time")
    plt.ylabel("Mel bins")
    plt.title(args.title if args.title else args.npy_path.name)
    plt.tight_layout()

    out_path = args.save_path if args.save_path else args.npy_path.with_suffix(".png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    print(f"Saved image: {out_path}")

    if not args.no_show:
        plt.show()
    else:
        plt.close()


if __name__ == "__main__":
    main()
