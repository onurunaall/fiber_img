"""Measures the period and orientation of the fiber-core lattice in the sim_MCF input images.

SGARNet's SpectralGate damps the lattice peaks in the frequency domain. It needs two numbers:
    period: 1 / frequency of the first-order spectral peaks, in input pixels
    angle:  angle of one first-order peak in degrees (the other five are 60 degrees apart),
            measured from the column-frequency axis towards the row-frequency axis

Usage:
    python estimate_lattice_period.py --dir_X /workspace/data/sim_MCF_Train --plot lattice_spectrum.png

It prints the train.py flags to use, for example:
    --sgarnet_lattice_period 5.3012 --sgarnet_lattice_angle 12.04
Check the plot: the six circles must sit on the six brightest spots around the centre.
"""
import argparse
import math
import sys

import numpy as np

from imageDatastore import default_loader, make_dataset
from lib import common


# SGARNet applies the gate after 4 encoders, i.e. at 1/16 of the input resolution.
BOTTLENECK_DOWN_SCALE = 16
# Standard deviation of the mask peaks (gate_bandwidth 0.06 / 2), in cycles per bottleneck pixel.
GATE_PEAK_SIGMA = 0.03


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure the core lattice period and angle for SGARNet.")
    parser.add_argument("--dir_X", required=True, type=str, help="folder with the sim_MCF input images")
    parser.add_argument("--num_images", default=100, type=int, help="number of images to average (spread evenly)")
    parser.add_argument("--max_period", default=64.0, type=float, help="ignore lattice periods longer than this (px)")
    parser.add_argument("--plot", default="", type=str, help="save the averaged spectrum with the found peaks here")
    return parser.parse_args()


def load_mean_power_spectrum(paths: list[str]) -> np.ndarray:
    total = None
    for path in paths:
        img = default_loader(path, n_colors=1)
        img = common.set_channel(img, n_channels=1)[0][:, :, 0].astype(np.float64)
        if total is None:
            shape = img.shape
            window = np.outer(np.hanning(shape[0]), np.hanning(shape[1]))
            total = np.zeros(shape)
        elif img.shape != shape:
            raise ValueError(f"All images must have the same size: {path} is {img.shape}, the first was {shape}")

        # The window keeps the image borders from smearing the spectrum.
        img = (img - img.mean()) * window
        total += np.abs(np.fft.fft2(img)) ** 2
    return total / len(paths)


def excess_over_ring_median(log_power: np.ndarray, radius: np.ndarray, bin_width: float) -> np.ndarray:
    """log power minus the median log power of its ring: lattice peaks stand out, smooth image content does not."""
    ring = np.round(radius / bin_width).astype(int).ravel()
    values = log_power.ravel()
    order = np.argsort(ring, kind="stable")
    rings, starts = np.unique(ring[order], return_index=True)
    median = np.zeros(ring.max() + 1)
    for r, chunk in zip(rings, np.split(values[order], starts[1:])):
        median[r] = np.median(chunk)
    return (values - median[ring]).reshape(log_power.shape)


def locate_peak(excess: np.ndarray, log_power: np.ndarray, fu: float, fv: float, search: int = 2) -> tuple[float, float, float]:
    """Finds the peak near (fu, fv) and refines it to sub-bin precision. Returns (fu, fv, excess)."""
    H, W = excess.shape
    r0, c0 = round(fv * H), round(fu * W)
    best = None
    for dr in range(-search, search + 1):
        for dc in range(-search, search + 1):
            value = excess[(r0 + dr) % H, (c0 + dc) % W]
            if best is None or value > best[0]:
                best = (value, r0 + dr, c0 + dc)
    value, r, c = best

    def parabola_offset(left: float, centre: float, right: float) -> float:
        denominator = left - 2 * centre + right
        return 0.0 if denominator >= 0 else 0.5 * (left - right) / denominator

    dr = parabola_offset(log_power[(r - 1) % H, c % W], log_power[r % H, c % W], log_power[(r + 1) % H, c % W])
    dc = parabola_offset(log_power[r % H, (c - 1) % W], log_power[r % H, c % W], log_power[r % H, (c + 1) % W])
    wrap = lambda f: ((f + 0.5) % 1.0) - 0.5
    return wrap((c + dc) / W), wrap((r + dr) / H), float(value)


def six_peaks(excess, log_power, radius_f, angle_deg):
    peaks = []
    for k in range(6):
        t = math.radians(angle_deg + 60 * k)
        peaks.append(locate_peak(excess, log_power, radius_f * math.cos(t), radius_f * math.sin(t)))
    return peaks


def save_plot(path, log_power, peaks, period, angle):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    H, W = log_power.shape
    fu = np.fft.fftshift(np.fft.fftfreq(W))
    fv = np.fft.fftshift(np.fft.fftfreq(H))
    extent = (fu[0] - 0.5 / W, fu[-1] + 0.5 / W, fv[-1] + 0.5 / H, fv[0] - 0.5 / H)

    plt.figure(figsize=(8, 8 * H / W + 0.5))
    plt.imshow(np.fft.fftshift(log_power) / math.log(10), cmap="gray", extent=extent, origin="upper")
    plt.scatter([p[0] for p in peaks], [p[1] for p in peaks], s=160, facecolors="none", edgecolors="red",
                label="found first-order peaks")
    second = [(2 / period * math.cos(math.radians(angle + 60 * k)), 2 / period * math.sin(math.radians(angle + 60 * k)))
              for k in range(6)]
    second = [p for p in second if abs(p[0]) < 0.5 and abs(p[1]) < 0.5]
    if second:
        plt.scatter([p[0] for p in second], [p[1] for p in second], marker="x", color="orange",
                    label="second order (from the fit)")
    plt.xlabel("column frequency (cycles / px)")
    plt.ylabel("row frequency (cycles / px)")
    plt.title(f"Mean log10 power spectrum, period {period:.4f} px, angle {angle:.2f} deg")
    plt.legend(loc="upper right")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved {path}")


def main() -> None:
    args = parse_args()

    paths = sorted(make_dataset(args.dir_X))
    if not paths:
        sys.exit(f"No images in {args.dir_X}")
    count = min(args.num_images, len(paths))
    paths = [paths[i] for i in np.linspace(0, len(paths) - 1, count).round().astype(int)]

    power = load_mean_power_spectrum(paths)
    H, W = power.shape
    log_power = np.log(power + 1e-12 * power.max())
    fv = np.fft.fftfreq(H)[:, None]
    fu = np.fft.fftfreq(W)[None, :]
    radius = np.sqrt(fu ** 2 + fv ** 2)
    angle = np.degrees(np.arctan2(fv, fu))
    excess = excess_over_ring_median(log_power, radius, bin_width=1.0 / min(H, W))

    searchable = (radius >= 1.0 / args.max_period) & (radius <= 0.5)
    strongest = np.unravel_index(np.argmax(np.where(searchable, excess, -np.inf)), excess.shape)
    f1, a1, e1 = radius[strongest], angle[strongest], excess[strongest]
    print(f"{count} images of {H} x {W} px from {args.dir_X}")
    print(f"Strongest peak: radius {f1:.5f} cycles/px (period {1 / f1:.3f} px), angle {a1:.2f} deg, "
          f"{10 * e1 / math.log(10):.1f} dB above its ring")

    # The strongest peak is normally first order. If it is a higher order of a hexagonal lattice,
    # the first-order peaks are at half its radius, or at 1/sqrt(3) of it and rotated by 30 degrees.
    threshold = max(math.log(4), 0.2 * e1)
    candidates = [
        ("strongest peak is second order", f1 / 2, a1),
        ("strongest peak is sqrt(3) order", f1 / math.sqrt(3), a1 + 30),
        ("strongest peak is first order", f1, a1),
    ]
    chosen = None
    for name, f, a in candidates:
        peaks = six_peaks(excess, log_power, f, a)
        weakest = min(p[2] for p in peaks)
        passed = 1.0 / f <= args.max_period and weakest >= threshold
        print(f"  check '{name}': weakest of 6 peaks {10 * weakest / math.log(10):.1f} dB "
              f"(need {10 * threshold / math.log(10):.1f} dB) -> {'yes' if passed else 'no'}")
        if passed and chosen is None:
            chosen = peaks

    if chosen is None:
        sys.exit("No six-fold (hexagonal) peak pattern found. The lattice may not be hexagonal, or --max_period "
                 "is too small. SGARNet's SpectralGate assumes a hexagonal lattice.")

    radii = np.array([math.hypot(p[0], p[1]) for p in chosen])
    angles = np.array([math.degrees(math.atan2(p[1], p[0])) for p in chosen])
    period = 1.0 / radii.mean()
    # Mean of angles modulo 60 degrees: average the unit vectors of 6 * angle.
    lattice_angle = (math.degrees(np.angle(np.exp(1j * np.radians(6 * angles)).sum())) / 6) % 60

    print("\nFirst-order peaks (column freq, row freq, radius, angle, height above ring):")
    for (pu, pv, e), r, a in zip(chosen, radii, angles):
        print(f"  ({pu:+.5f}, {pv:+.5f})  r {r:.5f}  {a:+8.2f} deg  {10 * e / math.log(10):5.1f} dB")

    spread = radii.std() / radii.mean()
    bottleneck_offset = BOTTLENECK_DOWN_SCALE * radii.std()
    print(f"\nLattice period: {period:.4f} px (spread of the 6 radii: {100 * spread:.2f} %)")
    print(f"Lattice angle:  {lattice_angle:.2f} deg (peaks at {lattice_angle:.2f} + k * 60 deg)")
    print(f"If the cores form a regular hexagonal lattice, their centre-to-centre distance is "
          f"{2 / math.sqrt(3) * period:.3f} px.")
    print(f"At the SGARNet bottleneck (1/{BOTTLENECK_DOWN_SCALE} resolution) the radius spread moves the peaks by "
          f"~{bottleneck_offset:.4f} cycles/px; the mask peaks have a standard deviation of {GATE_PEAK_SIGMA}.")
    if bottleneck_offset > 0.5 * GATE_PEAK_SIGMA:
        print("WARNING: the 6 peaks are not on one circle precisely enough for the bottleneck gate. "
              "The lattice may be distorted; the gate will only partly cover the aliased peaks.")

    # A flip maps the angle a to -a (or 180 - a), a transpose maps it to 90 - a. The set a + k * 60 stays the
    # same for flips when a is a multiple of 30 degrees, and for the transpose when a is 15 + a multiple of 30.
    flips_keep = abs(((lattice_angle + 15) % 30) - 15) < 0.5
    transpose_keeps = abs((lattice_angle % 30) - 15) < 0.5
    print(f"Augmentation: flips {'keep' if flips_keep else 'do NOT keep'} the peaks on the gate angles; "
          f"rot90 (transpose) {'keeps' if transpose_keeps else 'does NOT keep'} them "
          f"(it moves the angle to {(90 - lattice_angle) % 60:.2f} deg).")

    print("\nUse these train.py flags:")
    print(f"    --sgarnet_lattice_period {period:.4f} --sgarnet_lattice_angle {lattice_angle:.2f}")

    if args.plot:
        save_plot(args.plot, log_power, chosen, period, lattice_angle)


if __name__ == "__main__":
    main()
