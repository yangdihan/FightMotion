import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
from constants import COLOR_RANGES

# from mpl_toolkits.mplot3d import Axes3D


def plot_3d_color_spectrum(color_name, num_samples=1000):
    """
    Plot a 3D color spectrum for the given color name.

    :param color_name: Name of the color to plot (key in COLOR_RANGES).
    :param num_samples: Number of samples to generate for each range.
    """
    if color_name not in COLOR_RANGES:
        raise ValueError(f"Color name '{color_name}' not found in COLOR_RANGES")

    h_samples = []
    s_samples = []
    v_samples = []
    rgb_colors = []

    for lower, upper in COLOR_RANGES[color_name]:
        h_range = np.random.uniform(lower[0], upper[0], num_samples)
        s_range = np.random.uniform(lower[1], upper[1], num_samples)
        v_range = np.random.uniform(lower[2], upper[2], num_samples)
        h_samples.extend(h_range)
        s_samples.extend(s_range)
        v_samples.extend(v_range)

        hsv_samples = np.stack((h_range, s_range, v_range), axis=-1).astype(np.uint8)
        rgb_samples = cv2.cvtColor(hsv_samples[None, :, :], cv2.COLOR_HSV2RGB)[0]
        rgb_colors.extend(rgb_samples / 255.0)

    h_samples = np.array(h_samples)
    s_samples = np.array(s_samples)
    v_samples = np.array(v_samples)
    rgb_colors = np.array(rgb_colors)

    # print(h_samples.shape, rgb_colors.shape)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    scatter = ax.scatter(h_samples, s_samples, v_samples, c=rgb_colors, marker="o")
    ax.set_xlabel("Hue")
    ax.set_ylabel("Saturation")
    ax.set_zlabel("Value")
    ax.set_title(f"3D Color Spectrum for {color_name.capitalize()}")
    ax.set_xlim(0, 180)  # Constrain Hue to 0-180
    ax.set_ylim(0, 255)  # Constrain Saturation to 0-255
    ax.set_zlim(0, 255)  # Constrain Value to 0-255
    plt.show()
    return


def check_color_ranges_coverage():
    coverage_mask = np.zeros(
        (180, 256, 256), dtype=bool
    )  # HSV ranges: H [0, 179], S [0, 255], V [0, 255]

    for ranges in COLOR_RANGES.values():
        for lower, upper in ranges:
            h_mask = (lower[0] <= np.arange(180)) & (np.arange(180) <= upper[0])
            s_mask = (lower[1] <= np.arange(256)) & (np.arange(256) <= upper[1])
            v_mask = (lower[2] <= np.arange(256)) & (np.arange(256) <= upper[2])

            coverage_mask[np.ix_(h_mask, s_mask, v_mask)] = True

    if not np.all(coverage_mask):
        uncovered_color_plot(coverage_mask)
    else:
        print("All possible HSV values covered")
    return


def uncovered_color_plot(coverage_mask):
    h_samples, s_samples, v_samples = np.where(~coverage_mask)
    indices = np.random.choice(
        len(h_samples), size=min(100, len(h_samples)), replace=False
    )
    h_samples = h_samples[indices]
    s_samples = s_samples[indices]
    v_samples = v_samples[indices]

    hsv_samples = np.stack((h_samples, s_samples, v_samples), axis=-1).astype(np.uint8)
    rgb_samples = cv2.cvtColor(hsv_samples[None, :, :], cv2.COLOR_HSV2RGB)[0]
    rgb_colors = rgb_samples / 255.0

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    scatter = ax.scatter(
        h_samples, s_samples, v_samples, c=rgb_colors, marker="o", alpha=0.7
    )
    ax.set_xlabel("Hue")
    ax.set_ylabel("Saturation")
    ax.set_zlabel("Value")
    ax.set_title("Uncovered Color Regions in HSV Space")
    ax.set_xlim(0, 180)
    ax.set_ylim(0, 255)
    ax.set_zlim(0, 255)
    plt.show()
    return


# Example usage, assuming COLOR_RANGES is defined
check_color_ranges_coverage()


# Example usage
plot_3d_color_spectrum("red")
