import math
import numpy as np
from visualisations_settings import (ALPHA, FRACTIONS_FOR_BUCKETS, TICKS, DGAN_FILES,
                                                           DATA_PATH, N2C_STATS_NAME, PDF_IMG_NUMBER)


def get_buckets(buckets_number=1000):
    if buckets_number < 100:
        raise ValueError('buckets number has to be greater than 100')
    buckets = [math.ceil(buckets_number*f) for f in FRACTIONS_FOR_BUCKETS]
    buckets_sum = sum(buckets)
    return buckets + [buckets_number-2*buckets_sum] + buckets[::-1]


def normalized_histogram(ar, buckets):
    sorted_ar = sorted(ar)
    start_idx = 0
    border_values = [sorted_ar[start_idx]]
    for idx, bucket_value in enumerate(buckets):
        if idx < (len(buckets)/2) - 1:
            border_values.append(sorted_ar[start_idx + bucket_value])
        else:
            border_values.append(sorted_ar[start_idx + bucket_value - 1])
        start_idx += bucket_value
    return border_values


def create_plot(ax, ticks, mse_n2c, lines):
    n2c_mse_value = [mse_n2c]*len(ticks)
    for idx in range(0, len(lines)-1):


        ax.fill_between(ticks, lines[idx], lines[idx+1],
                         facecolor="orange", # The fill color
                         color='blue',       # The outline color
                         alpha=ALPHA[idx])          # Transparency of the fill
    ax.plot(ticks, n2c_mse_value, '--', color='red', label='mse_N2C')
    ax.legend()
    return ax


def load_stats_data_for_visualisations():
    n2c_data = np.load(DATA_PATH + N2C_STATS_NAME)

    dgan_data = []
    for filename in DGAN_FILES:
        loaded_data = np.load(DATA_PATH + filename)
        dgan_data.append(loaded_data[:PDF_IMG_NUMBER])

    dgan_data = list(zip(*dgan_data))
    return dgan_data, n2c_data


def mse_to_gt_by_image_idx(ax, idx):
    dgan_data, n2c_data = load_stats_data_for_visualisations()
    buckets = get_buckets()
    lines = [normalized_histogram(model_data, buckets) for model_data in dgan_data[idx]]
    lines = list(zip(*lines))
    return create_plot(ax, TICKS, n2c_data[idx], lines)

