import numpy as np
import pickle as pkl
from matplotlib import pyplot as plt
import math
from mpl_toolkits.axes_grid1 import ImageGrid
from train import scale
from visualisations.charts.visualisations_settings import (DATA_PATH, DGAN_IMAGES_FILE_PATTERN, IMAGESIZE,
                                                           N2C_IMAGES_FILE_PATTERN, TICKS)


def load_images(image_number, chosen_model):
    with open(DATA_PATH + DGAN_IMAGES_FILE_PATTERN.format(model_number=chosen_model, image_number=image_number), 'rb') as f:
        clean, noisy, dgan_denoised = pkl.load(f)
        dgan_denoised = np.reshape(dgan_denoised, [1, IMAGESIZE, IMAGESIZE, 3])
    with open(DATA_PATH + N2C_IMAGES_FILE_PATTERN.format(image_number=image_number), 'rb') as f:
        n2c_image = pkl.load(f)
    return clean, noisy, dgan_denoised, n2c_image


def prepare_images(image_idx):
    all_images = {}
    for epoch in TICKS:
        clean, noisy, dgan_denoised, n2c_denoised = load_images(image_idx, epoch)
        noisy = scale(noisy)
        clean = scale(clean)
        dgan_denoised = scale(dgan_denoised)
        n2c_denoised = scale(n2c_denoised)
        dgan_mse = (clean-dgan_denoised)**2
        n2c_mse = (clean-n2c_denoised)**2
        all_images[epoch] = {'clean': clean, 'noisy': noisy, 'dgan_denoised': dgan_denoised,
                             'dgan_mse': dgan_mse, 'n2c_denoised': n2c_denoised, 'n2c_mse': n2c_mse}

    images_names = ['clean', 'noisy', 'dgan_denoised', 'dgan_mse', 'n2c_denoised', 'n2c_mse']
    return images_names, all_images


def create_grid_plot(axis, all_images, images_names):
    grid = ImageGrid(axis, 211,  # similar to subplot(111)
                     nrows_ncols=(5, 6),  # creates 2x2 grid of axes
                     axes_pad=0.1,  # pad between axes in inch.
                     )
    for idx, subax in enumerate(grid):
        # Iterating over the grid returns the Axes.
        model_number = TICKS[math.floor(idx / len(images_names))]
        image_name = images_names[idx % len(images_names)]

        im = all_images[model_number][image_name]
        subax.imshow(np.squeeze(im), cmap='gray')
        plt.setp(subax.get_xticklabels(), visible=False)
        plt.setp(subax.get_yticklabels(), visible=False)
        subax.tick_params(axis=u'both', which=u'both', length=0)
        if idx < len(images_names):
            subax.set_title(image_name)
        if idx % len(images_names) == 0:
            subax.set_ylabel(str(model_number))


def get_image_grid(ax, image_idx):
    images_names, all_images = prepare_images(image_idx)
    return create_grid_plot(ax, all_images, images_names)


