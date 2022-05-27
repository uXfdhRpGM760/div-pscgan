from settings import IMAGESIZE

import numpy as np
import math
import tensorflow as tf


def best_of_n(model, noisy, clean, images_number=100):
    z_val = np.random.normal(0, 1, [images_number, IMAGESIZE, IMAGESIZE, 1])
    z_val /= np.linalg.norm(z_val, axis=(1, 2))[:, None, None]
    images = np.tile(tf.reshape(noisy, [1] + noisy.shape), [images_number, 1, 1, 1])
    gen_input = np.concatenate([images, z_val], axis=-1)
    out = model(gen_input)
    mses = np.mean(((out-clean)**2), axis=(1,2,3))
    indices = np.argsort(mses)
    best = mses[indices[0]]
    return out, mses, best, np.mean(mses)


def create_image_grid(images):
    size = math.sqrt(len(images))
    if int(size) != size:
        raise ValueError("Length of images has to be n**2")
    vis = np.reshape(images, [size, size, IMAGESIZE, IMAGESIZE])
    vis = np.concatenate(vis, axis=1)
    vis = np.concatenate(vis, axis=1)
    return vis


def best_of_n_batch(model, batch_noisy, batch_clean, images_number=5):
    bests, mses, means = [], [], []
    for idx, (noisy, clean) in enumerate(zip(batch_noisy, batch_clean)):
        out, mses_image, best, mean = best_of_n(model, noisy, clean, images_number=images_number)
        if idx == 0:
            out_return = out
        bests.append(best)
        means.append(mean)
        mses += list(mses_image)
    return out_return, sum(bests)/len(bests), mses, sum(means)/len(means)
