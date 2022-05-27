from dataset import add_noise
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

from settings import (BATCH_SIZE, BUFFER_SIZE, CELEBA_PATH, IMAGESIZE, CELEBA_BOUNDING_BOX)



def process_datapoint(x):
    image = x['image']
    if CELEBA_BOUNDING_BOX:
        image = tf.image.crop_to_bounding_box(image, *CELEBA_BOUNDING_BOX)
    else:
        image = tf.image.resize(image, [IMAGESIZE, IMAGESIZE], method='mitchellcubic')
    image = (np.float(image) - 127.5) / 127.5
    noisy1 = add_noise(image)
    noisy2 = add_noise(image)
    return image, noisy1, noisy2


def get_raw_dataset(split='train'):
    return tfds.load('celeb_a', split=split, shuffle_files=False, download=False, data_dir=CELEBA_PATH)


def get_dataset(split='train', batch=True, shuffle=True):
    ds = tfds.load('celeb_a', split=split, shuffle_files=False, download=False, data_dir=CELEBA_PATH)
    ds = ds.map(process_datapoint)
    if shuffle:
        ds = ds.shuffle(BUFFER_SIZE)
    if batch:
        ds = ds.batch(BATCH_SIZE)
    return ds


def get_datasets(batch_trainset=False, batch_valset=False,  batch_testset=False):
    train_dataset = get_dataset(split='train', batch=batch_trainset)
    val_dataset = get_dataset(split='validation', shuffle=False, batch=batch_valset)
    test_dataset = get_dataset(split='test', shuffle=False, batch=batch_testset)
    return train_dataset, val_dataset, test_dataset

