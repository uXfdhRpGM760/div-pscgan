import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_probability as tfp
from functools import partial

from settings import BATCH_SIZE, BUFFER_SIZE, STDDEV, DATASET, NOISE_TYPE, SPLIT

if NOISE_TYPE == 'LOGNORMAL':
    LOGNORM_DISTR = tfp.distributions.LogNormal(0, STDDEV)


def get_data_full_test(name):
    ds_train = tfds.load(name=name, split="train", shuffle_files=True)
    ds_test = tfds.load(name=name, split="test", shuffle_files=False)
    ds_train = [x['image'] for x in ds_train]
    ds_train, ds_val = ds_train[:50000], ds_train[50000:]
    ds_test = [x['image'] for x in ds_test]
    return np.stack(ds_train), np.stack(ds_val), np.stack(ds_test)


def get_data_hdn(name):
    ds_train = tfds.load(name=name, split="train", shuffle_files=True)
    ds_test = tfds.load(name=name, split="test", shuffle_files=False)
    ds_train = [x['image'] for x in ds_train]
    ds_test = [x['image'] for x in ds_test]
    ds_val, ds_test = ds_test[:-100], ds_test[-100:]
    return np.stack(ds_train), np.stack(ds_val), np.stack(ds_test)


def get_data_hdn_real(name):
    ds_train = tfds.load(name=name, split="train", shuffle_files=True)
    ds_test = tfds.load(name=name, split="test", shuffle_files=False)
    ds_train = [x['image'] for x in ds_train]
    ds_test = [x['image'] for x in ds_test]
    ds_train, ds_val = ds_train[:54000], ds_train[54000:]
    ds_test = ds_test[-100:]
    return np.stack(ds_train), np.stack(ds_val), np.stack(ds_test)


def get_data(name):
    if SPLIT == 'hdn':
        return get_data_hdn(name)
    if SPLIT == 'hdn_real':
        return get_data_hdn_real(name)
    if SPLIT == 'full_test':
        return get_data_full_test(name)
    raise ValueError('No such split')


def add_noise(img, *args, **kwargs):
    if NOISE_TYPE == 'GAUSSIAN':
        return add_noise_gaussian(img, *args, **kwargs)
    elif NOISE_TYPE == 'S&P':
        return add_noise_sp(img, *args, **kwargs)
    elif NOISE_TYPE == 'LOGNORMAL':
        return add_noise_lognormal(img, *args, **kwargs)


def add_noise_lognormal(img, *args, **kwargs):
    lognorm_noise = LOGNORM_DISTR.sample(img.shape)
    #normalisation
    return img + lognorm_noise - 1


def add_noise_sp(img, *args, **kwargs):
    uniform_tensor = tf.random.uniform([32,32,1])
    map_salt = uniform_tensor < 0.25
    map_pepper = uniform_tensor > 0.75
    img = tf.where(map_salt, 1., img)
    img = tf.where(map_pepper, -1., img)
    return img


def add_noise_gaussian(next_elem, loc=0, scale=STDDEV, normalize=False):
    noise = tf.random.normal(shape=next_elem.shape, mean=loc, stddev=scale)
    noised = next_elem + noise
    if normalize:
        noised = (noised - np.min(noised)) / np.ptp(noised)
    return noised


def process_image_double(img, std=STDDEV):
    """
    Create a datset out of an image adding two instances of noise two it.
    It will be used by dataset map, so the same two instances of noise will be used in every
    epoch, for the same images. It also returns the clean image for tb statistics
    """
    img_noise1 = add_noise(img, scale=std)
    img_noise2 = add_noise(img, scale=std)
    return img, img_noise1, img_noise2


def process_image_double_no_original(img):
    img_noise1 = add_noise(img)
    img_noise2 = add_noise(img)
    return img_noise1, img_noise2


def process_images_mnist(images, pad=True):
    images = images.reshape(images.shape[0], 28, 28, 1).astype('float32')
    images = (images - 127.5) / 127.5  # Normalize the images to [-1, 1]
    if pad:
        #if we want to use deeper unet we can pad to 32 to make it 2**n
        images = np.pad(images, [(0, 0), (2, 2), (2, 2), (0,0)], 'constant', constant_values=(-1))
    return images


def create_dataset_mnist(images, shuffle=True, batch=True):
    dataset = tf.data.Dataset.from_tensor_slices(images)
    batch_and_shuffle(dataset, shuffle, batch)
    return dataset


def batch_and_shuffle(dataset, shuffle=True, batch=True, batch_size=BATCH_SIZE):
    if shuffle:
        dataset = dataset.shuffle(BUFFER_SIZE)
    if batch:
        dataset = dataset.batch(batch_size)
    return dataset


def get_dataset_mnist_kmnist(get_original=True, dataset='mnist', std=STDDEV, batch_valset=False):
    sets = get_data(dataset)
    sets = [process_images_mnist(s) for s in sets]
    sets = [tf.data.Dataset.from_tensor_slices(s) for s in sets]
    if get_original:
        mapping_funct = partial(process_image_double, std=std)
    else:
        mapping_funct = process_image_double_no_original
    train_dataset, val_dataset, test_dataset = [s.map(mapping_funct) for s in sets]
    train_dataset = batch_and_shuffle(train_dataset, batch=False)
    val_dataset = batch_and_shuffle(val_dataset, shuffle=False, batch=batch_valset)
    test_dataset = batch_and_shuffle(test_dataset, shuffle=False, batch=False)
    return train_dataset, val_dataset, test_dataset


def get_dataset_mnist_n2n(get_original=True, std=STDDEV):
    return get_dataset_mnist_kmnist(get_original=get_original, dataset='mnist', std=std)


def get_dataset_kmnist_n2n(get_original=True, std=STDDEV):
    return get_dataset_mnist_kmnist(get_original=get_original, dataset='kmnist', std=std)


if __name__ == "__main__":
    train_dataset, val_dataset, test_dataset = get_dataset_kmnist_n2n()
    for d in train_dataset.take(1):
        print(d[0].shape)
        print(d[1].shape)
        print(d[1].shape)
