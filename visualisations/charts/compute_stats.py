import sys

from dataset import get_dataset_mnist_n2n, get_dataset_kmnist_n2n
from dataset_celeba import get_dataset as get_dataset_celeba
import numpy as np
import pickle

import os

import numpy
from numpy import cov
from numpy import trace
from numpy import iscomplexobj
from numpy.random import randint
from scipy.linalg import sqrtm
from settings import DATASET, IMAGESIZE, STDDEV as STD
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
import tensorflow as tf

DGAN_MODEL_PATH = ''
N2C_MODEL_PATH = ''


OUTPUT_PATH = ''


#DGAN_TICKS = list(range(0,71,10))
DGAN_MODELS_TO_USE = [15,16,17,18,19]
N2C_MODEL_TO_USE = 151
PDF_IMG_NUMBER = 30

def create_gen_input(img):
    z = tf.random.normal([img.shape[0]] + [IMAGESIZE, IMAGESIZE, 1], 0, 1)
    z /= np.linalg.norm(z, axis=(1, 2))[:, None, None]
    return tf.concat([img, z], axis=-1)

def get_validation_data():
    val_dataset = get_dataset()
    val_data = []
    for clean, noisy, _ in val_dataset.take(50):
        val_data.append([clean, noisy])
    return val_data



def get_dataset(dataset=DATASET):
    if dataset == 'kmnist':
        train_dataset, val_dataset, test_dataset = get_dataset_kmnist_n2n(std=STD)
    if dataset == 'MNIST':
        train_dataset, val_dataset, test_dataset = get_dataset_mnist_n2n(std=STD)
    elif dataset == 'celeba':
        val_dataset = get_dataset_celeba(split='validation')
    return val_dataset


def best_of_n(model, noisy, clean, images_number=1000):
    #TODO this might need batching on GPU
    z_val = np.random.normal(0, 1, [images_number, IMAGESIZE, IMAGESIZE, 1])
    z_val /= np.linalg.norm(z_val, axis=(1, 2))[:, None, None]
    images = np.tile(np.reshape(noisy, [1] + noisy.shape), [images_number, 1, 1, 1])
    gen_input = np.concatenate([images, z_val], axis=-1)
    mini_dataset = tf.data.Dataset.from_tensor_slices(gen_input)
    md = mini_dataset.batch(10)
    output = []
    for inp in md:
        output = output + [model(inp)]
    output = tf.concat(output, axis=0)
    mses = np.mean(((output-clean)**2), axis=(1, 2, 3))
    indices = np.argsort(mses)
    best = output[indices[0]]
    return output, mses, best


def load_generator(checkpoint_path):
    from models_celeba import create_generator, create_discriminator

    generator_scheduler = tf.keras.optimizers.schedules.ExponentialDecay(
        0.0004,
        decay_steps=100000,
        decay_rate=0.9)
    discriminator_scheduler = tf.keras.optimizers.schedules.ExponentialDecay(
        0.0004,
        decay_steps=100000,
        decay_rate=0.9)
    generator = create_generator()
    discriminator = create_discriminator()
    generator_optimizer = tf.keras.optimizers.Adam(generator_scheduler)
    discriminator_optimizer = tf.keras.optimizers.Adam(discriminator_scheduler)
    ckpt = tf.train.Checkpoint(step=tf.Variable(0), generator_optimizer=generator_optimizer, generator_model=generator,
                               discriminator_optimizer=discriminator_optimizer, discriminator_model=discriminator)
    ckpt.restore(checkpoint_path)
    return generator

def compute_stats(val_data, images_number=1000):
    for model_number in DGAN_MODELS_TO_USE:
        checkpoint_path = os.path.join(DGAN_MODEL_PATH, 'ckpt-' + str(model_number))
        generator = load_generator(checkpoint_path)
        model_stats = []
        for idx, (clean, noisy) in enumerate(val_data):
            print(idx)
            _, mses, best = best_of_n(generator, noisy, clean, images_number=images_number)
            model_stats.append(mses)
            if idx < PDF_IMG_NUMBER:
                filename = 'images_' + str(model_number) + '_' + str(idx);
                path = os.path.join(OUTPUT_PATH, filename)
                if not os.path.exists(OUTPUT_PATH):
                    os.makedirs(OUTPUT_PATH)
                with open(path, 'wb') as f:
                    pickle.dump([clean, noisy, best], f)
        model_stats = np.stack(model_stats)
        np.save(os.path.join(OUTPUT_PATH,  'stats_' + str(model_number)), model_stats)


def compute_stats_n2c(val_data):
    model = tf.keras.models.load_model(N2C_MODEL_PATH)
    model_stats = []
    for idx, (clean, noisy) in enumerate(val_data):
        noisy = np.expand_dims(noisy, axis=0)
        out = model(noisy)
        if idx < PDF_IMG_NUMBER:
            filename = 'n2c_images' + '_' + str(idx)
            path = os.path.join(OUTPUT_PATH, filename)
            with open(path, 'wb') as f:
                pickle.dump(out, f)
        print(idx, np.mean(((out-clean)**2)))
        model_stats.append(np.mean(((out-clean)**2), axis=(1, 2, 3)))

    model_stats = np.array(model_stats)
    np.save(os.path.join(OUTPUT_PATH, 'n2c_stats'), model_stats)


def get_codes_batch(model, batch):
    batch = tf.image.resize(batch, (299,299))
    batch = preprocess_input(batch)
    return model(batch)


def get_codes(inception_model, images_dataset, prediction_model=None, dgan=False):
    all_codes = []
    for batch in images_dataset:
        if prediction_model is not None:
            inp = batch[1]
            if dgan:
                inp = create_gen_input(inp)
            inp = prediction_model(inp)
        else:
            inp = batch[0]
        all_codes.append(get_codes_batch(inception_model, inp))
    return np.concatenate(all_codes, axis=0)


def calculate_fid(act1, act2):
    # calculate mean and covariance statistics
    mu1, sigma1 = act1.mean(axis=0), cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), cov(act2, rowvar=False)
    # calculate sum squared difference between means
    ssdiff = numpy.sum((mu1 - mu2)**2.0)
    # calculate sqrt of product between cov
    covmean = sqrtm(sigma1.dot(sigma2))
    # check and correct imaginary numbers from sqrt
    if iscomplexobj(covmean):
        covmean = covmean.real
    # calculate score
    fid = ssdiff + trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid


def calculate_fid_n2c(inception_model, original_codes):
    n2c_model = tf.keras.models.load_model(N2C_MODEL_PATH)
    dataset = get_dataset()
    n2c_codes = get_codes(inception_model, dataset, n2c_model)
    return calculate_fid(original_codes, n2c_codes)


def calculate_fid_dGAN(inception_model, original_codes):
    fids = []
    for model_number in DGAN_MODELS_TO_USE:
        checkpoint_path = os.path.join(DGAN_MODEL_PATH, 'ckpt-' + str(model_number))
        generator = load_generator(checkpoint_path)
        dataset = get_dataset()
        dgan_codes = get_codes(inception_model, dataset, generator, dgan=True)
        fids.append(calculate_fid(original_codes, dgan_codes))
    return fids

def calculate_fid_for_model(generator):
    inception_model = InceptionV3(include_top=False, pooling='avg', input_shape=(299, 299, 3))
    dataset = get_dataset()
    original_codes = get_codes(inception_model, dataset)
    dataset = get_dataset()
    dgan_codes = get_codes(inception_model, dataset, generator, dgan=True)
    return calculate_fid(original_codes, dgan_codes)

def calculate_fids():
    inception_model = InceptionV3(include_top=False, pooling='avg', input_shape=(299, 299, 3))
    dataset = get_dataset()
    original_codes = get_codes(inception_model, dataset)
    fid_n2c = calculate_fid_n2c(inception_model, original_codes)
    fids_dgan = calculate_fid_dGAN(inception_model, original_codes)
    np.save(os.path.join(OUTPUT_PATH, 'fid_n2c'), fid_n2c)
    np.save(os.path.join(OUTPUT_PATH, 'fids_dgan'), fids_dgan)



if __name__ == "__main__":
    val_data = get_validation_data()
    calculate_fids()





