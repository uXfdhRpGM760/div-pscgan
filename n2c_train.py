import tensorflow as tf
import json
import numpy as np
from dataset import get_dataset_mnist_n2n, get_dataset_kmnist_n2n, add_noise
import settings
from settings import (SMOOTH_MARGIN, STDDEV, GENERATOR_LR, DISCRIMINATOR_LR, NEW_NOISE_EVERY_EPOCH,
                      IMG_LOGGING_INTERVAL, FLIP_LABELS, FLIP_PART, EPOCHS_NO, NO_VARIANTS, BATCH_SIZE, DATASET)

from models_n2n import unet_for_n2n
from utils import get_new_model_log_paths
import os

tf.random.set_seed(
    543
)


LOGPATH, CHECKPOINT_PATH = get_new_model_log_paths()
writer = tf.summary.create_file_writer(LOGPATH)
mse = tf.keras.losses.MeanSquaredError()




def compute_loss(output, compare_to):
    return tf.math.reduce_mean(mse(output, compare_to))


def scale(image):
    scaled = image - tf.math.reduce_min(image)
    return scaled / tf.math.reduce_max(scaled)


def create_summaries(loss, step):
    with writer.as_default():
        tf.summary.scalar('mse', loss, step)


def create_val_summaries(clean, noisy1, noisy2, cleaned, mse, step):
    with writer.as_default():
        tf.summary.image('val_clean', scale(clean), step)
        tf.summary.image('val_noisy1', scale(noisy1), step)
        tf.summary.image('val_noisy2', scale(noisy2), step)
        tf.summary.image('val_cleaned', scale(cleaned), step)
        tf.summary.scalar('val_mse_to_gt', mse, step)


def create_train_image_summaries(clean, noisy1, noisy2, cleaned, step):
    stepi = int(step / IMG_LOGGING_INTERVAL)
    with writer.as_default():
        tf.summary.image('clean', scale(clean), stepi)
        tf.summary.image('noisy1', scale(noisy1), stepi)
        tf.summary.image('noisy2', scale(noisy2), stepi)
        tf.summary.image('cleaned', scale(cleaned), stepi)


def train_step(clean, noisy_images1, noisy_images2, model, optimizer, step):
    with tf.GradientTape(persistent=True) as tape:
        denoised_images = model(noisy_images1, training=True)
        loss = compute_loss(denoised_images, clean)
        create_summaries(loss, step)
        #if step % IMG_LOGGING_INTERVAL == 0:
        #    create_train_image_summaries(clean, noisy_images1, noisy_images2, denoised_images, step)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss


def val_step(clean, noisy_images1, noisy_images2, model, step):
    cleaned = model(noisy_images1)
    loss = compute_loss(cleaned, clean)
    create_val_summaries(clean, noisy_images1, noisy_images2, cleaned, loss, step)


def train(train_dataset, val_dataset, model, epochs):
    if not os.path.exists(CHECKPOINT_PATH):
        os.makedirs(CHECKPOINT_PATH)
    optimizer = tf.keras.optimizers.Adam(GENERATOR_LR)
    step = 0
    for epoch in range(epochs):
        for clean, noisy1, noisy2 in train_dataset:
            if NEW_NOISE_EVERY_EPOCH:
                noisy1 = add_noise(clean)
                noisy2 = add_noise(clean)
            _ = train_step(clean, noisy1, noisy2, model, optimizer, step)
            step += 1
        new_checkpoint_path = os.path.join(CHECKPOINT_PATH, str(epoch))
        model.save(new_checkpoint_path)
        for clean, noisy1, noisy2 in val_dataset.take(1):
            val_step(clean, noisy1, noisy2, model, step)


def train_loop():
    if DATASET == 'mnist':
        train_dataset, val_dataset, _ = get_dataset_mnist_n2n()
    elif DATASET == 'kmnist':
        train_dataset, val_dataset, _ = get_dataset_kmnist_n2n()
    train_dataset = train_dataset.batch(BATCH_SIZE)
    val_dataset = val_dataset.batch(BATCH_SIZE)
    model = unet_for_n2n()
    train(train_dataset, val_dataset, model, EPOCHS_NO)


if __name__ == "__main__":
    with open(os.path.join(LOGPATH, 'params.json'), 'w') as f:
        data = {k: v for k, v in vars(settings).items() if k.isupper()}
        data['MODE'] = 'n2c'
        f.write(json.dumps(data))
    train_loop()
