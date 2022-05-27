import argparse
import tensorflow as tf
import json
import numpy as np
from dataset import get_dataset_mnist_n2n, get_dataset_kmnist_n2n, add_noise
from dataset_celeba import get_datasets as get_datasets_celeba
import settings
from settings import (SMOOTH_MARGIN, STDDEV, GENERATOR_LR, DISCRIMINATOR_LR, IMAGESIZE, SCALAR_LOGGING_INTERVAL,
                      DIVERSIFY, DIVERSIFY_LOSS_MULTIPLIER, IMG_LOGGING_INTERVAL, FLIP_LABELS,
                      FLIP_PART, EPOCHS_NO, TIMES_DISCRIMINATOR_TRAINED, TIMES_GENERATOR_TRAINED,
                      DATASET, N2C, MSE_LOSS, MSE_LOSS_MULTIPLIER, BATCH_SIZE,
                      INCREASE_PACK_INTERVAL, DIV_LOSS_MAX, LR_DECAY_RATE, DECAY_STEPS,
                      CHANNELS,PACK_FACTORS, NEW_NOISE_EVERY_EPOCH)


from models_mnist import create_generator, create_discriminator
from models_celeba import (create_generator as create_generator_celeba,
                           create_discriminator as create_discriminator_celeba)
from utils import get_new_model_log_paths, factors
from visualisations.validation import best_of_n_batch
from visualisations.charts.compute_stats import calculate_fid_for_model
import os

tf.random.set_seed(
    543
)

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=False)
generator_filename = 'model_checkpoints/generator'
discriminator_filename = 'model_checkpoints/discriminator'


LOGPATH, CHECKPOINT_PATH = get_new_model_log_paths()
writer = tf.summary.create_file_writer(LOGPATH)
mse = tf.keras.losses.MeanSquaredError()
mae = tf.keras.losses.MeanAbsoluteError()


def compute_div_loss_pixels(images_first, images_second):
    axes = [1,2] if CHANNELS==1 else [1,2,3]
    return tf.reduce_mean(tf.abs(images_first - images_second), axis=axes)


def compute_div_loss_net_outputs(gen_output_first, gen_output_second):
    outputs = [tf.reshape(tf.abs(o1 - o2), [o1.shape[0], -1]) for o1, o2 in zip(gen_output_first, gen_output_second)]
    return tf.reduce_mean(tf.concat(outputs, axis=-1), axis=-1)


def compute_image_div_loss(first_run, second_run):
    return compute_div_loss_pixels(first_run, second_run)


def get_mixed_scores(discriminator, real, fake):
    epsilon = tf.random.uniform([real.shape[0], 1, 1, 1])
    epsilon = tf.tile(epsilon, [1, IMAGESIZE, IMAGESIZE, 1])
    interpolated = epsilon * real + (1-epsilon) * fake
    mixed_scores = discriminator(interpolated)
    return mixed_scores, interpolated


def gradient_penalty(disc_tape, mixed_scores, interpolated_images):
    mixed_grad = disc_tape.gradient(mixed_scores, interpolated_images)
    mixed_norm = tf.norm(tf.reshape(mixed_grad, [mixed_scores.shape[0], -1]), axis=1)
    return tf.reduce_mean(tf.square(mixed_norm - 1))


def noisy_labels(y, p_flip):
    original_shape = y.shape
    y = np.squeeze(y).copy()
    n_select = int(p_flip * y.shape[0])
    flip_ix = np.random.choice([i for i in range(y. shape[0])], size=n_select)
    y[flip_ix] = 1 - y[flip_ix]
    return np.reshape(y, original_shape)


def noisy_labels(y, p_flip):
    original_shape = y.shape
    y = np.squeeze(y).copy()
    n_select = int(p_flip * y.shape[0])
    flip_ix = np.random.choice([i for i in range(y. shape[0])], size=n_select)
    y[flip_ix] = 1 - y[flip_ix]
    return np.reshape(y, original_shape)


def discriminator_loss(real_output, fake_output):
    real_labels = tf.random.uniform(real_output.shape, minval=1-SMOOTH_MARGIN, maxval=1.)
    fake_labels = tf.random.uniform(real_output.shape, minval=0., maxval=SMOOTH_MARGIN)
    if FLIP_LABELS:
        real_labels = noisy_labels(real_labels, FLIP_PART)
        fake_labels = noisy_labels(fake_labels, FLIP_PART)
    real_loss = tf.math.reduce_mean(cross_entropy(real_labels, real_output))
    fake_loss = tf.math.reduce_mean(cross_entropy(fake_labels, fake_output))
    total_loss = real_loss + fake_loss
    return total_loss, real_loss, fake_loss


def generator_loss(fake_outputs):
    adv_loss = tf.math.reduce_mean(cross_entropy(tf.ones_like(fake_outputs), fake_outputs))
    return adv_loss

def\
        scale(image):
    scaled = image - tf.math.reduce_min(image)
    return scaled / tf.math.reduce_max(scaled)


def create_generator_summaries(gen_loss, gen_adv_loss, div_loss, mse_loss_value, distance_ratio, div_multiplier,
                               mse_loss_multiplier,pack_size, step):
    with writer.as_default():
        tf.summary.scalar('gen_loss', gen_loss, step)
        tf.summary.scalar('gen_adv_loss', gen_adv_loss, step)
        tf.summary.scalar('div_loss', div_loss, step)
        tf.summary.scalar('mse_loss_value', mse_loss_value, step)
        tf.summary.scalar('distance_ratio', distance_ratio, step)
        tf.summary.scalar('div_multiplier', div_multiplier, step)
        tf.summary.scalar('pack_size', pack_size, step)
        tf.summary.scalar('mse_loss_multiplier', mse_loss_multiplier, step)


def create_discriminator_summaries(disc_loss, real_loss, fake_loss,  mean_fake_response, mean_true_response, gp, step):
    with writer.as_default():
        tf.summary.scalar('disc_loss', disc_loss, step)
        tf.summary.scalar('real_loss', real_loss, step)
        tf.summary.scalar('fake_loss', fake_loss, step)
        tf.summary.scalar('gp_loss', gp, step)
        tf.summary.scalar('mean_fake_response', mean_fake_response, step)
        tf.summary.scalar('mean_true_response', mean_true_response, step)


def create_mean_summaries(clean, random_variants, step):
    stepi = int(step / IMG_LOGGING_INTERVAL)
    mean_image = tf.reduce_mean(random_variants, axis=0)
    mean_mse = mse(mean_image, clean)
    with writer.as_default():
        tf.summary.scalar('mean_mse_to_gt', mean_mse, stepi)


def create_val_summaries(clean, noisy, random_variants, variation, mses_to_gt, best_mse, fid, step):
    mean_image = tf.reshape(tf.reduce_mean(random_variants, axis=0), [1, IMAGESIZE, IMAGESIZE, CHANNELS]) 
    with writer.as_default():
        tf.summary.image('val_clean', scale(clean), step)
        tf.summary.image('val_noisy', scale(noisy), step)
        tf.summary.image('val_random_variant1', scale(tf.reshape(random_variants[0], [1, IMAGESIZE, IMAGESIZE, CHANNELS])), step)
        tf.summary.image('val_random_variant2', scale(tf.reshape(random_variants[1], [1, IMAGESIZE, IMAGESIZE, CHANNELS])), step)
        tf.summary.image('val_mean_denoised', scale(mean_image), step)
        tf.summary.scalar('images_variation', variation, step)
        tf.summary.histogram('val_mses_to_gt', mses_to_gt, step)
        tf.summary.scalar('val_mean_mse_to_gt', mses_to_gt, step)
        tf.summary.scalar('val_best_mse', best_mse, step)
        tf.summary.scalar('fid', fid, step)


def create_image_summaries(clean, noisy1, noisy2, random_variant1, random_variant2, step):
    stepi = int(step / IMG_LOGGING_INTERVAL)
    with writer.as_default():
        tf.summary.image('clean', scale(clean), stepi)
        tf.summary.image('noisy1', scale(noisy1), stepi)
        tf.summary.image('noisy2', scale(noisy2), stepi)
        tf.summary.image('random_variant1', scale(random_variant1), stepi)
        tf.summary.image('random_variant2', scale(random_variant2), stepi)
        tf.summary.image('random_var_diff', scale(random_variant1-random_variant2), stepi)


def create_gen_input(img):
    z = tf.random.normal([img.shape[0]] + [IMAGESIZE, IMAGESIZE, 1], 0, 1)
    z /= np.linalg.norm(z, axis=(1, 2))[:, None, None]
    return tf.concat([img, z], axis=-1)


def train_step(clean, noisy_images1, noisy_images2, generator, discriminator, generator_optimizer,
               discriminator_optimizer, step, times_generator_trained, times_discriminator_trained,
               div_multiplier, mse_loss_multiplier, pack_size, diversify):
    if NEW_NOISE_EVERY_EPOCH:
        noisy_images1 = add_noise(clean)
        noisy_images2 = add_noise(clean)
    if N2C:
        input_images1 = noisy_images1
        input_images2 = clean
    else:
        input_images1 = noisy_images1
        input_images2 = noisy_images2

    input_images1 = tf.tile(input_images1, [pack_size, 1, 1, 1])
    input_images2 = tf.tile(input_images2, [pack_size, 1, 1, 1])
    with tf.GradientTape(persistent=True) as gen_tape, tf.GradientTape(persistent=True) as disc_tape:
        generator_input = create_gen_input(input_images1)
        generator_output = generator(generator_input, training=True)
        denoised_images = generator_output
        if not N2C:
            generated_images = denoised_images + tf.random.normal(denoised_images.shape, mean=0., stddev=STDDEV)
        else:
            generated_images = denoised_images
        real_disc_input = tf.concat([input_images1, input_images2], axis=-1)
        fake_disc_input = tf.concat([input_images1, generated_images], axis=-1)
        real_output = discriminator(real_disc_input, training=True)
        fake_output = discriminator(fake_disc_input, training=True)
        gp = 0
        gen_loss = generator_loss(fake_output)
        gen_adv_loss = gen_loss
        disc_loss, real_loss, fake_loss = discriminator_loss(real_output, fake_output)
        if diversify and pack_size != 1:
            idx = noisy_images2.shape[0]
            g_out_dist = compute_image_div_loss(denoised_images[:idx], denoised_images[idx:2*idx])
            g_z_dist = tf.reduce_mean(tf.abs(generator_input[:idx,:,:,-1] - generator_input[idx:2*idx, :, :, -1]), axis=[1,2])
            distance_ratio = tf.reduce_mean(g_out_dist / g_z_dist)
            if DIV_LOSS_MAX:
                div_loss = mae(g_out_dist / g_z_dist, tf.cast(tf.tile([DIV_LOSS_MAX],
                                                                      (g_out_dist / g_z_dist).shape),
                                                              tf.float32))
                div_loss /= DIV_LOSS_MAX
                gen_loss += div_loss * div_multiplier
            else:
                div_loss = distance_ratio * div_multiplier
                gen_loss -= div_loss
        else:
            div_loss = 0
            distance_ratio = 0
        denoised_images = tf.reshape(denoised_images, [pack_size] + noisy_images1.shape)
        if MSE_LOSS:
            mean_denoised_images = tf.reduce_mean(denoised_images, axis=0)
            mse_loss_value = tf.reduce_mean(
                tf.keras.losses.mean_squared_error(
                    mean_denoised_images, input_images2[:noisy_images1.shape[0]]))
            mse_loss_value = mse_loss_value * mse_loss_multiplier
            gen_loss += mse_loss_value
        mean_fake_response = tf.math.reduce_mean(fake_output)
        mean_true_response = tf.math.reduce_mean(real_output)
        if step is not None and step % SCALAR_LOGGING_INTERVAL == 0:
            create_generator_summaries(gen_loss, gen_adv_loss, div_loss, mse_loss_value, distance_ratio, div_multiplier, mse_loss_multiplier,
                                       pack_size, step)
            create_discriminator_summaries(disc_loss, real_loss, fake_loss, mean_fake_response, mean_true_response, gp, step)
        denoised1 = denoised_images[0]
        if denoised_images.shape[0] == 1:
            denoised2 = denoised_images[0]
        else:
            denoised2 = denoised_images[1]
        if step is not None and step % IMG_LOGGING_INTERVAL == 0:
            create_image_summaries(clean, noisy_images1, noisy_images2, denoised1, denoised2, step)
            create_mean_summaries(clean, denoised_images, step)
    for _ in range(times_generator_trained):
        gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
        generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    if _ in range(times_discriminator_trained):
        discriminator_loss_gradients = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
        discriminator_optimizer.apply_gradients(zip(discriminator_loss_gradients, discriminator.trainable_variables))
    return gen_loss, disc_loss


def train(train_dataset, val_dataset, generator, discriminator, generator_optimizer,
          discriminator_optimizer,  epochs, params_dict, generator_scheduler, discriminator_scheduler):

    if not os.path.exists(CHECKPOINT_PATH):
        os.makedirs(CHECKPOINT_PATH)
    mse_loss_multiplier = 0
    step = 0
    pack_size_idx = 0

    with open(os.path.join(LOGPATH, 'params.json'), 'w') as f:
        settings_dict = vars(settings)
        data = {k: v for k, v in settings_dict.items() if k.isupper()}
        data.update({k.upper(): v for k, v in params_dict.items()})
        f.write(json.dumps(data))
    pack_factors = sorted(list(factors(BATCH_SIZE)))
    ckpt = tf.train.Checkpoint(step=tf.Variable(0), generator_optimizer=generator_optimizer, generator_model=generator,
                               discriminator_optimizer=discriminator_optimizer, discriminator_model=discriminator)
    ckpt_manager = tf.train.CheckpointManager(ckpt, CHECKPOINT_PATH, max_to_keep=None)

    for epoch in range(epochs):
        if epoch % INCREASE_PACK_INTERVAL == 0 and epoch != 0:
            pack_size_idx = -1 if pack_size_idx == len(pack_factors) else pack_size_idx + 1
        if MSE_LOSS:
            mse_loss_multiplier = params_dict['mse_loss_multiplier']
        if PACK_FACTORS:
            pack_size = pack_factors[pack_size_idx]
        else:
            pack_size = 4
        batched_train_dataset = train_dataset.batch(int(BATCH_SIZE/pack_size))
        batched_val_dataset = val_dataset.batch(int(BATCH_SIZE / pack_size))
        variation = 0
        best_mses = []
        mean_mses_to_gt = []
        for clean, noisy1, noisy2 in batched_train_dataset:
            _ = train_step(clean, noisy1, noisy2, generator, discriminator, generator_optimizer,
                           discriminator_optimizer, step, times_generator_trained=TIMES_GENERATOR_TRAINED,
                           times_discriminator_trained=TIMES_DISCRIMINATOR_TRAINED,
                           div_multiplier=params_dict['diversify_loss_multiplier'],
                           mse_loss_multiplier=mse_loss_multiplier, pack_size=pack_size,
                           diversify=params_dict['diversify'])

            if step % IMG_LOGGING_INTERVAL == 0:
                with writer.as_default():
                    tf.summary.scalar('gen_lr', generator_scheduler(step), step)
                    tf.summary.scalar('disc_lr', discriminator_scheduler(step), step)
            step += 1
        ckpt.step.assign(epoch)
        ckpt_manager.save()
        for clean, noisy1, noisy2 in batched_val_dataset.take(5):
            variants, best_mse, mses_to_gt, mean_mse_to_gt = best_of_n_batch(generator, noisy1, clean)
            best_mses.append(best_mse)
            mean_mses_to_gt.append(mean_mse_to_gt)
            if DATASET == 'celeba':
                fid = calculate_fid_for_model(generator)
            else:
                fid = 0
            mean_best_mse = sum(best_mses)/len(best_mses)
            mean_mse_to_gt_all = sum(mean_mses_to_gt)/len(mean_mses_to_gt)
        create_val_summaries(clean, noisy1, variants, variation, mean_mse_to_gt_all, mean_best_mse, fid, epoch)


def create_optimizers():
    if params_dict['lr_decay_rate']:
        generator_scheduler = tf.keras.optimizers.schedules.ExponentialDecay(
            params_dict['generator_lr'],
            decay_steps=params_dict['decay_steps'],
            decay_rate=params_dict['lr_decay_rate'])
        discriminator_scheduler = tf.keras.optimizers.schedules.ExponentialDecay(
            params_dict['discriminator_lr'],
            decay_steps=params_dict['decay_steps'],
            decay_rate=params_dict['lr_decay_rate'])
    else:
        generator_scheduler = params_dict['generator_lr']
        discriminator_scheduler = params_dict['discriminator_lr']
    generator_optimizer = tf.keras.optimizers.Adam(generator_scheduler)
    discriminator_optimizer = tf.keras.optimizers.Adam(discriminator_scheduler)
    return generator_optimizer, discriminator_optimizer, generator_scheduler, discriminator_scheduler

def train_loop(params_dict):
    if DATASET == 'mnist':
        train_dataset, val_dataset, _ = get_dataset_mnist_n2n()
        generator = create_generator()
        discriminator = create_discriminator()
    elif DATASET == 'kmnist':
        train_dataset, val_dataset, _ = get_dataset_kmnist_n2n()
        generator = create_generator()
        discriminator = create_discriminator()
    elif DATASET == 'celeba':
        train_dataset, val_dataset, _ = get_datasets_celeba()
        generator = create_generator_celeba()
        discriminator = create_discriminator_celeba()
    else:
        raise ValueError('Wrong dataset')

    generator_optimizer, discriminator_optimizer, generator_scheduler, discriminator_scheduler = create_optimizers()
    if params_dict['restore_checkpoint'] != '':
        ckpt = tf.train.Checkpoint(generator_optimizer=generator_optimizer, generator_model=generator,
                                   discriminator_optimizer=discriminator_optimizer, discriminator_model=discriminator)
        ckpt.restore(params_dict['restore_checkpoint'])
    train(train_dataset, val_dataset, generator, discriminator, generator_optimizer,
          discriminator_optimizer, EPOCHS_NO, params_dict, generator_scheduler, discriminator_scheduler)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='train denoising GAN')
    parser.add_argument('--generator_lr', type=float, nargs=1,
                        help='Generator learning rate', default=[GENERATOR_LR])
    parser.add_argument('--discriminator_lr', type=float, nargs=1,
                        help='Discriminator learning rate', default=[DISCRIMINATOR_LR])
    parser.add_argument('--lr_decay_rate', type=float, nargs=1,
                        help='Learning rate decay', default=[LR_DECAY_RATE])
    parser.add_argument('--decay_steps', type=int, nargs=1,
                        help='Decay steps', default=[DECAY_STEPS])
    parser.add_argument('--mse_loss_multiplier', type=float, nargs=1,
                        help='MSE loss multiplier', default=[MSE_LOSS_MULTIPLIER])
    parser.add_argument('--diversify', type=bool, nargs=1,
                        help='If we should diversify images during training', default=[DIVERSIFY])
    parser.add_argument('--diversify_loss_multiplier', type=float, nargs=1,
                        help='Diversify loss multiplier', default=[DIVERSIFY_LOSS_MULTIPLIER])
    parser.add_argument('--restore_checkpoint', type=str, nargs=1,
                        help='Path to the checkpoint we should start from', default=[''])
    args = parser.parse_args()
    params_dict = {'decay_steps': args.decay_steps[0], 'discriminator_lr': args.discriminator_lr[0],
                   'diversify': args.diversify[0], 'diversify_loss_multiplier': args.diversify_loss_multiplier[0],
                   'generator_lr': args.generator_lr[0], 'lr_decay_rate': args.lr_decay_rate[0],
                   'mse_loss_multiplier': args.mse_loss_multiplier[0], 'restore_checkpoint': args.restore_checkpoint[0]}
    train_loop(params_dict)
