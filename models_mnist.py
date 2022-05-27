import tensorflow as tf
from tensorflow.keras import layers, Model
initializer = tf.keras.initializers.HeNormal(seed=543)
from tensorflow.keras.constraints import Constraint


def double_conv_block_down(initializer, x, filters_first, filters_second):
    x = tf.keras.layers.Conv2D(filters_first, 3, strides=1, padding='same',
                               kernel_initializer=initializer)(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(filters_second, 3, strides=2, padding='same',
                               kernel_initializer=initializer)(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    return x


def single_conv_block_down(initializer, x, filters):
    x = tf.keras.layers.Conv2D(filters, 3, strides=2, padding='same',
                               kernel_initializer=initializer)(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    return x


def last_conv_block_up(initializer, skip, x, filters_first, filters_second):
    x = tf.keras.layers.Concatenate()([skip, x])
    x = tf.keras.layers.Conv2D(filters_first, 3, strides=1, padding='same',
                               kernel_initializer=initializer)(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(filters_second, 3, strides=1, padding='same',
                               kernel_initializer=initializer)(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    return x


def double_conv_block_up(initializer, skip, x, filters_first, filters_second):
    if skip is not None:
        x = tf.keras.layers.Concatenate()([skip, x])
    x = tf.keras.layers.Conv2DTranspose(filters_first, 3, strides=1, padding='same',
                               kernel_initializer=initializer)(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2DTranspose(filters_second, 3, strides=2, padding='same',
                               kernel_initializer=initializer)(x)
    x = tf.keras.layers.ReLU()(x)
    return x


def single_conv_block_up(initializer, skip, x, filters):
    x = tf.keras.layers.Concatenate()([skip, x])
    x = tf.keras.layers.Conv2DTranspose(filters, 3, strides=2, padding='same',
                                   kernel_initializer=initializer)(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    return x


def no_pool(input, initializer, filters_first, filters_second):
    x = tf.keras.layers.Conv2D(filters_first, 3, strides=1, padding='same', kernel_initializer=initializer)(input)
    x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(filters_second, 3, strides=1, padding='same', kernel_initializer=initializer)(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    return x


def create_generator():
    #input1 = tf.keras.layers.Input(shape=[None, None, 1]) #noisy_image
    #input2 = tf.keras.layers.Input(shape=NOISE_DIMENSION) #z
    #inputs = tf.keras.layers.Input(shape=[None, None, 2])
    #noise = layers.Dense(1024)(input2)
    #noise = tf.keras.layers.LeakyReLU(alpha=0.1)(noise)
    #noise = tf.reshape(noise, [-1, 32, 32, 1])
    #inputs = tf.keras.layers.Concatenate()([input1, noise])
    inputs = tf.keras.layers.Input(shape=[None, None, 2])

    x = no_pool(inputs, initializer, 16, 16)
    skips = [x] #32x32x16

    #down sample:
    x = double_conv_block_down(initializer, x, filters_first=32, filters_second=32) #
    skips.append(x) #16x16
    x = double_conv_block_down(initializer, x, filters_first=64, filters_second=64)  #
    skips.append(x)  # 8x8
    x = double_conv_block_down(initializer, x, filters_first=128, filters_second=128)  #
    skips.append(x)  # 4x4
    #middle code
    x = double_conv_block_down(initializer, x, filters_first=128, filters_second=128) #2x2

    x = double_conv_block_up(initializer, None, x, filters_first=128, filters_second=128) #4x4
    x = double_conv_block_up(initializer, skips[-1], x, filters_first=128, filters_second=128) #8x8
    x = double_conv_block_up(initializer, skips[-2], x, filters_first=64, filters_second=64)#16x16
    x = double_conv_block_up(initializer, skips[-3], x, filters_first=32, filters_second=32)#32x32

    x = tf.keras.layers.Concatenate()([skips[0], x])
    x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Conv2D(1, 3, strides=1, padding='same',
                               kernel_initializer=initializer)(x)
    x = tf.keras.activations.tanh(x)
    return tf.keras.Model(inputs=inputs, outputs=x)


def create_discriminator_wasserstein():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, (3, 3), strides=(1, 1),
                            input_shape=[32, 32, 2], kernel_initializer=initializer))
    model.add(layers.LeakyReLU())
    model.add(layers.BatchNormalization())

    model.add(layers.Conv2D(64, (3, 3), strides=(2, 2), kernel_initializer=initializer))
    model.add(layers.LeakyReLU())
    model.add(layers.BatchNormalization())

    model.add(layers.Conv2D(128, (3, 3), strides=(1, 1), kernel_initializer=initializer))
    model.add(layers.LeakyReLU())
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(128, (3, 3), strides=(2, 2), kernel_initializer=initializer))
    model.add(layers.LeakyReLU())
    model.add(layers.BatchNormalization())


    model.add(layers.Conv2D(264, (3, 3), strides=(2, 2), kernel_initializer=initializer))
    model.add(layers.LeakyReLU())
    model.add(layers.BatchNormalization())

    model.add(layers.Flatten())
    model.add(layers.Dense(1))
    #model.add(layers.Activation('sigmoid'))

    return model


def create_discriminator():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, (3, 3), strides=(1, 1),
                            input_shape=[32, 32, 2]))
    model.add(layers.LeakyReLU())
    model.add(layers.BatchNormalization())

    model.add(layers.Conv2D(64, (3, 3), strides=(2, 2)))
    model.add(layers.LeakyReLU())
    model.add(layers.BatchNormalization())

    model.add(layers.Conv2D(128, (3, 3), strides=(1, 1)))
    model.add(layers.LeakyReLU())
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(128, (3, 3), strides=(2, 2)))
    model.add(layers.LeakyReLU())
    model.add(layers.BatchNormalization())


    model.add(layers.Conv2D(264, (3, 3), strides=(2, 2)))
    model.add(layers.LeakyReLU())
    model.add(layers.BatchNormalization())

    model.add(layers.Flatten())
    model.add(layers.Dense(1))
    model.add(layers.Activation('sigmoid'))

    return model


def n2s_conv_block(x, units, last_activation='relu'):
    ox = x
    x = tf.keras.layers.Conv2D(units, 3, strides=1, padding='same')(x)
    x = tf.keras.layers.BatchNormalization(epsilon=1e-05, momentum=0.1)(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.01)(x)
    x = tf.keras.layers.Conv2D(units, 3, strides=1, padding='same')(x)
    x = tf.keras.layers.BatchNormalization(epsilon=1e-05, momentum=0.1)(x)
    x_start = tf.keras.layers.Add()([x[:, :, :, 0:min(ox.shape[3], x.shape[3])], ox[:, :, :, 0:min(ox.shape[3], x.shape[3])]])
    x_end = x[:, :, :, min(ox.shape[3], x.shape[3]):]
    tf.keras.layers.Concatenate([x_start, x_end])
    if last_activation == 'relu':
        x = tf.keras.layers.LeakyReLU(alpha=0.01)(x)
    else:
        x = tf.keras.activations.sigmoid(x)
    return x


def baby_unet():
    """
    This is the model that noise2self uses
    """
    inputs = tf.keras.layers.Input(shape=[32, 32, 1])
    c1 = n2s_conv_block(inputs, 16)
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2), padding='same')(c1)
    c2 = n2s_conv_block(x, 32)
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2), padding='same')(c2)
    x = n2s_conv_block(x, 32)
    x = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation="bilinear")(x)
    x = tf.keras.layers.Concatenate()([x, c2])
    x = n2s_conv_block(x, 32)
    x = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation="bilinear")(x)
    x = tf.keras.layers.Concatenate()([x, c1])
    x = n2s_conv_block(x, 16)
    x = n2s_conv_block(x, 1)
    return tf.keras.Model(inputs=inputs, outputs=x)

def autoencoder():
    inputs = tf.keras.layers.Input(shape=[32, 32, 1])
    x = n2s_conv_block(inputs, 16)
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2), padding='same')(x)
    x = n2s_conv_block(x, 32)
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2), padding='same')(x)
    code = n2s_conv_block(x, 1, last_activation='sigmoid')
    x = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation="bilinear")(code)
    x = n2s_conv_block(x, 32)
    x = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation="bilinear")(x)
    x = n2s_conv_block(x, 16)
    x = n2s_conv_block(x, 1)
    return tf.keras.Model(inputs=inputs, outputs=x)

class Autoencoder(Model):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = tf.keras.Sequential([
            layers.Input(shape=(32, 32, 1)),
            layers.Conv2D(16, (3, 3), activation='relu', padding='same', strides=2),
            layers.Conv2D(8, (3, 3), activation='relu', padding='same', strides=2)])

        self.decoder = tf.keras.Sequential([
            layers.Conv2DTranspose(8, kernel_size=3, strides=2, activation='relu', padding='same'),
            layers.Conv2DTranspose(16, kernel_size=3, strides=2, activation='relu', padding='same'),
            layers.Conv2D(1, kernel_size=(3, 3), activation='sigmoid', padding='same')])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded




class Between(Constraint):
    def __init__(self, min_value, max_value):
        self.min_value = min_value
        self.max_value = max_value

    def __call__(self, w):
        return tf.clip_by_value(w, self.min_value, self.max_value)

    def get_config(self):
        return {'min_value': self.min_value,
                'max_value': self.max_value}

if __name__ == '__main__':
    gen = create_generator()
    print(gen([tf.random.normal([1, 32, 32, 1], 0, 1), tf.random.normal([1,64])]))


    #disc = create_discriminator()
    #print(disc(tf.random.normal([1, 32, 32, 2], 0, 1)))
    """
    ae = autoencoder()
    ae.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError())
    train_dataset, val_dataset = get_dataset_mnist_n2n(get_original=False)
    ae.fit(train_dataset,
           epochs=10,
           shuffle=True,
           validation_data=val_dataset)
    ae.save('autoencoder')
    """





