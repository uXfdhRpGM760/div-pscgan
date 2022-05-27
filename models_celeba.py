import tensorflow as tf


b_init = tf.constant_initializer(0.0)  # bias initializer
w_init = tf.initializers.TruncatedNormal(stddev=0.02)  # kernel_initializer
settings_momentum = 0.9
settings_epsilon = 1e-5
settings_disc_filters = [32, 64, 64, 128, 128, 256, 512]
shrink_ratios = [32, 16, 8, 4, 2, 1]
settings_gen_filters = [512, 256, 128, 64, 32]
settings_image_shape = (128, 128, 3)

def ReLU():
    return tf.keras.layers.ReLU()

def instance_norm(x, epsilon=1e-8):
    assert len(x.shape) == 4 # NCHW
    orig_dtype = x.dtype
    x = tf.cast(x, tf.float32)
    x -= tf.reduce_mean(x, axis=[2,3], keepdims=True)
    epsilon = tf.constant(epsilon, dtype=x.dtype, name='epsilon')
    x *= tf.math.rsqrt(tf.reduce_mean(tf.square(x), axis=[2, 3], keepdims=True) + epsilon)
    x = tf.cast(x, orig_dtype)
    return x


def leaky_ReLU(leak=0.2):
    return tf.keras.layers.LeakyReLU(alpha=leak)


def batch_norm(axis=-1):
    return tf.keras.layers.BatchNormalization(momentum=settings_momentum,
                                              epsilon=settings_epsilon,
                                              axis=axis)


def flatten():
    return tf.keras.layers.Flatten()


def linear(units, use_bias=False):
    return tf.keras.layers.Dense(units=units,
                                 kernel_initializer=w_init,
                                 use_bias=use_bias,
                                 bias_initializer=b_init)


def conv_layer(filters, kernel_size, strides, use_bias=False):
    return tf.keras.layers.Conv2D(filters=filters,
                                  kernel_size=kernel_size,
                                  strides=strides,
                                  padding='same',
                                  data_format='channels_last',
                                  kernel_initializer=w_init,
                                  use_bias=use_bias,
                                  bias_initializer=b_init)


def transpose_conv_layer(filters, kernel_size, strides, use_bias=False):
    return tf.keras.layers.Conv2DTranspose(filters=filters,
                                           kernel_size=kernel_size,
                                           strides=strides,
                                           padding='same',
                                           data_format='channels_last',
                                           kernel_initializer=w_init,
                                           use_bias=use_bias,
                                           bias_initializer=b_init)


class ConvBlock(tf.keras.Model):
    def __init__(self, filters1, filters2):
        super(ConvBlock, self).__init__(name='conv_block')
        self.kernel_size = (3, 3)
        self.strides1 = (1, 1)
        self.strides2 = (2, 2)
        self.filters1, self.filters2 = filters1, filters2
        self.conv1 = conv_layer(self.filters1, self.kernel_size, self.strides1)
        self.conv2 = conv_layer(self.filters2, self.kernel_size, self.strides2)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.leaky_ReLU1 = leaky_ReLU()
        self.ReLU1 = ReLU()
        self.leaky_ReLU2 = leaky_ReLU()
        self.ReLU2 = ReLU()

    def call(self, inputs, leaky=False, training=False, return_skip=False, input_skips=None):
        inputs = self.conv1(inputs)
        inputs = self.leaky_ReLU1(inputs) if leaky else self.ReLU1(inputs)
        inputs = self.bn1(inputs)
        skip = inputs
        if input_skips is not None:
            inputs = tf.concat([inputs, input_skips], axis=-1)
        inputs = self.conv2(inputs)
        inputs = self.leaky_ReLU2(inputs) if leaky else self.ReLU2(inputs)
        output = self.bn2(inputs)
        if return_skip:
            return output, skip
        return output


class TransposeConvBlock(tf.keras.Model):
    def __init__(self, filters1, filters2, last=False):
        super(TransposeConvBlock, self).__init__(name='transpose_conv_block')
        self.last = last
        self.kernel_size = (3, 3)
        self.strides1 = (2, 2)
        self.strides2 = (1, 1)
        self.filters1, self.filters2 = filters1, filters2
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.transpose_conv1 = transpose_conv_layer(self.filters1, self.kernel_size, self.strides1)
        self.transpose_conv2 = transpose_conv_layer(self.filters2, self.kernel_size, self.strides2)
        self.act1 = ReLU()
        self.act2 = ReLU() if last else tf.nn.tanh

    def call(self, inputs, skips=None, training=False):
        inputs = self.transpose_conv1(inputs)
        inputs = self.act1(inputs)
        inputs = self.bn1(inputs)
        inputs = tf.concat([inputs, skips], axis=-1) if skips is not None else inputs
        inputs = self.transpose_conv2(inputs)
        output = self.act2(inputs)
        if not self.last:
            output = self.bn2(output)
        return output


class Discriminator(tf.keras.Model):
    # class for discriminator
    def __init__(self):
        super(Discriminator, self).__init__(name='Discriminator')
        self.disc_filters = settings_disc_filters
        self.leaky_ReLU = leaky_ReLU()
        self.conv_blocks = [ConvBlock(self.disc_filters[i], self.disc_filters[i]) for i in range(1, len(self.disc_filters), 1)]
        self.flatten = flatten()
        self.linear = linear(units=1, use_bias=True)

    def call(self, inputs, training=False):
        skips = []
        for idx, conv_block in enumerate(self.conv_blocks[:-1]):
            inputs, skip = conv_block(inputs, leaky=True, training=training, return_skip=True)
            skip = tf.keras.layers.MaxPool2D([shrink_ratios[idx], shrink_ratios[idx]])(skip)
            skips.append(skip)
        skips = tf.concat(skips, axis=-1)
        inputs = self.conv_blocks[-1](inputs, leaky=True, training=training, input_skips=skips)

        inputs = self.flatten(inputs)
        inputs = self.linear(inputs)
        output = tf.keras.activations.sigmoid(inputs)
        return output


class Generator(tf.keras.Model):
    # class for generator
    def __init__(self):
        super(Generator, self).__init__(name='Generator')
        self.gen_filters = settings_gen_filters
        self.img_shape = settings_image_shape
        self.batch_norm = batch_norm()
        self.ReLU = ReLU()
        self.conv_blocks = [ConvBlock(self.gen_filters[i], self.gen_filters[i]) for i in reversed(range(len(self.gen_filters)))]
        self.transpose_conv_blocks = [TransposeConvBlock(self.gen_filters[i], self.gen_filters[i]) for i in
                                      range(len(self.gen_filters))]
        self.last_conv = conv_layer(3, 3, 1)
        self.last_batch_norm = batch_norm()

    def call(self, inputs, training=False):
        skips = []
        # inputs = ConvBlock(strides=(1, 1), filters=self.gen_filters[-1])
        for conv_block in self.conv_blocks:
            inputs, skip = conv_block(inputs, training=training, return_skip=True)
            skips.append(skip)
        for transpose_conv_block in self.transpose_conv_blocks:
            inputs = transpose_conv_block(inputs, skips.pop(-1), training=training)
        inputs = self.last_conv(inputs)
        inputs = self.last_batch_norm(inputs)
        inputs = tf.nn.tanh(inputs)
        return inputs

def create_generator():
    return Generator()

def create_discriminator():
    return Discriminator()


if __name__ == '__main__':
    import numpy as np
    discriminator = Discriminator()
    discriminator.build([1,128, 128, 6])
    #print(discriminator.summary())
    print(discriminator(np.random.normal(0, 1, [1, 128, 128, 6])))

    generator = Generator()
    generator.build([1,128, 128, 4])
    #print(generator.summary())
    print(generator(np.random.normal(0, 1, [1, 128, 128, 4])))



    #import numpy as np
    #discriminator(np.random.normal(0,1,[1,128,128,2]))
