import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2


class Encoder(tf.keras.layers.Layer):
    def __init__(self):
        super(Encoder, self).__init__()
        self.encoder_conv_1 = tf.keras.layers.Conv2D(10, 3, (2, 2), padding='same',
                                                     kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.1),
                                                     bias_initializer=tf.keras.initializers.RandomNormal(stddev=0.1),
                                                     input_shape=(128, 128, 3))
        self.encoder_conv_2 = tf.keras.layers.Conv2D(10, 3, (2, 2), padding='same',
                                                     kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.1),
                                                     bias_initializer=tf.keras.initializers.RandomNormal(stddev=0.1))
        self.encoder_conv_3 = tf.keras.layers.Conv2D(10, 3, (2, 2), padding='same',
                                                     kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.1),
                                                     bias_initializer=tf.keras.initializers.RandomNormal(stddev=0.1))

    @tf.function
    def call(self, images):

        layer_1_output = self.encoder_conv_1(images)
        layer_1_output = tf.nn.leaky_relu(layer_1_output, 0.2)
        layer_2_output = self.encoder_conv_2(layer_1_output)
        layer_2_output = tf.nn.leaky_relu(layer_2_output, 0.2)
        layer_3_output = self.encoder_conv_3(layer_2_output)
        return layer_3_output

class Decoder(tf.keras.layers.Layer):
    def __init__(self):
        super(Decoder, self).__init__()

        self.decoder_deconv_1 = tf.keras.layers.Conv2DTranspose(10, 3, (2, 2), padding='same',
                                                              kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.1),
                                                              bias_initializer=tf.keras.initializers.RandomNormal(stddev=0.1))
        self.decoder_deconv_2 = tf.keras.layers.Conv2DTranspose(10, 3, (2, 2),
                                                                padding='same',
                                                                kernel_initializer=tf.keras.initializers.RandomNormal(
                                                                    stddev=0.1),
                                                                bias_initializer=tf.keras.initializers.RandomNormal(
                                                                    stddev=0.1))
        self.decoder_deconv_3 = tf.keras.layers.Conv2DTranspose(3, 3, (2, 2),
                                                                padding='same',
                                                                kernel_initializer=tf.keras.initializers.RandomNormal(
                                                                    stddev=0.1),
                                                                bias_initializer=tf.keras.initializers.RandomNormal(
                                                                    stddev=0.1))

    @tf.function
    def call(self, encoder_output):

        layer_1_output = self.decoder_deconv_1(encoder_output)
        layer_1_output = tf.nn.leaky_relu(layer_1_output, 0.2)
        layer_2_output = self.decoder_deconv_2(layer_1_output)
        layer_2_output = tf.nn.leaky_relu(layer_2_output, 0.2)
        layer_3_output = self.decoder_deconv_3(layer_2_output)

        return layer_3_output


class AutoEncoder(tf.keras.Model):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    @tf.function
    def call(self, images):

        encoded_images = self.encoder(images)
        decoded_images = self.decoder(encoded_images)
        return decoded_images

    @tf.function
    def loss_function(self, encoded, originals):
        encoded = tf.dtypes.cast(encoded, tf.float32)
        originals = tf.dtypes.cast(originals, tf.float32)
        return tf.reduce_sum((encoded - originals) ** 2)
