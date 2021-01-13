import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
from .autoencoder import AutoEncoder, Encoder, Decoder

class RandomEncoder(Encoder):
    def __init__(self):
        super(RandomEncoder, self).__init__()
        self.hidden_layer = tf.keras.layers.Dense(2 * 16 * 16 * 10)

    @tf.function
    def call(self, images):
        embedding = Encoder.call(self, images)
        tf.print(tf.shape(embedding))
        embedding = tf.nn.leaky_relu(embedding, 0.2)
        embedding = tf.reshape(embedding, (tf.shape(embedding)[0], -1))
        embedding = self.hidden_layer(embedding)
        mean, logvar = tf.split(embedding, num_or_size_splits=2, axis=1)
        embedding =  tf.random.normal(shape=mean.shape) * tf.exp(logvar * .5) + mean
        return embedding, mean, logvar

''' 
This is almost entirely based on the tensorflow tutorial
on Convolutional Variational Autoencoders, found here:
https://www.tensorflow.org/tutorials/generative/cvae    
'''


class VarAutoEncoder(AutoEncoder):
    def __init__(self):
        super(VarAutoEncoder, self).__init__()
        self.encoder = RandomEncoder()

    @tf.function
    def call(self, images):
        embedding, mean, logvar = self.encoder(images)
        encoded_images = tf.reshape(embedding, (-1, 16, 16, 10))
        decoded_images = self.decoder(encoded_images)
        return decoded_images, embedding, mean, logvar 
   
    def log_normal_pdf(self, sample, mean, logvar):
        log2pi = tf.math.log(2. * np.pi)
        return tf.reduce_sum(
            -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
            axis=-1)

    @tf.function
    def loss_function(self, results, encodings, originals, mean, logvar):
        SSD = AutoEncoder.loss_function(self, results, originals)
        cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=results, labels=originals)
        logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
        logpz = self.log_normal_pdf(encodings, 0., 0.)
        logqz_x = self.log_normal_pdf(encodings, mean, logvar)
        return SSD + -tf.reduce_mean(logpx_z + logpz - logqz_x)
