import os.path
from abc import ABC

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import funcs as ff


# Class to implement VAE
class VAE(tf.keras.Model):
    def __init__(self,
                 number_of_classes: int = 2,
                 latent_dim: int = 100,
                 input_shape=(128, 128, 1),
                 backbone: tf.keras.Model = None):
        """Create a simple model to be trained from scratch if 'dummy_model' is true;
           otherwise, the model has to be loaded from file using the suitable method.
           The latent space dimension is chosen as follows:
            :return latent_dim - outputs for the mean values
            :return latent_dim - outputs for the standard deviations
            :return num_classes - outputs for the classification logits (setting number_of_classes = 0
            basically doesn't allow classification)"""
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = None
        self.decoder = None
        self.num_classes = number_of_classes
        self.backbone = None
        self.vae = None

        #ENCODER
        if backbone is None:
            self.backbone = ff.make_std_cnn(
                num_classes=1,
                n_filters=8,
                conv_depth=3,
                fc_layers=2
            )
        else:
            self.backbone = backbone # tf.keras.Model(backbone.input, backbone.layers[-1].output)

        # freeze layers
        for layer in self.backbone.layers:
            layer.trainable = False

        def sampling(args):
            z_mean, z_log_sigma = args
            epsilon = tf.random.normal(shape=tf.shape(z_mean), mean=0., stddev=1.)
            return z_mean + tf.math.exp(z_log_sigma) * epsilon

        def sampling_with_logits(args):
            z_mean, z_log_sigma, lgts = args
            epsilon = tf.random.normal(shape=tf.shape(z_mean), mean=0., stddev=1.)
            return z_mean + tf.math.exp(z_log_sigma) * epsilon

        # tanh function scaled to output in the range (0, 1). I just like it more than sigmoid
        def my_tanh(x):
            return 0.5*(tf.nn.tanh(x)+1.)

        #mean and std deviation layers, i use leaky_relu to try to prevent weights to zero out themselves
        z_mean = tf.keras.layers.Dense(units=self.latent_dim, activation=tf.nn.leaky_relu, name='z_mean')(self.backbone.layers[-1].output)
        z_log_sigma = tf.keras.layers.Dense(units=self.latent_dim, activation=tf.nn.leaky_relu, name='z_logsigma')(self.backbone.layers[-1].output)

        logits = None
        z = tf.keras.layers.Lambda(sampling, output_shape=(self.latent_dim,), name='sampling-Z')([z_mean, z_log_sigma])
        self.encoder = tf.keras.Model(self.backbone.input, [z, z_mean, z_log_sigma], name='Encoder')

        if number_of_classes > 2:
            logits = tf.keras.layers.Dense(units=1, activation=my_tanh, name='binary-logits')(self.backbone.layers[-1].output)
            z = tf.keras.layers.Lambda(sampling_with_logits, output_shape=(self.latent_dim,), name='sampling-Z')([z_mean, z_log_sigma, logits])
            self.encoder = tf.keras.Model(self.backbone.input, [z, z_mean, z_log_sigma], name='Encoder+Logits')
        elif number_of_classes == 2:
            logits = tf.keras.layers.Dense(units=self.num_classes, activation=my_tanh, name='logits')(self.backbone.layers[-1].output)
            z = tf.keras.layers.Lambda(sampling_with_logits, output_shape=(self.latent_dim,), name='sampling-Z')([z_mean, z_log_sigma, logits])
            self.encoder = tf.keras.Model(self.backbone.input, [z, logits, z_mean, z_log_sigma], name='Encoder_wLogit')


        #DECODER
        self.decoder = ff.make_deeper_decoder(final_images_size=input_shape, input_size=self.latent_dim)
        vae_output = self.decoder(z)

        if logits is not None:
            self.vae = tf.keras.Model(self.backbone.input, [vae_output, z_mean, z_log_sigma, logits], name='VariationalAutoencoder_wLogits')
        else:
            self.vae = tf.keras.Model(self.backbone.input, [vae_output, z_mean, z_log_sigma], name='VariationalAutoencoder')

    def summary(self, expected_dims: list = [128, 128, 3], backbone=False):

        if backbone:
            print('Backbone : ')
            tmp = self.backbone(tf.expand_dims(tf.zeros(expected_dims), axis=0))
            self.backbone.summary()

        # print('Encoder : ')
        # tmp = self.encoder(tf.expand_dims(tf.zeros(expected_dims), axis=0))
        # self.encoder.summary()

        print('\n\nDecoder : ')
        tmp = self.decoder(tf.expand_dims(tf.zeros([self.latent_dim]), axis=0))
        self.decoder.summary()

        print('\n\nVariational Autoencoder : ')
        tmp = self.vae(tf.expand_dims(tf.zeros(expected_dims), axis=0))
        self.vae.summary()

    def encode(self, x):
        return self.encoder(x)

    def sampling(self, z_mean, z_logsigma):
        """VAE Re-parameterization
            Re-parameterization trick by sampling from an isotropic unit Gaussian.
            Arguments :
            z_mean, z_logsigma (tensor): mean and log of standard deviation of latent distribution (Q(z|X))
            Returns :
            z (tensor): sampled latent vector"""
        # By default, random.normal is "standard" (ie. mean=0 and std=1.0)
        batch, latent_dim = z_mean.shape
        epsilon = tf.random.normal(shape=(batch, latent_dim))
        z = z_mean + tf.math.exp(0.5 * z_logsigma) * epsilon
        return z

    def decode(self, z):
        return self.decoder(z)

    def call(self, x):
        return self.vae(x)

    def save_models(self, path, suffix=None, mask=None):
        """Save model"""
        if mask is None:
            mask = [True, True, True]
        if suffix is None:
            suffix = ''

        if mask[0]:
            self.encoder.save(os.path.join(path, 'encoder'+suffix+'.hdf5'))
        if mask[1]:
            self.decoder.save(os.path.join(path, 'decoder'+suffix+'.hdf5'))
        if mask[2]:
            self.vae.save(os.path.join(path, 'vae'+suffix+'.hdf5'))

    @staticmethod
    def loss(true_img, recon_img, mu, logsigma, kl_weight=0.0005):
        """Loss function for a Variational Autoencoder using Kullback-Leibler divergence to regularize
            the distribution (make it similar to ~N(0, 1)) :
            \n>> loss = ||x-x_reconstructed|| + 0.5*∑(µ^2 + σ - log(σ) - 1)
            \n true_img : input image
            \n recon_img : image decoded from latent space samples
            \n mu : latent space distribution mean
            \n logsigma : log(latent space distribution variance)
            \n kl_weight : Kullback-Leibler coefficient (weight for the latent space loss)"""

        reconstruction_loss = tf.reduce_sum(tf.abs(true_img - recon_img), axis=(1, 2, 3))
        latent_loss = 0.5 * tf.reduce_sum(tf.math.exp(logsigma) + tf.math.pow(mu, 2) - 1 - logsigma, axis=-1)

        return kl_weight * latent_loss + reconstruction_loss, reconstruction_loss, latent_loss
