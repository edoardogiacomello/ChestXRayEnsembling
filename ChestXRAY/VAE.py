from abc import ABC

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import funcs as ff


# Class to implement VAE
class VAE(tf.keras.Model):
    def __init__(self, number_of_classes=2, dummy_model=True, latent_dim=100):
        """Create a simple model to be trained from scratch if 'dummy_model' is true;
                    otherwise, the model has to be loaded from file using the suitable method.
                    The latent space dimension is chosen as follows :
                    -self.latent_dim outputs for the mean values
                    -self.latent_dim outputs for the standard deviations
                    -self.num_classes outputs for the classification logits (
                        setting number_of_classes = 0 basically doesn't allow classification)"""
        super(VAE, self).__init__()
        self.latent_dim = None
        self.encoder = None
        self.decoder = None
        self.num_classes = number_of_classes

        if dummy_model:
            self.latent_dim = latent_dim
            encoder_outputs = None
            if self.num_classes == 2:
                encoder_outputs = 2 * self.latent_dim + 1
            elif self.num_classes > 2:
                encoder_outputs = 2 * self.latent_dim + self.num_classes
            elif self.num_classes < 2 or (self.num_classes is None):
                encoder_outputs = 2 * self.latent_dim

            self.encoder = ff.cnn_to_encoder(
                model=ff.make_std_cnn(
                    num_classes=1,
                    n_filters=8,
                    conv_depth=6,
                    fc_layers=3
                ),
                latent_space_dim=encoder_outputs
            )
            self.decoder = ff.make_std_decoder(
                conv_steps=3,
                final_images_size=(128, 128))

    def summary(self):
        print('Encoder : ')
        tmp = self.encoder(tf.expand_dims(tf.zeros([128, 128, 3]), axis=0))
        self.encoder.summary()
        print('Decoder : ')
        tmp = self.decoder(tf.expand_dims(tf.zeros([self.latent_dim]), axis=0))
        self.decoder.summary()

    def encode(self, x):
        """function to feed images into encoder, encode the latent space, and output
            mean and variance of the latent space"""
        # encoder output
        encoder_output = self.encoder(x)

        # classification prediction
        y_logits = encoder_output[:, 0:self.num_classes]
        # latent variable distribution parameters
        z_mean = encoder_output[:, self.num_classes:self.latent_dim+self.num_classes]
        z_logsigma = encoder_output[:, self.latent_dim+self.num_classes:]

        return z_mean, z_logsigma, y_logits

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

    def re_parameterize(self, z_mean, z_logsigma):
        """VAE re-parametrization: given a mean and logsigma, sample latent variables"""
        z = self.sampling(z_mean, z_logsigma)
        return z

    def decode(self, z):
        """Decode the latent space and output reconstruction"""
        reconstruction = self.decoder(z)
        return reconstruction

    def call(self, x):
        """The call function will be used to pass inputs x through the VAE"""
        # Encode input to a prediction and latent space
        z_mean, z_logsigma, y_logits = self.encode(x)

        # use the re-parametrization trick to sample the latent space
        z = self.re_parameterize(z_mean=z_mean, z_logsigma=z_logsigma)

        # get a reconstructed image from the decoder
        recon = self.decode(z=z)
        return z_mean, z_logsigma, recon, y_logits

    def get_logits(self, x):
        """Predicts the class of the autoencoder"""
        z_mean, z_logsigma, logits = self.encode(x)
        return logits

    def predict(self, x):
        """Classes probability of a certain input"""
        _, _, logits = self.encode(x)

        return tf.exp(logits) / tf.reduce_sum(tf.exp(logits), axis=1)

    def save_models(self):
        """Save model"""
        self.encoder.save('models/encoder')
        self.decoder.save('models/decoder')

    def load_model(self, encoder_path='models/encoder', decoder_path='models/decoder'):
        """load model from file : necessary if a dummy_model has not been created"""
        self.encoder = tf.keras.models.load_model(encoder_path)
        self.decoder = tf.keras.models.load_model(decoder_path)

    @staticmethod
    def vae_loss_function(true_img, recon_img, mu, logsigma, kl_weight=0.0005):
        """Loss function for a Variational Autoencoder using Kullback-Leibler divergence to regularize
            the distribution (make it similar to ~N(0, 1)) :
            \n>> loss = ||x-x_reconstructed|| + 0.5*∑(µ^2 + σ - log(σ) - 1)
            \n true_img : input image
            \n recon_img : image decoded from latent space samples
            \n mu : latent space distribution mean
            \n logsigma : log(latent space distribution variance)
            \n kl_weight : Kullback-Leibler coefficient (weight for the latent space loss)"""

        reconstruction_loss = tf.reduce_mean(tf.abs(true_img - recon_img))
        latent_loss = 0.5 * tf.reduce_sum(tf.math.exp(logsigma) + tf.math.pow(mu, 2) - 1 - logsigma, axis=1)

        return kl_weight * latent_loss + reconstruction_loss

