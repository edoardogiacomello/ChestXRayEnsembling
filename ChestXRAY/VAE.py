import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import funcs as ff


# Class to implement VAE
class VAE(tf.keras.Model):
    def __init__(self, dummy_model = True, latent_dim=100):
        super(VAE, self).__init__()
        self.latent_dim = None
        self.encoder = None
        self.decoder = None

        # Create a simple model to be trained from scratch if 'dummy_model' is true;
        # otherwise, the model has to be loaded from file using the suitable method
        if dummy_model:
            self.latent_dim = latent_dim
            num_encoder_dims = 2 * self.latent_dim + 1
            self.encoder = ff.cnn_to_encoder(model=ff.make_std_cnn(), latent_space_dim=num_encoder_dims)
            self.decoder = ff.make_std_decoder()

    def summary(self):
        print('Encoder : ')
        tmp = self.encoder(tf.expand_dims(tf.zeros([128, 128, 3]), axis=0))
        self.encoder.summary()
        print('Decoder : ')
        tmp = self.decoder(tf.expand_dims(tf.zeros([100]), axis=0))
        self.decoder.summary()

    # function to feed images into encoder, encode the latent space, and output
    # mean and variance of the latent space
    def encode(self, x):
        # encoder output
        encoder_output = self.encoder(x)

        # classification prediction
        # y_logit = tf.expand_dims(encoder_output[:, 0], -1)
        # latent variable distribution parameters
        z_mean = encoder_output[:, 1:self.latent_dim + 1]
        z_logsigma = encoder_output[:, self.latent_dim + 1:]

        return z_mean, z_logsigma  # , y_logit

    # # VAE Reparameterization ###
    # Reparameterization trick by sampling from an isotropic unit Gaussian.
    # # Arguments
    # z_mean, z_logsigma (tensor): mean and log of standard deviation of latent distribution (Q(z|X))
    # # Returns
    # z (tensor): sampled latent vector
    def sampling(self, z_mean, z_logsigma):
        # By default, random.normal is "standard" (ie. mean=0 and std=1.0)
        batch, latent_dim = z_mean.shape
        epsilon = tf.random.normal(shape=(batch, latent_dim))

        z = z_mean + tf.math.exp(0.5 * z_logsigma) * epsilon
        return z

    # VAE reparameterization: given a mean and logsigma, sample latent variables
    def reparameterize(self, z_mean, z_logsigma):
        z = self.sampling(z_mean, z_logsigma)
        return z

    # Decode the latent space and output reconstruction
    def decode(self, z):
        reconstruction = self.decoder(z)
        return reconstruction

    # The call function will be used to pass inputs x through the core VAE
    def call(self, x):
        # Encode input to a prediction and latent space
        z_mean, z_logsigma = self.encode(x)

        # use the re-parametrization trick to sample the latent space
        z = self.reparameterize(z_mean=z_mean, z_logsigma=z_logsigma)

        # get a reconstructed image from the decoder
        recon = self.decode(z=z)
        return z_mean, z_logsigma, recon

    # Predicts the class of the autoencoder
    def predict(self, x):
        predicted_class, z_mean, z_logsigma = self.encode(x)
        return predicted_class

    # Save model
    def save_models(self):
        self.encoder.save('models/encoder')
        self.decoder.save('models/decoder')

    # load model from file : necessary if a dummy_model has not been created
    def load_model(self, encoder_path='models/encoder', decoder_path='models/decoder'):
        self.encoder = tf.keras.models.load_model(encoder_path)
        self.decoder = tf.keras.models.load_model(decoder_path)

    '''Loss function for a Variational Autoencoder;
    true_img : input image
    recon_img : image decoded from latent space samples
    mu : latent space distribution mean
    logsigma : log(latent space distribution variance)
    kl_weight : Kullback-Leibler weight (weight for the latent space loss)'''
    @staticmethod
    def loss_function(true_img, recon_img, mu, logsigma, kl_weight=0.0005):

        reconstruction_loss = tf.reduce_mean(tf.abs(true_img - recon_img))
        latent_loss = 0.5 * tf.reduce_sum(tf.math.exp(logsigma) + tf.math.pow(mu, 2) - 1 - logsigma, axis=1)

        return kl_weight*latent_loss + reconstruction_loss
