import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import functools
import cv2


# Standard CNN classifier with customizable number of convolutional and fully connected layers;
def make_std_cnn(num_classes=1, n_filters=8, conv_depth=4, fc_layers=1):
    model = tf.keras.Sequential()
    for i in range(conv_depth):
        model.add(tf.keras.layers.Conv2D(
            filters=n_filters,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
        ))
        model.add(tf.keras.layers.ReLU())
        model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
        n_filters *= 2

    model.add(tf.keras.layers.Flatten())

    units = 256
    for i in range(fc_layers):
        model.add(tf.keras.layers.Dense(units=units))

    if num_classes > 1:
        model.add(tf.keras.layers.Dense(units=num_classes, activation=tf.nn.softmax))
    else:
        model.add(tf.keras.layers.Dense(units=num_classes, activation=None))

    return model


# Transform a CNN in an encoder
# it basically strips the last fc (Dense) layer and substitutes it with a fc layer of the
# wanted latend space
def cnn_to_encoder(model, latent_space_dim=100):
    encoder = tf.keras.models.Sequential(model.layers[:-1])
    encoder.add(tf.keras.layers.Dense(units=latent_space_dim, activation=tf.nn.relu))
    return encoder


# Standard images decoder
def make_std_decoder(conv_steps=3, final_images_size=(128, 128)):
    # sequential model definition, number of filters fixed
    decoder = tf.keras.Sequential()
    initial_filters = 16

    # rearranging the latent layer in images of a size such that the size in the final layer is
    # the one wanted, considering the numbers of convolutional layers with stride 2
    init_dim = np.floor(final_images_size[0] / (tf.math.pow(2, conv_steps)))
    decoder.add(
        tf.keras.layers.Dense(units=init_dim * init_dim * initial_filters * 7, activation='relu'))
    decoder.add(tf.keras.layers.Reshape(target_shape=(init_dim, init_dim, initial_filters * 7)))

    for i in range(conv_steps):
        initial_filters = np.floor(initial_filters / 2)
        decoder.add(tf.keras.layers.Conv2DTranspose(padding='same', activation='relu', strides=2, kernel_size=3,
                                                    filters=initial_filters))

    decoder.add(tf.keras.layers.Conv2D(kernel_size=3, activation='relu', strides=1, filters=3))

    return decoder


# Class to implement GradCAM
class GradCAM:
    def __init__(self, model, class_idx, layer_name=None):
        # store the model, the class index used to measure the class
        # activation map, and the layer to be used when visualizing
        # the class activation map
        self.model = model
        self.class_idx = class_idx
        self.layer_name = layer_name
        # if the layer name is None, attempt to automatically find
        # the target output layer
        if self.layer_name is None:
            self.layer_name = self.find_target_layer()

    def find_target_layer(self):
        # attempt to find the final convolutional layer in the network
        # by looping over the layers of the network in reverse order
        for layer in reversed(self.model.layers):
            # check to see if the layer has a 4D output
            if len(layer.output_shape) == 4:
                return layer.name
        # otherwise, we could not find a 4D layer so the GradCAM
        # algorithm cannot be applied
        raise ValueError(
            "Could not find 4D layer - No suitable convolutional layer in the model. Cannot apply GradCAM.")

    def compute_heatmap(self, image, eps=1e-8):
        # construct our gradient model by supplying (1) the inputs
        # to our pre-trained model, (2) the output of the (presumably)
        # final 4D layer in the network, and (3) the output of the
        # softmax activations from the model
        grad_model = tf.keras.Model(
            inputs=[self.model.inputs],
            outputs=[self.model.get_layer(self.layer_name).output,
                     self.model.output])

        # record operations for automatic differentiation
        with tf.GradientTape() as tape:
            # cast the image tensor to a float-32 data type, pass the
            # image through the gradient model, and grab the loss
            # associated with the specific class index
            inputs = tf.cast(image, tf.float32)
            (conv_outputs, predictions) = grad_model(inputs)
            loss = predictions[:, self.class_idx]
        # use automatic differentiation to compute the gradients
        grads = tape.gradient(loss, conv_outputs)

        # compute the guided gradients
        cast_conv_outputs = tf.cast(conv_outputs > 0, "float32")
        cast_grads = tf.cast(grads > 0, "float32")
        guided_grads = cast_conv_outputs * cast_grads * grads

        # the convolution and guided gradients have a batch dimension
        # (which we don't need) so let's grab the volume itself and
        # discard the batch
        conv_outputs = conv_outputs[0]
        guided_grads = guided_grads[0]

        # compute the average of the gradient values, and using them
        # as weights, compute the ponderation of the filters with
        # respect to the weights
        weights = tf.reduce_mean(guided_grads, axis=(0, 1))
        cam = tf.reduce_sum(tf.multiply(weights, conv_outputs), axis=-1)

        # grab the spatial dimensions of the input image and resize
        # the output class activation map to match the input image
        # dimensions
        (w, h) = (image.shape[2], image.shape[1])
        heatmap = cv2.resize(cam.numpy(), (w, h))
        # normalize the heatmap such that all values lie in the range
        # [0, 1], scale the resulting values to the range [0, 255],
        # and then convert to an unsigned 8-bit integer
        numer = heatmap - np.min(heatmap)
        denom = (heatmap.max() - heatmap.min()) + eps
        heatmap = numer / denom
        heatmap = (heatmap * 255).astype("uint8")

        # return the resulting heatmap to the calling function
        return heatmap

    def overlay_heatmap(self, heatmap, image, alpha=0.5,
                        colormap=cv2.COLORMAP_INFERNO):
        # apply the supplied color map to the heatmap and then
        # overlay the heatmap on the input image
        heatmap_ = cv2.applyColorMap(heatmap, colormap)
        output = cv2.addWeighted(image, alpha, heatmap, 1 - alpha, 0)
        # return a 2-tuple of the color mapped heatmap and the output,
        # overlaid image
        return heatmap_, output


# Class to implement VAE
class VAE(tf.keras.Model):
    def __init__(self, latent_dim=100):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim

        # Define the number of outputs for the encoder. Recall that we have
        # `latent_dim` latent variables, as well as a supervised output for the
        # classification.
        num_encoder_dims = 2 * self.latent_dim + 1

        # tmp_model = make_std_cnn()
        self.encoder = cnn_to_encoder(model=make_std_cnn(), latent_space_dim=num_encoder_dims)
        self.decoder = make_std_decoder()

    def Summary(self):
        print('Encoder : ')
        tmp = self.encoder(tf.expand_dims(tf.zeros([128, 128, 3]), axis=0))
        self.encoder.summary()
        print('Decoder : ')
        tmp = self.decoder(tf.expand_dims(tf.zeros([100]), axis=0))
        self.decoder.summary()


    # function to feed images into encoder, encode the latent space, and output
    #   classification probability
    def encode(self, x):
        # encoder output
        encoder_output = self.encoder(x)

        # classification prediction
        y_logit = tf.expand_dims(encoder_output[:, 0], -1)
        # latent variable distribution parameters
        z_mean = encoder_output[:, 1:self.latent_dim + 1]
        z_logsigma = encoder_output[:, self.latent_dim + 1:]

        return y_logit, z_mean, z_logsigma

    # VAE Reparameterization ###

    """Reparameterization trick by sampling from an isotropic unit Gaussian.
  # Arguments
      z_mean, z_logsigma (tensor): mean and log of standard deviation of latent distribution (Q(z|X))
  # Returns
      z (tensor): sampled latent vector
  """
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
        y_logit, z_mean, z_logsigma = self.encode(x)

        z = self.reparameterize(z_mean=z_mean, z_logsigma=z_logsigma)

        recon = self.decode(z=z)
        return y_logit, z_mean, z_logsigma, recon

    # Predict face or not face logit for given input x
    def predict(self, x):
        y_logit, z_mean, z_logsigma = self.encode(x)
        return y_logit
