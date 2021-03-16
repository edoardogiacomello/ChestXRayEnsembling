import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import functools
import cv2

# Standard CNN classifier with customizable number of convolutional and fully connected layers; 
def make_std_cnn(num_classes=1, n_filters=4, conv_depth=3, fc_layers=2):
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
    
    units=64
    for i in range(fc_layers):
        model.add(tf.keras.layers.Dense(units=units))
    
    if num_classes > 1:
        model.add(tf.keras.layers.Dense(units=num_classes, activation = tf.nn.softmax))
    else:
        model.add(tf.keras.layers.Dense(units=num_classes, activation = None))

    return model

# Transform a CNN in an encoder
# it basically strip the last fc (Dense) layer and substitutes it with a fc layer of the
# wanted latend space
def cnn_to_encoder(model, latent_space_dim = 100):
    model.layers.pop()
    model.layers.add(tf.keras.layers.Dense(units=latent_space_dim, activation=tf.nn.relu))
    return model

# Standard images decoder
def make_std_decoder(latent_space_dim, conv_steps = 3, final_images_size=(128, 128)):    
    # sequential model definition, number of filters fixed
    decoder = tf.keras.Sequential()
    initial_filters = 16
    
    # rearranging the latent layer in images of a size such that the size in the final layer is
    # the one wanted, considering the numbers of convolutional layers with stride 2
    init_dim = np.floor(final_image_size[0] / (tf.math.pow(2, conv_steps)))
    decoder.add(tf.keras.Dense(units=init_dims*init_dims*initial_filters*latent_space_dim, activation='relu'))
    decoder.add(tf.keras.Reshape(target_shape=(init_dims, init_dims, intial_filters*latent_space_dim)))   
    
    for i in range(conv_steps):
        initial_filters = np.floor(initial_filters/2)
        decoder.add(tf.keras.Conv2DTranspose(padding='same', activation='relu'), strides=2, kernel_size=3)  
    
    decoder.add(tf.keras.Conv2D(kernel_size = 3, activation='relu', strides=1))
    
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
        raise ValueError("Could not find 4D layer - No suitable convolutional layer in the model. Cannot apply GradCAM.")

    def compute_heatmap(self, image, eps=1e-8):
        # construct our gradient model by supplying (1) the inputs
        # to our pre-trained model, (2) the output of the (presumably)
        # final 4D layer in the network, and (3) the output of the
        # softmax activations from the model
        gradModel = Model(
            inputs=[self.model.inputs],
            outputs=[self.model.get_layer(self.layer_name).output,
                     self.model.output])

        # record operations for automatic differentiation
        with tf.GradientTape() as tape:
            # cast the image tensor to a float-32 data type, pass the
            # image through the gradient model, and grab the loss
            # associated with the specific class index
            inputs = tf.cast(image, tf.float32)
            (convOutputs, predictions) = gradModel(inputs)
            loss = predictions[:, self.class_idx]
        # use automatic differentiation to compute the gradients
        grads = tape.gradient(loss, convOutputs)

        # compute the guided gradients
        castConvOutputs = tf.cast(convOutputs > 0, "float32")
        castGrads = tf.cast(grads > 0, "float32")
        guidedGrads = castConvOutputs * castGrads * grads

        # the convolution and guided gradients have a batch dimension
        # (which we don't need) so let's grab the volume itself and
        # discard the batch
        convOutputs = convOutputs[0]
        guidedGrads = guidedGrads[0]

        # compute the average of the gradient values, and using them
        # as weights, compute the ponderation of the filters with
        # respect to the weights
        weights = tf.reduce_mean(guidedGrads, axis=(0, 1))
        cam = tf.reduce_sum(tf.multiply(weights, convOutputs), axis=-1)

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
    num_encoder_dims = 2*self.latent_dim + 1

    tmp_model = make_std_cnn()
    self.encoder = cnn_to_encoder(tmp_model, num_encoder_dims)
    self.decoder = make_std_decoder(latent_dim)

  # function to feed images into encoder, encode the latent space, and output
  #   classification probability 
  def encode(self, x):
    # encoder output
    encoder_output = self.encoder(x)

    # classification prediction
    y_logit = tf.expand_dims(encoder_output[:, 0], -1)
    # latent variable distribution parameters
    z_mean = encoder_output[:, 1:self.latent_dim+1]
    z_logsigma = encoder_output[:, self.latent_dim+1:]

    return y_logit, z_mean, z_logsigma

  # VAE reparameterization: given a mean and logsigma, sample latent variables
  def reparameterize(self, z_mean, z_logsigma):
    # TODO: call the sampling function defined above
    z = sampling(z_mean, z_logsigma)
    return z

  # Decode the latent space and output reconstruction
  def decode(self, z):
    # TODO: use the decoder to output the reconstruction
    reconstruction = self.decoder(z)
    return reconstruction

  # The call function will be used to pass inputs x through the core VAE
  def call(self, x): 
    # Encode input to a prediction and latent space
    y_logit, z_mean, z_logsigma = self.encode(x)

    # TODO: reparameterization
    z = self.reparameterize(z_mean=z_mean, z_logsigma=z_logsigma)

    # TODO: reconstruction
    recon = self.decode(z=z)
    return y_logit, z_mean, z_logsigma, recon

  # Predict face or not face logit for given input x
  def predict(self, x):
    y_logit, z_mean, z_logsigma = self.encode(x)
    return y_logit
