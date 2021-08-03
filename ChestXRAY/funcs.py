import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import functools
import cv2
import time
import os

from IPython import display as ipythondisplay


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
        model.add(tf.keras.layers.Dense(
            units=units,
            activation=tf.nn.relu,
            kernel_initializer=tf.initializers.glorot_normal,
            kernel_regularizer=tf.nn.dropout
        )
        )

    if num_classes > 1:
        model.add(tf.keras.layers.Dense(units=num_classes, activation=tf.nn.softmax))
    else:
        model.add(tf.keras.layers.Dense(units=num_classes, activation=None))

    return model


# Transform a CNN in an encoder
# it basically strips the last fc (Dense) layer and substitutes it with a fc layer of the
# wanted latend space
def cnn_to_encoder(model: tf.keras.models.Model, latent_space_dim: int = 100, keep_last: bool = False) -> object:
    ipt = model.input
    hidden = model.output if keep_last else model.layers[-2].output
    new_layer = tf.keras.layers.Dense(units=latent_space_dim, activation='relu')
    otp = new_layer(hidden)
    encoder = tf.keras.models.Model(ipt, otp)
    return encoder


def cnn_to_deeper_encoder(model: tf.keras.models.Model, latent_space_dim: int = 100, keep_last: bool = False) -> object:
    ipt = model.input
    hidden = model.output if keep_last else model.layers[-2].output
    new_layer1 = tf.keras.layers.Dense(units=latent_space_dim, activation='relu')(hidden)
    otp = tf.keras.layers.Dense(units=latent_space_dim, activation='relu')(new_layer1)
    encoder = tf.keras.models.Model(ipt, otp)
    return encoder


# Standard images decoder
def make_std_decoder(conv_steps=3, final_images_size=(128, 128)):
    # sequential model definition, number of filters fixed
    decoder = tf.keras.Sequential()
    initial_filters = 512

    # rearranging the latent layer in images of a size such that the size in the final layer is
    # the one wanted, considering the numbers of convolutional layers with stride 2
    init_dim = int(np.floor(final_images_size[0] / (tf.math.pow(2, conv_steps))))
    decoder.add(tf.keras.layers.Dense(units=init_dim * init_dim * initial_filters, activation='relu'))
    decoder.add(tf.keras.layers.Reshape(target_shape=(init_dim, init_dim, initial_filters)))
    print('decoder initial dim : ', init_dim)

    for i in range(conv_steps):
        decoder.add(tf.keras.layers.Conv2DTranspose(padding='same', activation='relu', strides=2, kernel_size=3,
                                                    filters=initial_filters))
        initial_filters = np.floor(initial_filters / 2)

    decoder.add(tf.keras.layers.Conv2D(kernel_size=3, activation='relu', strides=1, filters=3, padding='same'))

    return decoder


def make_deeper_decoder(final_images_size=(128, 128), input_size=100):
    # sequential model definition, number of filters fixed
    # decoder = tf.keras.Sequential()
    inpt = tf.keras.layers.Input(shape=(input_size))
    filters = 512
    conv_steps = 7
    # rearranging the latent layer in images of a size such that the size in the final layer is
    # the one wanted, considering the numbers of convolutional layers with stride 2
    # init_dim = int(np.floor(final_images_size[0] / (tf.math.pow(2, conv_steps))))
    dense1 = tf.keras.layers.Dense(units=filters, activation='relu', name='dense-layer')(inpt)
    # dense2 = tf.keras.layers.Dense(units=init_dim * init_dim * 9, activation='relu')(dense1)
    rshp = tf.keras.layers.Reshape(target_shape=(2, 2, filters // 4), name='reshape-to-imgs')(dense1)
    # print('decoder initial dim : ', init_dim)

    previous_layer = tf.keras.layers.Conv2D(padding='same', activation='relu', strides=1, kernel_size=1,
                                                     filters=filters, name='first-convolutional')(rshp)
    for i in range(conv_steps):

        filters = max(np.floor(filters / 2), 32)

        next_layer = tf.keras.layers.Conv2DTranspose(padding='same', activation='relu', strides=1, kernel_size=3,
                                                     filters=filters, name='conv-transpose-{}'.format(i))(previous_layer)
        up_layer = tf.keras.layers.UpSampling2D(name='upsampling{}'.format(i))(next_layer)
        previous_layer = up_layer

    output = tf.keras.layers.Conv2D(kernel_size=3, activation='relu', strides=1, filters=3, padding='same', name='tmp-output')(
        previous_layer)
    rsz = tf.keras.layers.experimental.preprocessing.Resizing(final_images_size[0], final_images_size[1], name='resizing')(output)
    decoder = tf.keras.Model(inpt, rsz, name='Decoder')
    return decoder


def normalize_tensor(tensor: tf.Tensor):
    max = tf.reduce_max(tensor)
    min = tf.reduce_min(tensor)
    return (tensor - min) / (max - min)


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


def display_model(model):
    tf.keras.utils.plot_model(model,
                              to_file='tmp.png',
                              show_shapes=True)
    return ipythondisplay.Image('tmp.png')


class TrainingUtility:
    def __init__(self, model_names_list: list = None, smoothing=.75, patience=10, starting_epoch: int = 1):
        if model_names_list is None:
            model_names_list = ['model-1']

        self.train_loss = []
        self.val_loss = []
        self.train_accuracy = []
        self.val_accuracy = []
        self.loss_plot_scale = 'semilogy'
        self.fig1 = self.ax1 = self.ax2 = None
        self.smoothing_factor = smoothing
        self.modelnames = model_names_list
        self.patience = patience
        self.epoch = starting_epoch

    def plot_loss(self, scale='semilogy', xlabel='epoch', ylabel='loss', plot_color='#CAF625', bkgrd_color='#06260A'):
        self.loss_plot_scale = scale
        self.fig1 = plt.figure(1)
        # self.fig.add_subplot(2, 1, 1, sharex=True)
        self.ax1 = plt.subplot(211)
        plt.cla()
        self.ax1.tick_params(axis='x', colors=plot_color)
        self.ax1.tick_params(axis='y', colors=plot_color)
        self.ax1.title.set_color(plot_color)
        self.ax1.yaxis.label.set_color(plot_color)
        self.ax1.xaxis.label.set_color(plot_color)
        self.ax1.set_facecolor(bkgrd_color)
        if self.loss_plot_scale is None:
            self.ax1.plot(range(len(self.train_loss)), self.train_loss, 'r-', self.val_loss, 'y-')
        elif self.loss_plot_scale == 'semilogx':
            self.ax1.semilogx(range(len(self.train_loss)), self.train_loss, 'r-', self.val_loss, 'y-')
        elif self.loss_plot_scale == 'semilogy':
            self.ax1.semilogy(range(len(self.train_loss)), self.train_loss, 'r-', self.val_loss, 'y-')
        elif self.loss_plot_scale == 'loglog':
            self.ax1.loglog(range(len(self.train_loss)), self.train_loss, 'r-', self.val_loss, 'y-')
        else:
            raise ValueError("unrecognized parameter scale {}".format(self.loss_plot_scale))

        plt.legend(['training', 'validation'])
        plt.title('Loss Plot')
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

    def plot_metric(self, ylabel='Accuracy', plot_color='#FF46C7', bkgrd_color='#06260A'):
        self.ax2 = plt.subplot(212)
        plt.cla()
        self.ax2.tick_params(axis='x', colors=plot_color)
        self.ax2.tick_params(axis='y', colors=plot_color)
        self.ax2.title.set_color(plot_color)
        self.ax2.yaxis.label.set_color(plot_color)
        self.ax2.xaxis.label.set_color(plot_color)
        self.ax2.set_facecolor(bkgrd_color)
        plt.ylabel(ylabel)
        plt.title(ylabel + 'Plot')
        plt.plot(range(len(self.val_accuracy)),
                 self.train_accuracy, 'r-',
                 self.val_accuracy, 'y-')
        plt.legend(['training', 'validation'])

    def show_plots(self):
        ipythondisplay.clear_output(wait=False)
        ipythondisplay.display(plt.show())

    def add_loss(self, train_value, val_value=None):
        self.train_loss.append(
            self.smoothing_factor * self.train_loss[-1] + (1 - self.smoothing_factor) * train_value if len(
                self.train_loss) > 0 else train_value)
        if val_value is not None:
            self.val_loss.append(
                self.smoothing_factor * self.val_loss[-1] + (1 - self.smoothing_factor) * val_value if len(
                    self.val_loss) > 0 else val_value)

    def add_metric(self, train_value, val_value=None):
        self.train_accuracy.append(train_value)
        if val_value is not None:
            self.val_accuracy.append(val_value)

    def plot_imgs(self, imgs: tf.Tensor, imgs2: tf.Tensor = None, plot_dim=(2, 2)):
        fig = plt.figure(2)
        imgs = normalize_tensor(imgs)
        imgs2 = normalize_tensor(imgs2)
        plt.title('Images', color='red')
        for i in range(imgs.shape[0]):
            plt.subplot(plot_dim[0], plot_dim[1], i + 1)
            plt.imshow(imgs[i, :, :, :])
            plt.axis('off')

        fig = plt.figure(3)
        plt.title('Original Images', color='red')
        for i in range(imgs2.shape[0]):
            plt.subplot(plot_dim[0], plot_dim[1], i + 1)
            plt.imshow(imgs2[i, :, :, :])
            plt.axis('off')

    def save_model(self, models, path='model', suffix=None):
        if suffix is None:
            suffix = '-epoch_{}.hdf5'.format(self.epoch)
        if not os.path.isdir(path):
            os.mkdir(path)

        for model, name in zip(models, self.modelnames):
            filename = os.path.join(path, name + suffix)
            model.save(filename)

    def earlystopping_overfitting(self, divergence_check=False):
        """
        This function check if the validation loss is diverging too much from the training loss or it is even increasing,
        showing signs of overfitting.
        :param divergence_check: check for excessive divergence as well
        :return: boolean which tell if he requirement has been met (True) or not (False)
        """

        # if less than 'patience' epochs are elapsed, just ignore and go on
        if len(self.train_loss) < self.patience:
            return False

        stop_cond_1 = False
        stop_cond_2 = False

        if divergence_check:
            mean_difference = np.mean(
                np.abs(self.train_loss[::-1][:self.patience] - self.val_loss[::-1][:self.patience]))
            mean_training_loss = np.mean(self.train_loss[::-1][:self.patience])
            diff = np.abs(mean_difference / mean_training_loss)
            stop_cond_1 = diff >= .75
            print(
                "*Callback_EarlyStopping_Overfitting* Mean difference between training and validation loss is %f times the training loss" % diff)

        previous_validations = self.val_loss[::-1][1:self.patience + 1]
        last_validations = self.val_loss[::-1][:self.patience]
        s = np.sum(last_validations - previous_validations)

        if s > 0:
            print("*Callback_EarlyStopping_Overfitting* !!!! Validation loss is INCREASING over the last %d epochs" % (
                self.patience))
            print("*Callback_EarlyStopping_Overfitting* Mean d_val-loss/d_epoch : %f" % s)
            stop_cond_2 = True
        else:
            print("*Callback_EarlyStopping_Overfitting* Validation loss is decreasing over the last %d epochs" % (
                self.patience))
            print("*Callback_EarlyStopping_Overfitting* Mean d_val-loss/d_epoch : %f" % s)

        return stop_cond_1 or stop_cond_2

    def earlystopping_loss(self, min_delta: float = 0.05):
        """
        This fuction checks for minimum percentage improvements over previous loss through moving average.
        :param loss_list: history of losses
        :param min_delta: minimum improvement required
        :return: boolean which tell if he requirement has been met (True) or not (False)
        """
        # No early stopping for 2*self.patience epochs
        if len(self.train_loss) < self.patience:
            return False
        # Mean loss for last self.patience epochs and second-last self.patience epochs
        mean_previous = np.mean(self.train_loss[::-1][self.patience:2 * self.patience])  # second-last
        mean_recent = np.mean(self.train_loss[::-1][:self.patience])  # last
        # you can use relative or absolute change
        delta_abs = np.abs(mean_recent - mean_previous)  # abs change
        delta_abs = np.abs(delta_abs / mean_previous)  # relative change
        if delta_abs < min_delta:
            print("*Callback_EarlyStopping_Loss* Loss didn't change much from last %d epochs" % (self.patience))
            print("*Callback_EarlyStopping_Loss* Percent change in loss value:", delta_abs * 1e2)
            return True
        else:
            print("*Callback_EarlyStopping_Loss* Loss changed enough from last %d epochs" % (self.patience))
            print("*Callback_EarlyStopping_Loss* Percent change in loss value:", delta_abs * 1e2)
            return False
