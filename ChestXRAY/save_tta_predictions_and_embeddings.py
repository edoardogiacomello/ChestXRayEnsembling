import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
else:
    print("No GPU found, model running on CPU")
import tensorflow_addons as tfa
from skimage.transform import warp, AffineTransform
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import pandas as pd
import math
import seaborn as sns
tf.random.set_seed(42)
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt


def record_parser(example, image_size=224):
    example_fmt = {
        'label': tf.io.FixedLenFeature([14], tf.float32),
        'image': tf.io.FixedLenFeature([],tf.string, default_value='')}
    parsed = tf.io.parse_single_example(example, example_fmt)
    image = tf.io.decode_png(parsed["image"],channels=3)
    image.set_shape([image_size, image_size, 3])
    image = tf.image.convert_image_dtype(image, tf.float32)
    return image, parsed['label']

def normalize_image(img,labels):
    imagenet_mean = np.array([0.485, 0.456, 0.406])
    imagenet_std = np.array([0.229, 0.224, 0.225])
    img = (img - imagenet_mean) / imagenet_std
    return img,labels

def make_dataset(filename, image_size=224):
  base_path = 'datasets/'
  full_path = os.path.join(base_path,filename)
  dataset = tf.data.TFRecordDataset(full_path)
  parser = lambda x, size=image_size: record_parser(x, image_size=size)
  parsed_dataset = dataset.map(parser,num_parallel_calls = tf.data.experimental.AUTOTUNE)
  parsed_dataset = parsed_dataset.map(normalize_image,num_parallel_calls = tf.data.experimental.AUTOTUNE)
  return parsed_dataset

batch_size = 64

@tf.function
def apply_random_scaling(image, minval=-.02, maxval=.02):
    param = tf.random.uniform([], minval=minval, maxval=maxval)
    
    return output, param

def apply_random_shearing(image, minval=-5., maxval=5.):
    #param = tf.random.uniform([], minval=tf.math.atan(minval/image.shape[1]), maxval=tf.math.atan(maxval/image.shape[1]))
    #param = tf.random.uniform([], minval=tf.math.atan(), maxval=tf.math.atan(maxval/image.shape[1]))
    param = np.random.uniform(low=minval, high=maxval)
    output = warp(np.array(image), AffineTransform(shear=np.arctan(param/image.shape[1])).inverse)
    return output, param
@tf.function
def apply_random_rotation(image, minval=-7, maxval=7):
    param = tf.random.uniform([], minval=minval, maxval=maxval)
    output = tfa.image.rotate(image, param*math.pi/180.0, interpolation='BILINEAR')
    return output, param


@tf.function
def apply_tta(image, labels, magnitude=tf.constant(1.0)):
    
    if magnitude <= 0.0:
        return {'x': [image]*10, 'y': [labels]*10, 'method': ['NO_TTA']*10, 'param':[0.0]*10}
    
    scaling_min=-.02*magnitude
    scaling_max=.02*magnitude
    rotation_min=-7*magnitude
    rotation_max=7*magnitude
    augmented_samples = {'x': list(), 'y': list(), 'method': list(), 'param':list()}
    
    output = image
    param = tf.constant(0.0)
    
    # NO-FLIP
    augmented_samples['x'].append(image)
    augmented_samples['y'].append(labels)
    augmented_samples['param'].append(param)
    augmented_samples['method'].append('NOFLIP/ORIGINAL')
        
    # NO-FLIP + ROTATION
    for i in range(1, 3):
        param = tf.random.uniform([], minval=rotation_min, maxval=rotation_max)
        output = tfa.image.rotate(image, param*math.pi/180.0, interpolation='BILINEAR')
        augmented_samples['x'].append(output)
        augmented_samples['y'].append(labels)
        augmented_samples['param'].append(param)
        augmented_samples['method'].append('NOFLIP/ROTATION')    
    
    # NO-FLIP + SCALING
    for i in range(1, 3):
        param = tf.random.uniform([], minval=scaling_min, maxval=scaling_max)
        source_size = image.shape
        target_size = tf.cast(source_size[0]*(1.0+param), tf.int32), tf.cast(source_size[1]*(1.0+param), tf.int32)
        output = tf.image.resize_with_crop_or_pad(tf.image.resize(image, target_size), source_size[0], source_size[1])
        
        augmented_samples['x'].append(output)
        augmented_samples['y'].append(labels)
        augmented_samples['param'].append(param)
        augmented_samples['method'].append('NOFLIP/SCALING') 
    
    flip = tf.image.flip_left_right(image)
    param = tf.constant(0.0)
    
    # NO-FLIP
    augmented_samples['x'].append(flip)
    augmented_samples['y'].append(labels)
    augmented_samples['param'].append(param)
    augmented_samples['method'].append('FLIP/ORIGINAL')
        
    # NO-FLIP + ROTATION
    for i in range(1, 3):
        param = tf.random.uniform([], minval=rotation_min, maxval=rotation_max)
        output = tfa.image.rotate(flip, param*math.pi/180.0, interpolation='BILINEAR')
        augmented_samples['x'].append(output)
        augmented_samples['y'].append(labels)
        augmented_samples['param'].append(param)
        augmented_samples['method'].append('FLIP/ROTATION')    
    
    # NO-FLIP + SCALING
    for i in range(1, 3):
        param = tf.random.uniform([], minval=scaling_min, maxval=scaling_max)
        source_size = flip.shape
        target_size = tf.cast(source_size[0]*(1.0+param), tf.int32), tf.cast(source_size[1]*(1.0+param), tf.int32)
        output = tf.image.resize_with_crop_or_pad(tf.image.resize(flip, target_size), source_size[0], source_size[1])
        augmented_samples['x'].append(output)
        augmented_samples['y'].append(labels)
        augmented_samples['param'].append(param)
        augmented_samples['method'].append('FLIP/SCALING') 
    
    return augmented_samples
    
    
def load_dataset_with_tta(filename, image_size=224, tta_magnitude=1.0):
    base_path = 'datasets/'
    full_path = os.path.join(base_path,filename)
    dataset = tf.data.TFRecordDataset(full_path)
    parser = lambda x, size=image_size: record_parser(x, image_size=size)
    parsed_dataset = dataset.map(parser,num_parallel_calls = tf.data.experimental.AUTOTUNE)
    parsed_dataset = parsed_dataset.map(normalize_image,num_parallel_calls = tf.data.experimental.AUTOTUNE)
    custom_tta = lambda x, y: apply_tta(x, y, magnitude=tf.constant(tta_magnitude))
    parsed_dataset = parsed_dataset.map(custom_tta,num_parallel_calls = tf.data.experimental.AUTOTUNE)
    return parsed_dataset

LOAD_FINETUNED_MODELS = True
ENABLE_BAYES = False

dummy_labels = ['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity', 'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis',
                'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices']

from tensorflow.keras.models import load_model

def load_models(base_path):
      models = []
      model_names = sorted(os.listdir(base_path))
      for model in model_names:
        print('loading model: {}'.format(model))
        models.append(load_model(os.path.join(base_path,model)))
      return models, model_names

if LOAD_FINETUNED_MODELS:
    model_names = ['DenseNet121','DenseNet169','DenseNet201','VGG16','VGG19','Xception','InceptionResNetV2']

    models = []
    for el in model_names:
      mod = load_model('ConditionalTraining/{}/FineTuning/model-05.hdf5'.format(el))
      models.append(mod)
else:
    base_path = './Models/'
    models, model_names = load_models(base_path)
    
embedding_models = list()
for model in models:
    x = model.layers[-2].output
    embedding_models.append(tf.keras.Model(inputs=model.input, outputs=x))
    
def apply_bayes_to_predictions(predictions):
  predictions[:,5] = predictions[:,3] * predictions[:,5]
  predictions[:,6] = predictions[:,3] * predictions[:,6]
  predictions[:,7] = predictions[:,3] * predictions[:,7]
  predictions[:,4] = predictions[:,3] * predictions[:,4]
  predictions[:,8] = predictions[:,3] * predictions[:,8]
  predictions[:,2] = predictions[:,1] * predictions[:,2]
  return predictions

def predict(model, input_, apply_bayes):
    predictions = model.predict(input_)
    if apply_bayes:
        return apply_bayes_to_predictions(predictions)
    else:
        return predictions
    
MAGNITUDES = [0.0, 1.0, 1.5, 2.0, 2.5, 5.0]
predictions = [list() for m in MAGNITUDES]
labels = [list() for m in MAGNITUDES]
tta_params = pd.DataFrame(columns=['sample', 'magnitude', 'method', 'param'])

for m, magnitude in enumerate(MAGNITUDES):
    dset = load_dataset_with_tta('test_set_cropped.tfrecords', tta_magnitude=magnitude)
    embeddings = [list() for mod in embedding_models]
    for r, row in enumerate(dset):
        print('\r Magnitude {} Sample {}'.format(m, r), end='')
        pred = list()
        mod_labels = list()
        for mod, (model, embedding_model) in enumerate(zip(models, embedding_models)):
            pred.append(predict(model, row['x'], ENABLE_BAYES))
            mod_labels.append(row['y'].numpy())
            embeddings[mod].append(embedding_model.predict(row['x']))
        predictions[m].append(np.stack(pred))
        labels[m].append(np.stack(mod_labels))
        for method, param in zip(row['method'].numpy(), row['param'].numpy()):
            tta_params = tta_params.append({'sample': r, 'magnitude': magnitude, 'method': method.decode('utf8'), 'param':param}, ignore_index=True)
        
    # Save the embeddings to avoid wasting memory
    for embd, modname in zip(embeddings, model_names):
        np.save('./Predictions/v2/embedding_test/testembed_{}-Magnitude-{}'.format(modname, magnitude), embd)
    
predictions = np.array(predictions).transpose(0, 1, 3, 2, 4) # Shape (Magnitudes, Samples, Augmentations, Models, Labels)
labels = np.array(labels).transpose(0, 1, 3, 2, 4) # Shape (Magnitudes, Samples, Models, Augmentations, Models, Labels)

np.save('./Predictions/v2/test_predictions_tta', predictions)
np.save('./Predictions/v2/test_labels_tta', labels)
tta_params.to_csv('Predictions/v2/tta_params.csv')
print("All is good.")