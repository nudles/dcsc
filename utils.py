from __future__ import division,print_function
import math, os, json, sys, re
import pickle
from glob import glob
import numpy as np
from matplotlib import pyplot as plt
from operator import itemgetter, attrgetter, methodcaller
from collections import OrderedDict
import itertools
from itertools import chain

import PIL
from PIL import Image
from numpy.random import random, permutation, randn, normal, uniform, choice
from numpy import newaxis
import scipy
from scipy import misc, ndimage
from scipy.ndimage.interpolation import zoom
from scipy.ndimage import imread
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import OneHotEncoder
from sklearn.manifold import TSNE

from IPython.lib.display import FileLink

import keras
from keras import backend as K
from keras.utils.data_utils import get_file
from keras.utils import np_utils
from keras.utils.np_utils import to_categorical
from keras.models import Sequential, Model
from keras.layers import Input, Embedding, Reshape, merge, LSTM, Bidirectional
from keras.layers import TimeDistributed, Activation, SimpleRNN, GRU
from keras.layers.core import Flatten, Dense, Dropout, Lambda
#from keras.regularizers import l2, activity_l2, l1, activity_l1
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD, RMSprop, Adam
#from keras.utils.layer_utils import layer_from_config
from keras.metrics import categorical_crossentropy, categorical_accuracy
from keras.layers.convolutional import *
from keras.preprocessing import image, sequence
from keras.preprocessing.text import Tokenizer

from shutil import move

np.set_printoptions(precision=4, linewidth=100)


to_bw = np.array([0.299, 0.587, 0.114])

def seperate_dog_cat(src, dst):
    """
    seperate images for dog and cat into two different subfolders under dst, i.e. dst/dog and dst/cat
    """
    imgs = [f for f in os.listdir(src) if os.path.isfile(os.path.join(src, f)) and not f.startswith('.')]
    
    dst_dog = os.path.join(dst, 'dog')
    dst_cat = os.path.join(dst, 'cat')
    if not os.path.exists(dst_dog):
        os.makedirs(dst_dog)
    if not os.path.exists(dst_cat):
        os.makedirs(dst_cat)
    
    for img in imgs:
        if 'dog' in img:
            move(os.path.join(src, img), dst_dog)
        if 'cat' in img:
            move(os.path.join(src, img), dst_cat)
    print('seperate done')


def split_data(src, dst, ratio=0.2):
    """
    move images from src into dst
    
    src is a folder with multiple subfolders, e.g. src/cat, src/dog.
    for each subfolder xxx, create a folder in dst/xxx
    and move ratio images from src/xxx to dst/xxx
    """
    dirs = [f for f in os.listdir(src) if os.path.isdir(os.path.join(src, f)) and not f.startswith('.')]
    for d in dirs:
        src_subdir = os.path.join(src, d)
        dst_subdir = os.path.join(dst, d)
        if not os.path.exists(dst_subdir):
            os.makedirs(dst_subdir)
        imgs = [f for f in os.listdir(src_subdir) if os.path.isfile(os.path.join(src_subdir, f)) and not f.startswith('.')]
        for img in imgs:
            if np.random.uniform() <= ratio:
                move(os.path.join(src_subdir, img), dst_subdir)
    print('split done')
    
def reorganize_dog_cat(root):
    seperate_dog_cat(os.path.join(root, "train"), os.path.join(root, "train"))
    split_data(os.path.join(root, "train"), os.path.join(root, "valid"))

def gray(img):
    if K.image_dim_ordering() == 'tf':
        return np.rollaxis(img, 0, 1).dot(to_bw)
    else:
        return np.rollaxis(img, 0, 3).dot(to_bw)

def to_plot(img):
    if K.image_dim_ordering() == 'tf':
        return np.rollaxis(img, 0, 1).astype(np.uint8)
    else:
        return np.rollaxis(img, 0, 3).astype(np.uint8)

def plot(img):
    plt.imshow(to_plot(img))


def floor(x):
    return int(math.floor(x))
def ceil(x):
    return int(math.ceil(x))

def plots(ims, figsize=(12,6), rows=1, interp=False, titles=None):
    if type(ims[0]) is np.ndarray:
        ims = np.array(ims).astype(np.uint8)
        if (len(ims.shape) == 4 and ims.shape[-1] != 3):
            ims = ims.transpose((0,2,3,1)) 
    f = plt.figure(figsize=figsize)
    cols = len(ims)//rows if len(ims) % 2 == 0 else len(ims)//rows + 1
    for i in range(len(ims)):
        sp = f.add_subplot(rows, cols, i+1)
        sp.axis('Off')
        if titles is not None:
            sp.set_title(titles[i], fontsize=16)
        plt.imshow(ims[i], interpolation=None if interp else 'none')


def do_clip(arr, mx):
    clipped = np.clip(arr, (1-mx)/1, mx)
    return clipped/clipped.sum(axis=1)[:, np.newaxis]


def get_batches(dirname, gen=image.ImageDataGenerator(), shuffle=True, batch_size=4, class_mode='categorical',
                target_size=(224,224)):
    return gen.flow_from_directory(dirname, target_size=target_size,
            class_mode=class_mode, shuffle=shuffle, batch_size=batch_size)


def onehot(x):
    return to_categorical(x)


def wrap_config(layer):
    return {'class_name': layer.__class__.__name__, 'config': layer.get_config()}


def copy_weights(from_layers, to_layers):
    for from_layer,to_layer in zip(from_layers, to_layers):
        to_layer.set_weights(from_layer.get_weights())

        

def get_data(path, target_size=(224,224)):
    batches = get_batches(path, shuffle=False, batch_size=1, class_mode=None, target_size=target_size)
    return np.concatenate([batches.next() for i in range(batches.nb_sample)])


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    (This function is copied from the scikit docs.)
    """
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print(cm)
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')