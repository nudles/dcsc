
# coding: utf-8

# In[ ]:


'''Trains a simple convnet on the MNIST dataset.
Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.

Adapted from https://github.com/fchollet/keras/blob/master/examples/mnist_cnn.py

MIT license.
'''

from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
K.set_image_data_format('channels_first')

class MnistCNN(object):
    def __init__(self):
        # input image dimensions
        self.img_rows, self.img_cols = 28, 28
        self.num_classes = 10
        
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3),
                         activation='relu',
                         input_shape=(1, self.img_rows, self.img_cols)))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(self.num_classes, activation='softmax'))
        self.model = model  
        
    def load_weights(self, ckpt_path):
        self.model.load_weights(ckpt_path)
        
    def save_weights(self, ckpt_path):
        self.model.save_weights(ckpt_path)
        
        
    def predict(self, x, batchsize):
        x = x.reshape(x.shape[0], 1, self.img_rows, self.img_cols)
        x = x.astype('float32')
        x /= 255
        return self.model.predict(x, batchsize)
        
    def train(self, epochs=12, batch_size=128):
        # the data, shuffled and split between train and test sets
        (x_train, y_train), (x_test, y_test) = mnist.load_data()

        #if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, self.img_rows, self.img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, self.img_rows, self.img_cols)
        input_shape = (1, self.img_rows, self.img_cols)


        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train /= 255
        x_test /= 255
        print('x_train shape:', x_train.shape)
        print(x_train.shape[0], 'train samples')
        print(x_test.shape[0], 'test samples')

        # convert class vectors to binary class matrices
        y_train = keras.utils.to_categorical(y_train, self.num_classes)
        y_test = keras.utils.to_categorical(y_test, self.num_classes)
    
        self.model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=keras.optimizers.Adadelta(),
                      metrics=['accuracy'])

        self.model.fit(x_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=1,
                  validation_data=(x_test, y_test))
        score = self.model.evaluate(x_test, y_test, verbose=0)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])

    def summary(self):
        self.model.summary()