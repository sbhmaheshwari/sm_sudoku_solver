# -*- coding: utf-8 -*-
''' In this file, I first train a classification model on MNIST, then on Chars74 dataset. Finally, Chars74 dataset
model works better for identifying Sudoku digits. Feel free to do data augmentation for Chars74 dataset images'''
# Commented out IPython magic to ensure Python compatibility.
from keras.datasets.mnist import load_data
from keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Flatten, Dense, Dropout, Input, Activation
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
from keras.models import Model, load_model
import keras.backend as K
import numpy as np
import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import confusion_matrix
import cv2
import os
import json
# %matplotlib inline

class CyclicLR(Callback):
    """    
    # Example
        ```python
            clr = CyclicLR(base_lr=0.001, max_lr=0.006,
                                step_size=2000., mode='triangular')
            model.fit(X_train, Y_train, callbacks=[clr])
        ```
    # Arguments
        base_lr: initial learning rate which is the
            lower boundary in the cycle.
        max_lr: upper boundary in the cycle. Functionally,
            it defines the cycle amplitude (max_lr - base_lr).
            The lr at any cycle is the sum of base_lr
            and some scaling of the amplitude; therefore 
            max_lr may not actually be reached depending on
            scaling function.
        step_size: number of training iterations per
            half cycle. Authors suggest setting step_size
            2-8 x training iterations in epoch.
        mode: one of {triangular, triangular2, exp_range}.
            Default 'triangular'.
            Values correspond to policies detailed above.
            If scale_fn is not None, this argument is ignored.
        gamma: constant in 'exp_range' scaling function:
            gamma**(cycle iterations)
        scale_fn: Custom scaling policy defined by a single
            argument lambda function, where 
            0 <= scale_fn(x) <= 1 for all x >= 0.
            mode paramater is ignored 
        scale_mode: {'cycle', 'iterations'}.
            Defines whether scale_fn is evaluated on 
            cycle number or cycle iterations (training
            iterations since start of cycle). Default is 'cycle'.
    """

    def __init__(self, base_lr=0.001, max_lr=0.006, step_size=2000., mode='triangular',
                 gamma=1., scale_fn=None, scale_mode='cycle'):
        super(CyclicLR, self).__init__()

        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.mode = mode
        self.gamma = gamma
        if scale_fn == None:
            if self.mode == 'triangular':
                self.scale_fn = lambda x: 1.
                self.scale_mode = 'cycle'
            elif self.mode == 'triangular2':
                self.scale_fn = lambda x: 1/(2.**(x-1))
                self.scale_mode = 'cycle'
            elif self.mode == 'exp_range':
                self.scale_fn = lambda x: gamma**(x)
                self.scale_mode = 'iterations'
        else:
            self.scale_fn = scale_fn
            self.scale_mode = scale_mode
        self.clr_iterations = 0.
        self.trn_iterations = 0.
        self.history = {}

        self._reset()

    def _reset(self, new_base_lr=None, new_max_lr=None,
               new_step_size=None):
        """Resets cycle iterations.
        Optional boundary/step size adjustment.
        """
        if new_base_lr != None:
            self.base_lr = new_base_lr
        if new_max_lr != None:
            self.max_lr = new_max_lr
        if new_step_size != None:
            self.step_size = new_step_size
        self.clr_iterations = 0.
        
    def clr(self):
        cycle = np.floor(1+self.clr_iterations/(2*self.step_size))
        x = np.abs(self.clr_iterations/self.step_size - 2*cycle + 1)
        if self.scale_mode == 'cycle':
            return self.base_lr + (self.max_lr-self.base_lr)*np.maximum(0, (1-x))*self.scale_fn(cycle)
        else:
            return self.base_lr + (self.max_lr-self.base_lr)*np.maximum(0, (1-x))*self.scale_fn(self.clr_iterations)
        
    def on_train_begin(self, logs={}):
        logs = logs or {}

        if self.clr_iterations == 0:
            K.set_value(self.model.optimizer.lr, self.base_lr)
        else:
            K.set_value(self.model.optimizer.lr, self.clr())        
            
    def on_batch_end(self, epoch, logs=None):
        
        logs = logs or {}
        self.trn_iterations += 1
        self.clr_iterations += 1

        self.history.setdefault('lr', []).append(K.get_value(self.model.optimizer.lr))
        self.history.setdefault('iterations', []).append(self.trn_iterations)

        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)
        
        K.set_value(self.model.optimizer.lr, self.clr())
		
class LRFinder(Callback):
    
    '''
    A simple callback for finding the optimal learning rate range for your model + dataset. 
    
    # Usage
        ```python
            lr_finder = LRFinder(min_lr=1e-5, 
                                 max_lr=1e-2, 
                                 steps_per_epoch=np.ceil(epoch_size/batch_size), 
                                 epochs=3)
            model.fit(X_train, Y_train, callbacks=[lr_finder])
            
            lr_finder.plot_loss()
        ```
    
    # Arguments
        min_lr: The lower bound of the learning rate range for the experiment.
        max_lr: The upper bound of the learning rate range for the experiment.
        steps_per_epoch: Number of mini-batches in the dataset. Calculated as `np.ceil(epoch_size/batch_size)`. 
        epochs: Number of epochs to run experiment. Usually between 2 and 4 epochs is sufficient. 
        
    # References
        Blog post: jeremyjordan.me/nn-learning-rate
        Original paper: https://arxiv.org/abs/1506.01186
    '''
    
    def __init__(self, min_lr=1e-5, max_lr=1e-2, steps_per_epoch=None, epochs=None):
        super().__init__()
        
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.total_iterations = steps_per_epoch * epochs
        self.iteration = 0
        self.history = {}
        
    def clr(self):
        '''Calculate the learning rate.'''
        x = self.iteration / self.total_iterations 
        return self.min_lr + (self.max_lr-self.min_lr) * x
        
    def on_train_begin(self, logs=None):
        '''Initialize the learning rate to the minimum value at the start of training.'''
        logs = logs or {}
        K.set_value(self.model.optimizer.lr, self.min_lr)
        
    def on_batch_end(self, epoch, logs=None):
        '''Record previous batch statistics and update the learning rate.'''
        logs = logs or {}
        self.iteration += 1

        self.history.setdefault('lr', []).append(K.get_value(self.model.optimizer.lr))
        self.history.setdefault('iterations', []).append(self.iteration)

        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)
            
        K.set_value(self.model.optimizer.lr, self.clr())
 
    def plot_lr(self):
        '''Helper function to quickly inspect the learning rate schedule.'''
        plt.plot(self.history['iterations'], self.history['lr'])
        plt.yscale('log')
        plt.xlabel('Iteration')
        plt.ylabel('Learning rate')
        plt.show()
        
    def plot_loss(self):
        '''Helper function to quickly observe the learning rate experiment results.'''
        plt.plot(self.history['lr'], self.history['loss'])
        plt.xscale('log')
        plt.xlabel('Learning rate')
        plt.ylabel('Loss')
        plt.show()

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')



''' MNIST model'''
(x_train, y_train), (x_test, y_test) = load_data()
x_train.shape, y_train.shape, x_test.shape, y_test.shape

#preprocess
x_train = x_train.astype('float32')/255
x_test = x_test.astype('float32')/255
x_train, x_test = x_train[:,:,:, np.newaxis], x_test[:,:,:, np.newaxis]

y_train = to_categorical(y_train, num_classes = 10)
y_test = to_categorical(y_test, num_classes=10)

def model(filters = 32, kernel_size = 3, h = 28, w = 28, num_classes = 10, metrics = ['accuracy']):
  inp = Input(shape = (h,w,1))
  
  x = Conv2D(filters, (kernel_size, kernel_size), kernel_initializer='he_normal', padding = 'same')(inp)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  x = MaxPooling2D()(x)

  x = Conv2D(filters*2, (kernel_size, kernel_size), kernel_initializer='he_normal', padding = 'same')(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  x = MaxPooling2D()(x)

  x = Flatten()(x)
  x = Dense(32)(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  x = Dropout(0.2)(x)

  x = Dense(num_classes)(x)
  x = Activation('softmax')(x)

  model = Model(input = inp, output = x)
  model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = metrics)

  return(model)

model_mnist = model()

batch_size = 32
epoch_size = x_train.shape[0]

# find range of learning rate to use
lr_finder = LRFinder(min_lr=1e-4, max_lr=1e-1, steps_per_epoch=np.ceil(epoch_size/batch_size), epochs=3)
model_mnist.fit(x_train, y_train, callbacks=[lr_finder])

lr_finder.plot_loss()

clr = CyclicLR(base_lr=0.001, max_lr=0.01, step_size=1000., mode='triangular')
early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
model_mnist = model()
model_mnist.fit(x_train, y_train, validation_data = [x_test, y_test], epochs = 50, callbacks=[clr, early_stopping], batch_size=batch_size)

model_mnist.save('gdrive/My Drive/Sudoku/'+'model_mnist'+'.hdf5')

model_mnist = load_model('gdrive/My Drive/Sudoku/'+'model_mnist'+'.hdf5')

y_pred = model_mnist.predict(x_test)
y_pred = y_pred.argmax(axis=1)

np.mean(y_pred == y_test.argmax(-1))


confusion_mtx = confusion_matrix(y_pred, y_test.argmax(-1)) 
# plot the confusion matrix
plot_confusion_matrix(confusion_mtx, classes = range(10))

#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------

"""Char74 dataset"""

# path for images (replace this with the path where the images are stored in your machine)
im_path = '/content/gdrive/My Drive/Sudoku'

x_train_total = len(os.listdir(im_path+'/chars_train'))
x_test_total = len(os.listdir(im_path+'/chars_test'))

img_rows, img_cols = [28]*2

# preprocessing images to make them similar to sudoku digits
x_train = []
for i in range(x_train_total):
  if i%500==0: print(i)
  path = im_path+'/chars_train/'+str(i)+'.png'
  img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
  r, th = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)
  fin = cv2.resize(th, (img_rows, img_cols), interpolation = cv2.INTER_LANCZOS4)
  x_train.append(fin)

#with open(im_path+'/chars_train.pkl', 'wb') as file:
#  pickle.dump(x_train, file, protocol = pickle.HIGHEST_PROTOCOL)

x_test = []
for i in range(x_test_total):
  if i%500==0: print(i)
  path = im_path+'/chars_test/'+str(i)+'.png'
  img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
  r, th = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)
  fin = cv2.resize(th, (img_rows, img_cols), interpolation = cv2.INTER_LANCZOS4)
  x_test.append(fin)

#with open(im_path+'/chars_test.pkl', 'wb') as file:
#  pickle.dump(x_test, file, protocol = pickle.HIGHEST_PROTOCOL)

with open(im_path+'/labels_train.json') as file:
  y_train = json.load(file)
  y_train = [i-1 for i in y_train]

with open(im_path + '/labels_test.json') as file:
  y_test = json.load(file)
  y_test = [i-1 for i in y_test]

y_train, y_test, x_train, x_test = np.array(y_train), np.array(y_test), np.array(x_train), np.array(x_test)

x_train, x_test = x_train[:,:,:,np.newaxis]/255, x_test[:,:,:,np.newaxis]/255

y_train, y_test = to_categorical(y_train, 10), to_categorical(y_test, 10)

batch_size = 64
epoch_size = x_train.shape[0]

model_ch74 = model()

# find optimal range for learning rate 
lr_finder = LRFinder(min_lr=1e-4, max_lr=1e-1, steps_per_epoch=np.ceil(epoch_size/batch_size), epochs=3)
model_ch74.fit(x_train, y_train, callbacks=[lr_finder])

lr_finder.plot_loss()

clr = CyclicLR(base_lr=1e-3, max_lr=2e-3, step_size=1000., mode='triangular')
early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
model_ch4 = model()
model_ch4.fit(x_train, y_train, validation_data = [x_test, y_test], epochs = 50, callbacks=[clr, early_stopping], batch_size=batch_size)

model_ch4.save(im_path+'/model_ch4'+'.hdf5')

y_pred = model_ch4.predict(x_test).argmax(-1)

np.mean(y_pred == y_test.argmax(-1))

confusion_mtx = confusion_matrix(y_pred, y_test.argmax(-1)) 
# plot the confusion matrix
plot_confusion_matrix(confusion_mtx, classes = range(10))

