import os
import pandas as pd 
import numpy as np 
import matplotlib.pylab as plt
import random 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import keras
from keras.preprocessing.image import ImageDataGenerator
from keras import layers, models, optimizers 
from keras import backend as K 
from keras.models import Sequential
from keras.callbacks import ReduceLROnPlateau, TensorBoard, EarlyStopping 
from keras.preprocessing import image


#(1)LOAD DATA FROM NUMPY(.NPZ FILE) ARRAYS
a_npz = np.load("..\\input\\a_images_arrays.npz")
a = a_npz['arr_0']

b_npz = np.load("..\\input\\b_labels.npz")
b = b_npz['arr_0'] 


#(2)SPLIT DATASET (TRAIN, VALIDATION AND TEST) 
A_train, A_test, b_train, b_test = train_test_split(a, b, test_size = 0.2, random_state = 1, stratify = b)

A_train, A_val, b_train, b_val = train_test_split(A_train, b_train, train_size = 0.875, random_state = 1, stratify = b_train)
print(np.array(A_train).shape)
print(np.array(A_val).shape)
print(np.array(A_test).shape)


#(3)SETTING UP CONVNET MODEL USING KERAS
###SETTING UP MODEL PARAMETERS FOR TRAINING PURPOSE 
K.image_data_format()

K.clear_session()

imgwidth, imgheight = 224, 224
imgdepth = 3
numb_train_samples = len(A_train)
numb_validation_samples = len(A_val)
epochs = 30
batch_size = 16	

model = models.Sequential()      #initializing layer 
model.add(layers.Conv2D(32, kernel_size =(3,3),
    input_shape = (imgwidth, imgheight, imgdepth)))
model.add(layers.BatchNormalization())
model.add(layers.Activation("relu"))
#model.add(layers.MaxPooling2D(pool_size =(3,3), strides =(2,2)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64, kernel_size =(3,3)))
model.add(layers.BatchNormalization())
model.add(layers.Activation("relu"))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(128, kernel_size = (3,3)))
model.add(layers.BatchNormalization())
model.add(layers.Activation("relu"))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(256, kernel_size = (3,3)))
model.add(layers.BatchNormalization())
model.add(layers.Activation("relu"))
model.add(layers.MaxPooling2D((2,2)))

#Fully connected Layer
model.add(layers.Flatten())
model.add(layers.Dropout(0.25))
model.add(layers.Dense(64))                      # 64 relu functions randomly initalized to try to match input to output...
model.add(layers.BatchNormalization())
model.add(layers.Activation("relu"))

#Final Output Layer
# Only a single output in the layer  
model.add(layers.Dense(1))                       # 1 neuron that is going to be the output 
model.add(layers.BatchNormalization())
model.add(layers.Activation("sigmoid"))

#(4)COMPILING THE MODEL
model.compile(loss = 'binary_crossentropy', 
    optimizer = optimizers.Adam(lr = 0.00146, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-08, decay = 0.0), 
    metrics = ['accuracy'])   

model.summary()


#(5)DATA AUGMENTATION(Fitting CNN model to images)
train_datagen = ImageDataGenerator(
        rescale=1./255,       #rescale all pixel values between 0 and 1  [1/225]
        #shear_range=0.2,
        #zoom_range=0.2,
        rotation_range=30,
        horizontal_flip=True)

val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow(np.array(A_train),
	b_train, batch_size = batch_size)

validation_set = val_datagen.flow(np.array(A_val),
	b_val, batch_size = batch_size)

test_set = test_datagen.flow(np.array(A_test), 
	b_test, batch_size = batch_size)


#(6)CALLBACKS TO BE USED DURING TRAINING THE MODEL 

#Model Check Point 
filepath = "weightsmodelk.best.hdf5"
checkpoint_callback = keras.callbacks.ModelCheckpoint(filepath, 
	monitor = 'val_acc',
	verbose = 1, save_best_only = True,
	mode = 'max')


#Visualize data using Tensorboard 
#Notes: "/logsm" makes a directory in the root directory; whereas "./logsm" makes one in the working directory
tb_callback = keras.callbacks.TensorBoard(log_dir ='./logsm', histogram_freq = 1, 
	write_graph = True, write_images = True)

callbacks_list = [checkpoint_callback, tb_callback]


#(7)TRAIN THE MODEL USING KERAS FIT_GENERATOR FUNCTION 
# #Fits the model on data generated batch-by-batch by a Python generator.
# #The generator is run in parallel to the model, for efficiency. 
# #For instance, this allows you to do real-time data augmentation on images on CPU in parallel to training your model on GPU.
history = model.fit_generator(
        training_set,
        steps_per_epoch=numb_train_samples // batch_size,
        epochs=epochs,
        validation_data=validation_set,
        validation_steps=numb_validation_samples // batch_size,
        verbose = 1, callbacks = callbacks_list)

#(8)SAVING THE MODEL 
model.save('modelk.h5')               



#(9)VISUALIZE THE RESULT ON TRAINING AND VALIDATION DATA USING MATPLOTLIB
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'blue', label='Training acc')
plt.plot(epochs, val_acc, 'red', label='Validation acc')
plt.xlabel("Number of Epochs")
plt.ylabel("Accuracy")
plt.title('Model Accuracy on Training and validation Sets')
plt.legend()
plt.figure()

plt.plot(epochs, loss, 'blue', label='Training loss')
plt.plot(epochs, val_loss, 'red', label='Validation loss')
plt.xlabel("Number of Epochs")
plt.ylabel("Loss")
plt.title('Model Loss on Training and validation Sets')
plt.legend()
plt.show()             





