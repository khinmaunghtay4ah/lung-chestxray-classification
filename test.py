from keras.models import load_model 
from keras.models import model_from_json 

import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split

#(6)LOAD DATA FROM NUMPY(.NPZ FILE) ARRAYS
a_npz = np.load("..\\input\\a_images_arrays.npz")
a = a_npz['arr_0']

b_npz = np.load("..\\input\\b_labels.npz")
b = b_npz['arr_0'] 


#(7)SPLIT DATASET (70% TRAINING, 10% FOR VALIDATION AND 20% TESTING)
#dataset split ratio problem got solved by setting parameter from 'test_size' to 'train_size' since trainset will
#sice trainset will be split again into train and validation sets 
#First split dataset into train(80%) and test(20%) 
A_train, A_test, b_train, b_test = train_test_split(a, b, test_size = 0.2, random_state = 1, stratify = b)
#From train, split it again into train(70%) and validation(10%) 
A_train, A_val, b_train, b_val = train_test_split(A_train, b_train, train_size = 0.875, random_state = 1, stratify = b_train)
print(np.array(A_train).shape)
print(np.array(A_val).shape)
print(np.array(A_test).shape)


new_model = load_model("modelk.h5")


#score: evaluation of a loss function in test set
#force kears model to display probability by dividng image array by 255.0
score, accuracy = new_model.evaluate(A_test/255.0,
				  b_test)
print('Test Score:', score)           
print('Test Accuracy:', accuracy)

#print(new_model.predict_classes(A_test))
#print(new_model.predict_proba(A_test))



