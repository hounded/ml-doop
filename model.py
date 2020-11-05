import tensorflow as tf
import numpy as np
import os
import math
from os.path import join, getsize

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

device = 'fd6c34c6-92d3-45b6-ba63-731a691bdbc8'
MODE = 'folder'  # or 'file', if you choose a plain text file (see above).
DATASET_PATH = 'imgs/'+device  # the dataset file or root folder path.

# print(len(next(os.walk(DATASET_PATH))[1]))
# Image Parameters
N_CLASSES = len(next(os.walk(DATASET_PATH))[1])  # CHANGE HERE, total number of classes
IMG_HEIGHT = 128  # CHANGE HERE, the image height to be resized to
IMG_WIDTH = 128  # CHANGE HERE, the image width to be resized to
CHANNELS = 3  # The 3 color channels, change to 1 if grayscale
num_epochs = 25


def read_images(dataset_path):
    images, labels, test_images, test_labels = list(), list(), list(), list()
    label = 0
    last_img_index = -1
    classes = sorted(next(os.walk(dataset_path))[1])

    for c in classes:

        c_dir = join(dataset_path, c)
        # print(c,label)
        total_files = len(os.listdir(c_dir))
        num_files_test = math.floor(total_files*0.25)
        walk = next(os.walk(c_dir))
        # Add each image to the training set
        for idx,sample in enumerate(walk[2]):
            if sample.endswith('.npy'):
                if idx>=num_files_test:
                    images.append(np.load(os.path.join(c_dir, sample)))
                    labels.append(label)
                else:
                    test_images.append(np.load(os.path.join(c_dir, sample)))  
                    test_labels.append(label) 
        label += 1

    # converting lists into np arrays
    Y = np.array(labels)
    X = np.array(images)
    Y_test = np.array(test_labels)
    X_test = np.array(test_images)

            
    return  X, Y, X_test, Y_test

x_train, y_train, x_test, y_test = read_images(DATASET_PATH)

# print(y_train.size)


print('Shape of x_train:', x_train.shape)
print('shape of x_test:', x_test.shape)
print('Shape of y_train:', y_train.shape)
print('shape of y_test:', y_test.shape)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D

# Creating a Sequential Model and adding the layers
model = Sequential()
input_shape = (IMG_HEIGHT, IMG_WIDTH, 3)
model.add(Conv2D(32, kernel_size=(5,5), input_shape=input_shape, activation=tf.nn.relu))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten()) # Flattening the 2D arrays for fully connected layers
model.add(Dense(1024))
model.add(Dropout(0.25))
model.add(Dense(N_CLASSES,activation=tf.nn.softmax))


# #Compiling and Fitting the Model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(x=x_train,y=y_train, epochs=num_epochs)

# print(model)

''' Model is ready for evaluation. Now we just need the test data to evaluate the model.'''

# import matplotlib.pyplot as plt
#
# plt.imsave('imag0.png', x_test[0, :, :, :])
# plt.imsave('imag1.png', x_test[1, :, :, :])

print('Evaluation Results: ', model.evaluate(x_test, y_test))

# pred = model.predict(x_test[0].reshape(1,IMG_HEIGHT, IMG_WIDTH, 3))
# print("Predicted Model's Results class 0: ", pred.argmax())
# pred = model.predict(x_test[1].reshape(1,IMG_HEIGHT, IMG_WIDTH, 3))
# print("Predicted Model's Results class 1: ", pred.argmax())