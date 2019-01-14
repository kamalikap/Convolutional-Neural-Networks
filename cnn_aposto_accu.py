#Convolutional Neural Network

#Installing Theano

#Installing Tensorflow

#Installing Keras


#Part 1- Building the CNN

#importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.optimizers import Adam
from keras.callbacks import Callback
from keras.preprocessing.image import ImageDataGenerator
from keras import backend
import numpy as np
from keras.preprocessing import image
import os
import time

# Image dimensions
img_width, img_height = 150, 150 
 
"""
    Creates a CNN model
    p: Dropout rate
    input_shape: Shape of input 
"""

def create_model(p, input_shape=(32, 32, 3)):
    # Initialising the CNN
    model = Sequential()
    # Convolution + Pooling Layer 
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=input_shape, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # Convolution + Pooling Layer 
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # Convolution + Pooling Layer 
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # Convolution + Pooling Layer 
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    # Flattening
    model.add(Flatten())
    # Fully connection
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(p))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(p/2))
    model.add(Dense(1, activation='sigmoid'))
    
    # Compiling the CNN
    optimizer = Adam(lr=1e-3)
    metrics=['accuracy']
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=metrics)
    return model
"""
    Fitting the CNN to the images.
"""
def run_training(bs=32, epochs=10):
    
    train_datagen = ImageDataGenerator(rescale = 1./255, 
                                       shear_range = 0.2, 
                                       zoom_range = 0.2, 
                                       horizontal_flip = True)
    test_datagen = ImageDataGenerator(rescale = 1./255)
 
    training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (img_width, img_height),
                                                 batch_size = bs,
                                                 class_mode = 'binary')
                                                 
    test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (img_width, img_height),
                                            batch_size = bs,
                                            class_mode = 'binary')
                                            
    model = create_model(p=0.6, input_shape=(img_width, img_height, 3))                                  
    model.fit_generator(training_set,
                         steps_per_epoch=8000/bs,
                         epochs = epochs,
                         validation_data = test_set,
                         validation_steps = 2000/bs)
def main():
    start_time1= time.time()
    run_training(bs=32, epochs=100)
    print("------- %s seconds ------" % (time.time()- start_time1))
    
    #Part 3 - Making new predictions
    test_image = image.load_img('dataset/single_prediction/cat_or_dog_1.jpg', target_size=(64, 64))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    result = classifier.predict(test_image)
    training_set.class_indices
    if result [0][0] == 1:
        prediction = 'dog'
    else:
        prediction = 'cat'

 
""" Main """
if __name__ == "__main__":
    main()


"""

Output-
Epoch 100/100
250/250 [==============================] - 178s 713ms/step - loss: 0.1342 - acc: 0.9476 - val_loss: 0.2455 - val_acc: 0.9030
------- 50436.33285212517 seconds ------

"""




















