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
import time

#Initialising the CNN
classifier = Sequential()

#Step 1- Convolution
classifier.add(Conv2D(32, (3, 3), input_shape =(64, 64, 3), activation = 'relu'))

#Part 2 - Pooling
classifier.add(MaxPooling2D(pool_size= (2, 2)))

#adding a secod convolutional layer.
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size= (2, 2)))

#Part 3- Flattening
classifier.add(Flatten())

#Part 4 - Fully connected
classifier.add(Dense(units =128, activation='relu'))
classifier.add(Dense(units = 1, activation='sigmoid'))

#Compiling the CNN
classifier.compile(optimizer = 'adam', loss= 'binary_crossentropy', metrics = ['accuracy'])

#Part 2 - Fitting the CNN to the images
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                target_size=(64, 64),
                                                batch_size=32,
                                                class_mode='binary')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size=(64, 64),
                                            batch_size=32,
                                            class_mode='binary')

start_time= time.time()
classifier.fit_generator(training_set,
                        steps_per_epoch=8000,
                        epochs=25,
                        validation_data=test_set,
                        validation_steps=2000)
print("------- %s seconds ------" % (time.time()- start_time))

#Part 3 - Making new predictions
import numpy as np
from keras.preprocessing import image
test_image = image.load_img('dataset/single_prediction/cat_ordog_1.jpg', target_size=(64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)
training_set.class_indices
if result [0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'


"""
Using one convolutional layer.
Output-
step - loss: 0.0163 - acc: 0.9950 - val_loss: 2.2177 - val_acc: 0.7357
------- 38144.28951406479 seconds ------ 

"""



















