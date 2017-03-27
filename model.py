import argparse
import csv
import cv2
import numpy as np
import os
import random

from keras import backend as K
from keras.callbacks import EarlyStopping, TensorBoard
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Cropping2D, Dense, Flatten, Lambda, Dropout
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D 

if os.getcwd() == '/code':
    print('on floydhub')
    input_dir = '/input';
    output_dir = '/output';
else:
    print('on macbook')
    input_dir = './data'
    output_dir = '.'

parser = argparse.ArgumentParser()
parser.add_argument('--batch-size', type=int, default=32)
parser.add_argument('--epochs', type=int, default=5)
args = parser.parse_args()

epochs = args.epochs
batch_size = args.batch_size

model_file = output_dir + '/model.h5'

def create_simple(model):
    model.add(Flatten())
    model.add(Dense(1))

def create_lenet(model):
    model.add(Conv2D(6, 5, 5, activation='relu'))
    model.add(MaxPooling2D())
    model.add(Conv2D(16, 5, 5, activation='relu'))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(120))
    model.add(Dense(84))
    model.add(Dense(1))

def create_newnet(model):
    #model.add(Conv2D(1, 3, 3, activation='relu'))
    model.add(Conv2D(16, 3, 3, activation='relu'))
    model.add(Conv2D(16, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2)))
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(32))
    model.add(Dense(1))

samples = []
with open(input_dir + '/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        if line[0] == 'center':
            print('skipping header row', line)
            continue

        samples.append(line)

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1:
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            #print('offset:', offset)
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                filename = batch_sample[0].split('/')[-1]
                #print('loading: ', filename)
                file_path = input_dir + '/IMG/' + filename
                # randomly flip the images for data augmentation
                if random.random() < 0.5:
                    center_image = np.fliplr(cv2.imread(file_path))
                    center_angle = -float(batch_sample[3])
                else:
                    center_image = cv2.imread(file_path)
                    center_angle = float(batch_sample[3])
                #print(file_path, center_angle, center_image.shape, center_image[0][:10])
                images.append(center_image)
                angles.append(center_angle)

            X_train = np.array(images)
            y_train = np.array(angles)
            #yield shuffle(X_train, y_train)
            yield X_train, y_train


train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)

input_shape=(160,320,3)
print('input_shape:', input_shape)
print('epochs:', epochs)
print('batch size:', batch_size)

model = Sequential()
model.add(Cropping2D(cropping=((72,20), (0,0)), input_shape=input_shape))
model.add(Lambda(lambda x: (x / 255.0) - 0.5))

#create_simple(model)
#create_lenet(model)
create_newnet(model)

model.summary()

optimizer = Adam(lr=0.0001)
model.compile(loss='mse', optimizer=optimizer)

es = EarlyStopping(monitor='val_loss', patience=0)
#tb = TensorBoard(log_dir=output_dir + '/logs', histogram_freq=1, write_graph=True, write_images=True)
callbacks = [es]

model.fit_generator(train_generator, samples_per_epoch=len(train_samples),
        validation_data=validation_generator,
        nb_val_samples=len(validation_samples), nb_epoch=epochs,
        callbacks=callbacks)

model.save(model_file)

