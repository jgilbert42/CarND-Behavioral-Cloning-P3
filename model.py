import argparse
import csv
import cv2
import numpy as np
import os
import random
import time

from keras.callbacks import EarlyStopping, TensorBoard
from keras.models import Sequential
from keras.layers import Cropping2D, Dense, Flatten, Lambda, Dropout
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D 
from keras.optimizers import Adam

# switch a few variables based on running locally or on floydhub
# udacity data: hcXxcX5zKqFGYxHaGvat3N
# data 2: cy722Tfo9rAYrhGvPmMhwi
if os.getcwd() == '/code':
    print('on floydhub')
    input_dir = '/input'
    output_dir = '/output'
    fit_verbose = 2
    cache_images = True
else:
    print('on macbook')
    input_dir = './data'
    output_dir = '.'
    fit_verbose = 1
    cache_images = False

parser = argparse.ArgumentParser()
parser.add_argument('--batch-size', type=int, default=32)
parser.add_argument('--epochs', type=int, default=5)
parser.add_argument('--samples-per-epoch', type=int, default=0)
parser.add_argument('--input-dir', type=str, default=None)
args = parser.parse_args()

epochs = args.epochs
batch_size = args.batch_size
samples_per_epoch = args.samples_per_epoch
if args.input_dir:
    input_dir = args.input_dir

print('input dir:', input_dir)

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
#    model.add(Dropout(0.5))
    model.add(Dense(120))
    model.add(Dense(84))
    model.add(Dense(1))

def create_nvidia(model):
    model.add(Dropout(0.5))
    model.add(Conv2D(24, 5, 5, subsample=(2,2), activation='relu'))
    model.add(Conv2D(36, 5, 5, subsample=(2,2), activation='relu'))
    model.add(Conv2D(48, 5, 5, subsample=(2,2), activation='relu'))
    model.add(Conv2D(64, 3, 3, activation='relu'))
    model.add(Conv2D(64, 3, 3, activation='relu'))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))

class Sample:
    def __init__(self, filename, angle, flip=False):
        self.filename = filename 
        self.angle = float(angle)
        self.flip = bool(flip)

image_cache = {}

def load_image(filename):
    file_path = input_dir + '/IMG/' + filename
    return cv2.imread(file_path)

# get and image and optionally cache.  Caching images was implemented to work around a slow
# data network on floydhub.
def get_image(filename):
    if cache_images:
        if filename not in image_cache:
            image_cache[filename] = load_image(filename)

        return image_cache[filename]
    else:
        return load_image(filename)

samples = []

# add a sample to the list with both regular and flipped image and angle.  This
# is to avoid loading duplicate images all at once.
def add_sample(filename, angle):
    #print('add sample(filename: ', filename, ', angle:', angle)
    samples.append(Sample(filename, angle))
    samples.append(Sample(filename, angle, True))

print('loading images')
start_time = time.time()
with open(input_dir + '/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    cam_offset = 0.2
    for line in reader:
        if line[0] == 'center':
            #print('skipping header row', line)
            continue

        angle = float(line[3])

        # center
        add_sample(line[0].split('/')[-1], angle)
        # left
        add_sample(line[1].split('/')[-1], angle + cam_offset)
        # right
        add_sample(line[2].split('/')[-1], angle - cam_offset)


print('done loading images, {} samples, elapsed {:.3f}'.format(len(samples), time.time() - start_time))

# Incrementally load batches of images for use with fit_generator.  Flip based on sample flag.
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
                #print('loading:', batch_sample.filename, ', angle:', batch_sample.angle, ', flip:', batch_sample.flip)
                file_path = input_dir + '/IMG/' + batch_sample.filename
                # randomly flip the images for data augmentation
                #center_image = cv2.imread(file_path)
                center_image = get_image(batch_sample.filename)
                center_angle = float(batch_sample.angle)

                if batch_sample.flip:
                    center_image = np.fliplr(center_image)
                    center_angle = -center_angle

                #print(file_path, center_angle, center_image.shape, center_image[0][:10])
                images.append(center_image)
                angles.append(center_angle)

            X_train = np.array(images)
            y_train = np.array(angles)
            #yield shuffle(X_train, y_train)
            yield X_train, y_train


# split the training and validation data
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

if samples_per_epoch:
    val_samples = int(samples_per_epoch/4)
else:
    samples_per_epoch = len(train_samples)
    val_samples = len(validation_samples)

print('samples per epoch:', samples_per_epoch)
print('validation samples:', val_samples)

# create the generators for training and validation data
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)

input_shape=(160,320,3)
print('input_shape:', input_shape)
print('epochs:', epochs)
print('batch size:', batch_size)

# create the base model that includes image processing that's used by all
# models
model = Sequential()
model.add(Cropping2D(cropping=((64,30), (0,0)), input_shape=input_shape))
model.add(Lambda(lambda x: (x / 255.0) - 0.5))

# Create a model

#create_simple(model)
#create_lenet(model)
#create_jnet(model)
create_nvidia(model)

model.summary()

model.compile(loss='mse', optimizer='adam')

es = EarlyStopping(monitor='val_loss', patience=0)
#tb = TensorBoard(log_dir=output_dir + '/logs', histogram_freq=1, write_graph=False, write_images=True)
callbacks = [es]

# train the model
model.fit_generator(train_generator, samples_per_epoch=samples_per_epoch,
        validation_data=validation_generator,
        nb_val_samples=val_samples, nb_epoch=epochs,
        callbacks=callbacks, verbose=fit_verbose)

model.save(model_file)

