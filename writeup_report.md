**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image2]: ./data3/IMG/center_2017_03_29_23_08_42_969.jpg "Center Driving"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

My model is based on the nvidia DAVE-2 model and consists of a convolution neural network with 5x5 and 3x3 filter sizes and depths between 24 and 64 (model.py starting on line 63 in the function created_nvidia) 

The model includes RELU layers to introduce nonlinearity (code lines 65-69 using the activation parameter), and the data is normalized in the model using a Keras lambda layer (code line 181). 

####2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 64).  The dropout is at the beginning of the network after cropping and normalization and keeps 50% of pixels.

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 160). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 190).

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of attempting center lane driving and driving the opposite direction on the track.

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to follow the lectures and then use the included nvidia architecture.  I increased the cropping and added the Dropout later.

Once I had implemented the model and training with a generator, the mean squared error for both training and validation always started 0.02 or less and decreased for several epochs.  However, The loss amount didn't seem to have much correlation to driving performance.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set.

To combat overfitting, I modified the model with a Dropout layer.

I used left/right flipped images and the left/right cameras to augment the data.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py lines 65-69) consisted of a convolution neural network:

____________________________________________________________________________________________________
Layer (type)                     Output Shape          Param #     Connected to                     
====================================================================================================
cropping2d_1 (Cropping2D)        (None, 66, 320, 3)    0           cropping2d_input_1[0][0]         
____________________________________________________________________________________________________
lambda_1 (Lambda)                (None, 66, 320, 3)    0           cropping2d_1[0][0]               
____________________________________________________________________________________________________
dropout_1 (Dropout)              (None, 66, 320, 3)    0           lambda_1[0][0]                   
____________________________________________________________________________________________________
convolution2d_1 (Convolution2D)  (None, 31, 158, 24)   1824        dropout_1[0][0]                  
____________________________________________________________________________________________________
convolution2d_2 (Convolution2D)  (None, 14, 77, 36)    21636       convolution2d_1[0][0]            
____________________________________________________________________________________________________
convolution2d_3 (Convolution2D)  (None, 5, 37, 48)     43248       convolution2d_2[0][0]            
____________________________________________________________________________________________________
convolution2d_4 (Convolution2D)  (None, 3, 35, 64)     27712       convolution2d_3[0][0]            
____________________________________________________________________________________________________
convolution2d_5 (Convolution2D)  (None, 1, 33, 64)     36928       convolution2d_4[0][0]            
____________________________________________________________________________________________________
flatten_1 (Flatten)              (None, 2112)          0           convolution2d_5[0][0]            
____________________________________________________________________________________________________
dense_1 (Dense)                  (None, 100)           211300      flatten_1[0][0]                  
____________________________________________________________________________________________________
dense_2 (Dense)                  (None, 50)            5050        dense_1[0][0]                    
____________________________________________________________________________________________________
dense_3 (Dense)                  (None, 10)            510         dense_2[0][0]                    
____________________________________________________________________________________________________
dense_4 (Dense)                  (None, 1)             11          dense_3[0][0]                    
====================================================================================================
Total params: 348,219
Trainable params: 348,219
Non-trainable params: 0


####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded 3-ish laps on track one using center-ish lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded a couple of laps in the reverse direction to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would increase generalization of the model.

After the collection process, I had 7519 image data points. I then preprocessed this data by flipping horizontal using the numpy filplr method.  I also used the left and right camera images offset by 0.2 to increase the data set.  The training ultimately used 38572 samples for training and 9644 for validation.

The model cropped the images to remove extraneous data.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was between 3-6 as evidenced by the loss starting to incraase.  I used an adam optimizer so that manually training the learning rate wasn't necessary.  I used the Keras EarlyStopping callback with a patience of 0 to stop training when the loss started increasing.
