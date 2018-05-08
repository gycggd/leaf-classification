## Introduction

This is our leaf-classification project page for CSIT6000G, the dataset is from Kaggle competetion Leaf Classification.


## Project Structure

There are four directories and several files in the project: src/images/models/logs

>
	|-- images
	|-- logs
	|   |-- combined.log
	|   |-- image.log
	|   `-- numerical.log
	|-- models
	|   |-- combined_model
	|   |-- image_model
	|   `-- numerical_model
	|-- src
	|   |-- combined_model.ipynb
	|   |-- generate_data.py
	|   |-- log_analysis.ipynb
	|   |-- model_structure.ipynb
	|   |-- show_data.ipynb
	|   |-- show_model.ipynb
	|   |-- tf_train_mt_combined.py
	|   |-- tf_train_mt_image.py
	|   `-- tf_train_mt_numerical.py
	|-- tfrecords
	|   |-- train_data_1.tfrecords
	|   `-- val_data_1.tfrecords
	|-- test.csv
	`-- train.csv


Here we'll explain the files in the directories.

### images

In image we put all images in the dataset, for convenience, we put them in the Github project although it's not recommended.

The images files are named in `{ID}.jpg` format

### models

In models we put all our tensorflow models that we trained on [Meituan Deep Learning Service](https://www.mtyun.com/)

For each model, we train for 200 epoches, each epoch contains 28 batches with batch size 32 (except that the last batch with size 27).

We always save the model with the lowest validation loss.

The models directory contains two models:

* Combined model that uses both images and numerical features
* Image model that uses only images

### logs

Since we output many useful values during training, we download the logs from Meituan for analysis use.

The log for each epoch is as following:

>
	Epoch 174, training accuracy 1, training loss 0.0339188
	Best val loss 0.076045691967, save model to hdfs://default/user/mt_tenant_48517799/leaf/combined_model/best_combined_model.ckpt
	Validation accuracy 1, validation loss 0.0760457

### src

In src we have some Jupyter notebook files and some Python source code files. 

The Jupyter notebook files are for display use:

* [show_data.ipynb](https://github.com/gycggd/leaf-classification/blob/master/src/show_data.ipynb): this file gives a first glance at the data given, including the images and the feature data frame. It also shows the augmented data after rotating/flip/scaling.
* [model_structure.ipynb](https://github.com/gycggd/leaf-classification/blob/master/src/model_structure.ipynb): this file draw the structure of our model using matplotlib, here are generated model structures, you can check details in the file if interested.
* [log_analysis.ipynb](https://github.com/gycggd/leaf-classification/blob/master/src/log_analysis.ipynb): this file extracts train_accuracy/train_loss/validation_accuracy/validation_loss after the run of each epoch, and then draws charts of the data
* [show_model.ipynb](https://github.com/gycggd/leaf-classification/blob/master/src/show_model.ipynb): this file shows the weights of the convolutional layers, I expected to find the first layer as edge detectors, but failed.
* [combined_model](https://github.com/gycggd/leaf-classification/blob/master/src/combined_model.ipynb): this file gives the entire process that runs our combined model, you can check this file to find how it works.

The Python source code files:

* [generate_data.py](https://github.com/gycggd/leaf-classification/blob/master/src/generate_data.py): this file generates augmented images for the train set, and save them in a `.tfrecord` file, we generate data for 200 epochs, and then manually upload it to Meituan Deep Learning Platform (it only supports tensorflow and numpy, so we can't use keras to generate on its platform). 
* [tf_train_mt_combined.py](https://github.com/gycggd/leaf-classification/blob/master/src/tf_train_mt_combined.py): this file is to run the combined model on Meituan Deep Learning Platform
* [tf_train_mt_image.py](https://github.com/gycggd/leaf-classification/blob/master/src/tf_train_mt_image.py): this file is to run the iamge model on Meituan Deep Learning Platform
* [tf_train_mt_numerical.py](https://github.com/gycggd/leaf-classification/blob/master/src/tf_train_mt_numerical.py): this file is to run the numerical model on Meituan Deep Learning Platform

## Preprocessing

### Original images

Here are original images for 99 classes:

![Original images](https://github.com/gycggd/leaf-classification/blob/master/web_pics/original.png?raw=true)

We can see that they are different in image size, rotation degrees, scale. We are going to deal with this by resizing images and generate more train images.

### Resize images

Here we want to resize all images into 96x96 and make the leaf at the center of the image. Here are the steps:

* Resize the image with scale 96/max(width, length) so that the longer border of the image is 96
* Allocate a 96x96 array, then put the image at the middle of the array, fill remaining positions with 0 since the background color is black

Here are resized images:

![Resized images](https://github.com/gycggd/leaf-classification/blob/master/web_pics/resized.png?raw=true)

Core code here:

``` Python
def resize_img(img, max_dim=96):
    max_axis = np.argmax(img.size)
    scale = max_dim / img.size[max_axis]
    return img.resize((int(img.size[0] * scale), int(img.size[1] * scale)))

def resize(img, max_dim=96, center=True):
    for i, id in enumerate(ids):
        img = resize_img(img, max_dim=max_dim)
        img_array = img_to_array(img)
        X = zeros((max_dim, max_dim))
        h, w = img_array.shape[:2]
        if center:
            h1 = (max_dim - h) >> 1
            h2 = h1 + h
            w1 = (max_dim - w) >> 1
            w2 = w1 + w
        else:
            h1, h2, w1, w2 = 0, h, 0, w
        X[h1:h2, w1:w2][:] = img_array
    return np.around(X / 255)
```

### Data augmentation

We augmented the images by rotating the image a little (filling border with nearest color), scaling the images and flipping the images by x-axis or y-axis.

Core code here:

``` Python
from keras.preprocessing.image import ImageDataGenerator
img_generator = ImageDataGenerator(rotation_range=20, zoom_range=0.2, horizontal_flip=True,
                            vertical_flip=True, fill_mode='nearest')
imgen_train = img_generator.flow(train_images, y_train, batch_size=32)
```

Here are some of the augmented images:

![Augmented images](https://github.com/gycggd/leaf-classification/blob/master/web_pics/augmented.png?raw=true)


## Models

### Combined Model

![Combined model structure](https://github.com/gycggd/leaf-classification/blob/master/web_pics/combined_model.png?raw=true "Combined model structure")

We use two conv layers followed by max pooling layer, then concatenate the output with the numerical features, feed the concatenated array into the following two fully connected layers. In hidden layers we use ReLU as activation function and in output layer we use softmax function and use cross entropy loss function.

Structure:

* Input 96\*96 tensor
* 8@5x5 conv layer => 96x96x5
* 2x2 max pooling => 48x48x5
* 32@5x5 conv layer => 48x48x32
* 2x2 max pooling => 24x24x32
* Flatten => 18432
* Concatenate 192 numerical features => 18624
* Fully connected 18624x100 => 100
* Fully conencted 100x99 => 99

We implement this model in both Keras and Tensorflow:

Tensorflow code here (`bias_variable`/`weight_variable`/`max_pool_2x2` are self defined functions):

``` Python
image = tf.placeholder(tf.float32, (None, 96, 96, 1))
numerical = tf.placeholder(tf.float32, (None, 192))
label = tf.placeholder(tf.float32, (None, 99))
keep_prob = tf.placeholder(tf.float32)

W_conv1 = weight_variable([5, 5, 1, 8], name='W_conv1')
b_conv1 = bias_variable([8], name='b_conv1')
h_conv1 = tf.nn.relu(conv2d(image, W_conv1) + b_conv1)

h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5, 5, 8, 32], name='W_conv2')
b_conv2 = bias_variable([32], name='b_conv2')
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

h_pool2 = max_pool_2x2(h_conv2)

h_pool2_flat = tf.reshape(h_pool2, [-1, 24 * 24 * 32])

concated = tf.concat([h_pool2_flat, numerical], axis=1)

W_fc1 = weight_variable([24 * 24 * 32 + 192, 100], name='W_fc1')
b_fc1 = bias_variable([100], name='b_fc1')

h_fc1 = tf.nn.relu(tf.matmul(concated, W_fc1) + b_fc1)

h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([100, 99], name='W_fc2')
b_fc2 = bias_variable([99], name='b_fc2')

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(label, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
```

Keras code here:

``` Python
image = Input(shape=(96, 96, 1), name='image')
x = Convolution2D(8, (5, 5), input_shape=(96, 96, 1))(image)
x = (Activation('relu'))(x)
x = (MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))(x)

x = Convolution2D(32, (5, 5))(x)
x = Activation('relu')(x)
x = (MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))(x)
x = Flatten()(x)

numerical = Input(shape=(192,), name='numerical')
concatenated = concatenate([x, numerical])

x = Dense(100, activation='relu')(concatenated)
x = Dropout(0.5)(x)

out = Dense(99, activation='softmax')(x)

model = Model(inputs=[image, numerical], outputs=out)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
```


The result of this model as following, this model gives the best result, nearly 100% accuracy and very low validation loss:

![Result of combined model](https://github.com/gycggd/leaf-classification/blob/master/web_pics/stat_combined.png?raw=true "Stat of combined model")

Here we also show the weights of the first conv layer:

![First conv weights](https://github.com/gycggd/leaf-classification/blob/master/web_pics/visual_combined_conv1.png?raw=true)

First 5 of 32 weights of the second conv layer:

![Second conv weights](https://github.com/gycggd/leaf-classification/blob/master/web_pics/visual_combined_conv2_1.png?raw=true)
![Second conv weights](https://github.com/gycggd/leaf-classification/blob/master/web_pics/visual_combined_conv2_2.png?raw=true)
![Second conv weights](https://github.com/gycggd/leaf-classification/blob/master/web_pics/visual_combined_conv2_3.png?raw=true)
![Second conv weights](https://github.com/gycggd/leaf-classification/blob/master/web_pics/visual_combined_conv2_4.png?raw=true)
![Second conv weights](https://github.com/gycggd/leaf-classification/blob/master/web_pics/visual_combined_conv2_5.png?raw=true)

Unfortunately, we can't understand the weight images.

### Image Model

![Image model structure](https://github.com/gycggd/leaf-classification/blob/master/web_pics/image_model.png?raw=true "Image model structure")

The image model is the same as the combined layer except the concatenate step.

Structure:

* Input 96\*96 tensor
* 8@5x5 conv layer => 96x96x5
* 2x2 max pooling => 48x48x5
* 32@5x5 conv layer => 48x48x32
* 2x2 max pooling => 24x24x32
* Flatten => 18432
* Fully connected 18432 => 100
* Fully conencted 100x99 => 99

The result of this model as following, the result is not good, less than 70% accuracy and high validation loss, the training process is also not stable:

![Result of image model](https://github.com/gycggd/leaf-classification/blob/master/web_pics/stat_image.png?raw=true "Stat of image model")

### Numerical Model

The numerical model is just a neural network that have 1 hidden layer.

The result of this model as following, works well:

![Result of numerical model](https://github.com/gycggd/leaf-classification/blob/master/web_pics/stat_numerical.png?raw=true "Stat of numerical model")
