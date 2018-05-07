## Introduction

This is our leaf-classification project page for CSIT6000G, the dataset is from Kaggle competetion Leaf Classification.


## Structure

There are four directories in the project: src/images/models/logs

Here we'll explain the files in the directories.

### images

In image we put all images in the dataset, for convenience, we put them in the Github project although it's not recommended.

### models

In models we put all our tensorflow models that we trained on [Meituan Deep Learning Service](https://www.mtyun.com/)

### logs

Since we output many useful values during training, we download the logs from Meituan for analysis use.

### src

In src we have some Jupyter notebook files and some Python source code files. 

The Jupyter notebook files are for display use:

* [show_data.ipynb](https://github.com/gycggd/leaf-classification/blob/master/src/show_data.ipynb): this file gives a first glance at the data given, including the images and the feature data frame. It also shows the augmented data after rotating/flip/scaling.
* [model_structure.ipynb](https://github.com/gycggd/leaf-classification/blob/master/src/model_structure.ipynb): this file draw the structure of our model using matplotlib, here are generated model structures, you can check details in the file if interested.
![Combined model structure](https://github.com/gycggd/leaf-classification/blob/master/web_pics/combined_model.png?raw=true "Combined model structure")
![Image model structure](https://github.com/gycggd/leaf-classification/blob/master/web_pics/image_model.png?raw=true "Image model structure")
* [log_analysis.ipynb](https://github.com/gycggd/leaf-classification/blob/master/src/log_analysis.ipynb): this file extracts train_accuracy/train_loss/validation_accuracy/validation_loss after the run of each epoch, and then draws charts of the data
* [show_model.ipynb](https://github.com/gycggd/leaf-classification/blob/master/src/show_model.ipynb): this file shows the weights of the convolutional layers, I expected to find the first layer as edge detectors, but failed.
* [combined_model](https://github.com/gycggd/leaf-classification/blob/master/src/combined_model.ipynb): this file gives the entire process that runs our combined model, you can check this file to find how it works.