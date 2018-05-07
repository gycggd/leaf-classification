import os
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array, load_img
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder, StandardScaler


def load_numeric_training(standardize=True):
    data = pd.read_csv('../train.csv')
    ID = data.pop('id')
    y = data.pop('species')
    y = LabelEncoder().fit(y).transform(y)
    X = StandardScaler().fit(data).transform(data) if standardize else data.values
    return ID.values, X, y


def load_numeric_test(standardize=True):
    data = pd.read_csv('../test.csv')
    ID = data.pop('id')
    test = StandardScaler().fit(data).transform(data) if standardize else data.values
    return ID.values, test


def resize_img(img, max_dim=96):
    max_axis = np.argmax(img.size)
    scale = max_dim / img.size[max_axis]
    return img.resize((int(img.size[0] * scale), int(img.size[1] * scale)))


def load_img_data(ids, max_dim=96, center=True):
    X = np.empty((len(ids), max_dim, max_dim, 1))
    for i, id in enumerate(ids):
        img = load_img('../images/{}.jpg'.format(id), grayscale=True)
        img = resize_img(img, max_dim=max_dim)
        x = img_to_array(img)
        h, w = x.shape[:2]
        if center:
            h1 = (max_dim - h) >> 1
            h2 = h1 + h
            w1 = (max_dim - w) >> 1
            w2 = w1 + w
        else:
            h1, h2, w1, w2 = 0, h, 0, w
        X[i][h1:h2, w1:w2][:] = x
    return np.around(X / 255)


def load_train_data(split=0.9, random_state=7):
    ID, X_num_train, y = load_numeric_training()
    X_img_train = load_img_data(ID)
    sss = StratifiedShuffleSplit(n_splits=1, train_size=split, test_size=1 - split, random_state=random_state)
    train_idx, val_idx = next(sss.split(X_num_train, y))
    ID_tr, X_num_tr, X_img_tr, y_tr = ID[train_idx], X_num_train[train_idx], X_img_train[train_idx], y[train_idx]
    ID_val, X_num_val, X_img_val, y_val = ID[val_idx], X_num_train[val_idx], X_img_train[val_idx], y[val_idx]
    return (ID_tr, X_num_tr, X_img_tr, y_tr), (ID_val, X_num_val, X_img_val, y_val)


def load_test_data():
    ID, X_num_test = load_numeric_test()
    X_img_test = load_img_data(ID)
    return ID, X_num_test, X_img_test


print('Loading train data ...')
(ID_train, X_num_tr, X_img_tr, y_tr), (ID_val, X_num_val, X_img_val, y_val) = load_train_data()

# Prepare ID-to-label and ID-to-numerical dictionary
ID_y_dic, ID_num_dic = {}, {}
for i in range(len(ID_train)):
    ID_y_dic[ID_train[i]] = y_tr[i]
    ID_num_dic[ID_train[i]] = X_num_tr[i, :]

print('Loading test data ...')
ID_test, X_num_test, X_img_test = load_test_data()

# Convert label to categorical/one-hot
ID_train, y_tr, y_val = to_categorical(ID_train), to_categorical(y_tr), to_categorical((y_val))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _float32_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def write_val_data():
    val_data_path = '../tfrecords/val_data_1.tfrecords'
    if os.path.exists(val_data_path):
        print('Warning: old file exists, removed.')
        os.remove(val_data_path)

    val_image, val_num, val_label = X_img_val.astype(np.bool), X_num_val.astype(np.float64), y_val.astype(np.bool)
    print(val_image.shape, val_num.shape, val_label.shape)
    val_writer = tf.python_io.TFRecordWriter(val_data_path)
    print('Writing data into tfrecord ...')
    for i in range(len(val_image)):
        image, num, label = val_image[i], val_num[i], val_label[i]
        feature = {'image': _bytes_feature(image.tostring()),
                   'num': _bytes_feature(num.tostring()),
                   'label': _bytes_feature(label.tostring())}
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        val_writer.write(example.SerializeToString())

    print('Done!')


def write_train_data():
    imgen = ImageDataGenerator(rotation_range=20, zoom_range=0.2, horizontal_flip=True,
                               vertical_flip=True, fill_mode='nearest')
    imgen_train = imgen.flow(X_img_tr, ID_train, batch_size=32, seed=7)

    print('Generating augmented images')
    all_images = []
    all_ID = []
    p = True
    for i in range(28 * 200):
        print('Generating augmented images for epoch {}, batch {}'.format(i // 28, i % 28))
        X, ID = imgen_train.next()
        all_images.append(X)
        all_ID.append(np.argmax(ID, axis=1))

    all_images = np.concatenate(all_images).astype(np.bool)
    all_ID = np.concatenate(all_ID)
    all_y = np.zeros(all_ID.shape)
    all_nums = np.zeros((all_ID.shape[0], X_num_tr.shape[1]))
    for i in range(len(all_ID)):
        all_nums[i, :] = ID_num_dic[all_ID[i]]
        all_y[i] = ID_y_dic[all_ID[i]]
    all_y = to_categorical(all_y).astype(np.bool)

    print('Data shapes:')
    print('Image:', all_images.shape)
    print('Label:', all_y.shape)
    print('Numerical:', all_nums.shape)

    train_data_path = '../tfrecords/train_data_1.tfrecords'
    if os.path.exists(train_data_path):
        print('Warning: old file exists, removed.')
        os.remove(train_data_path)

    # compression = tf.python_io.TFRecordCompressionType.GZIP
    # train_writer = tf.python_io.TFRecordWriter(train_data_path, options=tf.python_io.TFRecordOptions(compression))
    train_writer = tf.python_io.TFRecordWriter(train_data_path)

    print('Writing data into tfrecord ...')
    for i in range(len(all_images)):
        if i % 891 == 0:
            print('Writing {} th epoch data ...'.format(i // 891))
        image, num, label = all_images[i], all_nums[i], all_y[i]
        feature = {'image': _bytes_feature(image.tostring()),
                   'num': _bytes_feature(num.tostring()),
                   'label': _bytes_feature(label.tostring())}
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        train_writer.write(example.SerializeToString())

    print('Done!')


write_val_data()
