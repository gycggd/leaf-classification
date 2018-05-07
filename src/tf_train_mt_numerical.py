import tensorflow as tf
import numpy as np
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='',
                    help='input data path')
parser.add_argument('--model_dir', type=str, default='',
                    help='output model path')
FLAGS, _ = parser.parse_known_args()
train_data_path = os.path.join(FLAGS.data_dir, "train_data_1.tfrecords")
val_data_path = os.path.join(FLAGS.data_dir, "val_data_1.tfrecords")


def max_pool_2x2(x):
    """max_pool_2x2 downsamples a feature map by 2X."""
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


def weight_variable(shape):
    """weight_variable generates a weight variable of a given shape."""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    """bias_variable generates a bias variable of a given shape."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


numerical = tf.placeholder(tf.float32, (None, 192))
label = tf.placeholder(tf.float32, (None, 99))
keep_prob = tf.placeholder(tf.float32)

W_fc1 = weight_variable([192, 100])
b_fc1 = bias_variable([100])

h_fc1 = tf.nn.relu(tf.matmul(numerical, W_fc1) + b_fc1)

h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([100, 99])
b_fc2 = bias_variable([99])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(label, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    def decode(serialized_example):
        feature_dic = {'image': tf.FixedLenFeature([], tf.string),
                       'num': tf.FixedLenFeature([], tf.string),
                       'label': tf.FixedLenFeature([], tf.string)}
        features = tf.parse_single_example(serialized_example, features=feature_dic)
        return features['image'], features['num'], features['label']


    class DataReader:
        def __init__(self, data_path):
            self.dataset = tf.data.TFRecordDataset([data_path]).map(decode)
            self.iterator = self.dataset.make_one_shot_iterator()
            self.iter_image, self.iter_num, self.iter_label = self.iterator.get_next()
            self.cnt = 0

        def get_all(self):
            all_image, all_num, all_label = [], [], []
            try:
                while True:
                    bytes_image, bytes_num, bytes_label = sess.run((self.iter_image, self.iter_num, self.iter_label))
                    image, num, label = np.fromstring(bytes_image, dtype=np.bool), \
                                        np.fromstring(bytes_num, dtype=np.float64), \
                                        np.fromstring(bytes_label, dtype=np.bool)
                    # print(image.shape, num.shape, label.shape)
                    image = np.reshape(image, (1, 96, 96, 1))
                    all_image.append(image)
                    all_num.append(num)
                    all_label.append(label)
                    self.cnt += 1
            except tf.errors.OutOfRangeError:
                pass
            all_image, all_num, all_label = np.concatenate(all_image), np.array(
                all_num), np.array(all_label)
            # print(all_image.shape, all_num.shape, all_label.shape)
            return all_image, all_num, all_label

        def get_batch(self):
            batch_image, batch_num, batch_label = [], [], []
            try:
                while True:
                    bytes_image, bytes_num, bytes_label = sess.run((self.iter_image, self.iter_num, self.iter_label))
                    image, num, label = np.fromstring(bytes_image, dtype=np.bool), \
                                        np.fromstring(bytes_num, dtype=np.float64), \
                                        np.fromstring(bytes_label, dtype=np.bool)
                    # print(image.shape, num.shape, label.shape)
                    image = np.reshape(image, (1, 96, 96, 1))
                    batch_image.append(image)
                    batch_num.append(num)
                    batch_label.append(label)
                    self.cnt += 1
                    if self.cnt % 32 == 0 or self.cnt % 891 == 0:
                        break
            except tf.errors.OutOfRangeError:
                pass
            batch_image, batch_num, batch_label = np.concatenate(batch_image), np.array(
                batch_num), np.array(batch_label)
            # print(batch_image.shape, batch_num.shape, batch_label.shape)
            return batch_image, batch_num, batch_label


    sess.run(tf.global_variables_initializer())
    train_data_reader = DataReader(data_path=train_data_path)
    val_data_reader = DataReader(data_path=val_data_path)
    val_image, val_num, val_label = val_data_reader.get_all()
    for i in range(28 * 200):
        batch_image, batch_num, batch_label = train_data_reader.get_batch()
        if i % 28 == 0:
            train_accuracy = accuracy.eval(feed_dict={
                numerical: batch_num, label: batch_label, keep_prob: 1.0})
            train_loss = cross_entropy.eval(feed_dict={
                numerical: batch_num, label: batch_label, keep_prob: 1.0})
            print('Epoch %d, training accuracy %g, training loss %g' % (i // 32, train_accuracy, train_loss))
            val_accuracy = accuracy.eval(feed_dict={
                numerical: val_num, label: val_label, keep_prob: 1.0})
            val_loss = cross_entropy.eval(feed_dict={
                numerical: val_num, label: val_label, keep_prob: 1.0})
            print('Validation accuracy %g, validation loss %g' % (val_accuracy, val_loss))
        train_step.run(feed_dict={numerical: batch_num, label: batch_label, keep_prob: 0.5})
