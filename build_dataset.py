import os
import tensorflow as tf
from tensorflow_vgg import vgg16
from tensorflow_vgg import utils
import numpy as np
import pickle


def batch_iter(data, labels, batch_size, shuffle=True):
    data_size = len(data)
    num_batches = ((data_size-1)//batch_size) + 1
    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_x = data[shuffle_indices]
        shuffled_y = labels[shuffle_indices]
    else:
        shuffled_x = data
        shuffled_y = labels
    for batch_num in range(num_batches):
        start_index = batch_num*batch_size
        end_index = min((batch_num+1)*batch_size, data_size)
        yield shuffled_x[start_index:end_index], shuffled_y[start_index:end_index]


def path_to_img(path):
    img = utils.load_image(path)
    return img.reshape((1, 224, 224, 3))


def main():
    data_dir = 'train/'
    contents = os.listdir(data_dir)
    img_paths = [os.path.join(data_dir, content) for content in contents]
    labels = [content.split('.')[0] for content in contents]

    targets = []
    relu6_result = None
    with tf.Session() as sess:
        vgg = vgg16.Vgg16()
        with tf.name_scope('inputs'):
            vgg_inputs = tf.placeholder(tf.float32, [None, 224, 224, 3])
            vgg.build(vgg_inputs)
        for batch_x, batch_y in batch_iter(img_paths, labels, 50, False):
            images = np.concatenate([path_to_img(path)
                                     for path in batch_x])
            vgg_feed_dict = {vgg_inputs: images}
            relu6_batch = sess.run(vgg.relu6, feed_dict=vgg_feed_dict)
            if relu6_result is None:
                relu6_result = relu6_batch
            else:
                relu6_result = np.concatenate((relu6_result, relu6_batch))

            labels_batch = [[1, 0] if label == 'cat' else [0, 1]
                            for label in batch_y]
            targets.extend(labels_batch)

        targets = np.array(targets)
        with open('data.pkl', 'wb') as f:
            pickle.dump(relu6_result, f)
        with open('label.pkl', 'wb') as f:
            pickle.dump(targets, f)


if __name__ == '__main__':
    main()
