import pickle
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow_vgg import vgg16
from tensorflow_vgg import utils
import matplotlib.pyplot as plt
from build_dataset import batch_iter


# build _network
inputs = tf.placeholder(tf.float32, [None, 4096])
labels = tf.placeholder(tf.int32, [None, 2])

fc_1 = tf.layers.dense(inputs, 1024, activation=tf.nn.relu)

logits = tf.layers.dense(fc_1, 2)

loss = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

predicted = tf.nn.softmax(logits)
correct_pred = tf.equal(tf.argmax(predicted,1),tf.argmax(labels,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32))


# get data
with open('label.pkl', 'rb') as f:
    y = pickle.load(f)
with open('data.pkl', 'rb') as f:
    x = pickle.load(f)

# split test and train
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2)


epochs = 20
iteration = 0
batch_size = 100
saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(epochs):
        for x_batch, y_batch in batch_iter(x_train, y_train,batch_size,True):
            feed={inputs:x_batch,labels:y_batch}
            sess.run(optimizer, feed_dict=feed)
            if iteration % 100 == 0:
                test_feed = {inputs:x_test, labels:y_test}
                acc = sess.run(accuracy,feed_dict=test_feed)
                print(acc)
            iteration += 1
    saver.save(sess, "checkpoints/catvsdog.ckpt")

# val
original_img = utils.load_image('test/202.jpg')
img = original_img.reshape(1,224,224,3)
vgg = vgg16.Vgg16()
with tf.Session() as sess:

    vgg_inputs = tf.placeholder(tf.float32, [None, 224, 224, 3])
    with tf.name_scope('content_vgg'):
        vgg.build(vgg_inputs)
    feed_dict = {vgg_inputs:img}
    code = sess.run(vgg.relu6, feed_dict=feed_dict)
saver = tf.train.Saver()
with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('checkpoints'))
    feed = {inputs: code}
    prediction = sess.run(predicted, feed_dict=feed).squeeze()
    # squeeze:[[1,0]]-->[1,0]
    print(prediction)
plt.imshow(original_img)
plt.show()
plt.bar(np.arange(2), prediction)
plt.xticks(np.arange(2), ['cat', 'dog'])
plt.show()

