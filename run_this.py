import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pickle
from DWTfliter import  dwtfilter
import time
import normalization
import cv2
from skimage import io, color, data
np.set_printoptions(threshold=np.inf)

# Import MNIST data
# from tensorflow.examples.tutorials.mnist import input_data
# mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
# Parameters

# pylab.figure()
# pylab.plot(train_data[0][0][0], 'y--', label='butterworth')
# pylab.legend(loc='best')
# pylab.show()

# print a
# is_training = True
eachpic = '../data1/1/new/0.jpg'
im = io.imread(eachpic, as_gray=False)
im = color.rgb2gray(im)
rows, cols = im.shape
for i in range(rows):
    for j in range(cols):
        im[i, j] = 0 if im[i, j] <= 0.5 else 1
im = im.astype(np.float32)
im = np.reshape(im, (480, 640, 1))

class autoencoder():
    def __init__(
            self,
            train_data = None,
            batch_size = 32,
            learning_rate = 0.005,
            training_epochs = 10,
            time_scale = 1800,
            param_file = True,
            is_train = False
                 ):
# Network Parameters
        self.train = train_data
        self.batch_size = batch_size
        self.lr = learning_rate
        self.training_epochs = training_epochs
        self.time_scale = time_scale

        self.build()
        self.saver = tf.train.Saver()
        self.sess = tf.Session()

        if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
            init = tf.initialize_all_variables()
        else:
            init = tf.global_variables_initializer()
        self.sess.run(init)

        if is_train is True:
            if param_file is True:
                self.saver.restore(self.sess, "./params/train.ckpt")
                print("loading neural-network params...")
                self.learn()
            else:
                self.learn()
        else:
            self.saver.restore(self.sess, "./params/train.ckpt")
            self.show()

    def build(self):

            self.input = tf.placeholder(tf.float32, shape = [None, self.time_scale, 180], name='csi_input')
            self.tag = tf.placeholder(tf.float32, shape = [None, 480, 640, 1], name ='image_origin')
            self.output= tf.placeholder(tf.float32, shape = [None, 480, 640,1], name='image_output')

            with tf.variable_scope('rnn'):
                nl1 = 512
                w_initializer = tf.random_normal_initializer(0., 0.1)
                b_initializer = tf.constant_initializer(0.)

                with tf.variable_scope('l1'):
                    X = tf.reshape(self.input, [-1, 180], name='preprocess')
                    self.w1 = tf.get_variable('w1', [180, nl1], initializer=w_initializer,)
                    self.b1 = tf.get_variable('b1', [nl1, ], initializer=b_initializer,)
                    result = tf.matmul(X,self.w1) + self.b1
                    result_in = tf.reshape(result, [-1, self.time_scale, nl1], name='preprocessed')

                with tf.variable_scope('lstm_cell'):
                    lstm_cell = tf.contrib.rnn.BasicLSTMCell(nl1, state_is_tuple=True)
                    # init_state = lstm_cell.zero_state(self.batch_size, dtype=tf.float32)
                    lstm_cell = tf.contrib.rnn.MultiRNNCell([lstm_cell for _ in range(1)], state_is_tuple=True)
                    outputs, final_state = tf.nn.dynamic_rnn(lstm_cell, result_in, initial_state=None, dtype=tf.float32, time_major=False)
                    result_out = tf.unstack(tf.transpose(outputs, [1, 0, 2]))
                    lstm_cell_output = result_out[-1]

                    self.w2 = tf.get_variable('w1', [nl1, 480 * 640], initializer=w_initializer,)
                    self.b2 = tf.get_variable('b1', [480 * 640, ], initializer=b_initializer,)
                    self.output = tf.nn.sigmoid(tf.matmul(lstm_cell_output, self.w2) + self.b2)
                    self.output = tf.reshape(self.output, [-1, 480, 640, 1])

            with tf.variable_scope('encoder'):
                alpha = 0.01
                w_initializer = tf.random_normal_initializer(0., 0.1)
                b_initializer = tf.constant_initializer(0.1)

                # with tf.variable_scope('l1de'):
                #     layer1 = tf.layers.batch_normalization(lstm_cell_output, training=is_training)
                #     layer1 = tf.maximum(alpha * layer1, layer1)
                #     layer1 = tf.nn.dropout(layer1, keep_prob=0.8)
                #     #  de-convolution operation
                #     layer2 = tf.layers.conv2d_transpose(layer1, 1, 3, strides=2, padding='same', )
                #     layer2 = tf.layers.batch_normalization(layer2, training=is_training)
                #     layer2 = tf.maximum(alpha * layer2, layer2)
                #     layer2 = tf.nn.dropout(layer2, keep_prob=0.8)

                #     layer3 = tf.layers.conv2d_transpose(layer2, 1, 3, strides=2, padding='same', )
                #     layer3 = tf.layers.batch_normalization(layer3, training=is_training)
                #     layer3 = tf.maximum(alpha * layer3, layer3)
                #     layer3 = tf.nn.dropout(layer3, keep_prob=0.8)
                #     #  de-convolution operation
                #     layer4 = tf.layers.conv2d_transpose(layer3, 1, 3, strides=2, padding='same', )
                #     layer4 = tf.layers.batch_normalization(layer4, training=is_training)
                #     layer4 = tf.maximum(alpha * layer4, layer4)
                #     layer4 = tf.nn.dropout(layer4, keep_prob=0.8)
                #     self.output = tf.sigmoid(layer4)

                # self.W_e_conv1 = tf.get_variable('we1', [3, 3, 3, 32], initializer=w_initializer)
                # b_e_conv1 = tf.get_variable('be1', [32, ], initializer=b_initializer)
                # self.conv1 = tf.nn.relu(tf.add(self.conv2d(self.input, self.W_e_conv1), b_e_conv1))
                # print self.conv1.shape

                # self.W_e_conv2 = tf.get_variable('we2', [3, 3, 32, 128], initializer=w_initializer)
                # b_e_conv2 = tf.get_variable('be2', [128, ], initializer=b_initializer)
                # self.conv2 = tf.nn.relu(tf.add(self.conv2d(self.conv1, self.W_e_conv2), b_e_conv2))
                # print self.conv2.shape

                # self.W_e_conv3 = tf.get_variable('we3', [3, 3, 128, 512], initializer=w_initializer)
                # b_e_conv3 = tf.get_variable('be3', [512, ], initializer=b_initializer)
                # self.conv3 = tf.nn.relu(tf.add(self.conv2d(self.conv2, self.W_e_conv3), b_e_conv3))
                # print self.conv3.shape

                # self.output = tf.sigmoid(self.conv3)
                # self.output = tf.reshape(self.output, (-1, 480, 640, 1))
                # print self.output.shape


            with tf.variable_scope('loss'):
                    alpha, beta, rho = 5e-6, 7.5e-6, 0.08
                    # Wset = [self.W_e_conv1, self.W_e_conv2, self.W_e_conv3, self.W_e_conv4]
                    # results = [self.conv1, self.encoder]
                    # kldiv_loss = reduce(lambda x, y: x + y, map(lambda x: tf.reduce_sum(kldlv(rho, tf.reduce_mean(x, 0))), results))
                    # l2_loss = reduce(lambda x, y: x + y, map(lambda x: tf.nn.l2_loss(x), Wset))
                    # self.loss = tf.reduce_mean(tf.pow(self.input - self.input_reconstruct, 2))+ alpha * l2_loss + beta * kldiv_loss
                    # self.loss = tf.reduce_mean(tf.pow(self.output - self.tag, 2)) + alpha * l2_loss
                    self.loss = tf.reduce_mean(tf.pow(self.tag - self.output, 2))
                    # self.loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=self.output, labels=self.tag))

            with tf.variable_scope('train'):
                    self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
                    # self.optimizer = tf.train.GradientDescentOptimizer(self.lr).minimize(self.loss)

    def batch_Convert(self, csidata, image, batch_turn):
        csidata_batch, image_batch = None, None
        for index in range(self.batch_size):
            # print  ((batch_turn * self.batch_size + index )* 2)
            xs = csidata[1800 - self.time_scale + (batch_turn * self.batch_size + index) * 30 : 1800 + (batch_turn * self.batch_size + index )* 30, :]
            ys = image[batch_turn * self.batch_size + index ]

            csidata_batch = np.array([xs]) if csidata_batch is None else np.append(csidata_batch, [xs], axis=0)
            image_batch = np.array([ys]) if image_batch is None else np.append(image_batch, [ys], axis= 0)

        return csidata_batch, image_batch

    def learn(self):

        for j in range(2):
            for train_data in self.train:
                xs = train_data[0].astype(np.float32)
                xs = np.nan_to_num(xs)
                total_batch = int(len(train_data[1]) / self.batch_size)

                for i in range(800):
                    loss = 0
                    for batch_turn in range(total_batch):
                        batch_xs, batch_ys = self.batch_Convert(xs, train_data[1], batch_turn=batch_turn)
                        batch_xs = np.reshape(batch_xs, [-1, self.time_scale, 180])
                        batch_ys = batch_ys.astype(np.float32)
                        # for i in range(self.batch_size):
                        #     batch_ys[i] = batch_ys[i] - im
                        batch_ys = np.reshape(batch_ys, [-1, 480, 640, 1])
                        # batch_ys = batch_ys * 2 - 1
                        # batch_ys = np.array([i-im for i in batch_ys])

                        _, c = self.sess.run([self.optimizer, self.loss], feed_dict={self.input: batch_xs, self.tag: batch_ys})

                        loss += c
                        if np.any(np.isnan(batch_xs)):
                            print "Input Nan Type Error!! "
                        if np.any(np.isnan(batch_ys)):
                            print "Tag Nan Type Error!! "
                    if i % 5 == 0:
                        print("Total Epoch:", '%d' % (j), "Pic Rpoch:",'%d' % (i), "total cost=", "{:.9f}".format(loss))

        print("Optimization Finished!")
        self.saver.save(self.sess, "./params/train.ckpt")

        # xs = self.train[0].astype(np.float32)
        # xs = np.nan_to_num(xs)
        # total_batch = int(len(self.train[1]) / self.batch_size)
        # for i in range(500):
        #     loss = 0
        #     for batch_turn in range(total_batch):
        #         batch_xs, batch_ys = self.batch_Convert(xs, self.train[1], batch_turn)
        #         batch_xs = np.reshape(batch_xs, [-1, self.time_scale, 90])
        #
        #         batch_ys = batch_ys.astype(np.float32)
        # #         # batch_ys = batch_ys - im
        #         batch_ys = np.reshape(batch_ys, [-1, 480, 640, 1])
        #         # batch_ys = batch_ys * 2 - 1
        #         _, c = self.sess.run([self.optimizer, self.loss], feed_dict={self.input: batch_xs, self.tag: batch_ys})
        #
        #         loss +=c
        #         if np.any(np.isnan(batch_xs[0])):
        #             print "Input Nan Type Error!! "
        #         if np.any(np.isnan(batch_ys)):
        #             print "Tag Nan Type Error!! "
        #     print("Epoch:", '%d' % (i), "total cost=", "{:.9f}".format(loss))
        #
        # print("Optimization Finished!")
        # self.saver.save(self.sess, "./params/train.ckpt")

    def show(self):
        """
        display the performance of autoencoder
        :return: a autoencoder model using unsupervised learning
        """
        for train_data in self.train:
            xs = train_data[0].astype(np.float32)
            xs = np.nan_to_num(xs)
            total_batch = int(len(train_data[1]) / self.batch_size)

            for batch_turn in range(total_batch):
                batch_xs, batch_ys = self.batch_Convert(xs, train_data[1], batch_turn=batch_turn)
                batch_xs = np.reshape(batch_xs, [-1, self.time_scale, 180])

                output = self.sess.run(self.output, feed_dict={self.input: batch_xs})
                output = np.reshape(output, (-1, 480, 640 * 1))
                print output[0]
                # output = output.astype(np.float32)

                cv2.imshow("Image", output[0])
                cv2.waitKey(0)


    def conv2d(self,  x, W):
        return tf.nn.conv2d(x, W, strides=[1,2,2,1], padding='SAME')

    def deconv2d(self, x,W, output_shape):
        return tf.nn.conv2d_transpose(x, W, output_shape, strides=[1,2,2,1], padding = 'SAME')

# Building the encoder
"""
KL Divergence + L2 regularization
"""
def kldlv(rho, rho_hat):
    invrho = tf.subtract(tf.constant(1.), rho)
    invrhohat = tf.subtract(tf.constant(1.), rho_hat)
    logrho = tf.add(logfunc(rho, rho_hat), logfunc(invrho, invrhohat))
    return logrho
def logfunc(x, x2):
    return tf.multiply(x, tf.log(tf.div(x, x2)))

# results = [encoder_1, encoder_2, encoder_3]
# Wset = [weights['encoder_h0'], weights['encoder_h1'], weights['encoder_h2'],  tf.transpose(weights['encoder_h2']),  tf.transpose(weights['encoder_h1']), tf.transpose(weights['encoder_h0'])]
# kldiv_loss = reduce(lambda x,y : x+y, map(lambda x : tf.reduce_sum(kldlv(rho, tf.reduce_mean(x, 0))), results))
# l2_loss = reduce(lambda x, y : x + y, map(lambda x : tf.nn.l2_loss(x), Wset))
# Define loss and optimizer, minimize the squared error
# cost = tf.reduce_mean(tf.pow(y_true - y_pred, 2)) + alpha * l2_loss + beta * kldiv_loss
# Launch the graph

def package(train_data):
    # tn_data = np.hstack((np.transpose(csi[0]), np.transpose(csi[1])))
    # tn_data = np.hstack((tn_data, np.transpose(csi[2])))
    csi_rx1, csi_rx2, image = train_data[0], train_data[1], train_data[2]
    tn_data = np.hstack((np.transpose(csi_rx1[0]), np.transpose(csi_rx1[1])))
    tn_data = np.hstack((tn_data, np.transpose(csi_rx1[2])))
    tn_data = np.hstack((tn_data, np.transpose(csi_rx2[0])))
    tn_data = np.hstack((tn_data, np.transpose(csi_rx2[1])))
    tn_data = np.hstack((tn_data, np.transpose(csi_rx2[2])))

    return [tn_data, image]


if __name__ =="__main__":
    np.set_printoptions(threshold=np.inf)
    with open('../data1/1/training_data.pkl', 'rb') as handle:
        da = pickle.load(handle)

    with open('../data1/2/training_data.pkl', 'rb') as handle:
        db = pickle.load(handle)

    # for subcarrier in range(len(da[0][0])):
    #     da[0][0][subcarrier] = dwtfilter(da[0][0][subcarrier]).butterWorth()
    da[0][0] = normalization.MINMAXNormalization(da[0][0])
    da[1][0] = normalization.MINMAXNormalization(da[1][0])


    # for subcarrier in range(len(dc[0][0])):
    #     dc[0][0][subcarrier] = dwtfilter(dc[0][0][subcarrier]).butterWorth()
    db[0][0] = normalization.MINMAXNormalization(db[0][0])
    db[1][0] = normalization.MINMAXNormalization(db[1][0])
    # dd[0][0] = normalization.MINMAXNormalization(dd[0][0])
    # de[0][0] = normalization.MINMAXNormalization(de[0][0])

    print da[0].shape
    print len(da)
    data1 = package(da)
    data2 = package(db)

    # data3 = package(csi3, dc)
    # data4 = package(csi4, dd)
    # data5 = package(csi5, de)

    train_data = [data1, data2]
    # train_data = train_data[:-1]
    # train_data = data1
    # print train_data[0][1].shape
    # with open('./training_data/train.pkl', 'wb') as handle:
    #     pickle.dump(train_data, handle, -1)
    # time.sleep(10)

    # with open('./training_data/train.pkl', 'rb') as handle:
    #     training_data = pickle.load(handle)
    # train_data = train_data[:-1]
    # autoencoder(train_data=train_data)
    autoencoder(train_data=train_data)