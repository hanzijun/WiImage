import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pickle
from DWTfliter import  dwtfilter
import normalization
import cv2
# import os
# os.environ['CUDA_VISIBLE_DEVICES']='0'
#
# config = tf.ConfigProto(allow_soft_placement=True)
# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
# config.gpu_options.allow_growth = True
# tf.device('/gpu:2')
class autoencoder():
    def __init__(
            self,
            train_data = None,
            batch_size = 16,
            learning_rate = 0.01,
            training_epochs = 100,
            time_scale = 60,
            param_file = True,
            is_train = False
                 ):

        self.train = train_data
        self.batch_size = batch_size
        self.lr = learning_rate
        self.is_train = is_train
        self.training_epochs = training_epochs
        self.time_scale = time_scale

        self.build()
        print "Neural networks build!"
        self.saver = tf.train.Saver()
        self.sess = tf.Session()
        # sess = tf.Session(config=config)

        # if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
        #     init = tf.initialize_all_variables()
        # else:
        #     init = tf.global_variables_initializer()
        init = tf.global_variables_initializer()
        self.sess.run(init)

        if is_train is True:
            if param_file is True:
                self.saver.restore(self.sess, "./params/train.ckpt")
                print("loading neural-network params...")
                self.learn()
            else:
                print "learning initialization!"
                self.learn()
        else:
            self.saver.restore(self.sess, "./params/train.ckpt")
            self.show()

    def build(self):

            self.input = tf.placeholder(tf.float32, shape = [None, 30, self.time_scale, 4], name='csi_input')
            self.tag = tf.placeholder(tf.float32, shape = [None, 480, 640, 1], name ='image_origin')
            self.output= tf.placeholder(tf.float32, shape = [None, 480, 640,1], name='image_output')

            # with tf.variable_scope('rnn'):
            #     nl1 = 512
            #     w_initializer = tf.random_normal_initializer(0., 0.1)
            #     b_initializer = tf.constant_initializer(0.)
            #
            #     with tf.variable_scope('l1'):
            #         X = tf.reshape(self.input, [-1, 120], name='preprocess')
            #         self.w1 = tf.get_variable('w1', [120, nl1], initializer=w_initializer,)
            #         self.b1 = tf.get_variable('b1', [nl1, ], initializer=b_initializer,)
            #         result = tf.matmul(X,self.w1) + self.b1
            #         result_in = tf.reshape(result, [-1, self.time_scale, nl1], name='preprocessed')
            #
            #     with tf.variable_scope('lstm_cell'):
            #         lstm_cell = tf.contrib.rnn.BasicLSTMCell(nl1, state_is_tuple=True)
            #         # init_state = lstm_cell.zero_state(self.batch_size, dtype=tf.float32)
            #         lstm_cell = tf.contrib.rnn.MultiRNNCell([lstm_cell for _ in range(1)], state_is_tuple=True)
            #         outputs, final_state = tf.nn.dynamic_rnn(lstm_cell, result_in, initial_state=None, dtype=tf.float32, time_major=False)
            #         result_out = tf.unstack(tf.transpose(outputs, [1, 0, 2]))
            #         lstm_cell_output = result_out[-1]
            #
            #         self.w2 = tf.get_variable('w1', [nl1, 480 * 640], initializer=w_initializer,)
            #         self.b2 = tf.get_variable('b1', [480 * 640, ], initializer=b_initializer,)
            #         self.output = tf.nn.sigmoid(tf.matmul(lstm_cell_output, self.w2) + self.b2)
            #         self.output = tf.reshape(self.output, [-1, 480, 640, 1])

            with tf.variable_scope('CNN'):
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

                self.W_e_conv1 = tf.get_variable('w1', [3, 3, 4, 8], initializer=w_initializer)
                b_e_conv1 = tf.get_variable('b1', [8, ], initializer=b_initializer)
                self.conv1 = tf.nn.relu(tf.add(self.conv2d(self.input, self.W_e_conv1), b_e_conv1))
                print self.conv1.shape

                self.W_e_conv2 = tf.get_variable('w2', [3, 3, 8, 32], initializer=w_initializer)
                b_e_conv2 = tf.get_variable('b2', [32, ], initializer=b_initializer)
                self.conv2 = tf.nn.relu(tf.add(self.conv2d(self.conv1, self.W_e_conv2), b_e_conv2))
                print self.conv2.shape

                self.W_e_conv3 = tf.get_variable('w3', [3, 3, 32, 64], initializer=w_initializer)
                b_e_conv3 = tf.get_variable('b3', [64, ], initializer=b_initializer)
                self.conv3 = tf.nn.relu(tf.add(self.conv2d(self.conv2, self.W_e_conv3), b_e_conv3))
                print self.conv3.shape
                self.conv3 = tf.reshape(self.conv3, [-1, 4 * 8 * 64])

                # self.output = tf.layers.dense(self.conv3, 480 * 640, activation=tf.sigmoid)
                self.w2 = tf.get_variable('w4', [4 * 8 * 64, 60 * 80 * 4], initializer=w_initializer, )
                self.b2 = tf.get_variable('b4', [60 * 80 * 4, ], initializer=b_initializer,)
                encoder = tf.nn.relu(tf.matmul(self.conv3, self.w2) + self.b2)
                encoder = tf.reshape(encoder, [-1, 60, 80, 4])

                decoder_1 = tf.layers.conv2d_transpose(encoder, 64, 3, strides=2, padding='same', )
                decoder_1 = tf.layers.batch_normalization(decoder_1, training=self.is_train)
                decoder_1 = tf.maximum(alpha * decoder_1, decoder_1)

                decoder_2 = tf.layers.conv2d_transpose(decoder_1, 32, 3, strides=2, padding='same', )
                decoder_2 = tf.layers.batch_normalization(decoder_2, training=self.is_train)
                decoder_2 = tf.maximum(alpha * decoder_2, decoder_2)

                decoder_3 = tf.layers.conv2d_transpose(decoder_2, 1, 3, strides=2, padding='same', )
                decoder_3 = tf.layers.batch_normalization(decoder_3, training=self.is_train)
                decoder_3 = tf.maximum(alpha * decoder_3, decoder_3)

                self.output = tf.reshape(decoder_3, [-1, 480, 640, 1])

            with tf.variable_scope('loss'):
                """
                KL Divergence + L2 regularization
                """
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

            xs = csidata[:,(batch_turn * self.batch_size + index) * 2 : self.time_scale + (batch_turn * self.batch_size + index )* 2,:]
            ys = image[batch_turn * self.batch_size + index]

            csidata_batch = np.array([xs]) if csidata_batch is None else np.append(csidata_batch, [xs], axis=0)
            image_batch = np.array([ys]) if image_batch is None else np.append(image_batch, [ys], axis= 0)

        return csidata_batch, image_batch

    def learn(self):

        for j in range(self.training_epochs):
            for train_data in self.train:
                xs = train_data[0].astype(np.float32)
                xs = np.nan_to_num(xs)
                total_batch = int(len(train_data[1]) / self.batch_size)

                for i in range(2000):
                    loss = 0
                    for batch_turn in range(total_batch):
                        batch_xs, batch_ys = self.batch_Convert(xs, train_data[1], batch_turn=batch_turn)
                        batch_xs = np.reshape(batch_xs, [-1, 30, self.time_scale, 4])

                        batch_ys = batch_ys.astype(np.float32)
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
                batch_xs = np.reshape(batch_xs, [-1, 30, self.time_scale, 4])

                output = self.sess.run(self.output, feed_dict={self.input: batch_xs})
                output = np.reshape(output, (-1, 480, 640 * 1))
                #print output[0]
                # output = output.astype(np.float32)

                cv2.imshow("Image", output[0])
                cv2.waitKey(0)

    def conv2d(self,  x, W):
        return tf.nn.conv2d(x, W, strides=[1,2,2,1], padding='SAME')

    def deconv2d(self, x,W, output_shape):
        return tf.nn.conv2d_transpose(x, W, output_shape, strides=[1,2,2,1], padding = 'SAME')

def kldlv(rho, rho_hat):
    invrho = tf.subtract(tf.constant(1.), rho)
    invrhohat = tf.subtract(tf.constant(1.), rho_hat)
    logrho = tf.add(logfunc(rho, rho_hat), logfunc(invrho, invrhohat))
    return logrho
def logfunc(x, x2):
    return tf.multiply(x, tf.log(tf.div(x, x2)))

def batchNormalization(data):
    for each_item in range(len(data)):
        data[each_item] = normalization.MINMAXNormalization(data[each_item])

def package(train_data):
    """
    The form applicable to RNN networks
    tn_data = np.hstack((np.transpose(csi_rx1[0]), np.transpose(csi_rx1[1])))
    tn_data = np.hstack((tn_data, np.transpose(csi_rx2[0])))
    tn_data = np.hstack((tn_data, np.transpose(csi_rx2[1])))
    """
    csi_rx1, csi_rx2, image = train_data[0], train_data[1], train_data[2]
    tn_data = np.append(csi_rx1, csi_rx2,axis=0)
    tn_data = np.transpose(tn_data, [1,2,0])

    return [tn_data, image]


if __name__ =="__main__":
    np.set_printoptions(threshold=np.inf)
    with open('/home/wifi/train_data/training_data_7.pkl', 'rb') as handle:
        da = pickle.load(handle)

    with open('/home/wifi/train_data/training_data_8.pkl', 'rb') as handle:
        db = pickle.load(handle)

    # with open('/home/wifi/train_data/training_data_3.pkl', 'rb') as handle:
    #     dc = pickle.load(handle)

    # with open('../data/4/training_data_4.pkl', 'rb') as handle:
    #     dd = pickle.load(handle)
    #
    # with open('../data/5/training_data_5.pkl', 'rb') as handle:
    #     de = pickle.load(handle)
    #
    # with open('../data/6/training_data_6.pkl', 'rb') as handle:
    #     df = pickle.load(handle)

    batchNormalization(da[0])
    batchNormalization(da[1])
    # batchNormalization(da[2])

    batchNormalization(db[0])
    batchNormalization(db[1])
    # batchNormalization(db[2])

    # batchNormalization(dc[0])
    # batchNormalization(dc[1])
    # batchNormalization(dc[2])

    data1 = package(da)
    data2 = package(db)
    # data3 = package(dc)
    print data1[0].shape

    train_data = [data1, data2]
    autoencoder(train_data=train_data)