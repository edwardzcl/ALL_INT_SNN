# -*- coding: utf-8 -*-
"""
Created on Fri May  4 18:25:59 2018

@author: ***
Training script for spiking LeNet with quantization level k=2 on MNIST.
"""

#! /usr/bin/python
# -*- coding: utf-8 -*-

import time
import tensorflow as tf
import tensorlayer as tl 
import os
import ast
print(os.getcwd())

tf.reset_default_graph()

import argparse
parser = argparse.ArgumentParser()
# resume from previous checkpoint
parser.add_argument('--resume', type=ast.literal_eval, default=False)
# training or inference
parser.add_argument('--mode', type=str, default='training')
args = parser.parse_args()

print(args.resume, args.mode)

X_train, y_train, X_val, y_val, X_test, y_test = tl.files.load_mnist_dataset(shape=(-1, 28, 28, 1))
# X_train, y_train, X_test, y_test = tl.files.load_cropped_svhn(include_extra=False)

sess = tf.InteractiveSession()

batch_size = 200

x = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
y_ = tf.placeholder(tf.int64, shape=[None])

model_file_name = "./model_mnist_advanced.ckpt"

def model(x, is_train=True, reuse=False):
    # In BNN, all the layers inputs are binary, with the exception of the first layer.
    # ref: https://github.com/itayhubara/BinaryNet.tf/blob/master/models/BNN_cifar10.py
    with tf.variable_scope("binarynet", reuse=reuse):
        net = tl.layers.InputLayer(x, name='input')
        #net = tl.layers.Conv2d(net, n_filter=15, filter_size=(5, 5), strides=(1, 1), padding='SAME', b_init=None, name='bcnn0')
        #net = tl.layers.BatchNormLayer(net, act=tf.nn.relu, is_train=is_train, name='bn0')
        #net = tl.layers.Quant_Layer(net, k)
        #net = tl.layers.SignLayer(net)

        net = tl.layers.Conv2d(net, n_filter=32, filter_size=(5, 5), strides=(1, 1), padding='VALID', b_init=None, name='bcnn1')
        #net = tl.layers.MaxPool2d(net, (2, 2), (2, 2), padding='SAME', name='pool2')
        net = tl.layers.BatchNormLayer(net, act=tf.nn.relu, is_train=is_train, name='bn1')
        #net = tl.layers.Quant_Layer(net, k)
        #net = tl.layers.SignLayer(net)
        #net.outputs = (net.outputs+1)/2

        net = tl.layers.Conv2d(net, n_filter=32, filter_size=(2, 2), strides=(2, 2), padding='SAME', b_init=None, name='bcnn2')
        #net = tl.layers.MaxPool2d(net, (2, 2), (2, 2), padding='SAME', name='pool1')
        net = tl.layers.BatchNormLayer(net, act=tf.nn.relu, is_train=is_train, name='bn2')
        #net = tl.layers.Quant_Layer(net, k)
        #net = tl.layers.SignLayer(net)
        #net.outputs = (net.outputs+1)/2

        net1 = tl.layers.Conv2d(net, n_filter=64, filter_size=(5, 5), strides=(1, 1), padding='VALID', b_init=None, name='bcnn3')
        #net = tl.layers.MaxPool2d(net, (2, 2), (2, 2), padding='SAME', name='pool2')
        net = tl.layers.BatchNormLayer(net1, act=tf.nn.relu, is_train=is_train, name='bn3')
        #net = tl.layers.Quant_Layer(net, k)
        #net = tl.layers.SignLayer(net)
        
        #net = tl.layers.SignLayer(net)
        #net.outputs = (net.outputs+1)/2

        net = tl.layers.Conv2d(net, n_filter=64, filter_size=(2, 2), strides=(2, 2), padding='SAME', b_init=None, name='bcnn4')
        #net = tl.layers.MaxPool2d(net, (2, 2), (2, 2), padding='SAME', name='pool2')
        net = tl.layers.BatchNormLayer(net, act=tf.nn.relu, is_train=is_train, name='bn4')
        #net = tl.layers.Quant_Layer(net, k)
        #net = tl.layers.SignLayer(net)
        
        #net = tl.layers.SignLayer(net)
        #net.outputs = (net.outputs+1)/2

        net = tl.layers.Conv2d(net, n_filter=1000, filter_size=(4, 4), strides=(1, 1), padding='VALID', b_init=None, name='fc1')
        #net = tl.layers.MaxPool2d(net, (2, 2), (2, 2), padding='SAME', name='pool2')
        net = tl.layers.BatchNormLayer(net, act=tf.nn.relu, is_train=is_train, name='bn5')
        #net = tl.layers.Quant_Layer(net, k)

        net = tl.layers.FlattenLayer(net)
        # net = tl.layers.DropoutLayer(net, 0.8, True, is_train, name='drop1')
        #net = tl.layers.DenseLayer(net, n_units=256, b_init=None, name='dense')
        #net = tl.layers.BatchNormLayer(net, act=tf.nn.relu, is_train=is_train, name='bn5')
        #net = tl.layers.SignLayer(net)


        # net = tl.layers.DropoutLayer(net, 0.8, True, is_train, name='drop2')
        #net = tl.layers.SignLayer(net)
        #net.outputs = (net.outputs+1)/2

        net = tl.layers.DenseLayer(net, 10, b_init=None, name='bout')
        #net = tl.layers.BatchNormLayer(net, is_train=is_train, name='bno')
    return net, net1


# define inferences
net_train, _ = model(x, is_train=True, reuse=False)
net_test,net1 = model(x, is_train=False, reuse=True)

# cost for training
y = net_train.outputs
cost = tl.cost.cross_entropy(y, y_, name='xentropy')

# cost and accuracy for evalution
y2 = net_test.outputs
cost_test = tl.cost.cross_entropy(y2, y_, name='xentropy2')
correct_prediction = tf.equal(tf.argmax(y2, 1), y_)
acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# define the optimizer
train_params = tl.layers.get_variables_with_name('binarynet', True, True)
train_op = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(cost, var_list=train_params)

# initialize all variables in the session
tl.layers.initialize_global_variables(sess)

if args.resume:
    print("Load existing model " + "!" * 10)
    saver = tf.train.Saver()
    saver.restore(sess, model_file_name)

net_train.print_params()
net_train.print_layers()

n_epoch = 200
print_freq = 1

# print(sess.run(net_test.all_params)) # print real values of parameters

if args.mode == 'training':
   for epoch in range(n_epoch):
       start_time = time.time()
       for X_train_a, y_train_a in tl.iterate.minibatches(X_train, y_train, batch_size, shuffle=True):
           sess.run(train_op, feed_dict={x: X_train_a, y_: y_train_a})

       if epoch + 1 == 1 or (epoch + 1) % print_freq == 0:
           print("Epoch %d of %d took %fs" % (epoch + 1, n_epoch, time.time() - start_time))
           train_loss, train_acc, n_batch = 0, 0, 0
           for X_train_a, y_train_a in tl.iterate.minibatches(X_train, y_train, batch_size, shuffle=True):
               err, ac = sess.run([cost_test, acc], feed_dict={x: X_train_a, y_: y_train_a})
               train_loss += err
               train_acc += ac
               n_batch += 1
           print("   train loss: %f" % (train_loss / n_batch))
           print("   train acc: %f" % (train_acc / n_batch))
           #print(sess.run(threshold))
           val_loss, val_acc, n_batch = 0, 0, 0
           for X_val_a, y_val_a in tl.iterate.minibatches(X_val, y_val, batch_size, shuffle=True):
               err, ac = sess.run([cost_test, acc], feed_dict={x: X_val_a, y_: y_val_a})
               val_loss += err
               val_acc += ac
               n_batch += 1
           print("   val loss: %f" % (val_loss / n_batch))
           print("   val acc: %f" % (val_acc / n_batch))
           #print(sess.run(threshold))

           if (epoch + 1) % (print_freq) == 0:
               print("Save model " + "!" * 10)
               saver = tf.train.Saver()
               save_path = saver.save(sess, model_file_name)
               # you can also save model into npz
               tl.files.save_npz(net_train.all_params, name='model_mnist.npz', sess=sess)
               # and restore it as follow:
               # tl.files.load_and_assign_npz(sess=sess, name='model.npz', network=network)


print('Evaluation')
test_loss, test_acc, n_batch = 0, 0, 0
#for X_test_a, y_test_a in tl.iterate.minibatches(X_test, y_test, batch_size, shuffle=True):
err, ac = sess.run([cost_test, acc], feed_dict={x: X_test[:,:,:,:], y_: y_test[:]})
test_loss += err
test_acc += ac
#    n_batch += 1
x = sess.run(net1.outputs[0,:,0,0],feed_dict={x: X_test[:,:,:,:], y_: y_test[:]})
#    break
print("   test loss: %f" % (test_loss / 1))
print("   test acc: %f" % (test_acc / 1))
print(x)
