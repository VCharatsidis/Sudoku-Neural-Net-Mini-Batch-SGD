# -*- coding: utf-8 -*-
"""
Created on Tue Aug  7 16:07:23 2018

@author: vcharatsidis
"""
import tensorflow as tf
import numpy as np
from Sudoku import SolvedSudoku


hardestSudoku = [[8, 1, 2, 7, 5, 3, 6, 4, 9],
                 [9, 4, 3, 6, 8, 2, 1, 7, 5],
                 [6, 7, 5, 4, 9, 1, 2, 8, 3],
                 [1, 5, 4, 2, 3, 7, 8, 9, 6],
                 [3, 6, 9, 8, 4, 5, 7, 2, 1],
                 [2, 8, 7, 1, 6, 9, 5, 3, 4],
                 [5, 2, 1, 9, 7, 4, 3, 6, 8],
                 [4, 3, 8, 5, 2, 6, 9, 1, 7],
                 [7, 9, 6, 3, 1, 8, 4, 5, 2]]


reducer = SolvedSudoku(2000)
test_reducer = SolvedSudoku(1)

nodes1 = 256
nodes2 = 128
nodes3 = 128
nodes4 = 128
nodes5 = 128
nodes6 = 128
nodes7 = 128
nodes8 = 128

batch_size = 128
numbers_to_predict = 10

x = tf.placeholder("float32", [None, 810], name="reducedBoards")
y = tf.placeholder("float32", [None, numbers_to_predict*10], name="solutions")


def nnmodel(data):
    hl1 = {'weights': tf.Variable(tf.random_normal([810, nodes1])),
           'biases': tf.Variable(tf.random_normal([nodes1]))}

    hl2 = {'weights': tf.Variable(tf.random_normal([nodes1, nodes2])),
           'biases': tf.Variable(tf.random_normal([nodes2]))}

    hl3 = {'weights': tf.Variable(tf.random_normal([nodes2, nodes3])),
           'biases': tf.Variable(tf.random_normal([nodes3]))}

    hl4 = {'weights': tf.Variable(tf.random_normal([nodes3, nodes4])),
           'biases': tf.Variable(tf.random_normal([nodes4]))}

    hl5 = {'weights': tf.Variable(tf.random_normal([nodes4, nodes5])),
           'biases': tf.Variable(tf.random_normal([nodes5]))}

    hl6 = {'weights': tf.Variable(tf.random_normal([nodes5, nodes6])),
           'biases': tf.Variable(tf.random_normal([nodes6]))}

    hl7 = {'weights': tf.Variable(tf.random_normal([nodes6, nodes7])),
           'biases': tf.Variable(tf.random_normal([nodes7]))}

    hl8 = {'weights': tf.Variable(tf.random_normal([nodes7, nodes8])),
           'biases': tf.Variable(tf.random_normal([nodes8]))}

    output_layer = {'weights': tf.Variable(tf.random_normal([nodes8, numbers_to_predict*10])),
                    'biases': tf.Variable(tf.random_normal([numbers_to_predict*10]))}

    lay1 = tf.matmul(data, hl1['weights']) + hl1['biases']
    lay1 = tf.nn.sigmoid(lay1)

    lay2 = tf.matmul(lay1, hl2['weights']) + hl2['biases']
    lay2 = tf.nn.sigmoid(lay2)

    lay3 = tf.matmul(lay2, hl3['weights']) + hl3['biases']
    lay3 = tf.nn.sigmoid(lay3)

    lay4 = tf.matmul(lay3, hl4['weights']) + hl4['biases']
    lay4 = tf.nn.sigmoid(lay4)

    lay5 = tf.matmul(lay4, hl5['weights']) + hl5['biases']
    lay5 = tf.nn.sigmoid(lay5)

    lay6 = tf.matmul(lay5, hl6['weights']) + hl6['biases']
    lay6 = tf.nn.sigmoid(lay6)

    lay7 = tf.matmul(lay6, hl7['weights']) + hl7['biases']
    lay7 = tf.nn.sigmoid(lay7)

    lay8 = tf.matmul(lay7, hl8['weights']) + hl8['biases']
    lay8 = tf.nn.sigmoid(lay8)

    output = tf.matmul(lay8, output_layer['weights']) + output_layer['biases']

    print("hi")
    print(output.shape)

    return output

def one_hot(array):
    targets = np.array(np.asarray(array).reshape(-1))
    one_hot = np.eye(10)[targets]
    return one_hot


def prepare_data(x, reshape_num):
    ohx = one_hot(x)
    x_reshaped = np.reshape(ohx, reshape_num)
    x_final = np.asarray(x_reshaped)

    return x_final

def test_batch():
    batch_x = []
    batch_y = []

    #num_per_board * len(reducers) = batch_size = 128
    for i in range(batch_size):
        xs, ys = test_reducer.board_reduction(numbers_to_predict)

        x_prepared = prepare_data(xs, 810)
        batch_x.append(x_prepared)

        y_prepared = prepare_data(ys, numbers_to_predict*10)
        batch_y.append(y_prepared)

    return np.asarray(batch_x), np.asarray(batch_y)

def next_batch():
    batch_x = []
    batch_y = []

    #num_per_board * len(reducers) = batch_size = 128
    for i in range(batch_size):
        xs, ys = reducer.board_reduction(numbers_to_predict)

        x_prepared = prepare_data(xs, 810)
        batch_x.append(x_prepared)

        y_prepared = prepare_data(ys, numbers_to_predict*10)
        batch_y.append(y_prepared)

        #print(xs)
        #print(ys)

    return np.asarray(batch_x), np.asarray(batch_y)



def train_nn(x):
    prediction = nnmodel(x)

    cost = tf.losses.mean_squared_error(labels=y, predictions=prediction)   # 0.4-0.6
    # cost = tf.losses.absolute_difference(labels = y, predictions = prediction) 0.4-0.8
    # cost = tf.losses.log_loss(labels = y, predictions = prediction) nan

    optimizer = tf.train.AdamOptimizer().minimize(cost)

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())

        for i in range(500001):
            b_x, b_y = next_batch()
            _, c = sess.run([optimizer, cost], feed_dict={x: b_x, y: b_y})

            if i % 1000 == 0:
                print("iteration : " + str(i) + ", cost : " + str(c))

        for i in range(11):
            b_x, b_y = test_batch()

            accuracy = tf.losses.mean_squared_error(labels=y, predictions=prediction)
            test_accuracy = sess.run(accuracy, feed_dict={x: b_x, y: b_y})

            print("iteration  : " + str(i) + ", test_accuracy : " + str(test_accuracy))

train_nn(x)