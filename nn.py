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

otherSudoku = [[5, 3, 4, 6, 7, 8, 9, 1, 2],
               [6, 7, 2, 1, 9, 5, 3, 4, 8],
               [1, 9, 8, 3, 4, 2, 5, 6, 7],
               [8, 5, 9, 7, 6, 1, 4, 2, 3],
               [4, 2, 6, 8, 5, 3, 7, 9, 1],
               [7, 1, 3, 9, 2, 4, 8, 5, 6],
               [9, 6, 1, 5, 3, 7, 2, 8, 4],
               [2, 8, 7, 4, 1, 9, 6, 3, 5],
               [3, 4, 5, 2, 8, 6, 1, 7, 9]]

hardestSudoku_fixed = [[True, False, False, False, False, False, False, False, False],
                       [False, False, True, True, False, False, False, False, False],
                       [False, True, False, False, True, False, True, False, False],
                       [False, True, False, False, False, True, False, False, False],
                       [False, False, False, False, True, True, True, False, False],
                       [False, False, False, True, False, False, False, True, False],
                       [False, False, True, False, False, False, False, True, True],
                       [False, False, True, True, False, False, False, True, False],
                       [False, True, False, False, False, False, True, False, False]]


def one_hot(array):
    targets = np.array(np.asarray(array).reshape(-1))
    one_hot = np.eye(10)[targets]
    return one_hot


reducer = SolvedSudoku(otherSudoku, hardestSudoku_fixed)
test_reducer = SolvedSudoku(otherSudoku, hardestSudoku_fixed)

nodes1 = 64
nodes2 = 32
# nodes3 = 32
# nodes4 = 32
# nodes5 = 32

x = tf.placeholder("float32", [None, 810], name="reducedBoards")
y = tf.placeholder("float32", [None, 810], name="solutions")


def nnmodel(data):
    hl1 = {'weights': tf.Variable(tf.random_normal([810, nodes1])),
           'biases': tf.Variable(tf.random_normal([nodes1]))}

    hl2 = {'weights': tf.Variable(tf.random_normal([nodes1, nodes2])),
           'biases': tf.Variable(tf.random_normal([nodes2]))}

    # hl3 = {'weights': tf.Variable(tf.random_normal([nodes2, nodes3])),
    #        'biases': tf.Variable(tf.random_normal([nodes3]))}
    #
    # hl4 = {'weights': tf.Variable(tf.random_normal([nodes3, nodes4])),
    #        'biases': tf.Variable(tf.random_normal([nodes4]))}
    #
    # hl5 = {'weights': tf.Variable(tf.random_normal([nodes4, nodes5])),
    #        'biases': tf.Variable(tf.random_normal([nodes5]))}

    output_layer = {'weights': tf.Variable(tf.random_normal([nodes2, 810])),
                    'biases': tf.Variable(tf.random_normal([810]))}

    lay1 = tf.matmul(data, hl1['weights']) + hl1['biases']
    lay1 = tf.nn.sigmoid(lay1)

    lay2 = tf.matmul(lay1, hl2['weights']) + hl2['biases']
    lay2 = tf.nn.sigmoid(lay2)

    # lay3 = tf.matmul(lay2, hl3['weights']) + hl3['biases']
    # lay3 = tf.nn.sigmoid(lay3)
    #
    # lay4 = tf.matmul(lay3, hl4['weights']) + hl4['biases']
    # lay4 = tf.nn.sigmoid(lay4)
    #
    # lay5 = tf.matmul(lay4, hl5['weights']) + hl5['biases']
    # lay5 = tf.nn.sigmoid(lay5)

    output = tf.matmul(lay2, output_layer['weights']) + output_layer['biases']

    print("hi")
    print(output.shape)

    return output


def prepare_data(x):
    ohx = one_hot(x)
    x_reshaped = np.reshape(ohx, 810)
    x_final = np.asarray(x_reshaped)

    return x_final

def next_batch(batch_size, reducer):
    batch_x = []
    batch_y = []
    for j in range(batch_size):
        xs = reducer.board_to_row(reducer.board_reduction(35))
        x_prepared = prepare_data(xs)
        batch_x.append(x_prepared)

        ys = reducer.board_to_row(reducer.solution)
        y_prepared = prepare_data(ys)
        batch_y.append(y_prepared)

    return np.asarray(batch_x), np.asarray(batch_y)


def train_nn(x):
    batch_size = 128
    prediction = nnmodel(x)

    #v = tf.reshape(prediction, [81, 10])
    #kapa = tf.Print(v, [v], "prediction : ", summarize=1000000)
    #kapa = tf.reshape(kapa, [1,810])

    cost = tf.losses.mean_squared_error(labels=y, predictions=prediction)   # 0.4-0.6
    # cost = tf.losses.absolute_difference(labels = y, predictions = prediction) 0.4-0.8
    # cost = tf.losses.log_loss(labels = y, predictions = prediction) nan

    optimizer = tf.train.AdamOptimizer().minimize(cost)

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())

        for i in range(300001):
            b_x, b_y = next_batch(batch_size, reducer)
            _, c = sess.run([optimizer, cost], feed_dict={x: b_x, y: b_y})

            if i % 1000 == 0:
                #print(xs)
                #print("")
                #print(ys)
                print("cost " + str(c))
                print("iteration " + str(i))


        for i in range(11):
            b_x, b_y = next_batch(batch_size, test_reducer)

            accuracy = tf.losses.mean_squared_error(labels=y, predictions=prediction)
            test_accuracy = sess.run(accuracy, feed_dict={x: b_x, y: b_y})

            print("test")
            print(xs)

            print(ys)
            print("test_accuracy " + str(test_accuracy))
            print("iteration " + str(i))

train_nn(x)