import tensorflow as tf
import numpy as np
from Sudoku import SolvedSudoku

# hardestSudoku = [[8, 1, 2, 7, 5, 3, 6, 4, 9],
#                  [9, 4, 3, 6, 8, 2, 1, 7, 5],
#                  [6, 7, 5, 4, 9, 1, 2, 8, 3],
#                  [1, 5, 4, 2, 3, 7, 8, 9, 6],
#                  [3, 6, 9, 8, 4, 5, 7, 2, 1],
#                  [2, 8, 7, 1, 6, 9, 5, 3, 4],
#                  [5, 2, 1, 9, 7, 4, 3, 6, 8],
#                  [4, 3, 8, 5, 2, 6, 9, 1, 7],
#                  [7, 9, 6, 3, 1, 8, 4, 5, 2]]
#
# hardestSudoku_fixed = [[True, False, False, False, False, False, False, False, False],
#                        [False, False, True, True, False, False, False, False, False],
#                        [False, True, False, False, True, False, True, False, False],
#                        [False, True, False, False, False, True, False, False, False],
#                        [False, False, False, False, True, True, True, False, False],
#                        [False, False, False, True, False, False, False, True, False],
#                        [False, False, True, False, False, False, False, True, True],
#                        [False, False, True, True, False, False, False, True, False],
#                        [False, True, False, False, False, False, True, False, False]]


reducer = SolvedSudoku(2000)
test_reducer = SolvedSudoku(1)
numbers_to_predict = 10
batch_size = 128

sess = tf.InteractiveSession()

x = tf.placeholder("float32", [None, 810], name="reducedBoards")
y = tf.placeholder("float32", [None, numbers_to_predict*10], name="solutions")

x_board = tf.reshape(x, [-1, 9, 90, 1], name="x_board")

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


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


#Define layers in the NN

#Box filter
W_box = weight_variable([3, 30, 1, 64])
b_box = bias_variable([64])

# #Row filter
# W_row = weight_variable([1, 90, 1, 1])
# b_row = bias_variable([1])
#
# #Column filter
# W_column = weight_variable(([90, 1, 1, 1]))
# b_column = bias_variable([1])

conv1_box = tf.nn.relu(tf.nn.conv2d(x_board, W_box, strides=[1, 3, 30, 1], padding='SAME') + b_box)
# conv1_row = tf.nn.relu(tf.nn.conv2d(x_board, W_row, strides=[1, 1, 9, 1], padding='SAME') + b_row)
# conv1_column = tf.nn.relu(tf.nn.conv2d(x_board, W_column, strides=[1, 9, 1, 1], padding='SAME') + b_column)
#
# conv1_box = tf.reshape(conv1_box, [-1, 3 * 3 * 1])
# conv1_row = tf.reshape(conv1_row, [-1, 3 * 3 * 1])
# conv1_column = tf.reshape(conv1_column, [-1, 3 * 3 * 1])

conv1 = conv1_box
w_1x1 = weight_variable([1, 1, 64, 32])
b_1x1 = bias_variable([32])
conv1x1 = tf.nn.relu(tf.nn.conv2d(conv1, w_1x1, strides=[1, 1, 1, 1], padding='SAME') + b_1x1)
#Layer 1
# W_conv1 = weight_variable([3, 10, 1, 4])
# b_conv1 = bias_variable([4])
#
# conv1 = tf.nn.relu(tf.nn.conv2d(x_board, W_conv1, strides=[1, 1, 10, 1], padding='SAME') + b_conv1)
# #pool1 = max_pool_2x2(conv1)
#
# #Layer 2
# W_conv2 = weight_variable([1, 3, 4, 4])
# b_conv2 = bias_variable([4])
#
# conv2 = tf.nn.relu(tf.nn.conv2d(conv1, W_conv2, strides=[1, 1, 1, 1], padding='SAME') + b_conv2)
# #pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
#
# #Fully Connected Layer
# W_dense = weight_variable([7 * 7 * 4, 1024])
W_dense = weight_variable([3 * 3 * 32, 1024])
b_dense = bias_variable([1024])
#
# #Connect pool2 with fully_connected
# conv2 = tf.reshape(conv2, [-1, 7 * 7 * 4])
# #pool2 = tf.reshape(pool2, [-1, 7 * 7 * 64])
# dense = tf.nn.sigmoid(tf.matmul(conv2, W_dense) + b_dense)
conv1 = tf.reshape(conv1x1, [-1, 3 * 3 * 32])

dense = tf.nn.sigmoid(tf.matmul(conv1, W_dense) + b_dense)

#Readout layer
W_output = weight_variable(([1024, numbers_to_predict*10]))
b_output = bias_variable([numbers_to_predict*10])

y_conv = tf.matmul(dense, W_output) + b_output

#Loss
cost = tf.losses.mean_squared_error(labels=y, predictions=y_conv)
#Optimizer
optimizer = tf.train.AdamOptimizer().minimize(cost)

sess.run(tf.global_variables_initializer())

import time

num_steps = 500001
display_every = 1000

#Start timer
start_time = time.time()
end_time = time.time()

for i in range(num_steps):
    b_x, b_y = next_batch()

    optimizer.run(feed_dict={x: b_x, y: b_y})
    cost1 = cost.eval(feed_dict={x: b_x, y: b_y})

    if i % display_every == 0:
        print("iteration : " + str(i)+" cost : " + str(cost1))

for i in range(10):
    b_x, b_y = test_batch()

    optimizer.run(feed_dict={x: b_x, y: b_y})
    cost1 = cost.eval(feed_dict={x:b_x, y:b_y})
    print("cost test : " +str(cost))