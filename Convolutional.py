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

other1 = [[5, 3, 4, 6, 7, 8, 9, 1, 2],
          [6, 7, 2, 1, 9, 5, 3, 4, 8],
          [1, 9, 8, 3, 4, 2, 5, 6, 7],
          [8, 5, 9, 7, 6, 1, 4, 2, 3],
          [4, 2, 6, 8, 5, 3, 7, 9, 1],
          [7, 1, 3, 9, 2, 4, 8, 5, 6],
          [9, 6, 1, 5, 3, 7, 2, 8, 4],
          [2, 8, 7, 4, 1, 9, 6, 3, 5],
          [3, 4, 5, 2, 8, 6, 1, 7, 9]]

other2 = [[7, 3, 5, 6, 1, 4, 8, 9, 2],
          [8, 4, 2, 9, 7, 3, 5, 6, 1],
          [9, 6, 1, 2, 8, 5, 3, 7, 4],
          [2, 8, 6, 3, 4, 9, 1, 5, 7],
          [4, 1, 3, 8, 5, 7, 9, 2, 6],
          [5, 7, 9, 1, 2, 6, 4, 3, 8],
          [1, 5, 7, 4, 9, 2, 6, 8, 3],
          [6, 9, 4, 7, 3, 8, 2, 1, 5],
          [3, 2, 8, 5, 6, 1, 7, 4, 9]]

other3 = [[2, 9, 5, 7, 4, 3, 8, 6, 1],
          [4, 3, 1, 8, 6, 5, 9, 2, 7],
          [8, 7, 6, 1, 9, 2, 5, 4, 3],
          [3, 8, 7, 4, 5, 9, 2, 1, 6],
          [6, 1, 2, 3, 8, 7, 4, 9, 5],
          [5, 4, 9, 2, 1, 6, 7, 3, 8],
          [7, 6, 3, 5, 2, 4, 1, 8, 9],
          [9, 2, 8, 6, 7, 1, 3, 5, 4],
          [1, 5, 4, 9, 3, 8, 6, 7, 2]]

other4 = [[8, 2, 7, 1, 5, 4, 3, 9, 6],
          [9, 6, 5, 3, 2, 7, 1, 4, 8],
          [3, 4, 1, 6, 8, 9, 7, 5, 2],
          [5, 9, 3, 4, 6, 8, 2, 7, 1],
          [4, 7, 2, 5, 1, 3, 6, 8, 9],
          [6, 1, 8, 9, 7, 2, 4, 3, 5],
          [7, 8, 6, 2, 3, 5, 9, 1, 4],
          [1, 5, 4, 7, 9, 6, 8, 2, 3],
          [2, 3, 9, 8, 4, 1, 5, 6, 7]]

other5 = [[4, 8, 9, 3, 1, 5, 2, 6, 7],
          [1, 6, 2, 4, 9, 7, 3, 5, 8],
          [3, 5, 7, 2, 8, 6, 9, 1, 4],
          [8, 9, 5, 6, 3, 1, 4, 7, 2],
          [6, 2, 1, 7, 4, 8, 5, 9, 3],
          [7, 4, 3, 5, 2, 9, 1, 8, 6],
          [9, 1, 4, 8, 7, 3, 6, 2, 5],
          [2, 7, 6, 1, 5, 4, 8, 3, 9],
          [5, 3, 8, 9, 6, 2, 7, 4, 1]]

other6 = [[1, 5, 2, 4, 8, 9, 3, 7, 6],
          [7, 3, 9, 2, 5, 6, 8, 4, 1],
          [4, 6, 8, 3, 7, 1, 2, 9, 5],
          [3, 8, 7, 1, 2, 4, 6, 5, 9],
          [5, 9, 1, 7, 6, 3, 4, 2, 8],
          [2, 4, 6, 8, 9, 5, 7, 1, 3],
          [9, 1, 4, 6, 3, 7, 5, 8, 2],
          [6, 2, 5, 9, 4, 8, 1, 3, 7],
          [8, 7, 3, 5, 1, 2, 9, 6, 4]]

other7 = [[2, 4, 8, 3, 9, 5, 7, 1, 6],
          [5, 7, 1, 6, 2, 8, 3, 4, 9],
          [9, 3, 6, 7, 4, 1, 5, 8, 2],
          [6, 8, 2, 5, 3, 9, 1, 7, 4],
          [3, 5, 9, 1, 7, 4, 6, 2, 8],
          [7, 1, 4, 8, 6, 2, 9, 5, 3],
          [8, 6, 3, 4, 1, 7, 2, 9, 5],
          [1, 9, 5, 2, 8, 6, 4, 3, 7],
          [4, 2, 7, 9, 5, 3, 8, 6, 1]]

hardestSudoku_fixed = [[True, False, False, False, False, False, False, False, False],
                       [False, False, True, True, False, False, False, False, False],
                       [False, True, False, False, True, False, True, False, False],
                       [False, True, False, False, False, True, False, False, False],
                       [False, False, False, False, True, True, True, False, False],
                       [False, False, False, True, False, False, False, True, False],
                       [False, False, True, False, False, False, False, True, True],
                       [False, False, True, True, False, False, False, True, False],
                       [False, True, False, False, False, False, True, False, False]]

reducers = []
reducer_hard = SolvedSudoku(hardestSudoku, hardestSudoku_fixed)
reducer_1 = SolvedSudoku(other1, hardestSudoku_fixed)
reducer_2 = SolvedSudoku(other2, hardestSudoku_fixed)
reducer_3 = SolvedSudoku(other3, hardestSudoku_fixed)
reducer_4 = SolvedSudoku(other4, hardestSudoku_fixed)
reducer_5 = SolvedSudoku(other5, hardestSudoku_fixed)
reducer_6 = SolvedSudoku(other6, hardestSudoku_fixed)
reducer_7 = SolvedSudoku(other7, hardestSudoku_fixed)

reducers.append(reducer_hard)

# reducers.append(reducer_hard)
# reducers.append(reducer_hard)
# reducers.append(reducer_hard)
# reducers.append(reducer_hard)
# reducers.append(reducer_hard)
# reducers.append(reducer_hard)
# reducers.append(reducer_hard)

reducers.append(reducer_1)
reducers.append(reducer_2)
reducers.append(reducer_3)
reducers.append(reducer_4)
reducers.append(reducer_5)
reducers.append(reducer_6)
reducers.append(reducer_7)

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

def next_batch():
    batch_x = []
    batch_y = []

    num_reducers = len(reducers)
    num_per_board = batch_size // num_reducers

    #num_per_board * len(reducers) = batch_size = 128
    for i in range(num_per_board):
        for red in range(len(reducers)):
            xs = reducers[red].board_to_row(reducers[red].board_reduction(numbers_to_predict))
            x_prepared = prepare_data(xs, 810)
            batch_x.append(x_prepared)

            ys = reducers[red].solution
            y_prepared = prepare_data(ys, numbers_to_predict*10)
            batch_y.append(y_prepared)

    return np.asarray(batch_x), np.asarray(batch_y)


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 3, 30, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

#Define layers in the NN

#Layer 1
W_conv1 = weight_variable([3 ,30 ,1, 1])
b_conv1 = bias_variable([1])

conv1 = tf.nn.relu(conv2d(x_board, W_conv1) + b_conv1)
#pool1 = max_pool_2x2(conv1)

# #Layer 2
# W_conv2 = weight_variable([5, 5, 32, 64])
# b_conv2 = bias_variable([64])
#
# conv2 = tf.nn.relu(conv2d(pool1, W_conv2) + b_conv2)
# pool2 = max_pool_2x2(conv2)

#Fully Connected Layer
#W_dense = weight_variable([7 * 7 * 64, 1024])
W_dense = weight_variable([3 * 3 * 1, 1024])
b_dense = bias_variable([1024])

#Connect pool2 with fully_connected
#pool2 = tf.reshape(pool2, [-1, 7 * 7 * 64])
conv1 = tf.reshape(conv1, [-1, 3 * 3 * 1])
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

num_steps = 3000
display_every = 100

#Start timer
start_time = time.time()
end_time = time.time()

for i in range(num_steps):
    b_x, b_y = next_batch()

    _, cost = sess.run([optimizer, cost], feed_dict={x: b_x, y: b_y})

    if i % display_every == 0:
        print("iteration : " + str(i)+" cost : " + str(cost))

