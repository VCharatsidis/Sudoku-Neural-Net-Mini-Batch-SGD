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
test_reducer = SolvedSudoku(2000)
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
num_filters = 64
#Box filter
W = weight_variable([3, 30, 1, num_filters])
b = bias_variable([num_filters])

conv1 = tf.nn.sigmoid(tf.nn.conv2d(x_board, W, strides=[1, 3, 30, 1], padding='SAME') + b)

print(str(conv1.shape))

# w_1x1 = weight_variable([1, 1, 64, 32])
# b_1x1 = bias_variable([32])
# conv1x1 = tf.nn.sigmoid(tf.nn.conv2d(conv1, w_1x1, strides=[1, 1, 1, 1], padding='SAME') + b_1x1)

# #Fully Connected Layer
# W_dense = weight_variable([7 * 7 * 4, 1024])
W_dense = weight_variable([3 * 3 * num_filters, 1024])
b_dense = bias_variable([1024])

conv1 = tf.reshape(conv1, [-1, 3 * 3 * num_filters])

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
display_every = 5000

#Start timer
start_time = time.time()
end_time = time.time()

for i in range(num_steps):
    b_x, b_y = next_batch()

    optimizer.run(feed_dict={x: b_x, y: b_y})
    cost1 = cost.eval(feed_dict={x: b_x, y: b_y})

    if i % display_every == 0:
        print("iteration : " + str(i)+" cost : " + str(cost1))

end_time = time.time()
print("time elapsed : " + str(end_time - start_time))

avg_cost = 0

for i in range(1000):
    b_x, b_y = test_batch()

    optimizer.run(feed_dict={x: b_x, y: b_y})
    cost1 = cost.eval(feed_dict={x:b_x, y:b_y})
    avg_cost = avg_cost + cost1

    if i % 50 == 0:
        print("cost test : " + str(cost1) +" avg cost "+str(avg_cost / i))