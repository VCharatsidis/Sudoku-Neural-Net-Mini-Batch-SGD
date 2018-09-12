import tensorflow as tf
import numpy as np
from Sudoku import SolvedSudoku

reducer = SolvedSudoku(1)
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

#Layer 1
W_conv1 = weight_variable([3, 10, 1, 8])
b_conv1 = bias_variable([8])

conv1 = tf.nn.sigmoid(tf.nn.conv2d(x_board, W_conv1, strides=[1, 1, 10, 1], padding='SAME') + b_conv1)
#pool1 = max_pool_2x2(conv1)

#Layer 2
W_conv2 = weight_variable([1, 3, 8, 16])
b_conv2 = bias_variable([16])

conv2 = tf.nn.sigmoid(tf.nn.conv2d(conv1, W_conv2, strides=[1, 1, 1, 1], padding='SAME') + b_conv2)
#pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

shape = conv2.get_shape().as_list()
print("shape[1] : "+ str(shape[1]) + " shape[2] : "+ str(shape[2])+" shape[3] : " +str(shape[3]))
shape = shape[1] * shape[2] * shape[3]

conv2 = tf.reshape(conv2, [-1, shape])


#Fully Connected Layer
W_dense = weight_variable([9 * 9 * 16, 1024])
#W_dense = weight_variable([3 * 3 * 32, 1024])
b_dense = bias_variable([1024])
#
# #Connect pool2 with fully_connected
# conv2 = tf.reshape(conv2, [-1, 7 * 7 * 4])
# #pool2 = tf.reshape(pool2, [-1, 7 * 7 * 64])
# dense = tf.nn.sigmoid(tf.matmul(conv2, W_dense) + b_dense)
conv2 = tf.reshape(conv2, [-1, 9 * 9 * 16])

dense = tf.nn.sigmoid(tf.matmul(conv2, W_dense) + b_dense)

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

avg_cost = 0

for i in range(1000):
    b_x, b_y = test_batch()

    optimizer.run(feed_dict={x: b_x, y: b_y})
    cost1 = cost.eval(feed_dict={x:b_x, y:b_y})
    avg_cost = avg_cost + cost1

    if i % 50 == 0:
        print("cost test : " + str(cost1) +" avg cost "+str(avg_cost / i))