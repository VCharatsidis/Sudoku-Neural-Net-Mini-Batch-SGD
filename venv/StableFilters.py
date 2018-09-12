
import tensorflow as tf
import numpy as np
from Sudoku import SolvedSudoku

num_sudokus = 2000
reducer = SolvedSudoku(num_sudokus)
test_reducer = SolvedSudoku(num_sudokus//2)
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


#we have 3 different features, row column box
num_features = 3
num_filters = 8
#Box feature
W_box = weight_variable([3, 30, 1, num_filters])
b_box = bias_variable([num_filters])

#Row feature
W_row = weight_variable([1, 90, 1, num_filters])
b_row = bias_variable([num_filters])

#Column feature
W_column = weight_variable(([9, 10, 1, num_filters]))
b_column = bias_variable([num_filters])

conv1_box = tf.nn.sigmoid(tf.nn.conv2d(x_board, W_box, strides=[1, 3, 30, 1], padding='VALID') + b_box)
conv1_row = tf.nn.sigmoid(tf.nn.conv2d(x_board, W_row, strides=[1, 1, 90, 1], padding='VALID') + b_row)
conv1_column = tf.nn.sigmoid(tf.nn.conv2d(x_board, W_column, strides=[1, 9, 10, 1], padding='VALID') + b_column)

conv1_box = tf.reshape(conv1_box, [-1, 1, 9, num_filters])
conv1_row = tf.reshape(conv1_row, [-1, 1, 9, num_filters])
conv1_column = tf.reshape(conv1_column, [-1, 1, 9, num_filters])

x_refined = tf.get_variable("x_ref", [batch_size, 1, 1, num_filters])

for row in range(9):
    for col in range(9):
        extracted_row = tf.slice(conv1_row, [0, 0, row, 0], [batch_size, 1, 1, num_filters])
        extracted_col = tf.slice(conv1_column, [0, 0, col, 0], [batch_size, 1, 1, num_filters])

        # extracted_row = tf.reshape(extracted_row, [1, num_filters])
        # extracted_col = tf.reshape(extracted_col, [1, num_filters])
        #concat-----------------------
        row_col = tf.concat([extracted_row, extracted_col], 2)

        box = (row // 3) * 3 + col // 3
        print("row_col " + str(row_col.shape))
        extracted_box = tf.slice(conv1_box, [0, 0, box, 0], [batch_size, 1, 1, num_filters])
        #print("extracted_box[1] : " + str(extracted_box.shape[1]) + " extracted_box[2] : " + str(extracted_box.shape[2]) + " extracted_box[3] : " + str(extracted_box.shape[3]))
        # extracted_box = tf.reshape(extracted_box, [1, num_filters])

        shape_r = extracted_box.get_shape().as_list()
        #print(" shape row 0 : " + str(shape_r[0]) + " shape row 1 : " + str(shape_r[1]))

        # concat-----------------------
        row_col_box = tf.concat([row_col, extracted_box], 2)
        print("row_col_box reshaped " + str(row_col_box.shape))
        print("row_col_box[0] : " + str(row_col_box.shape[0]) + "row_col_box[1] : " + str(
            row_col_box.shape[1]) + " row_col_box[2] : " + str(row_col_box.shape[2]) + " row_col_box[3] : " + str(
            row_col_box.shape[3]))
        # concat-----------------------
        x_refined = tf.concat([x_refined, row_col_box], 2)

print("just made "+str(x_refined.shape))

print("x_refined[0] : " + str(x_refined.shape[0])+"x_refined[1] : " + str(x_refined.shape[1]) + " x_refined[2] : " + str(x_refined.shape[2]) + " x_refined[3] : " + str(x_refined.shape[3]))
#x_refined = tf.slice(x_refined, [0, 0, 0, 0], [-1, 1, 243, num_filters])
# Layer 2
W_conv2 = weight_variable([1, 3, num_filters, num_filters])
b_conv2 = bias_variable([num_filters])

#x_refined = tf.reshape(x_refined, [-1, 1, 243, num_filters])

print("x_refined reshaped "+str(x_refined.shape))
conv2 = tf.nn.sigmoid(tf.nn.conv2d(x_refined, W_conv2, strides=[1, 1, 3, 1], padding='VALID') + b_conv2)

shape = conv2.get_shape().as_list()
print(" conv2 shape[1] : "+ str(shape[1]) + " conv2  shape[2] : "+ str(shape[2])+" conv2  shape[3] : " +str(shape[3]))
shape = shape[1] * shape[2] * shape[3]

conv2 = tf.reshape(conv2, [-1, 9, 9, num_filters])
# Layer 3
W_conv3 = weight_variable([3, 3, num_filters, num_filters])
b_conv3 = bias_variable([num_filters])
conv3 = tf.nn.sigmoid(tf.nn.conv2d(conv2, W_conv3, strides=[1, 1, 1, 1], padding='VALID') + b_conv3)

shape = conv3.get_shape().as_list()
print(" conv3 shape[1] : "+ str(shape[1]) + " conv3  shape[2] : "+ str(shape[2])+" conv3  shape[3] : " +str(shape[3]))
shape = shape[1] * shape[2] * shape[3]

conv3 = tf.reshape(conv3, [-1, shape])

print("shape "+ str(shape))

# #Fully Connected Layer
W_dense = weight_variable([shape, 1024])
b_dense = bias_variable([1024])
dense = tf.nn.sigmoid(tf.matmul(conv3, W_dense) + b_dense)

# dense 2
W_dense2 = weight_variable([1024, 512])
b_dense2 = bias_variable([512])
dense2 = tf.nn.sigmoid(tf.matmul(dense, W_dense2) + b_dense2)

#Readout layer
W_output = weight_variable(([512, numbers_to_predict * 10]))
b_output = bias_variable([numbers_to_predict * 10])
y_conv = tf.matmul(dense2, W_output) + b_output

#Loss
cost = tf.losses.mean_squared_error(labels=y, predictions=y_conv)
#Optimizer
optimizer = tf.train.AdamOptimizer(0.002).minimize(cost)

sess.run(tf.global_variables_initializer())

import time

num_steps = 500001
display_every = 10000

#Start timer
start_time = time.time()
end_time = time.time()

for i in range(num_steps):
    b_x, b_y = next_batch()

    optimizer.run(feed_dict={x: b_x, y: b_y})
    cost1 = cost.eval(feed_dict={x: b_x, y: b_y})

    if i % display_every == 0:
        end_time = time.time()
        print("iteration : " + str(i)+", cost : " + str(cost1) + ", time elapsed : " + str(end_time - start_time))

end_time = time.time()
print("time elapsed : " + str(end_time - start_time))
avg_cost = 0

for i in range(5000):
    b_x, b_y = test_batch()

    optimizer.run(feed_dict={x: b_x, y: b_y})
    cost1 = cost.eval(feed_dict={x:b_x, y:b_y})
    avg_cost = avg_cost + cost1

    if i % 500 == 0:
        print("cost test : " + str(cost1) +" avg cost "+str(avg_cost / i))
