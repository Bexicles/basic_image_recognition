import tensorflow as tf
import control_panel
import data_setup
import nn_functions

img1 = control_panel.image_size[0]
img2 = control_panel.image_size[1]
n = control_panel.solutions_number
p = control_panel.image_pixels
a = control_panel.alpha
N = data_setup.t1   # number of 'images' in training data set

X_train = data_setup.Train_X  # training matrix of x values
Y_train = data_setup.Hot_train_Y  # training vector of y labels (one hot format)
X_test = data_setup.Test_X  # test matrix of x values
Y_test = data_setup.Hot_test_Y  # test vector of y labels (one hot format)


x = tf.placeholder(tf.float32, [None, p])   # placeholder for the input data, x
y_ = tf.placeholder(tf.float32, [None, n])  # placeholder for the actual values of y, in training data


# First convolutional layer
W_conv1 = nn_functions.weight_variable([5, 5, 1, 32])
b_conv1 = nn_functions.bias_variable([32])

x_image = tf.reshape(x, [-1, img1, img2, 1])

h_conv1 = tf.nn.relu(nn_functions.conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = nn_functions.max_pool_2x2(h_conv1)


# Second convolutional layer
W_conv2 = nn_functions.weight_variable([5, 5, 1, 64])
b_conv2 = nn_functions.bias_variable([64])

h_conv2 = tf.nn.relu(nn_functions.conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = nn_functions.max_pool_2x2(h_conv2)


# Densely connected layer
W_fc1 = nn_functions.weight_variable([7 * 7 * 64, 1024])
b_fc1 = nn_functions.bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)


# Final layer
W_fc2 = nn_functions.weight_variable([1024, n])
b_fc2 = nn_functions.bias_variable([n])

y_conv = tf.matmul(h_fc1, W_fc2) + b_fc2


cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.GradientDescentOptimizer(a).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

sess.run(train_step, feed_dict={x: X_train, y_: Y_train})
train_accuracy = accuracy.eval(feed_dict={x: X_train, y_: Y_train})
print("step %d, training accuracy %g"%( train_accuracy))

print("test accuracy %g"%accuracy.eval(feed_dict={x: X_test, y_: Y_test}))