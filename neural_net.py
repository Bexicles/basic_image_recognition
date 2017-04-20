import tensorflow as tf
import control_panel
import data_setup

n = control_panel.solutions_number
p = control_panel.image_pixels
a = control_panel.alpha
N = data_setup.t1   # number of 'images' in training data set

X_train = data_setup.Train_X  # training matrix of x values
Y_train = data_setup.Hot_train_Y  # training vector of y labels (one hot format)
X_test = data_setup.Test_X  # test matrix of x values
Y_test = data_setup.Hot_test_Y  # test vector of y labels (one hot format)


x = tf.placeholder(tf.float32, [None, p])   # placeholder for the input data, x

W = tf.Variable(tf.zeros([p, n]))
b = tf.Variable(tf.zeros([n]))

y = tf.nn.softmax(tf.matmul(x, W) + b)  # model to calculate learned values of y
y_ = tf.placeholder(tf.float32, [None, n])  # placeholder for the actual values of y, in training data

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))   # cross entropy of difference between calculated answer and actual answer
train_step = tf.train.GradientDescentOptimizer(a).minimize(cross_entropy)   # performs Gradient Descent to minimise cross entropy

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

sess.run(train_step, feed_dict={x: X_train, y_: Y_train})

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print(sess.run(accuracy, feed_dict={x: X_test, y_: Y_test}))
