import tensorflow as tf
import control_panel
import data_setup

n = control_panel.solutions_number  # number of solution classes
p = control_panel.image_pixels  # number of input pixels
a = control_panel.alpha # learning rate
N = data_setup.t1   # number of 'images' in training data set

n_hidden_1 = 256    # number of features in first layer
n_hidden_2 = 256    # number of features in second layer
n_hidden_3 = 256    # number of features in third layer

X_train = data_setup.Train_X  # training matrix of x values
Y_train = data_setup.Hot_train_Y  # training vector of y labels (one hot format)
X_test = data_setup.Test_X  # test matrix of x values
Y_test = data_setup.Hot_test_Y  # test vector of y labels (one hot format)

x = tf.placeholder(tf.float32, [None, p])   # placeholder for the input data, x
y_ = tf.placeholder(tf.float32, [None, n])  # placeholder for the actual values of y, in training data

# Function to build model
def multilayer_perceptron(x, weights, biases):
    layer_1 = tf.add(tf.matmul(x, weights['h1']),biases['b1'])
    layer_1 = tf.nn.relu(layer_1)   # RELU activation

    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']),biases['b2'])
    layer_2 = tf.nn.relu(layer_2)   # RELU activation

    layer_3 = tf.add(tf.matmul(layer_2, weights['h3']),biases['b3'])
    layer_3 = tf.nn.relu(layer_3)   # RELU activation

    output_layer = tf.matmul(layer_3, weights['h4']) + biases['b4'] # output layer, linear activation
    return output_layer


# Dictionaries to store layers' weights & biases
weights = {
    'h1': tf.Variable(tf.random_normal([p, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3])),
    'h4': tf.Variable(tf.random_normal([n_hidden_3, n])),
}

biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'b3': tf.Variable(tf.random_normal([n_hidden_3])),
    'b4': tf.Variable(tf.random_normal([n])),

}


# Model
y = multilayer_perceptron(x, weights, biases)

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
train_step = tf.train.AdamOptimizer(a).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

#writer = tf.train.SummaryWriter("Logs/", graph=tf.get_default_graph())

train_accuracy = accuracy.eval(feed_dict={ x: X_train, y_: Y_train})
print("training accuracy %g"%(train_accuracy))
train_step.run(feed_dict={ x: X_train, y_: Y_train})

print("test accuracy %g"%accuracy.eval(feed_dict={ x: X_test, y_: Y_test}))