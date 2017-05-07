import tensorflow as tf
import control_panel
import data_setup


n = control_panel.solutions_number  # number of solution classes
p = control_panel.image_pixels  # number of input pixels
a = control_panel.alpha # learning rate
N = data_setup.t1   # number of 'images' in training data set

n_hidden_1 = 256    # number of features in first layer
n_hidden_2 = 256    # number of features in second layer

X_train = data_setup.Train_X  # training matrix of x values
Y_train = data_setup.Hot_train_Y  # training vector of y labels (one hot format)
X_test = data_setup.Test_X  # test matrix of x values
Y_test = data_setup.Hot_test_Y  # test vector of y labels (one hot format)

X = tf.placeholder(tf.float32, [None, p], name="X")   # placeholder for the input data, x
Y = tf.placeholder(tf.float32, [None, n], name="Y")  # placeholder for the actual values of y, in training data


# Initialise weights & biases

def init_weights(shape, name):
    return tf.Variable(tf.random_normal(shape, stddev=0.01), name=name)


w1 = init_weights([p, n_hidden_1], "w1")
w2 = init_weights([n_hidden_1, n_hidden_2], "w2")
w3 = init_weights([n_hidden_2, n], "w3")

b1 = init_weights([n_hidden_1], "b1")
b2 = init_weights([n_hidden_2], "b2")
b3 = init_weights([n], "b3")


# Histograms to allow me to visualise weights & biases
tf.summary.histogram("w1 summary", w1)
tf.summary.histogram("w2 summary", w2)
tf.summary.histogram("output summary", w3)

tf.summary.histogram("b1 summary", b1)
tf.summary.histogram("b2 summary", b2)
tf.summary.histogram("output bias summary", b3)


# Function to build model
def multilayer_perceptron(x, w1, b1, w2, b2, w3, b3):
    with tf.name_scope("layer_1"):
        layer_1 = tf.add(tf.matmul(x, w1), b1)
        layer_1 = tf.nn.relu(layer_1)   # RELU activation

    with tf.name_scope("layer_2"):
        layer_2 = tf.add(tf.matmul(layer_1, w2), b2)
        layer_2 = tf.nn.relu(layer_2)   # RELU activation

    with tf.name_scope("output_layer"):
        output_layer = tf.matmul(layer_2, w3) + b3 # output layer, linear activation

    return output_layer



# Model
y = multilayer_perceptron(X, w1, b1, w2, b2, w3, b3)

with tf.name_scope("cost_function"):
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=y))
    train_step = tf.train.AdamOptimizer(a).minimize(cross_entropy)
    tf.summary.scalar("cost_function", cross_entropy)

with tf.name_scope("accuracy"):
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar("accuracy", accuracy)


# Create a session
with tf.Session() as sess:

    writer = tf.summary.FileWriter("./Logs/", sess.graph)
    merged = tf.summary.merge_all(key="summaries")

    tf.initialize_all_variables().run()


    for i in range(100):
        for start, end in zip(range(0, len(X_train), 128), range(128, len(X_train)+1, 128)):
            sess.run(train_step, feed_dict={X: X_train[start: end], Y: Y_train[start: end]})

        summary, acc = sess.run([merged, accuracy], feed_dict={X: X_test, Y: Y_test})
        writer.add_summary(summary, i)
        print(i, acc)