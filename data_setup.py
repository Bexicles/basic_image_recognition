import numpy as np
import control_panel

# Loads input data matrix from source file, then splits it into a training set & a test set
A = np.load("Data/image_data.npy")
np.random.shuffle(A)
A_split = np.vsplit(A, [int(A.shape[0] * control_panel.training_split), A.shape[0]])

Train = A_split[0]
Test = A_split[1]

# Splits out Ys and Xs for training and test data sets#
t1 = Train.shape[1]
t2 = Test.shape[1]

Train_split = np.hsplit(Train, [1, t1])
Train_X = Train_split[1]
Train_Y = Train_split[0]


Test_split = np.hsplit(Test, [1, t2])
Test_X = Test_split[1]
Test_Y = Test_split[0]

# Convert the y-vectors into one-hot matrices
Hot_train_Y = np.zeros((Train_Y.shape[0],Train_Y.shape[1]+1))
Hot_test_Y = np.zeros((Test.shape[0],Test_Y.shape[1]+1))

for i in range(0,Train_Y.shape[0]):
    if Train_Y[i] == 1:
        Hot_train_Y[i, 1] = 1

    else:
        Hot_train_Y[i, 0] = 1

for i in range(0,Test_Y.shape[0]):
    if Test_Y[i] == 1:
        Hot_test_Y[i, 1] = 1

    else:
        Hot_test_Y[i, 0] = 1
