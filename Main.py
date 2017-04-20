import numpy as np
import control_panel

# Loads input data matrix from source file, then splits it into a training set & a test set
A = np.load("Data/image_data.npy")
np.random.shuffle(A)
A_split = np.vsplit(A, [int(A.shape[0] * control_panel.training_split), A.shape[0]])

Train = A_split[0]
Test = A_split[1]


# Splits out Ys and Xs for training and test data sets
Train_split = np.hsplit(Train, [1, Train.shape[0]])
Train_X = Train_split[1]
Train_Y = Train_split[0]

Test_split = np.hsplit(Test, [1, Test.shape[0]])
Test_Y = Test_split[1]
Test_Y = Test_split[0]



