import numpy as np
import control_panel

n = control_panel.solutions_number  # number of possible solutions

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


def convert_to_one_hot(A):
    temp = np.zeros((A.shape[0],A.shape[1]+(n-1)))  # intialises a matrix of zeros with extra column

    for i in range(0, A.shape[0]):
        for j in range(0,n):
            if A[i] == j:
                temp[i, j] = 1
    return temp

Hot_train_Y = convert_to_one_hot(Train_Y)
Hot_test_Y = convert_to_one_hot(Test_Y)

print("Data Setup is now complete!")
