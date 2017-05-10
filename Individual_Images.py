import numpy as np
from PIL import Image
import control_panel

p = control_panel.image_pixels # number of features (pixels) in each image
X = np.zeros(shape=(1, p))  # initialise image matrix with zeros in one row


# Creates alpha array for selected image
def get_img_vector(FilePath, infile):
    img = Image.open(FilePath + infile)
    a = np.asarray(img).reshape(-1)     # reshape each matrix of image's pixel alpha values to a vector
    global X
    X = np.vstack([X, a])  # add vector of each image's pixel alpha values to new row of Image matrix
    X = np.delete(X, (0), axis=0)  # delete the first row of Image matrix (the zeros row)
    return X


