image_size = (28, 28)
image_pixels = image_size[0] * image_size[1]

training_split = 0.75   # proportion of original data set, set aside for training the model

alpha = 0.00001 # learning rate
solutions_number = 4   # no. of possible answers is two (me, dad, joan or peet; 0, 1, 2 or 3)
batch_size = 50