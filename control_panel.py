image_size = (128, 128)
image_pixels = image_size[0] * image_size[1]

training_split = 0.75   # proportion of original data set, set aside for training the model

alpha = 0.5 # learning rate
solutions_number = 1   # no. of possible answers is two (me or peet; 0 or 1)
batch_size = 30 # no. 'images' per batch
