Basic Image Recognition Project:
Taking images of myself and my husband from a webcam, I firstly wrote a script to process the images (convert to greyscale and resize to 128x128). To ensure I had enough data to train my neural network, I also wrote a script to duplicate and mirror each file through the y-axis.

I then converted these images into a matrix of alpha values, which I split into training and test sets.
I fed this data into a multi-layer neural network, (built with tensorflow), which now tests with 99% accuracy.

I have also made use of tensorboard summaries and histograms in order to understand how my neural net was working.
