import numpy as np
from tensorflow import constant as constensor


## idea for maxgroup, needs evaluated values in this state -> doesnt work in a network
def maxgroup(image, group, IMAGE_SIZE):
    # get the Tensor's values
    npimg = image.eval()

    # reshape the array so <group> feature-maps are put into one array together
    pairs = np.reshape(npimg, [-1, group, IMAGE_SIZE, IMAGE_SIZE])

    # calculate the elementwise max in each group
    max = np.amax(pairs, axis=1)

    return constensor(max)
