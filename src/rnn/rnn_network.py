# Just import everything into current namespace
from tensorpack import *
import tensorflow as tf
import numpy as np

rnn = tf.contrib.rnn

"""
Explanation:

Our model consist of 2 LSTM cells that read the sequence in opposite directions.
The input is a sequence of 128 dimensional vectors each containing character information corresponding to one frame of the sliding window.
E.g. an image of size 32x(32+ 2*stepsize) is converted to [[...], [...], [...]] by the cnn, where [...] is a vector of
128 values describing the respective frame.

The outputs of the 2 LSTM cells are then combined (via average?) to a vector of probability-vectors p_i.
Each probability-vector contains 128 probabilities for a given character.
Due to the unsegmented nature of the word image at the character level, the length of the LSTM outputs is not consistent with the length of a target word string.
We therefore apply the CTC approach to make reasonable character sequences out of the probability-vector (see https://www.tensorflow.org/versions/r0.12/api_docs/python/nn/connectionist_temporal_classification__ctc_)

The output of the CTC is then returned
"""

"""
TODOs:

- where do the probability vectors come from?
- implement CTC
"""

input = [] #sequence of vectors (128d)
frame_size = 128 #len(input[0])
batch_size = len(input)

lstm1 = tf.contrib.rnn.BasicLSTMCell(frame_size)
lstm2 = tf.contrib.rnn.BasicLSTMCell(frame_size)


hidden_state_1 = tf.zeros([frame_size, lstm1.state_size[0]])
current_state_1 = tf.zeros([frame_size, lstm1.state_size[0]])
state_1 = hidden_state_1, current_state_1

hidden_state_2, current_state_2, state_2 = hidden_state_1, current_state_1, state_1

print(state_1)


for i in range(batch_size):
    # The value of state is updated after processing each batch of words.
    output_1, state_1 = lstm1(input[i], state_1)
    output_2, state_2 = lstm2(input[len(input) - i], state_2)
