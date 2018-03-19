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

The output of the LSTM is then given into the CTC.
"""

"""Image break down; returns a sequence of frames"""

"""Application of CNN on every frame; returns a sequence of 128D-vectors"""
sequence_of_128D_vectors = []

"""RNN: 128LSTM_cells per layer, fully connected 37 neuron layer in the end"""


def build_rnn(input_images):
    #cut the picture in a sequence of frames:

    #apply the already trained CNN on each frame and return a sequence of 128D-vectors


    """Constants:"""
    seq_length = len(sequence_of_128D_vectors)
    num_LSTMs_per_layer = 128  # as described in the paper

    """RNN Cells"""
    # one cell
    single_LSTM_cell = rnn.BasicLSTMCell(num_units=num_neurons, activation=tf.nn.relu)
    # bidirectional LSTM with 128 layers each
    outputs_lstm, states_lstm = rnn.stack_bidirectional_rnn(
        cells_fw=[single_LSTM_cell for layer in range(num_LSTMs_per_layer)],
        cells_bw=[single_LSTM_cell for layer in range(num_LSTMs_per_layer)],
        inputs=X)

    """fully connected layer"""

    logits = tf.contrib.layers.fully_connected(
        inputs=outputs_lstm,
        num_outputs=37,  # these are the 36 characters plus the symbol for no character
        activation_fn=tf.nn.relu)

    #softmax as described in the paper:

    logits = tf.nn.softmax(logits, name='final_logits')

    return(logits)
