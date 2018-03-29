# Just import everything into current namespace
from tensorpack import *
import tensorflow as tf
import numpy as np

import tensorflow.contrib.rnn as rnn

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

The output of the LSTM is then given into the CTC. (This happens in the train class though)
"""



"""RNN: 128LSTM_cells per layer, fully connected 37 neuron layer in the end"""


def build_rnn(sequence_of_128D_vectors):
    """Constants:"""
    # TODO lenght not known at this point
    # seq_length = len(sequence_of_128D_vectors)

    num_LSTMs_per_layer = 128  # as described in the paper

    """RNN Cells"""
    # bidirectional LSTM with 128 layers each
    outputs_lstm, states_lstm = rnn.stack_bidirectional_rnn(
        cells_fw=[rnn.BasicLSTMCell(num_units=num_LSTMs_per_layer, activation=tf.nn.tanh)],
        cells_bw=[rnn.BasicLSTMCell(num_units=num_LSTMs_per_layer, activation=tf.nn.tanh)],
        inputs=[sequence_of_128D_vectors],
        dtype=tf.float32
    )

    """fully connected layer on top"""

    decoded_sequence = tf.contrib.layers.fully_connected(
        inputs=outputs_lstm,
        num_outputs=37,  # these are the 36 characters plus the symbol for no character
        activation_fn=tf.identity)

    """softmax as described in the paper"""

    decoded_sequence = tf.nn.softmax(decoded_sequence, name='final_logits')

    return decoded_sequence
