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





def build_rnn(inputs, sequence_length):
    """Constants:"""
    # TODO lenght not known at this point
    # seq_length = len(sequence_of_128D_vectors)

    # inputs = tf.expand_dims(inputs, 3)
    num_LSTMs_per_layer = 128  # as described in the paper

    """RNN Cells"""
    # bidirectional LSTM with 128 layers each
    # (outputs, _, _) = rnn.stack_bidirectional_rnn(
    #    cells_fw=[rnn.BasicLSTMCell(num_units=num_LSTMs_per_layer, activation=tf.nn.tanh)],
    #    cells_bw=[rnn.BasicLSTMCell(num_units=num_LSTMs_per_layer, activation=tf.nn.tanh)],
    #    inputs=[input],
    #    dtype=tf.float32
    # )

    outputs, _ = tf.nn.bidirectional_dynamic_rnn(rnn.BasicLSTMCell(num_units=num_LSTMs_per_layer, activation=tf.nn.tanh),
                                                 rnn.BasicLSTMCell(num_units=num_LSTMs_per_layer, activation=tf.nn.tanh),
                                                 inputs,
                                                 sequence_length=sequence_length,
                                                 dtype=tf.float32)


    # Concatenate outputs from fw and bw layer
    logits = tf.concat(outputs, 2)
    # print("RNN output shape: {}".format(logits.shape))


    # Logits contains the predicted character for every 36 possible frames.
    logits = tf.contrib.layers.fully_connected(
        inputs=logits,
        num_outputs=37,  # these are the 36 characters plus the symbol for no character
        activation_fn=tf.identity)

    # print("FC shape: {}".format(logits.shape))

    """softmax as described in the paper"""

    logits = tf.nn.softmax(logits, name='final_logits')

    return logits
