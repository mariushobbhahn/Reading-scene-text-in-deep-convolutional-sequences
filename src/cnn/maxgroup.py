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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import gen_array_ops

from tensorflow.python.layers import base


def maxgroup(inputs, group, IMAGE_SIZE, name=None):
  """Adds a maxgroup op
  "
   Arguments:
   inputs: Tensor input
   group: how many featur-maps should be grouped together
   axis: The dimension where max pooling will be performed. Default is the
   last dimension.
   name: Optional scope for name_scope.
   Returns:
    A `Tensor` representing the results of the pooling operation.
   Raises:
    ValueError: if num_units is not multiple of number of features.
  """
  return MaxGroup(IMAGE_SIZE, group=group, name=name)(inputs)


class MaxGroup(base.Layer):
  """Adds a maxgroup op
  "
  Arguments:
    inputs: Tensor input
    num_units: Specifies how many features will remain after maxout in the `axis` dimension
         (usually channel).
    This must be multiple of number of `axis`.
    axis: The dimension where max pooling will be performed. Default is the
    last dimension.
    name: Optional scope for name_scope.
  Returns:
    A `Tensor` representing the results of the pooling operation.
  Raises:
    ValueError: if num_units is not multiple of number of features.
  """

  def __init__(self,
         IMAGE_SIZE,
         group=2,
         name=None,
         **kwargs):
    super(MaxGroup, self).__init__(
      name=name, trainable=False, **kwargs)
    self.num_units = group

  def call(self, inputs):
    inputs = ops.convert_to_tensor(inputs)
    shape = inputs.shape
    pairing = gen_array_ops.reshape(inputs, [-1, self.num_units, IMAGE_SIZE, IMAGE_SIZE])

    if shape[0].value % self.num_units:
      raise ValueError('number of features({}) is not '
               'a multiple of group({})'
               .format(num_channels, self.num_units))

    outputs = math_ops.reduce_max(pairing, axis=1, keep_dims=False)

    return outputs
