# import numpy as np
# from tensorflow import constant as constensor
#
#
# ## idea for maxgroup, needs evaluated values in this state -> doesnt work in a network
# def maxgroupold(image, group, IMAGE_SIZE):
#     # get the Tensor's values
#     npimg = image.eval()
#
#     # reshape the array so <group> feature-maps are put into one array together
#     pairs = np.reshape(npimg, [-1, group, IMAGE_SIZE, IMAGE_SIZE])
#
#     # calculate the elementwise max in each group
#     max = np.amax(pairs, axis=1)
#
#     return constensor(max)

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow import get_variable_scope
from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import gen_array_ops

from tensorflow.python.layers import base

from tensorpack.models.common import layer_register

@layer_register(log_shape=True)
def maxgroup(inputs, group, image_size, axis=3, name=None):
    """Adds a maxgroup op
    "
     Arguments:
     inputs: Tensor input
     group: how many feature-maps should be grouped together
     axis: The dimension where max pooling will be performed. Default is the
     last dimension.
     name: Optional scope for name_scope.
     Returns:
      A `Tensor` representing the results of the pooling operation.
     Raises:
      ValueError: if num_units is not multiple of number of features.
    """
    return _MaxGroup(image_size=image_size, axis=axis, group=group, name=name)(inputs)


def MaxGroup(inputs, group, image_size, axis=3, name=None):
    return _MaxGroup(image_size=image_size, axis=axis, group=group, name=name).apply(inputs, scope=get_variable_scope())



class _MaxGroup(base.Layer):
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
                 group=2,
                 axis=3,
                 image_size=24,
                 name=None,
                 **kwargs):
        super(_MaxGroup, self).__init__(
            name=name, trainable=False, **kwargs)
        self.num_units = group
        self.size = image_size
        self.axis = axis


    def call(self, inputs):
        inputs = ops.convert_to_tensor(inputs)
        shape = inputs.shape

        if shape[self.axis] % self.num_units != 0:
            raise ValueError('number of features({}) is not '
                             'a multiple of group({})'
                             .format(shape[self.axis], self.num_units))

        out_channel = int(shape[self.axis].value / self.num_units)
        # Dealing with batches with arbitrary sizes
        batchsize = gen_array_ops.shape(inputs)[0]

        pairing = gen_array_ops.reshape(inputs, [batchsize, self.size, self.size, out_channel, self.num_units])
        outputs = math_ops.reduce_max(pairing, axis=self.axis+1, keepdims=False)

        return outputs



@layer_register(log_shape=True)
def pruneaxis(inputs, name=None):
  """
  prunes axis 1 and two. only call this if inputs.shape[1] == 1 == inputs.shape[2]
  input: Tensor of shape [batchsize, 1, 1, x]
  output: Tensor of shape [batchsize, x]
  """
  return PruneAxis(name=name)(inputs)

class PruneAxis(base.Layer):
  def __init__(self, name=None, **kwargs):
    super(PruneAxis, self).__init__(name=name, trainable=False, **kwargs)

  def call(self, inputs):
    inputs = ops.convert_to_tensor(inputs)
    shape = gen_array_ops.shape(inputs)
    batchsize = shape[0]
    out_channel = shape[3]
    return gen_array_ops.reshape(inputs, [batchsize, out_channel])


class ConvertLabel(base.Layer):
  def call(self, inputs):
    print("convert {}".format(inputs))
    inputs = ops.convert_to_tensor(inputs)
    shape = inputs.shape

    if shape[self.axis] % self.num_units != 0:
      raise ValueError('number of features({}) is not '
                       'a multiple of group({})'
                       .format(shape[self.axis], self.num_units))

    out_channel = int(shape[self.axis].value / self.num_units)
    # Dealing with batches with arbitrary sizes
    batchsize = gen_array_ops.shape(inputs)[0]

    pairing = gen_array_ops.reshape(inputs, [batchsize, self.size, self.size, out_channel, self.num_units])
    outputs = math_ops.reduce_max(pairing, axis=self.axis + 1, keep_dims=False)

    return outputs
