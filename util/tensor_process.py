from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import tensorflow as tf
import traceback
import os
import math
import numpy as np
import random
import nibabel
from tensorflow.contrib.image.ops import gen_image_ops
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops
from tensorflow.contrib.image.python.ops.image_ops import angles_to_projective_transforms
from tensorflow.contrib.image.python.ops.image_ops import _IMAGE_DTYPES
from scipy import ndimage
from util.data_loader import *
from util.data_process import *

def load_nifty_volume(filename):
    """Read a nifty image and return data array
    used as parameters tf.py_func for image loading
    input shape [W, H, D]
    output shape [D, H, W]
    """
    img = nibabel.load(filename)
    data = img.get_data()
    data = np.asarray(np.transpose(data, [2,1,0]), np.float32)
    return [data]

def transpose_tensor(x, direction = 'axial'):
    if(direction == 'axial'):
        output = x
    elif(direction == 'sagittal'):
        output = tf.transpose(x, perm = [2, 0, 1, 3])
    elif(direction == 'coronal'):
        output = tf.transpose(x, perm = [1, 0, 2, 3])
    else:
        raise(VlueError('unsupported direction {0:}'.format(direction)))
    return output

def random_flip_tensors_in_one_dim(x, d):
    """
    Random flip a tensor in one dimension
    x: a list of tensors
    d: a integer denoting the axis
    """
    r = tf.random_uniform([1], 0, 1)
    r = tf.less(r, tf.constant(0.5))
    r = tf.cast(r, tf.int32)
    y = []
    for xi in x:
        if(xi is None):
            y.append(None)
        else:
            xi_xiflip = tf.stack([xi, tf.reverse(xi, tf.constant([d]))])
            slice_begin = tf.concat([r, tf.zeros_like(tf.shape(xi))], -1)
            slice_size  = tf.concat([tf.constant([1]), tf.shape(xi)], -1)
            flip = tf.slice(xi_xiflip, slice_begin, slice_size)
            flip = tf.reshape(flip, tf.shape(xi))
            y.append(flip)
    return y

def transform_4d(images, transforms, interpolation="NEAREST"):
    """ A variant of  tensorflow.contrib.image.python.ops.image_ops.rotate.transform
     that only works on 4D tensors, without checking image rank.
     when the shape of images is unknown, len(image_or_images.get_shape()) will raise error.
    Applies the given transform(s) to the image(s).
    Args:
    images: A tensor of shape (num_images, num_rows, num_columns, num_channels)
       (NHWC), (num_rows, num_columns, num_channels) (HWC), or
       (num_rows, num_columns) (HW).
    transforms: Projective transform matrix/matrices. A vector of length 8 or
       tensor of size N x 8. If one row of transforms is
       [a0, a1, a2, b0, b1, b2, c0, c1], then it maps the *output* point
       `(x, y)` to a transformed *input* point
       `(x', y') = ((a0 x + a1 y + a2) / k, (b0 x + b1 y + b2) / k)`,
       where `k = c0 x + c1 y + 1`. The transforms are *inverted* compared to
       the transform mapping input points to output points.
    interpolation: Interpolation mode. Supported values: "NEAREST", "BILINEAR".
    Returns:
    Image(s) with the same type and shape as `images`, with the given
    transform(s) applied. Transformed coordinates outside of the input image
    will be filled with zeros.
    Raises:
    TypeError: If `image` is an invalid type.
    """
    images = ops.convert_to_tensor(images, name="images")
    transform_or_transforms = ops.convert_to_tensor(
      transforms, name="transforms", dtype=dtypes.float32)
    if images.dtype.base_dtype not in _IMAGE_DTYPES:
        raise TypeError("Invalid dtype %s." % images.dtype)

    if len(transform_or_transforms.get_shape()) == 1:
        transforms = transform_or_transforms[None]
    elif len(transform_or_transforms.get_shape()) == 2:
        transforms = transform_or_transforms
    else:
        raise TypeError("Transforms should have rank 1 or 2.")
    output = gen_image_ops.image_projective_transform(
      images, transforms, interpolation=interpolation.upper())
    return output

def rotate(images, angles, interpolation="NEAREST"):
    """A variant of  tensorflow.contrib.image.python.ops.image_ops.rotate
     that only works on 4D tensors, without checking image rank.
     when the shape of images is unknown, len(image_or_images.get_shape()) will raise error.

    Rotate image(s) by the passed angle(s) in radians.
    Args:
    images: A tensor of shape (num_images, num_rows, num_columns, num_channels)
       (NHWC), (num_rows, num_columns, num_channels) (HWC), or
       (num_rows, num_columns) (HW).
    angles: A scalar angle to rotate all images by, or (if images has rank 4)
       a vector of length num_images, with an angle for each image in the batch.
    interpolation: Interpolation mode. Supported values: "NEAREST", "BILINEAR".
    Returns:
    Image(s) with the same type and shape as `images`, rotated by the given
    angle(s). Empty space due to the rotation will be filled with zeros.
    Raises:
    TypeError: If `image` is an invalid type.
    """
    images = ops.convert_to_tensor(images, name="images")

    image_height = math_ops.cast(array_ops.shape(images)[1], dtypes.float32)[None]
    image_width  = math_ops.cast(array_ops.shape(images)[2], dtypes.float32)[None]
    output = transform_4d(images,
                          angles_to_projective_transforms(angles, image_height, image_width),
                          interpolation=interpolation)
    return output

def get_bounding_box_of_4d_tensor(label, margin):
    """ Get the 4D bounding boxe generated from nonzero region of the label
        if the nonzero region is null, return the tesor size
        inputs:
            label: a 4D tensor
            margin: a list of margin of each dimension
        outputs:
            indices_min: the lower bound in each dimension of bounding box
            indices_max: the upper bound in each dimension of bounding box
        """
    # find bounding box first
    max_idx = tf.subtract(tf.shape(label), tf.constant([1,1,1,1], tf.int32))
    margin = tf.constant(margin)
    mask = tf.not_equal(label, tf.constant(0, dtype=tf.int32))
    indices = tf.cast(tf.where(mask), tf.int32)
    indices_min = tf.reduce_min(indices, reduction_indices=[0]) # infinity if indices is null
    indices_min = tf.subtract(indices_min, margin)
    indices_min = tf.maximum(indices_min, tf.constant([0,0,0,0], tf.int32))
    indices_min = tf.minimum(indices_min, max_idx)
    
    indices_max = tf.reduce_max(indices, reduction_indices=[0]) # minus infinity if indces is null
    indices_max = tf.add(indices_max, margin)
    indices_max = tf.minimum(indices_max,
                            tf.subtract(tf.shape(label),
                            tf.constant([1,1,1,1], tf.int32)))
    indices_max = tf.maximum(indices_max, tf.constant([0,0,0,0]))
    
    # in case the nonzero region is null, switch the indices_min and indices_max
    indices_sum = tf.add(indices_min, indices_max)
    indices_sub = tf.subtract(indices_min, indices_max)
    indices_abs = tf.abs(indices_sub)
    indices_min = tf.div(tf.subtract(indices_sum, indices_abs), tf.constant([2,2,2,2]))
    indices_max = tf.div(tf.add(indices_sum, indices_abs), tf.constant([2,2,2,2]))
    return [indices_min, indices_max]

def crop_4d_tensor_with_bounding_box(input_tensor, idx_min, idx_max):
    """Crop a 4D tensor with a bounding box
    inputs:
        input_tensor: the input 4D tensor
        idx_min: the lower bound in each dimension of the bounding box
        idx_max: the upper bound in each dimension of the bounding box
    outputs:
        output_tensor: the cropped region of input_tensor
    """
    size = tf.subtract(idx_max, idx_min)
    size = tf.add(size, tf.constant([1,1,1,1], tf.int32))
    output_tensor  = tf.slice(input_tensor, idx_min, size)
    return output_tensor

def pad_tensor_to_desired_shape(inpt_tensor, outpt_shape, mode="REFLECT"):
    """ Pad a tensor to desired shape
        if the input size is larger than output shape, then return the input tensor
    inputs: 
        inpt_tensor: the input tensor
        outpt_shape: the desired shape after padding
    outputs:
        outpt_tensor: the padded tensor, with 'REFLECT' mode
    """
    inpt_shape = tf.shape(inpt_tensor)
    shape_sub  = tf.subtract(inpt_shape, outpt_shape)
    flag = tf.cast(tf.less(shape_sub, tf.zeros_like(shape_sub)), tf.int32)
    flag = tf.scalar_mul(tf.constant(-1), flag)
    pad  = tf.multiply(shape_sub, flag)
    pad_l = tf.scalar_mul(tf.constant(0.5), tf.cast(pad, tf.float32))
    pad_l = tf.cast(pad_l, tf.int32)
    pad_r = pad - pad_l
    pad_lr = tf.stack([pad_l, pad_r], axis = 1)
    outpt_tensor = tf.pad(inpt_tensor, pad_lr, mode = mode)
    return outpt_tensor

def itensity_normalize_3d_volume(volume):
    pixels = volume[volume > 0]
    mean = pixels.mean()
    std  = pixels.std()
    out = (volume - mean)/std
    out_random = np.random.normal(0, 1, size = volume.shape)
    out[volume == 0] = out_random[volume == 0]
    return out

def itensity_normalize_4d_array(input_array):
    num_channels = input_array.shape[3]
    output_array = input_array
    for i in range(num_channels):
        temp_array = input_array[:,:,:,i]
        output_array[:,:,:,i] = itensity_normalize_3d_volume(temp_array)
    return [output_array]
