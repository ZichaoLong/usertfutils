#!/usr/bin/env python
# -*- coding: utf-8 -*-
#%%
import tensorflow as tf
from numpy import *
import numpy as np
from improc_tools.kernel_basis_filter import diff_monomial_coe,diff_op_default_coe
from . import precision_control
from . import custom_op
__all__ = ['wrap_filter2d_tensor', ]
#__all__ = ['diff_monomial_coe_tensor', 'wrap_filter2d_tensor', 'diff_op_default_coe_tensor', 'diff_op_default_filter_tensor']
#%%
def diff_monomial_coe_tensor(shape=None, name=None, trainable=True, x_order=0, y_order=0, x_vers=None, y_vers=None):
    """
    a tensor version of diff_monomial_coe
    """
    ker = diff_monomial_coe(shape=shape, x_order=x_order, y_order=y_order, x_vers=x_vers, y_vers=y_vers)
    return tf.get_variable(name=name, initializer=ker, dtype=precision_control.TENSOR_PRECISION(), trainable=trainable)
def wrap_filter2d_tensor(initializer, *, name=None, trainable=True):
    """
    initializer: a ndarray for tf.get_variable or a tensor(have not got to be a tf.Variable).
    If initializer is a tensor, the parameters "name" and "trainable" would not be used
    """
    if isinstance(initializer, ndarray):
        assert initializer.ndim == 4
        ker_tensor = tf.get_variable(
                name=name,
                dtype=precision_control.TENSOR_PRECISION(), 
                trainable=trainable, 
                initializer=initializer.astype(precision_control.NUMPY_PRECISION())
                )
    else:
        ker_tensor = initializer
    def f(x, name=None, boundary=None):
        return custom_op.tfconv2d(x, ker_tensor, name=name, boundary=boundary)
    return f
def diff_op_default_coe_tensor(shape, op='laplace', name=None, trainable=True):
    initializer = diff_op_default_coe(shape, op=op)
    if initializer.ndim == 2:
        initializer = reshape(initializer, [*initializer.shape,1,1])
    ker_tensor = tf.get_variable(
            name=name,
            dtype=precision_control.TENSOR_PRECISION(), 
            trainable=trainable,
            initializer=initializer.astype(precision_control.NUMPY_PRECISION())
            )
    return ker_tensor
def diff_op_default_filter_tensor(shape, op='laplace', name=None, trainable=True):
    """
    Args:
        op: a string from ['dx','dy','laplace','grad','div']
    """
    ker_tensor = diff_op_default_coe_tensor(shape, op='laplace', name=name, trainable=trainable)
    return wrap_filter2d_tensor(initializer=ker_tensor)

#%%
def test():
    current_precision = precision_control.get_precision()
    precision_control.set_precision(tf.float64)
    ker_tensor = diff_op_default_coe_tensor(shape=[5,5], op='laplace', name='laplace')
    laplace = wrap_filter2d_tensor(initializer=ker_tensor)
    u = tf.placeholder(name='u', shape=[10, 20, 20, 1], dtype=tf.float64)
    laplace_ker = tf.Variable(reshape(array([[0,1,0],[1,-4,1],[0,1,0]]), [3,3,1,1]), name='laplace_ker_true', dtype=tf.float64)
    v = custom_op.tfconv2d(u, laplace_ker, boundary='constant')
    v1 = laplace(u, name='laplace_u', boundary='constant')
    loss = tf.reduce_mean((v-v1)*2)
    sess = tf.Session()
    sess.run([ker_tensor.initializer, laplace_ker.initializer])
    print('err: ')
    print(sess.run([loss,], feed_dict={u:random.randn(10,20,20,1)}))
    precision_control.set_precision(current_precision)
    return None


