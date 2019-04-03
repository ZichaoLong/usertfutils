#!/usr/bin/env python
# -*- coding: utf-8 -*-
#%%
from numpy import *
import numpy as np
import tensorflow as tf
from .. import precision_control
from .. import kernel_basis_filter_tensor
__all__ = ['diff_moment_gen_2d', 'ID_bank', 'diff_bank']

#%%
def diff_moment_gen_2d(order, kernel_size, ver=0):
    """
    Args:
        order, kernel_size, ver
    """
    M = zeros([kernel_size,]*len(order))
    M[order[0],order[1]] = 1
    if ver == 0:
        return M
    M = M+nan
    for k in range(order[0]+order[1]+1):
        for j in range(k+1):
            M[j,k-j] = 0
    M[order[0],order[1]] = 1
    return M
#%%
def ID_bank(kernel_size):
    """
    Args:
        kernel_size
    Return:
        ID_filter, ID_tensor, ID_array, moment0, moment1
    """
    TENSOR_DTYPE = precision_control.TENSOR_PRECISION()
    m2f, f2m = kernel_basis_filter_tensor.switch_moment_filter_tensor(shape=[kernel_size, kernel_size])
    moment = diff_moment_gen_2d([0,0], kernel_size, ver=0)
    ID_moment = tf.Variable(moment, trainable=True, name='ID_moment', dtype=TENSOR_DTYPE)
    ID_tensor = m2f(ID_moment)[:,:,tf.newaxis,tf.newaxis]
    moment0 = {}
    moment1 = {}
    moment0[ID_moment] = diff_moment_gen_2d([0,0], kernel_size, ver=0)
    moment1[ID_moment] = diff_moment_gen_2d([0,0], kernel_size, ver=1)
    ID_filter = kernel_basis_filter_tensor.wrap_filter2d_tensor(ID_tensor)
    return ID_moment, ID_tensor, ID_filter, moment0, moment1
def _diff_bank(max_order, kernel_size):
    """
    Args:
        kernel_size, max_order
    Return:
        diff_moments, diff_tensors, diff_filters, moment0, moment1
    """
    TENSOR_DTYPE = precision_control.TENSOR_PRECISION()
    diff_moments = ndarray([max_order+1,]*2, dtype=np.object)
    diff_tensors = ndarray([max_order+1,]*2, dtype=np.object)
    diff_filters = ndarray([max_order+1,]*2, dtype=np.object)
    m2f, f2m = kernel_basis_filter_tensor.switch_moment_filter_tensor(shape=[kernel_size, kernel_size])
    for k in range(max_order+1):
        for j in range(k+1):
            moment = diff_moment_gen_2d([j,k-j], kernel_size, ver=0)
            diff_moments[j,k-j] = tf.Variable(
                    moment, trainable=True, 
                    name='order_'+str(k)+'_'+str(j)+'_'+str(k-j),
                    dtype=TENSOR_DTYPE
                    )
            diff_tensors[j,k-j] = \
                    m2f(diff_moments[j,k-j])[:,:,tf.newaxis,tf.newaxis]
            diff_filters[j,k-j] = \
                    kernel_basis_filter_tensor.wrap_filter2d_tensor(diff_tensors[j,k-j])
    moment0 = {}
    moment1 = {}
    for k in range(max_order+1):
        for j in range(k+1):
            moment = diff_moment_gen_2d([j,k-j], kernel_size, ver=0)
            moment0[diff_moments[j,k-j]] = moment
            moment = diff_moment_gen_2d([j,k-j], kernel_size, ver=1)
            moment1[diff_moments[j,k-j]] = moment
    return diff_moments, diff_tensors, diff_filters, moment0, moment1
def diff_bank(max_order, kernel_size):
    ID_moment, ID_tensor, ID_filter, A0, A1 = \
            ID_bank(kernel_size)
    diff_moments, diff_tensors, diff_filters, B0, B1 = \
            _diff_bank(max_order, kernel_size)
    moment0 = {**A0, **B0}
    moment1 = {**A1, **B1}
    return ID_moment, diff_moments, ID_tensor, diff_tensors, moment0, moment1
#%%


