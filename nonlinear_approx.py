#!/usr/bin/env python
# -*- coding: utf-8 -*-
#%%
from numpy import *
import numpy as np
import tensorflow as tf
from scipy.misc import factorial
from . import precision_control
__all__ = ['LagrangeInterp', 'LagrangeInterp_Fitter', 
        'SVR_RBF', 'SVR_RBF_Fitter', 
        'NN_Fitter', 'KNN_Fitter']

#%%
def preprocess_for_fitter(fitter,variables):
    def F(inputs, **kw):
        """
        Args: inputs
            inputs 是一个 N1 x N2 x ... Nt x m 的 tensor.
            其中 N1,N2,...,Nt 之一可以是 None(例如placeholder情形).
        Returns: outputs
            outputs 满足如下情形之一:
                case1. outputs shape = N1 x N2 x ... x Nt
                case2. outputs shape = N1 x N2 x ... x Nt x k & k>1
                case3. outputs 是一个 list, outputs 中的每个元素属于 case1 or case2 之一
        Note:
            F 通过调用装饰函数 preprocess_for_fitter 的参数 fitter 实现,
            fitter 相当于 F 只接收 N x m tensor的情形.
            对上面的 case1, outputs = F(inputs1) 相当于 
                >> inputs = tf.reshape(inputs1, [N1 x ... x Nt, m])
                >> outputs = tf.reshape(fitter(inputs), [N1,...,Nt])
            对于 case2,case3 做相同理解.
        """
        tensor_shape = inputs.get_shape()
        s = []
        for i in range(len(tensor_shape)):
            j = (-1 if tensor_shape[i].value is None else tensor_shape[i].value)
            s.append(j)
        if len(s) > 2:
            inputs = tf.reshape(inputs, [-1,s[-1]])
        infe_tmp = fitter(inputs, **kw)
        infe = []
        if len(s) > 2:
            if not isinstance(infe_tmp, list):
                if len(infe_tmp.get_shape()) == 1:
                    infe = tf.reshape(infe_tmp, s[:-1])
                else:
                    infe = tf.reshape(infe_tmp, [*s[:-1],infe_tmp.get_shape()[1].value])
            else:
                for i in range(len(infe_tmp)):
                    if len(infe_tmp[i].get_shape()) == 1:
                        infe.append(tf.reshape(infe_tmp[i], s[:-1]))
                    else:
                        infe = tf.reshape(infe_tmp, [*s[:-1],infe_tmp[i].get_shape()[1].value])
        else:
            infe = infe_tmp
        return infe
    F.__doc__ = F.__doc__ + 'docstring for fitter:\n' + fitter.__doc__
    F.variables = variables
    return F

#%%
def LagrangeInterp(inputs, interp_coe, nonlinear_rank, interp_order, mesh_bound, mesh_size, mesh_delta):
    ele2coe = zeros([interp_order+1,]*nonlinear_rank+[nonlinear_rank,], dtype=int32)
    perm = arange(nonlinear_rank+1, dtype=int32)
    perm[1:] = arange(nonlinear_rank, dtype=int32)
    perm[0] = nonlinear_rank
    ele2coe = transpose(ele2coe, axes=perm)
    for i in range(nonlinear_rank):
        perm = arange(nonlinear_rank+1, dtype=int32)
        perm[1] = i+1
        perm[i+1] = 1
        ele2coe = transpose(ele2coe, axes=perm)
        for j in range(interp_order+1):
            ele2coe[i,j] = j
        ele2coe = transpose(ele2coe, axes=perm)
    perm = arange(nonlinear_rank+1, dtype=int32)
    perm[:nonlinear_rank] = arange(1, nonlinear_rank+1, dtype=int32)
    perm[nonlinear_rank] = 0
    ele2coe = transpose(ele2coe, axes=perm)
    #%%
    sample_point_shift = (inputs-mesh_bound[newaxis,0,:])/mesh_delta[newaxis]
    element_indices = tf.floor(sample_point_shift)
    # shift element_indices to valid indices, i.e. 0:mesh_size resp.
    element_indices = tf.nn.relu(element_indices)
    element_indices = mesh_size[newaxis,:]-1-tf.nn.relu(mesh_size[newaxis,:]-1-element_indices)
    # 
    sample_point_shift = sample_point_shift-element_indices
    ##
    ##
    element_indices = tf.cast(element_indices, dtype=tf.int32)
    interp_coe_indices = tf.reshape(element_indices*interp_order, [-1,]+[1,]*nonlinear_rank+[nonlinear_rank,])+ele2coe[newaxis]
    interp_coe_resp = tf.gather_nd(interp_coe, interp_coe_indices)
    #%%
    base_function = ndarray(shape=[nonlinear_rank, interp_order+1], dtype=np.object)
    for i in range(nonlinear_rank):
        M = sample_point_shift[:,i,tf.newaxis]-(arange(interp_order+1)/interp_order)[newaxis,:]
        for j in range(interp_order+1):
            base_function[i,j] = (
                    tf.reduce_prod(M[:,:j], axis=1)*tf.reduce_prod(M[:,j+1:], axis=1)
                    *(interp_order**interp_order/factorial(j)/factorial(interp_order-j)*(-1)**(interp_order-j))
                  )
    TENSOR_DTYPE = precision_control.TENSOR_PRECISION()
    base = tf.constant(1, dtype=TENSOR_DTYPE)
    for i in range(nonlinear_rank):
        base_tmp0 = [0,]*(interp_order+1)
        for j in range(interp_order+1):
            base_tmp0[j] = base_function[i,j][:,tf.newaxis]
        base_tmp1 = tf.reshape(tf.concat(base_tmp0, axis=1), [-1,interp_order+1]+[1,]*i)
        if i == 0:
            base = base_tmp1
        else:
            base = base[:,tf.newaxis]*base_tmp1
    perm = arange(nonlinear_rank+1, dtype=int32)
    perm[1:] = arange(1,nonlinear_rank+1)[::-1]
    base = tf.transpose(base, perm=perm)
    inference = tf.reduce_sum(interp_coe_resp*base, axis=list(range(1,nonlinear_rank+1)))
    return inference
    ##base_tmp = base_function[nonlinear_rank-1]
    ##for i in range(nonlinear_rank-2,-1,-1):
    ##    base_tmp1 = [0,]*(interp_order+1)
    ##    base_tmp2 = [0,]*(interp_order+1)
    ##    perm = arange(nonlinear_rank-i, dtype=int32)
    ##    perm[0] = 1
    ##    perm[1] = 0
    ##    #perm[0:-1] = arange(nonlinear_rank-i-1, dtype=int32)+1
    ##    #perm[-1] = 0
    ##    for j in range(interp_order+1):
    ##        base_tmp1[j] = tf.reshape(base_function[i,j], [-1,]+[1,]*(nonlinear_rank-i-2))*base_tmp
    ##        base_tmp2[j] = tf.transpose(base_tmp1[j], perm=perm)
    ##    base_tmp = base_tmp2
    ##for i in range(interp_order+1):
    ##    base_tmp[i] = base_tmp[i][:,tf.newaxis]
    ##base = tf.concat(base_tmp, axis=1)
    ##inference = tf.reduce_sum(interp_coe_resp*base, axis=list(range(1,nonlinear_rank+1)))
    ##return inference
def LagrangeInterp_Fitter(interp_coe, interp_order, mesh_bound, mesh_size, collections=None):
    """
    Lagrange interpolation in R^m
    Args: 
        interp_coe: ndarray or tensor, interp_coe.shape = [N1, N2, ... , Nt, (m^interp_order)]
        interp_order: Lagrange interpolation order, a natural number
        mesh_bound: ndarray, mesh_bound.shape = [2,m]
        mesh_size: ndarray, mesh_size.shape = [m,]
        mesh_delta: ndarray, mesh_delta.shape = [m,]
    Return:
        A LagrangeInterp fitter
    """
    mesh_bound = array(mesh_bound)
    mesh_size = array(mesh_size)
    mesh_delta = (mesh_bound[1]-mesh_bound[0])/mesh_size
    TENSOR_DTYPE = precision_control.TENSOR_PRECISION()
    nonlinear_rank = mesh_size.shape[0]
    if isinstance(interp_coe, ndarray):
        interp_coe = tf.Variable(interp_coe, dtype=TENSOR_DTYPE, name='interp_coe', collections=collections)
    fitter = lambda inputs: LagrangeInterp(inputs, interp_coe, nonlinear_rank, interp_order, mesh_bound, mesh_size, mesh_delta)
    fitter.__doc__ = 'Lagrange Interpolation'
    return preprocess_for_fitter(fitter, interp_coe)

#%%
def Kernel_Fitter(kernel_type_func, w, Train_Point, trainable=False, collections=None, **kw):
    TENSOR_DTYPE = precision_control.TENSOR_PRECISION()
    if isinstance(Train_Point, ndarray):
        Train_Point = tf.Variable(Train_Point, dtype=TENSOR_DTYPE, trainable=trainable, collections=collections)
    s0 = w.get_shape()
    s1 = Train_Point.get_shape()
    assert len(s0) == len(s1)
    assert len(s0) > 1
    for i in range(len(s0)-1):
        assert s0[i].value == s1[i].value
    Train_Point = tf.reshape(Train_Point, [-1, s1[-1].value])
    w = tf.reshape(w, [-1, s0[-1].value])
    fitter = lambda inputs: kernel_type_func(inputs, w, Train_Point, **kw)
    fitter.__doc__ = kernel_type_func.__doc__
    variables = kw
    variables['w'] = w
    variables['Train_Point'] = Train_Point
    return preprocess_for_fitter(fitter, variables)
def SVR_RBF(inputs, w, Train_Point, *, b=0, gamma=1, **kw):
    """
    SVR approximation via RBF kernel
    """
    logK = tf.reduce_sum(tf.square(Train_Point[:,tf.newaxis,:]-inputs[tf.newaxis,:,:]), axis=2)
    K = tf.exp(-gamma*logK)
    w = tf.reshape(w, [1,-1])
    inference = (w @ K + b)[0,:]
    return inference
def SVR_RBF_Fitter(w, Train_Point, b, gamma, trainable=False, collections=None):
    """
    Args:
        w: tensor, N1 x N2 x ... x Nt x 1
        Train_Point: tensor or ndarray, N1 x N2 x ... x Nt x m
    Return:
        A SVR fitter via RBF kernel
    """
    TENSOR_DTYPE =  precision_control.TENSOR_PRECISION()
    b = (tf.reshape(b, []) if not isinstance(b, ndarray) else tf.Variable(b, dtype=TENSOR_DTYPE, collections=collections))
    return Kernel_Fitter(SVR_RBF, w, Train_Point, trainable=trainable, collections=collections, b=b, gamma=gamma)
SVR_Compactsupport = SVR_RBF
SVR_Compactsupport_Fitter = SVR_RBF_Fitter
#%%
def KNN(inputs, w, Train_Point, approx_order=0, *args, **kw):
    """
    1-nearest neiborhood approximation 
    """
    r = Train_Point.get_shape()[-1].value
    diff = Train_Point[:,tf.newaxis,:]-inputs[tf.newaxis,:,:]
    distance = tf.reduce_sum(tf.square(diff), axis=2)
    select = tf.reshape(tf.cast(tf.argmin(distance, axis=0), dtype=tf.int32), [-1,1])
    if approx_order == 0:
        return tf.gather_nd(w[:,0], select)
    else:
        w = tf.gather_nd(w, select)
        NearestPoint = tf.gather_nd(Train_Point, select)
        diff = NearestPoint-inputs
        return w[:,0]+ tf.reduce_sum(w[:,1:]*diff, axis=1)
def KNN_Fitter(w, Train_Point, approx_order=0, trainable=False, collections=None):
    """
    Return an 1-nearest neiborhood fitter in R^m
    Args:
        w: tensor, N1 x N2 x ... x Nt x (1+approx_order*m)
        Train_Point: tensor or ndarray, N1 x N2 x ... x Nt x m
    Return:
        an 1-nearest neiborhood fitter
    """
    return Kernel_Fitter(KNN, w, Train_Point, trainable=trainable, collections=collections, approx_order=approx_order)
#%%
def batch_norm_wrapper(nonlinear_rank, collections=None):
    TENSOR_DTYPE = precision_control.TENSOR_PRECISION()
    scale = tf.Variable(tf.ones([nonlinear_rank], dtype=TENSOR_DTYPE), 
            name='BN_scale', trainable=False, dtype=TENSOR_DTYPE, collections=collections)
    beta = tf.Variable(tf.zeros([nonlinear_rank], dtype=TENSOR_DTYPE), 
            name='BN_beta', trainable=False, dtype=TENSOR_DTYPE, collections=collections)
    pop_mean = tf.Variable(tf.zeros([nonlinear_rank], dtype=TENSOR_DTYPE), 
            name='pop_mean', trainable=False, dtype=TENSOR_DTYPE, collections=collections)
    pop_var = tf.Variable(tf.ones([nonlinear_rank], dtype=TENSOR_DTYPE), 
            name='pop_var', trainable=False, dtype=TENSOR_DTYPE, collections=collections)
    def batch_norm(train_inputs, test_inputs=None, epsilon=1e-4, decay=0.95):
        assert train_inputs.get_shape()[-1].value == nonlinear_rank
        batch_mean, batch_var = tf.nn.moments(train_inputs,[0])
        train_mean = tf.assign(pop_mean,
                               pop_mean * decay + batch_mean * (1 - decay))
        train_var = tf.assign(pop_var,
                              pop_var * decay + batch_var * (1 - decay))
        with tf.control_dependencies([train_mean, train_var]):
            train_outputs = tf.nn.batch_normalization(train_inputs,
                batch_mean, batch_var, beta, scale, epsilon)

        test_inputs = (train_inputs if test_inputs is None else test_inputs)
        test_outputs = tf.nn.batch_normalization(test_inputs,
            pop_mean, pop_var, beta, scale, epsilon)
        return train_outputs, test_outputs
    return batch_norm
def NN_Fitter(nonlinear_rank, widths, activation_func=tf.nn.relu, with_bn=True, collections=None, 
        providebn=None, isoutputbn=False):
    """
    Return an full connected neural network fitter in R^nonlinear_rank
    Args:
        nonlinear_rank: R^nonlinear_rank
        widths: 一个list, widths[:-1]是各隐层宽度,widths[-1]是输出层输出数
    Return:
        a NN fitter
    """
    TENSOR_DTYPE = precision_control.TENSOR_PRECISION()
    widths = [nonlinear_rank,]+widths
    w = []
    b = []
    isprovidebn = (False if providebn is None else True)
    bn = ([] if providebn is None else providebn)
    if with_bn and not isprovidebn:
        bn_wrapper = batch_norm_wrapper(widths[i], collections=collections)
        bn.append(bn_wrapper)
    for i in range(len(widths)-1):
        w_init = tf.truncated_normal(shape=widths[i:i+2], mean=0, stddev=1/widths[i], dtype=TENSOR_DTYPE)
        w.append(
                tf.Variable(w_init, dtype=TENSOR_DTYPE, collections=collections)
                )
        if not with_bn:
            b_init = tf.random_uniform(shape=[1,widths[i+1]], dtype=TENSOR_DTYPE)/widths[i+1]
            b.append(
                    tf.Variable(b_init, dtype=TENSOR_DTYPE, collections=collections)
                    )
        elif not isprovidebn:
            bn_wrapper = batch_norm_wrapper(widths[i+1], collections=collections)
            bn.append(bn_wrapper)
    scale = tf.Variable(random.rand(1,widths[-1]), name='scale', dtype=TENSOR_DTYPE, collections=collections)
    bias = tf.Variable(random.rand(1,widths[-1]), name='bias', dtype=TENSOR_DTYPE, collections=collections)
    variables = {}
    variables['w'] = w
    variables['scale'] = scale
    variables['bias'] = bias
    if not with_bn:
        variables['b'] = b
    def fitter(inputs, epsilon=1e-4, decay=0.95):
        """
        a NN fitter
        """
        if with_bn:
            train_outputs,test_outputs = bn[0](inputs)
        else:
            outputs = inputs
        for i in range(len(widths)-1):
            if with_bn:
                train_outputs, test_outputs = bn[i+1](train_outputs @ w[i], test_outputs @ w[i], collections=collections)
                train_outputs = activation_func(train_outputs)
                test_outputs = activation_func(test_outputs)
            else:
                outputs = activation_func(outputs @ w[i] + b[i])
        output_shape = [-1,widths[-1]] if widths[-1]>1 else [-1,]
        if with_bn:
            train_infe = tf.reshape(scale*train_outputs+bias, output_shape)
            test_infe = tf.reshape(scale*test_outputs+bias, output_shape)
            return [train_infe, test_infe]
        else:
            infe = tf.reshape(scale*outputs+bias, output_shape)
            return infe 
        if isoutputbn:
            return preprocess_for_fitter(fitter, variables), bn
        else:
            return preprocess_for_fitter(fitter, variables)

#%%

