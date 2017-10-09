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
class nonlinear_fitter:
    def __init__(self, fitter, *, trainable_vars=None, method=None, doc=None, dim=None, extra_vars=None):
        self.fitter = fitter
        self.trainable_vars = trainable_vars
        self.method = method
        self.__doc = doc
        self.dim = dim
        self.extra_vars = ({} if extra_vars is None else extra_vars)
    def __reps__(self):
        print('dimension: ', self.dim)
        print('approx method:', self.method)
        print('trainable variables:')
        print(self.trainable_vars)
        print('extra variables:')
        print(self.extra_vars)
    def __str__(self):
        self.__reps__()
        print(self.__doc)
    def __call__(self, inputs, **kw):
        return self.fitter(inputs, **kw)

def preprocess_for_fitter(fitter):
    """
    对 fitter 的包装, 作用是调整其 inputs,outputs 的 shape
    Args: 
        fitter, callable
    Return: 
        F, a decoration of fitter
    About fitter:
        Args: inputs, tensor, N x m
        Return: outputs, tensor, N or N x k
    About F:
        Args: inputs
            inputs 是一个 N1 x N2 x ... Nt x m 的 tensor, 其中 m,t>0, inputs 表示一系列 m 维点的坐标,
            其中 N1,N2,...,Nt 之一可以是 None(例如placeholder情形).
        Return: outputs
            outputs 满足如下情形之一:
                case1. outputs shape = N1 x N2 x ... x Nt
                case2. outputs shape = N1 x N2 x ... x Nt x k & k>1
    Note:
        F 内部调用 fitter 并对 inputs, outputs reshape实现对fitter的包装.
        对上面的 case1, outputs = F(inputs1) 相当于 
            >> inputs = tf.reshape(inputs1, [N1 x ... x Nt, m])
            >> outputs = tf.reshape(fitter(inputs), [N1,...,Nt])
        对于 case2 做相同理解.
    """
    def F(inputs, **kw):
        tensor_shape = inputs.get_shape()
        assert len(tensor_shape) > 1
        s = []
        for i in range(len(tensor_shape)):
            j = (-1 if tensor_shape[i].value is None else tensor_shape[i].value)
            s.append(j)
        if len(s) > 2:
            inputs = tf.reshape(inputs, [-1,s[-1]])
        outputs_tmp = fitter(inputs, **kw)
        if len(outputs_tmp.get_shape()) <= 1 or outputs_tmp.get_shape()[-1] == 1: # 前者默认输出维度为1
            outputs = tf.reshape(outputs_tmp, s[:-1])
        else: 
            outputs = tf.reshape(outputs_tmp, [*s[:-1],outputs_tmp.get_shape()[-1].value])
        return outputs
    return F

#%%
def LagrangeInterp(inputs, *, interp_coe, interp_dim, interp_order, mesh_bound, mesh_size, mesh_delta):
    """
    R^m中矩形网格Lagrange插值
    Args: 
        inputs: tensor, shape = [N,m], 其中N可以是None
        interp_dim: m 就是 interp_dim
        interp_coe: tensor, shape = mesh_size*interp_order+1
        interp_order: Lagrange插值次数
        mesh_bound: ndarray, R^m中各维度bound, shape = [2,m]
        mesh_size: ndarray, R^m中各坐标方向网格数, shape = [m,]
        mesh_delta: ndarray, R^m中各坐标方向网格尺度, shape = [m,], 可由mesh_bound,mesh_size推断出
    Return:
        outputs: tensor, shape = [N,], 根据interp_coe对inputs各点的插值结果
    """
    ele2coe = zeros([interp_order+1,]*interp_dim+[interp_dim,], dtype=int32) # 构造ele2coe: ele2coe[a_1,a_2,...,a_m] = array([a_1,a_2,...,a_m])
    perm = arange(interp_dim+1, dtype=int32)
    perm[1:] = arange(interp_dim, dtype=int32)
    perm[0] = interp_dim
    ele2coe = transpose(ele2coe, axes=perm)
    for i in range(interp_dim):
        perm = arange(interp_dim+1, dtype=int32)
        perm[1] = i+1
        perm[i+1] = 1
        ele2coe = transpose(ele2coe, axes=perm)
        for j in range(interp_order+1):
            ele2coe[i,j] = j
        ele2coe = transpose(ele2coe, axes=perm)
    perm = arange(interp_dim+1, dtype=int32)
    perm[:interp_dim] = arange(1, interp_dim+1, dtype=int32)
    perm[interp_dim] = 0
    ele2coe = transpose(ele2coe, axes=perm)
    #%%
    sample_point_shift = (inputs-mesh_bound[newaxis,0,:])/mesh_delta[newaxis]
    element_indices = tf.floor(sample_point_shift) # inputs各点所在单元, element_indices.shape = [N,m], i.e. inputs.shape
    element_indices = tf.nn.relu(element_indices) # 考虑到inputs可能超出mesh_bound, 需要把element_indices移至正常位置
    element_indices = mesh_size[newaxis,:]-1-tf.nn.relu(mesh_size[newaxis,:]-1-element_indices)
    # 
    sample_point_shift = sample_point_shift-element_indices
    ##
    ##
    element_indices = tf.cast(element_indices, dtype=tf.int32)
    interp_coe_indices = tf.reshape(element_indices*interp_order, [-1,]+[1,]*interp_dim+[interp_dim,])+ele2coe[newaxis]
    interp_coe_resp = tf.gather_nd(interp_coe, interp_coe_indices) # interp_coe_resp[n,a_1,a_2,...,a_m]表示inputs第n(0<=n<=N)个点坐在单元第[a_1,...,a_m]个系数, 0<=a_i<=interp_order, 每个点对应(interp_order+1)^m个系数.
    #%%
    base_function = ndarray(shape=[interp_dim, interp_order+1], dtype=np.object)
    for i in range(interp_dim):
        M = sample_point_shift[:,i,tf.newaxis]-(arange(interp_order+1)/interp_order)[newaxis,:]
        for j in range(interp_order+1):
            base_function[i,j] = (
                    tf.reduce_prod(M[:,:j], axis=1)*tf.reduce_prod(M[:,j+1:], axis=1)
                    *(interp_order**interp_order/factorial(j)/factorial(interp_order-j)*(-1)**(interp_order-j))
                  )
    TENSOR_DTYPE = precision_control.TENSOR_PRECISION()
    base = tf.constant(1, dtype=TENSOR_DTYPE)
    for i in range(interp_dim):
        base_tmp0 = [0,]*(interp_order+1)
        for j in range(interp_order+1):
            base_tmp0[j] = base_function[i,j][:,tf.newaxis]
        base_tmp1 = tf.reshape(tf.concat(base_tmp0, axis=1), [-1,interp_order+1]+[1,]*i)
        if i == 0:
            base = base_tmp1
        else:
            base = base[:,tf.newaxis]*base_tmp1
    perm = arange(interp_dim+1, dtype=int32)
    perm[1:] = arange(1,interp_dim+1)[::-1]
    base = tf.transpose(base, perm=perm)
    outputs = tf.reduce_sum(interp_coe_resp*base, axis=list(range(1,interp_dim+1)))
    return outputs
def LagrangeInterp_Fitter(*, interp_order, mesh_bound, interp_coe=None, mesh_size=None, collections=None):
    """
    基于interp_coe等输入参数构建R^m中矩形网格上Lagrange插值拟合器.
    此拟合器接收形如N1xN2x...xNtxm的inputs, 其中N1,N2,...,Nt可以有至多一个是None, t可取任意正整数.
    Args:
        interp_order: Lagrange插值次数
        mesh_bound: ndarray, R^m中各坐标方向的bound, shape = [2,m]
        interp_coe: ndarray or tensor, shape = mesh_size*interp_order+1, 当interp_coe为ndarray时,以此为初值构建一个tf.Variable作为插值系数. 若输入 interp_coe 为 None, 则由 mesh_size,interp_order 新建interp_coe.
        mesh_size: ndarray, R^m中各坐标方向网格数, shape = [m,]. 若输入 interp_coe 不为 None, 可由 interp_coe 推断出 mesh_size.
        collections: 内部生成interp_coe时额外加入的collections
    Return:
        a Lagrange interpolation fitter F,
            outputs = F(inputs), inputs.shape=[N1,N2,...,Nt,m], outputs.shape=[N1,...,Nt]
    """
    mesh_bound = array(mesh_bound)
    interp_dim = mesh_bound.shape[1]
    if mesh_size is None:
        assert (not interp_coe is None)
    if interp_coe is None:
        interp_coe = random.randn(*list(array(mesh_size, dtype=int32)*interp_order+1))
    TENSOR_DTYPE = precision_control.TENSOR_PRECISION()
    if isinstance(interp_coe, ndarray):
        interp_coe = tf.Variable(interp_coe, dtype=TENSOR_DTYPE, name='interp_coe', collections=collections)
    mesh_size = []
    for i in range(interp_dim):
        mesh_size.append((interp_coe.get_shape()[i].value-1)//interp_order)
    mesh_size = array(mesh_size)
    mesh_delta = (mesh_bound[1]-mesh_bound[0])/mesh_size
    fitter = lambda inputs: LagrangeInterp(inputs, interp_coe=interp_coe, interp_dim=interp_dim, interp_order=interp_order, mesh_bound=mesh_bound, mesh_size=mesh_size, mesh_delta=mesh_delta)
    extra_vars = {'interp_order': interp_order, 'mesh_bound': mesh_bound, 
            'mesh_size': mesh_size, 'mesh_delta': mesh_delta}
    return nonlinear_fitter(preprocess_for_fitter(fitter), trainable_vars=interp_coe, method='Lagrange Interpolation', doc=LagrangeInterp_Fitter.__doc__, dim=interp_dim, extra_vars=extra_vars)

#%%
def Kernel_Fitter(kernel_type_func, w, Train_Point, **kw):
    TENSOR_DTYPE = precision_control.TENSOR_PRECISION()
    s0 = w.get_shape()
    s1 = Train_Point.get_shape()
    assert len(s0) == len(s1)
    assert len(s0) > 1
    for i in range(len(s0)-1):
        assert s0[i].value == s1[i].value
    Train_Point = tf.reshape(Train_Point, [-1, s1[-1].value])
    w = tf.reshape(w, [-1, s0[-1].value])
    fitter = lambda inputs: kernel_type_func(inputs, w=w, Train_Point=Train_Point, **kw)
    return preprocess_for_fitter(fitter)
def SVR_RBF(inputs, w, Train_Point, *, b=0, gamma=1):
    """
    RBF kernel SVR
    """
    logK = tf.reduce_sum(tf.square(Train_Point[:,tf.newaxis,:]-inputs[tf.newaxis,:,:]), axis=2)
    K = tf.exp(-gamma*logK)
    w = tf.reshape(w, [1,-1])
    outputs = (w @ K + b)[0,:]
    return outputs
def SVR_RBF_Fitter(w, Train_Point, *, b, gamma, trainable=False, collections=None):
    """
    Args:
        w: tensor, N1 x N2 x ... x Nt x 1
        Train_Point: tensor or ndarray, N1 x N2 x ... x Nt x m
    Return:
        A RBF kernel SVR fitter
    """
    TENSOR_DTYPE =  precision_control.TENSOR_PRECISION()
    b = (tf.reshape(b, []) if not isinstance(b, ndarray) else tf.Variable(b, dtype=TENSOR_DTYPE, collections=collections))
    if isinstance(Train_Point, ndarray):
        Train_Point = tf.Variable(Train_Point, dtype=TENSOR_DTYPE, trainable=trainable, collections=collections)
    fitter = Kernel_Fitter(SVR_RBF, w=w, Train_Point=Train_Point, b=b, gamma=gamma)
    extra_vars = {'Train_Point': Train_Point, 'gamma': gamma}
    trainable_vars = {'w':w, 'b':b}
    all_trainable_vars = tf.trainable_variables()
    if Train_Point in all_trainable_vars:
        trainable_vars['Train_Point'] = Train_Point
    if gamma in all_trainable_vars:
        trainable_vars['gamma'] = gamma
    dim = Train_Point.get_shape()[-1].value
    return nonlinear_fitter(fitter, trainable_vars=trainable_vars, method='RBF kernel SVR', doc=SVR_RBF_Fitter.__doc__, dim=dim, extra_vars=extra_vars)
#%% TODO
# SVR_Compactsupport
# SVR_Compactsupport_Fitter
#%%
def KNN(inputs, w, Train_Point, *, approx_order=0):
    """
    1-nearest neiborhood regression 
    """
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
def KNN_Fitter(w, Train_Point, *, approx_order=0, trainable=False, collections=None):
    """
    Return an 1-nearest neiborhood fitter in R^m
    Args:
        w: tensor, N1 x N2 x ... x Nt x (1+approx_order*m)
        Train_Point: tensor or ndarray, N1 x N2 x ... x Nt x m
        approx_order: 0 or 1, 1-nearest neiborhood 局部常数回归或局部线性回归
    Return:
        an 1-nearest neiborhood fitter
    """
    TENSOR_DTYPE =  precision_control.TENSOR_PRECISION()
    if isinstance(Train_Point, ndarray):
        Train_Point = tf.Variable(Train_Point, dtype=TENSOR_DTYPE, trainable=trainable, collections=collections)
    fitter = Kernel_Fitter(KNN, w, Train_Point, approx_order=approx_order)
    extra_vars = {'Train_Point': Train_Point, 'approx_order': approx_order}
    trainable_vars = {'w':w}
    if Train_Point in tf.trainable_variables():
        trainable_vars['Train_Point'] = Train_Point
    dim = Train_Point.get_shape()[-1].value
    return nonlinear_fitter(fitter, trainable_vars=trainable_vars, method='1-nearest neiborhood regression', doc=KNN_Fitter.__doc__, dim=dim, extra_vars=extra_vars)
#%%
def batch_norm_wrapper(nonlinear_rank, collections=None): # will be removed from nonlinear_approx.py
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
def NN_Fitter(dim, widths, *, activation_func=tf.nn.relu, output_dim=None, collections=None):
    """
    Return an full connected neural network fitter in R^dim
    Args:
        dim, widths(a list), output_dim 是输入层、中间各隐层、输出层宽度. 当output_dim为None时, widths[-1]作为输出层宽度.
    Return:
        a NN fitter
    """
    TENSOR_DTYPE = precision_control.TENSOR_PRECISION()
    widths = [dim,]+widths+([] if output_dim is None else [output_dim,])
    w = []
    b = []
    for i in range(len(widths)-1):
        w_init = tf.truncated_normal(shape=widths[i:i+2], mean=0, stddev=1/sqrt(widths[i]), dtype=TENSOR_DTYPE)
        w.append(
                tf.Variable(w_init, dtype=TENSOR_DTYPE, collections=collections)
                )
        b_init = tf.random_uniform(shape=[1,widths[i+1]], dtype=TENSOR_DTYPE)
        b.append(
                tf.Variable(b_init, dtype=TENSOR_DTYPE, collections=collections)
                )
    scale = tf.Variable(ones([1,widths[-1]]), name='scale', dtype=TENSOR_DTYPE, collections=collections)
    bias = tf.Variable(random.rand(1,widths[-1]), name='bias', dtype=TENSOR_DTYPE, collections=collections)
    def fitter_(inputs):
        """
        a NN fitter
        """
        outputs = inputs
        for i in range(len(widths)-1):
            outputs = activation_func(outputs @ w[i] + b[i])
        output_shape = [-1,widths[-1]] if widths[-1]>1 else [-1,]
        outputs = tf.reshape(scale*outputs+bias, output_shape)
        return outputs 
    fitter = preprocess_for_fitter(fitter_)
    trainable_vars = {'w':w, 'b':b, 'scale':scale, 'bias':bias}
    extra_vars = None
    return nonlinear_fitter(fitter, trainable_vars=trainable_vars, method='NN', doc=NN_Fitter.__doc__, dim=dim, extra_vars=None)

#%%

