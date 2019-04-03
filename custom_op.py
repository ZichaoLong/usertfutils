#!/usr/bin/env python
# -*- coding: utf-8 -*-
#%%
import tensorflow as tf
from numpy import *
from scipy.signal import correlate2d
from . import precision_control
__all__ = ['py_func', 'tfconv2d']
#%%
def py_func(func, inp, name=None, grad=None): # make out what this function do, and you will know all
    rnd_name = 'PyFuncGrad' + str(random.randint(2**32))
    tf.RegisterGradient(rnd_name)(grad)
    g = tf.get_default_graph()
    with g.gradient_override_map({"PyFunc": rnd_name}):
        return tf.py_func(func, inp, precision_control.TENSOR_PRECISION(), stateful=True, name=name)
#%% pad
def user_pad(tensor, paddings, mode='CONSTANT', name=None, constant_value=0):
    if mode in ['CONSTANT', 'SYMMETRIC', 'REFLECT']:
        if constant_value == 0 and mode == 'CONSTANT':
            return tf.add(tf.pad(tensor-constant_value, paddings=paddings, mode=mode),constant_value,name=name)
        else:
            return tf.pad(tensor, paddings=paddings, mode=mode, name=name)
    else:
        assert mode == 'WRAP'
        i = 0
        indx = []
        for a,b in paddings:
            slice0 = slice(-a,None)
            slice1 = slice(0,b)
            if a+b > 0:
                tensor = tf.concat([tensor[indx+[slice0,]],tensor,tensor[indx+[slice1,]]], axis=i)
            indx = indx+[slice(None),]
            i += 1
        return (tensor if name is None else tf.identity(tensor, name=name))
#%%
#conv2d
def user_ndarray_corr2d(x, f, mode='full'):
    """
    x: [batch_size, height, width, in_channels]
    f: [ker_height, ker_width, in_channels, out_channels]
    mode: full,valid
    """
    #print(mode)
    batch_size = x.shape[0]
    in_channels = x.shape[3]
    out_channels = f.shape[3]
    h = x.shape[1]
    w = x.shape[2]
    ker_h = f.shape[0]
    ker_w = f.shape[1]
    if mode == 'full':
        y = zeros([batch_size, h+ker_h-1, w+ker_w-1, out_channels])
    elif mode == 'valid':
        y = zeros([batch_size, h-ker_h+1, w-ker_w+1, out_channels])
    for k in range(batch_size):
        for i in range(out_channels):
            for j in range(in_channels):
                y[k,:,:,i] += correlate2d(x[k,:,:,j], f[:,:,j,i], mode=mode)
    return y.astype(precision_control.NUMPY_PRECISION())
user_ndarray_corr2d_full = lambda x,f: user_ndarray_corr2d(x, f, mode='full')
user_ndarray_corr2d_valid = lambda x,f: user_ndarray_corr2d(x, f, mode='valid')
def custom_corr2d_full_grad(op, grad):
    return [
            tf.py_func(
                user_ndarray_corr2d_valid, 
                [
                    grad,
                    tf.transpose(op.inputs[1], [0,1,3,2])[::-1,::-1]
                    ], Tout=precision_control.TENSOR_PRECISION()
                ),
            tf.transpose(
                tf.py_func(
                    user_ndarray_corr2d_valid,
                    [
                        tf.transpose(grad, [3,1,2,0]),
                        tf.transpose(op.inputs[0], [1,2,0,3])
                        ], Tout=precision_control.TENSOR_PRECISION()
                    ), [1,2,3,0]
                )[::-1,::-1]
            ]
def custom_corr2d_valid_grad(op, grad):
    return [
            tf.py_func(
                user_ndarray_corr2d_full, 
                [
                    grad,
                    tf.transpose(op.inputs[1], [0,1,3,2])[::-1,::-1]
                    ], Tout=precision_control.TENSOR_PRECISION()
                ),
            tf.transpose(
                tf.py_func(
                    user_ndarray_corr2d_valid,
                    [
                        tf.transpose(op.inputs[0], [3,1,2,0]),
                        tf.transpose(grad, [1,2,0,3])
                        ], Tout=precision_control.TENSOR_PRECISION()
                    ), [1,2,0,3]
                )
            ]
def tfconv2d(x, f, name=None, boundary=None, userdef_tfconv2d=None, constant_value=0):
    """
    if boundary is None, then no padding for conv2d, 
    else boudary \in {['symmetric', 'constant', 'reflect', 'wrap'] and their upper}
    Usage:
        dtype = tf.float32
        x = tf.Variable(random.randn(10,100,100,2), dtype=dtype)
        f = tf.Variable(random.randn(5,5,2,9), dtype=dtype)
        b = tfconv2d(x, f, boundary='symmetric', userdef_tfconv2d=True)
        c = tfconv2d(x, f, boundary='symmetric', userdef_tfconv2d=False)
    """
    assert x.shape[1].value >= f.shape[1].value
    assert x.shape[2].value >= f.shape[2].value
    if userdef_tfconv2d is None:
        userdef_tfconv2d = (True if precision_control.TENSOR_PRECISION() is tf.float64 else False)
    if (boundary in ['constant', 'CONSTANT']) and (constant_value == 0): # 此情形直接调用卷积op,避免先pad再卷积的多余操作
        if not userdef_tfconv2d:
            return tf.nn.conv2d(x, f, name=name, strides=[1,1,1,1], padding='SAME')
        else:
            return tf.identity(tf.nn.conv3d(x[:,:,:,tf.newaxis,:],f[:,:,tf.newaxis,:,:], strides=[1,1,1,1,1],padding='SAME')[:,:,:,0,:], name=name)
    if not boundary is None:
        pad_top = (f.shape[0].value-1)//2
        pad_bottom = f.shape[0].value-1-pad_top
        pad_left = (f.shape[1].value-1)//2
        pad_right = f.shape[1].value-1-pad_left
        x = user_pad(x, paddings=[[0,0], [pad_top,pad_bottom], [pad_left,pad_right], [0,0]], mode=boundary.upper(), constant_value=constant_value)
    shape = [1,]*4
    shape[0] = (-1 if x.shape[0].value is None else x.shape[0].value)
    shape[1] = x.shape[1].value-f.shape[0].value+1
    shape[2] = x.shape[2].value-f.shape[1].value+1
    shape[3] = f.shape[3].value
    return (
            tf.identity(
                tf.nn.conv3d(x[:,:,:,tf.newaxis,:],f[:,:,tf.newaxis,:,:], strides=[1,1,1,1,1],padding='VALID')[:,:,:,0,:],
                name=name)
            # tf.reshape(
            #     py_func(user_ndarray_corr2d_valid, inp=[x,f], name='conv2d_py_func', grad=custom_corr2d_valid_grad),
            #     shape=shape,
            #     name=name)
            if userdef_tfconv2d
            else tf.nn.conv2d(x, f, name=name, strides=[1,1,1,1], padding='VALID')
            )

#%%
def test_tfconv2d():
    """
    compare custom conv2d operation "tfconv2d" with tf.nn.conv2d, under precision tf.float32
    x: a tensor initialized from random.randn(10,100,100,2)+1
    f: a tensor initialized from random.randn(5,5,2,9)+1
    b = tfconv2d(x, f, boundary='symmetric', userdef_tfconv2d=True)
    c = tfconv2d(x, f, boundary='symmetric', userdef_tfconv2d=False)
    lb = tf.reduce_sum(b**2)
    lc = tf.reduce_sum(c**2)
    """
    print(test_tfconv2d.__doc__)
    xx = random.randn(10,100,100,2)+1
    ff = random.randn(5,5,2,9)+1
    current_type = precision_control.TENSOR_PRECISION()
    precision_control.set_precision(tf.float32)
    x = tf.Variable(xx, dtype=precision_control.TENSOR_PRECISION())
    f = tf.Variable(ff, dtype=precision_control.TENSOR_PRECISION())
    b = tfconv2d(x, f, boundary='symmetric', userdef_tfconv2d=True)
    c = tfconv2d(x, f, boundary='symmetric', userdef_tfconv2d=False)
    lb = tf.reduce_sum(b**2)
    lc = tf.reduce_sum(c**2)
    from .optimizer_tools import objfunc_wrapper
    lb_wrap = objfunc_wrapper(lb)
    lc_wrap = objfunc_wrapper(lc)
    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()
    import time
    startt = time.time()
    b.eval()
    print('b.eval() elapsed time ', time.time()-startt)
    sstartt = time.time()
    c.eval()
    print('c.eval() elapsed time ', time.time()-sstartt)
    sstartt = time.time()
    lb_wrap.variables_grads[0].eval()
    lb_wrap.variables_grads[1].eval()
    print('grad(lb,x)+grad(lb,f) elapsed time ', time.time()-sstartt)
    sstartt = time.time()
    lc_wrap.variables_grads[0].eval()
    lc_wrap.variables_grads[1].eval()
    print('grad(lc,x)+grad(lc,f) elapsed time ', time.time()-sstartt)
    print('abs(b-c)/abs(c): ',
            (
                abs(b.eval()-c.eval())
                /
                abs(c.eval())
                ).mean()
            )
    print('mean(|grad(lb,x)-grad(lc,x)|/|grad(lc,x)|): ',
            (
                abs(lb_wrap.variables_grads[0].eval()-lc_wrap.variables_grads[0].eval())
                /
                abs(lc_wrap.variables_grads[0].eval())
                ).mean()
            )
    print('mean(|grad(lb,f)-grad(lc,f)|)/|grad(lc,f)|): ',
            (
                abs(lb_wrap.variables_grads[1].eval()-lc_wrap.variables_grads[1].eval())
                /
                abs(lc_wrap.variables_grads[1].eval())
                ).mean()
            )
    print('||mean(|grad(lb,x)-grad(lc,x)|, axis=0)||', linalg.norm(abs(lb_wrap.variables_grads[0].eval()-lc_wrap.variables_grads[0].eval()).mean(axis=0)))
    print('||mean(|grad(lb,f)-grad(lc,f)|, axis=3)||', linalg.norm(abs(lb_wrap.variables_grads[1].eval()-lc_wrap.variables_grads[1].eval()).mean(axis=3)))
    precision_control.set_precision(current_type)
#%%


