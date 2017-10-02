#!/usr/bin/env python
# -*- coding: utf-8 -*-
#%%
import tensorflow as tf
from numpy import *
from functools import reduce
from . import precision_control
__all__ = ['objfunc_wrapper', 'array2list', 'list2array']
#%%
def array2list(x, variables_size, variables_shape, iscopy=True):
    N = sum(variables_size)
    n = len(variables_size)
    x = reshape(x, (N,))
    x = (x.copy() if iscopy else x)
    l = []
    i = 0
    for j in range(n):
        l.append(reshape(x[i:i+variables_size[j]], variables_shape[j]))
        i += variables_size[j]
    return l
def list2array(l, variables_size, variables_shape):
    N = sum(variables_size)
    n = len(variables_size)
    x = zeros(N)
    i = 0
    for j in range(n):
        x[i:i+variables_size[j]] = reshape(l[j], (variables_size[j],))
        i += variables_size[j]
    return x

# optimize utils
def tfloss_meta(loss):
    trainable_vars = tf.trainable_variables()
    grads_and_vars = zip(tf.gradients(loss, trainable_vars), trainable_vars)
    variables_grads = []
    variables = []
    for gv in grads_and_vars:
        if not gv[0] is None:
            variables_grads.append(gv[0])
            variables.append(gv[1])
    variables_size = []
    variables_shape = []
    for v in variables:
        s = tuple(map(int, v.get_shape()))
        variables_size.append((1 if len(s) == 0 else reduce(multiply, s)))
        variables_shape.append(s)
    variables_num = len(variables)
    allsize = sum(variables_size)
    with tf.name_scope('array2tensors/'+loss.name.split(':')[0]):
        array2tensors_placeholder = []
        array2tensors_assignop = []
        for j in range(variables_num):
            array2tensors_placeholder.append(
                    tf.placeholder(
                        dtype=precision_control.TENSOR_PRECISION(),
                        shape=variables_shape[j],
                        name='placeholder/'+variables[j].name.split(':')[0]
                        )
                    )
            array2tensors_assignop.append(
                    tf.assign(
                        variables[j],
                        array2tensors_placeholder[j],
                        name='assign/'+variables[j].name.split(':')[0]
                        )
                    )
        #array2tensors_group_op = tf.group(*array2tensors_assignop, name='array2tensors')
    def tfloss_info():
        pass
    tfloss_info()
    tfloss_info.allsize = allsize
    tfloss_info.variables = variables
    tfloss_info.variables_grads = variables_grads
    tfloss_info.variables_size = variables_size
    tfloss_info.variables_shape = variables_shape
    tfloss_info.array2tensors_placeholder = array2tensors_placeholder
    tfloss_info.array2tensors_assignop = array2tensors_assignop
    tfloss_info.loss = loss
    tfloss_info.xopt_list = array2list(zeros((allsize,)), variables_size, variables_shape, iscopy=False)
    return tfloss_info
#%%
class objfunc_wrapper_class(object):
    """
    objfunc_wrapper_class
    几个主要成员变量：
    loss: tensorflow compute graph中目标函数loss function所代表的tensor
    variables: 一个由tensorflow variables组成的list
    variables_grads: 最顶层loss function对variables中各变量梯度的tensor
    xopt_list: list，由objfunc_wrapper_class实例所持有的variables取值, 类比于sess.run(variables)
    其他辅助成员变量:
    allsize: variables中各变量size之和
    variables_size,variables_shape: variables中各变量size及shape
    array2tensors_placeholder, array2tensors_assignop: 用于更新variables
    __init__:
        输入: 
        father_wrapper: 可以是一个objfunc_wrapper_class实例，也可由上述tfloss_meta生成
            father_wrapper 必须有如下attributes:
            'variables', 'variables_grads', 
            'loss', 'xopt_list' 
            'allsize', 'variables_size', 'variables_shape', 
            'array2tensors_placeholder', 'array2tensors_assignop', 
        registered_vars: list, father_wrapper.variables的子集, 
            初始化即执行self.variables = registered_vars并将self的成员、成员函数更新
        xopt_list: 可选，若xopt_list == None, 则self.xopt_list各元素
            按registered_vars对应继承自father_wrapper.xopt_list
    """
    def __init__(self, father_wrapper, registered_vars, xopt_list=None):
        assert isinstance(registered_vars, list)
        assert len(registered_vars) > 0
        registered_vars_indx = []
        for v in registered_vars:
            assert v in father_wrapper.variables
            registered_vars_indx.append(father_wrapper.variables.index(v))
        self.loss = father_wrapper.loss
        self.variables = []
        self.variables_grads = []
        self.variables_size = []
        self.variables_shape = []
        self.array2tensors_placeholder = []
        self.array2tensors_assignop = []
        self.xopt_list = []
        for i in registered_vars_indx:
            self.variables.append(father_wrapper.variables[i])
            self.variables_grads.append(father_wrapper.variables_grads[i]) 
            self.variables_size.append(father_wrapper.variables_size[i])
            self.variables_shape.append(father_wrapper.variables_shape[i])
            self.array2tensors_placeholder.append(father_wrapper.array2tensors_placeholder[i])
            self.array2tensors_assignop.append(father_wrapper.array2tensors_assignop[i])
            self.xopt_list.append(father_wrapper.xopt_list[i].copy())
        self.xopt_list = (xopt_list if not xopt_list is None else self.xopt_list)
        self.xopt_dict = dict(zip(self.variables, self.xopt_list))
        self.allsize = sum(self.variables_size)
        self.variables_num = len(registered_vars)
    def update(self, feed_dict=None, sess=None):
        """
        feed_dict是一个dict，feed_dict.keys()包含于self.variables, 
        依据feed_dict更新self.variables中部分变量的值.
        feed_dict缺省时，使用self.xopt_list更新self.variables
        """
        feed_dict = (dict(zip(self.variables, self.xopt_list)) if feed_dict is None else feed_dict)
        assignop_feed_dict = {}
        assignop = []
        for v in feed_dict.keys():
            assert v in self.variables
            i = self.variables.index(v)
            assignop.append(self.array2tensors_assignop[i])
            assignop_feed_dict[self.array2tensors_placeholder[i]] = feed_dict[v]
        sess = (tf.get_default_session() if sess is None else sess)
        sess.run(assignop, feed_dict=assignop_feed_dict)
        return None
    def set_xopt(self, xopt, iscopy=True):
        """
        利用xopt更新self.xopt_list
        其中:xopt可以是
            a ndarray with shape of [self.allsize,],
            or a list of ndarrays as xopt_list,
            or a dict of ndarrays with self.variables as dict keys
            若xopt是dict，xopt.keys()包含于(可以不相等)self.variables
        """
        if isinstance(xopt, list):
            self.xopt_list = []
            if not iscopy:
                self.xopt_list = xopt
            else:
                for x in xopt:
                    self.xopt_list.append(x.copy())
        elif isinstance(xopt, dict):
            self.xopt_list = self.d2l(xopt, iscopy)
        else:
            self.xopt_list = self.a2l(xopt, iscopy)
        self.xopt_dict = dict(zip(self.variables, self.xopt_list))
        return self.xopt_list
    def a2l(self, xopt, iscopy=True):
        return array2list(xopt, self.variables_size, self.variables_shape, iscopy)
    def l2a(self, xopt_list):
        return list2array(xopt_list, self.variables_size, self.variables_shape)
    def l2d(self, xopt_list):
        return dict(zip(self.variables, xopt_list))
    def d2l(self, xopt_dict, iscopy=True):
        xopt_list = []
        for v in self.variables:
            if v in xopt_dict:
                x = (xopt_dict[v].copy() if iscopy else xopt_dict[v])
                xopt_list.append(x)
            else:
                i = self.variables.index(v)
                x = (self.xopt_list[i].copy() if iscopy else self.xopt_list[i])
                xopt_list.append(x)
        return xopt_list
    def a2d(self, xopt, iscopy=True):
        return self.l2d(self.a2l(xopt, iscopy=iscopy))
    def d2a(self, xopt_dict):
        return self.l2a(self.d2l(xopt_dict, iscopy=False))
    def set_tfenv(self, x, feed_dict=None, sess=None, iscopy=False):
        sess = (tf.get_default_session() if sess is None else sess)
        feed_dict = {} if feed_dict is None else feed_dict
        feed = dict(zip(self.variables, array2list(x, self.variables_size, self.variables_shape, iscopy)))
        feed = {**feed, **feed_dict}
        return feed, sess
    def f(self, x, feed_dict=None, sess=None):
        """
        Args:
            x: 1-d ndarray, x.size=self.allsize
            feed_dict: extra feed_dict for forward propagation, e.g. batch_size of data
            sess: if sess is None, then tf.get_default_session() will be used
        Return:
            objfunc value resp. x
        """
        feed, sess = self.set_tfenv(x, feed_dict, sess)
        return self.loss.eval(session=sess, feed_dict=feed)
    def g(self, x, feed_dict=None, sess=None):
        """
        Args:
            x: 1-d ndarray, x.size=self.allsize
            feed_dict: extra feed_dict for forward propagation, e.g. batch_size of data
            sess: if sess is None, then tf.get_default_session() will be used
        Return:
            1-d ndarray version of tf.gradient(loss, partial_trainable_variables)
        """
        feed, sess = self.set_tfenv(x, feed_dict, sess)
        grads = sess.run(self.variables_grads, feed_dict=feed)
        xgrad = list2array(grads, self.variables_size, self.variables_shape)
        return xgrad

def objfunc_wrapper(father_wrapper, registered_vars=None, xopt_list=None):
    if not hasattr(father_wrapper, 'variables'):
        # in this case, father_wrapper is just a tensor of loss function
        father_wrapper = tfloss_meta(father_wrapper)
    if registered_vars is None:
        registered_vars = father_wrapper.variables
    return objfunc_wrapper_class(father_wrapper, registered_vars, xopt_list)
objfunc_wrapper.__doc__ = objfunc_wrapper_class.__doc__
#%%

def test():
    """
    use tfloss_meta to solve the ls_l2 and ls_l1 system, with lbfgs optimizer: 
        min_{a,b} ||X@a+b-y||_2^2  ---(1)
    and
        min_{a,b} ||X@a+b-y||_1    ---(2)
    where a_opt = arange(10), b_opt = 1, 
    X = random.randn(100,10), y = X@a+b+random.randn(100,1)*0.3
    """
    print(test.__doc__)
    a_true = reshape(arange(10), [10,1])
    b_true = 1
    with tf.name_scope('optimizer_tools'):
        x = tf.Variable(zeros((100,10)), dtype=tf.float64, trainable=False)
        y = tf.Variable(zeros((100,1)), dtype=tf.float64, trainable=False)
        a = tf.Variable(random.randn(10,1), tf.float64)
        b = tf.Variable([1,], dtype=tf.float64)
        loss2 = tf.reduce_mean((x@a+b-y)**2)
        loss1 = tf.reduce_sum(tf.abs(x@a+b-y))
        sess = tf.Session()
        sess.run([a.initializer, b.initializer])
        loss1wrapper = objfunc_wrapper(loss1)
        loss2wrapper = objfunc_wrapper(loss2)
        loss3wrapper = objfunc_wrapper(loss1wrapper, [a,])
    from scipy.optimize import fmin_l_bfgs_b as lbfgs
    data = random.randn(100,10)
    with sess.as_default():
        xopt, fopt, dict_opt = lbfgs(loss1wrapper.f, zeros(11), fprime=loss1wrapper.g, args=({x: data, y: data@a_true+b_true+random.randn(100,1)*0.3},))
        loss1wrapper.set_xopt(xopt)
        loss1wrapper.update()
        print('l1 opt: ')
        print('a: ', reshape(a.eval(), [10,]))
        print('b: ', b.eval())
        xopt, fopt, dict_opt = lbfgs(loss2wrapper.f, zeros(11), fprime=loss2wrapper.g, args=({x: data, y: data@a_true+b_true+random.randn(100,1)*0.3},))
        loss2wrapper.set_xopt(xopt)
        loss2wrapper.update()
        set_printoptions(suppress=True, precision=2)
        print('l2 opt: ')
        print('a: ', reshape(a.eval(), [10,]))
        print('b: ', b.eval())
        xopt, fopt, dict_opt = lbfgs(loss3wrapper.f, zeros(10), fprime=loss3wrapper.g, args=({x: data, y: data@a_true+b_true+random.randn(100,1)*0.3},))
        loss3wrapper.set_xopt(xopt)
        loss3wrapper.update()
        set_printoptions(suppress=True, precision=2)
        print('l3 opt: ')
        print('a: ', reshape(a.eval(), [10,]))
    return None
#%%

#%%

