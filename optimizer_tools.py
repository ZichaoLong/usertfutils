#!/usr/bin/env python
# -*- coding: utf-8 -*-
#%%
import tensorflow as tf
import functools
from numpy import *
from . import precision_control
__all__ = ['objfunc_wrapper', 'array2list', 'list2array', 'updater', 'xopt_check', 'grad_check', 'constraint_objfunc']
#%%
def array2list(x, variables_size, variables_shape, iscopy=True):
    N = int(sum(variables_size))
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
#%% updater
class updater(object):
    def __init__(self, fathter_updater=None, registered_vars=None):
        if fathter_updater is None:
            self.placeholder = {}
            self.assignment_op = {}
        else:
            self.placeholder = dict((v,fathter_updater.placeholder[v]) for v in registered_vars)
            self.assignment_op = dict((v,fathter_updater.assignment_op[v]) for v in registered_vars)
    def add(self, *tensors):
        for v in tensors:
            w = tf.placeholder(shape=v.get_shape(), dtype=v.dtype)
            self.placeholder[v] = w
            self.assignment_op[v] = tf.assign(v, w)
        return None
    def update(self, feed_dict=None, sess=None):
        """
        feed_dict.keys(): 需要更新的tensors
        feed_dict.values(): 对应的values
        """
        sess = (tf.get_default_session() if sess is None else sess)
        feed_dict = ({} if feed_dict is None else feed_dict)
        for v in feed_dict.keys():
            assert v in self.placeholder
        feed_dict_placeholder = dict((self.placeholder[v],feed_dict[v]) for v in feed_dict.keys())
        sess.run(list(self.assignment_op[v] for v in feed_dict.keys()), feed_dict=feed_dict_placeholder)
        return None
#%% optimize utils
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
        variables_size.append((1 if len(s) == 0 else functools.reduce(multiply, s)))
        variables_shape.append(s)
    variables_num = len(variables)
    allsize = int(sum(variables_size))
    with tf.name_scope('array2tensors/'+loss.name.split(':')[0]):
        variables_updater = updater()
        variables_updater.add(*variables)
    def tfloss_info():
        pass
    tfloss_info()
    tfloss_info.allsize = allsize
    tfloss_info.variables = variables
    tfloss_info.variables_grads = variables_grads
    tfloss_info.variables_size = variables_size
    tfloss_info.variables_shape = variables_shape
    tfloss_info.variables_updater = variables_updater
    tfloss_info.loss = loss
    tfloss_info.xopt_list = array2list(zeros((allsize,)), variables_size, variables_shape, iscopy=False)
    return tfloss_info

class objfunc_wrapper_class(object):
    """
    objfunc_wrapper_class
    几个主要成员变量：
        loss: tensorflow compute graph中目标函数loss function所代表的tensor
        variables: 一个由tensorflow variables组成的list
        variables_grads: 最顶层loss function对variables中各变量梯度的tensor
        xopt_list: list，variables values暂存值；sess.run(variables)为variables当前值
        xopt_dict: dict, 暂存值的dict版, 即dict(zip(variables, xopt_list))
    其他辅助成员变量:
        allsize: variables中各变量size之和
        variables_size,variables_shape: variables中各变量size及shape
        variables_updater: 用于更新variables
    __init__:
        输入: 
        father_wrapper: 可以是一个objfunc_wrapper_class实例，也可由上述tfloss_meta生成
            father_wrapper 必须有如下attributes:
            'variables', 'variables_grads', 
            'loss', 'xopt_list' 
            'allsize', 'variables_size', 'variables_shape', 
            'variables_updater'
        registered_vars: list, father_wrapper.variables的子集, 
            初始化即执行self.variables = registered_vars并将self的成员、成员函数更新
        xopt_list: 可选，若xopt_list == None, 则self.xopt_list各元素
            按registered_vars对应copy自father_wrapper.xopt_list
    """
    def __init__(self, father_wrapper, registered_vars, xopt_list=None):
        assert isinstance(registered_vars, list)
        # assert len(registered_vars) > 0
        registered_vars_indx = []
        for v in registered_vars:
            assert v in father_wrapper.variables
            registered_vars_indx.append(father_wrapper.variables.index(v))
        self.loss = father_wrapper.loss
        self.variables = []
        self.variables_grads = []
        self.variables_size = []
        self.variables_shape = []
        self.variables_updater = updater(father_wrapper.variables_updater, registered_vars)
        self.xopt_list = []
        for i in registered_vars_indx:
            self.variables.append(father_wrapper.variables[i])
            self.variables_grads.append(father_wrapper.variables_grads[i]) 
            self.variables_size.append(father_wrapper.variables_size[i])
            self.variables_shape.append(father_wrapper.variables_shape[i])
            self.xopt_list.append(father_wrapper.xopt_list[i].copy())
        self.xopt_list = (xopt_list if not xopt_list is None else self.xopt_list)
        self.xopt_dict = dict(zip(self.variables, self.xopt_list))
        self.allsize = int(sum(self.variables_size))
        self.variables_num = len(registered_vars)
    def update(self, feed_dict=None, sess=None):
        """
        依据feed_dict更新指定variables,
        feed_dict缺省时，使用暂存版self.xopt_list更新self.variables
        """
        feed_dict = (dict(zip(self.variables, self.xopt_list)) if feed_dict is None else feed_dict)
        self.variables_updater.update(feed_dict, sess)
        return None
    def get_xopt(self, dtype='dict', sess=None):
        """
        返回variables当前值dtype版
        """
        sess = (tf.get_default_session() if sess is None else sess)
        xopt_list = sess.run(self.variables)
        if dtype == 'list':
            return xopt_list
        elif dtype == 'dict':
            return dict(zip(self.variables, xopt_list))
        else:
            return self.l2a(xopt_list)
    def set_xopt(self, xopt, iscopy=True):
        """
        利用xopt更新暂存区self.xopt_list
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
        """
        xopt_dict.keys()包含于(可以不相等)self.variables,
        返回完整xopt_list, xopt_dict没有的地方用暂存区self.xopt_list补齐
        """
        xopt_list = []
        for v in self.variables:
            if v in xopt_dict:
                x = (xopt_dict[v].copy() if iscopy else xopt_dict[v])
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

#%% constraint utils
def constraint_check(x, obj ,constraint, checktype='xopt', iscopy=True):
    x_dict = obj.a2d(x, iscopy=iscopy)
    for v in constraint.keys():
        if not v in x_dict:
            continue
        m = constraint[v]
        if x_dict[v].shape != ():
            x_dict[v][~isnan(m)] = (m[~isnan(m)] if checktype == 'xopt' else 0)
            continue
        if isnan(m):
            continue
        x_dict_v_tmp = reshape(x_dict[v], [1,]) 
        assert shares_memory(x_dict[v], x_dict_v_tmp)
        x_dict_v_tmp[0] = (m if checktype == 'xopt' else 0)
    if iscopy:
        return obj.d2a(x_dict)
    else:
        return x
xopt_check = functools.partial(constraint_check, checktype='xopt')
grad_check = functools.partial(constraint_check, checktype='grad')
def constraint_objfunc(obj, constraint):
    def F(x, **kw):
        return obj.f(xopt_check(x, obj, constraint, iscopy=True), **kw)
    def G(x, **kw):
        return grad_check(obj.g(xopt_check(x, obj, constraint, iscopy=True), **kw), obj, constraint, iscopy=False)
    def proj(x, **kw):
        return xopt_check(x, obj, constraint, iscopy=True)
    return F, G, proj
#%%
#%%

#%%

