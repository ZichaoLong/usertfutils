#!/usr/bin/env python
# -*- coding: utf-8 -*-
#%%
from numpy import *
import tensorflow as tf
import usertfutils
from usertfutils import objfunc_wrapper, updater, constraint_objfunc
usertfutils.set_precision(tf.float64)
#%%
"""
use tfloss_meta to solve the ls_l2 and ls_l1 system, with lbfgs optimizer: 
    min_{a,b} ||X@a+b-y||_2^2  ---(1)
and
    min_{a,b} ||X@a+b-y||_1    ---(2)
where a_opt = arange(10), b_opt = 1, 
X = random.randn(100,10), y = X@a+b+random.randn(100,1)*0.3
"""
a_true = reshape(arange(10), [10,1])
b_true = 1
with tf.name_scope('optimizer_tools'):
    x = tf.Variable(zeros((100,10)), dtype=tf.float64, trainable=False)
    y = tf.Variable(zeros((100,1)), dtype=tf.float64, trainable=False)
    a = tf.Variable(random.randn(10,1), tf.float64)
    b = tf.Variable(1, dtype=tf.float64)
    loss2 = tf.reduce_mean((x@a+b-y)**2)
    loss1 = tf.reduce_sum(tf.abs(x@a+b-y))
    loss1wrapper = objfunc_wrapper(loss1)
    loss2wrapper = objfunc_wrapper(loss2)
    loss3wrapper = objfunc_wrapper(loss1wrapper, [a,])
sess = tf.InteractiveSession()
sess.run([a.initializer, b.initializer])
#%%
from scipy.optimize import fmin_l_bfgs_b as lbfgs
data = random.randn(100,10)
data_updater = updater()
data_updater.add(x,y)
with sess.as_default():
    data_updater.update({x: data, y: data@a_true+b_true+random.randn(100,1)*0.3})
    xopt, fopt, dict_opt = lbfgs(loss1wrapper.f, zeros(11), fprime=loss1wrapper.g)
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

    a_c = a_true.astype(np.float64).copy()
    a_c[:5] = np.nan
    b_c = 1
    constraint = {a:a_c,b:b_c}
    F,G,xopt_proj,grad_proj = constraint_objfunc(loss1wrapper, constraint)
    xopt, fopt, dict_opt = lbfgs(F, zeros(11), fprime=G)
    xopt = xopt_proj(xopt)
    loss1wrapper.set_xopt(xopt)
    loss1wrapper.update()
    print('constraint l1 opt: ')
    print('a: ', reshape(a.eval(), [10,]))
    print('b: ', b.eval())
    constraint = {}
    F,G,xopt_proj,grad_proj = constraint_objfunc(loss1wrapper, constraint)
    xopt, fopt, dict_opt = lbfgs(F, zeros(11), fprime=G)
    xopt = xopt_proj(xopt)
    loss1wrapper.set_xopt(xopt)
    loss1wrapper.update()
    print('constraint l1 opt: ')
    print('a: ', reshape(a.eval(), [10,]))
    print('b: ', b.eval())
#%%


