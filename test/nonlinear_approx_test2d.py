#!/usr/bin/env python
# -*- coding: utf-8 -*-
#%%
from numpy import *
import tensorflow as tf
from scipy.misc import factorial
import usertfutils
from usertfutils import objfunc_wrapper
from usertfutils.nonlinear_approx import LagrangeInterp_Fitter, SVR_RBF_Fitter, KNN_Fitter, NN_Fitter
usertfutils.set_precision(tf.float64)
from testutils import meshgen

#%%
mesh_bound = array([[0,0],[1,1]], dtype=float64)
mesh_size = array([1000,1000], dtype=int32)
xy = meshgen(mesh_bound, mesh_size)
dim = len(mesh_size)
#%% LagrangeInterp_Fitter
interp_order = 2
interp_mesh_size = array([40,]*dim, dtype=int32)
interp_coe = tf.Variable(zeros(interp_mesh_size*interp_order+1), dtype=usertfutils.TENSOR_PRECISION())
Nonlinear_Fitter = LagrangeInterp_Fitter(interp_order=interp_order, mesh_bound=mesh_bound, interp_coe=interp_coe)
#%% SVR_RBF_Fitter
Train_Point = xy[25::50,25::50]
train_sample_num = Train_Point.shape[0]*Train_Point.shape[1]
w = tf.Variable(zeros((Train_Point.shape[0], Train_Point.shape[1], 1)), dtype=usertfutils.TENSOR_PRECISION())
b = tf.Variable(0, dtype=usertfutils.TENSOR_PRECISION())
gamma = tf.placeholder(dtype=usertfutils.TENSOR_PRECISION(), shape=[])
Nonlinear_Fitter = SVR_RBF_Fitter(w, Train_Point, b=b, gamma=gamma)
#%% KNN_Fitter
Train_Point = xy[25::50,25::50]
train_sample_num = Train_Point.shape[0]*Train_Point.shape[1]
approx_order = 1
w = tf.Variable(zeros((Train_Point.shape[0], Train_Point.shape[1], 1+2*approx_order)), dtype=usertfutils.TENSOR_PRECISION())
gamma = tf.placeholder(dtype=usertfutils.TENSOR_PRECISION(), shape=[])
Nonlinear_Fitter = KNN_Fitter(w, Train_Point, approx_order=approx_order)
#%% NN_Fitter
def leakyrelu(alpha):
    def F(inputs):
        return tf.nn.relu(inputs)-alpha*tf.nn.relu(-inputs)
    return F
widths = [50,50,1]
Nonlinear_Fitter = NN_Fitter(dim, widths, activation_func=leakyrelu(0.01))
#Nonlinear_Fitter = NN_Fitter(dim, widths, activation_func=tf.nn.relu)
Nonlinear_Fitter = NN_Fitter(dim, widths, activation_func=tf.nn.sigmoid)

#%%
inputs = tf.placeholder(shape=[100,100,dim], dtype=usertfutils.TENSOR_PRECISION())
sample_label = tf.placeholder(shape=[100,100], dtype=usertfutils.TENSOR_PRECISION())
outputs = Nonlinear_Fitter(inputs)
fit_err = tf.abs(sample_label-outputs)
loss = tf.reduce_mean(tf.square(sample_label-outputs))
#%%

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
obj = objfunc_wrapper(loss)
xopt = (random.rand(obj.allsize)-0.5)/1e4
obj.set_xopt(xopt)
xopt_list = obj.a2l(xopt)
xopt_dict = obj.a2d(xopt)
obj.update()
#%%
def f(xx):
    return sin(xx[:,:,0]*1.8*pi)+cos(xx[:,:,1]*2.3*pi)
point_flat = reshape(xy, [-1,dim])
feed_dict = {}
try:
    feed_dict[gamma] = 500
except NameError:
    pass
for iii in range(50):
    feed_dict[inputs] = xy[random.choice(1000, 100)[:,newaxis],random.choice(1000,100)[newaxis,:]]
    feed_dict[sample_label] = f(feed_dict[inputs])
    
    from scipy.optimize import fmin_l_bfgs_b as lbfgs
    xopt, fopt, dict_opt = lbfgs(
            func=obj.f, 
            x0=xopt,
            fprime=obj.g, 
            args=(feed_dict,), 
            m=200,
            maxiter=10,
            iprint=10, 
            factr=1e1, pgtol=1e-16
            )
    obj.set_xopt(xopt)
    obj.update()
#%%
feed_dict[inputs] = xy[5::10,5::10]
feed_dict[sample_label] = f(feed_dict[inputs])
print(sqrt(loss.eval(feed_dict=feed_dict)).mean())
print(fit_err.eval(feed_dict=feed_dict).max())
infe = reshape(outputs.eval(feed_dict=feed_dict), [100, 100])
infe_true = reshape(feed_dict[sample_label], [100, 100])
import shelve
with shelve.open('results/infe') as db:
    db['infe'] = infe
    db['infe_true'] = infe_true
#%%
import shelve
with shelve.open('results/infe') as db:
    infe = db['infe']
    infe_true = db['infe_true']
h = plt.figure()
a = h.add_subplot(3,2,1)
a.imshow(infe_true)
a.set_title('true')
a = h.add_subplot(3,2,2)
a.imshow(infe)
a.set_title('inferenced')
indx = random.randint(100)
a = h.add_subplot(3,2,3)
a.plot(infe_true[indx])
a = h.add_subplot(3,2,4)
a.plot(infe[indx])
indx = random.randint(100)
a = h.add_subplot(3,2,5)
a.plot(infe_true[:,indx])
a = h.add_subplot(3,2,6)
a.plot(infe[:,indx])
#%%

