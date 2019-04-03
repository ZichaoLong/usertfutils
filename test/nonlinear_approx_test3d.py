#!/usr/bin/env python
# -*- coding: utf-8 -*-
#%%
from numpy import *
import tensorflow as tf
from scipy.misc import factorial
import usertfutils
from usertfutils import objfunc_wrapper
from usertfutils.nonlinear_approx import LagrangeInterp_Fitter, SVR_RBF_Fitter, NN_Fitter, KNN_Fitter
usertfutils.set_precision(tf.float64)
from testutils import meshgen
#%%
mesh_bound = array([[0,0,0],[1,1,1]], dtype=float64)
mesh_size = array([200,200,200], dtype=int32)
xy = meshgen(mesh_bound, mesh_size)
dim = len(mesh_size)
#%% LagrangeInterp_Fitter
interp_order = 2
interp_mesh_size = array([40,]*dim, dtype=int32)
interp_coe = tf.Variable(zeros(interp_mesh_size*interp_order+1), dtype=usertfutils.TENSOR_PRECISION())
Nonlinear_Fitter = LagrangeInterp_Fitter(interp_order=interp_order, mesh_bound=mesh_bound, interp_coe=interp_coe)
inputs_shape = [50,50,50]
#%% SVR_RBF_Fitter
Train_Point = xy[5::10,5::10,5::10]
train_sample_num = Train_Point.shape[0]*Train_Point.shape[1]*Train_Point.shape[2]
w = tf.Variable(zeros((Train_Point.shape[0], Train_Point.shape[1], Train_Point.shape[2], 1)), dtype=usertfutils.TENSOR_PRECISION())
b = tf.Variable(0, dtype=usertfutils.TENSOR_PRECISION())
gamma = tf.placeholder(dtype=usertfutils.TENSOR_PRECISION(), shape=[])
Nonlinear_Fitter = SVR_RBF_Fitter(w, Train_Point, b=b, gamma=gamma)
inputs_shape = [20,20,20]
#%% KNN_Fitter
Train_Point = xy[5::10,5::10,5::10]
train_sample_num = Train_Point.shape[0]*Train_Point.shape[1]*Train_Point.shape[2]
approx_order = 1
w = tf.Variable(zeros((Train_Point.shape[0], Train_Point.shape[1], Train_Point.shape[2], 1+2*approx_order)), dtype=usertfutils.TENSOR_PRECISION())
gamma = tf.placeholder(dtype=usertfutils.TENSOR_PRECISION(), shape=[])
Nonlinear_Fitter = KNN_Fitter(w, Train_Point, approx_order=approx_order)
inputs_shape = [20,20,20]
#%% NN_Fitter
def leakyrelu(alpha):
    def F(inputs):
        return tf.nn.relu(inputs)-alpha*tf.nn.relu(-inputs)
    return F
widths = [50,50,1]
Nonlinear_Fitter = NN_Fitter(dim, widths, activation_func=leakyrelu(0.01))
#Nonlinear_Fitter = NN_Fitter(dim, widths, activation_func=tf.nn.relu)
Nonlinear_Fitter = NN_Fitter(dim, widths, activation_func=tf.nn.sigmoid)
inputs_shape = [50,50,50]

#%%
inputs = tf.placeholder(shape=inputs_shape+[dim,], dtype=tf.float64)
sample_label = tf.placeholder(shape=inputs_shape, dtype=tf.float64)
inference = Nonlinear_Fitter(inputs)
if isinstance(inference, list):
    infe_test = inference[1]
    inference = inference[0]
fit_err = tf.abs(sample_label-inference)
loss = tf.reduce_mean(tf.square(sample_label-inference))
#%%

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
obj = objfunc_wrapper(loss)
xopt = (random.rand(obj.allsize)-0.5)/10,
#%%
def f(xx):
    return sin(xx[:,:,:,0]*8)+cos(sqrt(xx[:,:,:,1])*4)*sin(xx[:,:,:,2]*4)
IN,JN,KN = int(200/inputs_shape[0]), int(200/inputs_shape[1]), int(200/inputs_shape[2])
indx = zeros((IN*JN*KN,3),dtype=int32)
idx = 0
for i in range(IN):
    for j in range(JN):
        for k in range(KN):
            indx[idx] = array([i,j,k])*array(inputs_shape)
            idx += 1
#for i in range(IN*JN*KN):
for i in range(64):
    print(str(i)+'/'+str(IN*JN*KN))
    feed_dict = {}
    try:
        feed_dict[gamma] = 1000
    except NameError:
        pass
    feed_dict[inputs] = xy[
            indx[i,0]:indx[i,0]+inputs_shape[0],
            indx[i,1]:indx[i,1]+inputs_shape[1],
            indx[i,2]:indx[i,2]+inputs_shape[2]
            ]
    #feed_dict[inputs] = xy[
    #        random.choice(200,inputs_shape[0])[:,newaxis,newaxis],
    #        random.choice(200,inputs_shape[1])[newaxis,:,newaxis],
    #        random.choice(200,inputs_shape[2])[newaxis,newaxis,:]
    #        ]
    #feed_dict[inputs] = xy[
    #        random.randint(200/inputs_shape[0])+int(200/inputs_shape[0])*arange(0,inputs_shape[0],dtype=int32)[:,newaxis,newaxis],
    #        random.randint(200/inputs_shape[1])+int(200/inputs_shape[1])*arange(0,inputs_shape[1],dtype=int32)[newaxis,:,newaxis],
    #        random.randint(200/inputs_shape[2])+int(200/inputs_shape[2])*arange(0,inputs_shape[2],dtype=int32)[newaxis,newaxis,:]
    #        ]
    feed_dict[sample_label] = f(feed_dict[inputs])
    
    from scipy.optimize import fmin_l_bfgs_b as lbfgs
    xopt, fopt, dict_opt = lbfgs(
            func=obj.f, 
            x0=xopt,
            #x0=(random.rand(obj.allsize)-0.5)/10,
            fprime=obj.g, 
            args=(feed_dict,), 
            m=200,
            maxiter=5,
            iprint=5, 
            factr=1e1, pgtol=1e-16
            )
obj.set_xopt(xopt)
obj.update()
#%%
point_flat = reshape(xy, [-1,dim])
point_flat = xy[
        random.randint(200/inputs_shape[0])+int(200/inputs_shape[0])*arange(0,inputs_shape[0],dtype=int32)[:,newaxis,newaxis],
        random.randint(200/inputs_shape[1])+int(200/inputs_shape[1])*arange(0,inputs_shape[1],dtype=int32)[newaxis,:,newaxis],
        random.randint(200/inputs_shape[2])+int(200/inputs_shape[2])*arange(0,inputs_shape[2],dtype=int32)[newaxis,newaxis,:]
        ]
feed_dict[inputs] = point_flat
feed_dict[sample_label] = f(feed_dict[inputs])
print(sqrt(loss.eval(feed_dict=feed_dict)).mean())
print(fit_err.eval(feed_dict=feed_dict).max())
infe = inference.eval(feed_dict=feed_dict)
infe_true = feed_dict[sample_label]
import shelve
with shelve.open('results/infe') as db:
    db['infe'] = infe
    db['infe_true'] = infe_true
#%%
import shelve
with shelve.open('results/infe') as db:
    infe = db['infe']
    infe_true = db['infe_true']
#%%
h = plt.figure()
indx = random.randint(20)
a = h.add_subplot(4,2,1)
a.imshow(infe_true[indx])
a.set_title('true')
a = h.add_subplot(4,2,2)
a.imshow(infe[indx])
a.set_title('inferenced')
indx = random.randint(20)
a = h.add_subplot(4,2,3)
a.plot(infe_true[indx,indx])
a = h.add_subplot(4,2,4)
a.plot(infe[indx,indx])
indx = random.randint(20)
a = h.add_subplot(4,2,5)
a.plot(infe_true[indx,:,indx])
a = h.add_subplot(4,2,6)
a.plot(infe[indx,:,indx])
indx = random.randint(20)
a = h.add_subplot(4,2,7)
a.plot(infe_true[:,indx,indx])
a = h.add_subplot(4,2,8)
a.plot(infe[:,indx,indx])
#%%

