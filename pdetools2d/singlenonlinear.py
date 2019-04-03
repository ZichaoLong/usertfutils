#!/usr/bin/env python
# -*- coding: utf-8 -*-
#%%
from numpy import *
import numpy as np
import tensorflow as tf
import sympy
import improc_tools
from .. import precision_control
from .. import optimizer_tools
from .. import kernel_basis_filter_tensor
from .. import nonlinear_approx
from . import diff_utils
from . import vc_utils
from . import pdenet2d
#__all__ = ['singlenonlinear_eg1', 'singlenonlinear','singlenonlinear_config']
__all__ = []

#%%
def singlenonlinear_eg(mesh_bound, mesh_size, nonlinear_coefficient=5):
    mesh_bound = array(mesh_bound, dtype=float64)
    mesh_size = array(mesh_size, dtype=int32)
    dx = (mesh_bound[1]-mesh_bound[0])/mesh_size
    f = improc_tools.diff_monomial_coe(y_order=2,shape=[3,3])/(dx[0]**2)+improc_tools.diff_monomial_coe(x_order=2,shape=[3,3])/(dx[1]**2)
    f = reshape(f, [3,3,1,1])
    f = tf.Variable(f, dtype=precision_control.TENSOR_PRECISION(), trainable=False)
    laplace = kernel_basis_filter_tensor.wrap_filter2d_tensor(f)
    diffusion_velocity = 0.3
    extra_property = {'nonlinear_coefficient':tf.Variable(nonlinear_coefficient, dtype=precision_control.TENSOR_PRECISION(), trainable=False)}
    def dt_block(u0, dt):
        u1 = laplace(u0, boundary='CONSTANT')
        u1 = u0+dt*(diffusion_velocity*u1+extra_property['nonlinear_coefficient']*tf.sin(u0))
        return u1
    dt_max = min(dx[0]**2,dx[1]**2)/diffusion_velocity/4
    return pdenet2d.pdenet2d(dt_block, dt_max=dt_max, mesh_bound=mesh_bound, mesh_size=mesh_size, boundary='Dirichlet', extra_property=extra_property)

#%%
def singlenonlinear(config, mesh_bound, mesh_size, boundary='Dirichlet'):
    """
    可以拟合形如
        u_t = Lu+f(u)
    的pdenet.其中L是微分阶不超过2阶的线性算子,f是非线性函数
    Args:
        config: dict, config.keys():
            ID_tensor, diff_tensors: Lu中需要用到的微分算子
            vc_fitters: 用于拟合Lu中可能出现的变系数
            na_fitter： 用于拟合f
        boundary, mesh_bound, mesh_size: 参考meshgen.
    """
    ID_tensor = config['ID_tensor']
    diff_tensors = config['diff_tensors']
    vc_fitters = config['vc_fitters']
    na_fitter = config['na_fitter']
    mesh_bound = array(mesh_bound, dtype=float64)
    mesh_size = array(mesh_size, dtype=int32)
    xy = pdenet2d.meshgen(mesh_bound, mesh_size, boundary='Dirichlet', holdboundary=False, dtype=precision_control.TENSOR_PRECISION())
    dx = (mesh_bound[1]-mesh_bound[0])/mesh_size
    VC = []
    DIFF = []
    DIFF.append(ID_tensor)
    for k in range(3):
        for j in range(k+1):
            VC.append(vc_fitters[j,k-j](xy)[tf.newaxis,:,:,tf.newaxis])
            scale = 1/(dx[0]**j)/(dx[1]**(k-j))
            DIFF.append(diff_tensors[j,k-j]*scale)
    VC = tf.concat(VC, axis=3)
    DIFF = tf.concat(DIFF, axis=3)
    DIFF_FILTER = kernel_basis_filter_tensor.wrap_filter2d_tensor(DIFF)
    convboundary = ('wrap' if boundary.upper() == 'PERIODIC' else 'CONSTANT')
    def dt_block(u0, dt):
        u1 = DIFF_FILTER(u0, boundary=convboundary)
        # u_{n+1} = u_n+dt*(f(u_n)+DIFF_FILTER(u_n)*VC)
        u1 = u1[:,:,:,0:1]+dt*(na_fitter(u1[:,:,:,1:2])[:,:,:,tf.newaxis]+tf.reduce_sum(u1[:,:,:,2:]*VC[:,:,:,1:], axis=3, keep_dims=True))
        return u1
    extra_property={'ID_tensor': ID_tensor, 'diff_tensors': diff_tensors, 'vc_fitters': vc_fitters, 'na_fitter': na_fitter}
    return pdenet2d.pdenet2d(dt_block, dt_max=None, mesh_bound=mesh_bound, mesh_size=mesh_size, boundary=boundary, extra_property=extra_property)
#%%
def singlenonlinear_config(kernel_size, vc_config=None, na_config=None):
    vc_config_default = {'mesh_size':array([5,5]),'mesh_bound':array([[0,0],[2*pi,2*pi]]),'interp_order':2}
    vc_config = {} if vc_config is None else vc_config
    vc_config_default.update(vc_config)
    na_config_default = {'mesh_size':array([50,]),'mesh_bound':array([[-100,],[100,]]),'interp_order':4}
    na_config = {} if na_config is None else na_config
    na_config_default.update(na_config)
    ID_moment, diff_moments, ID_tensor, diff_tensors, moment0, moment1 = \
            diff_utils.diff_bank(max_order=2, kernel_size=kernel_size)
    vc_fitters,vc_pars = \
            vc_utils.vc_fitters_bank(max_order=2,**vc_config_default)
    na_fitter = nonlinear_approx.LagrangeInterp_Fitter(**na_config_default)
    trainable_vars = [ID_moment,]
    for k in range(3):
        for j in range(k+1):
            trainable_vars.append(diff_moments[j,k-j])
    for k in range(3):
        for j in range(k+1):
            trainable_vars.append(vc_pars[j,k-j])
    trainable_vars.append(na_fitter.trainable_vars)
    config = {}
    config['ID_tensor'] = ID_tensor
    config['diff_tensors'] = diff_tensors
    config['vc_fitters'] = vc_fitters
    config['na_fitter'] = na_fitter
    config['moment0'] = moment0
    config['moment1'] = moment1
    config['trainable_vars'] = trainable_vars
    config['ID_moment'] = ID_moment
    config['diff_moments'] = diff_moments
    return config

#%%


#%%


