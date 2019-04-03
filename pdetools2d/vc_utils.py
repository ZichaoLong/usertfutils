#!/usr/bin/env python
# -*- coding: utf-8 -*-
#%%
from numpy import *
import numpy as np
import tensorflow as tf
from .. import nonlinear_approx
__all__ = ['interp_config', 'vc_fitters_bank', 'vc_bank']
#%%
def interp_config():
    nonlinear_rank = 2
    mesh_size = array([5,]*nonlinear_rank, dtype=int32)
    mesh_bound = array([[0,2*pi],]*nonlinear_rank).transpose()
    nonlinear_config = {}
    nonlinear_config['mesh_size'] = mesh_size
    nonlinear_config['mesh_bound'] = mesh_bound
    nonlinear_config['interp_order'] = 2
    return nonlinear_config
def vc_fitters_bank(max_order, **kw):
    """
    Args:
        max_order, kw:
        (kw.keys(): interp_order, mesh_bound, mesh_size, interp_coe, 参考 nonlinear_approx.LagrangeInterp_Fitter)
    Return:
        vc_fitters, vc_pars
    """
    vc_fitters = ndarray([max_order+1,]*2, dtype=np.object)
    vc_pars = ndarray([max_order+1,]*2, dtype=np.object)
    nonlinear_config = interp_config()
    nonlinear_config.update(kw)
    for k in range(max_order+1):
        for j in range(k+1):
            vc_fitters[j,k-j] = nonlinear_approx.LagrangeInterp_Fitter(**nonlinear_config)
            vc_pars[j,k-j] = vc_fitters[j,k-j].trainable_vars
    return vc_fitters, vc_pars
def vc_bank(xy_tensor, vc_fitters):
    max_order = vc_fitters.shape[0]-1
    variant_coe = ndarray([max_order+1,]*2, dtype=np.object)
    for k in range(max_order+1):
        for j in range(k+1):
            tmp = vc_fitters[j,k-j](xy_tensor)
            variant_coe[j,k-j] = tmp[tf.newaxis,:,:,tf.newaxis]
    return variant_coe
#%%


