#!/usr/bin/env python
# -*- coding: utf-8 -*-
#%%
from numpy import *
import numpy as np
import numpy.fft as fft
import tensorflow as tf
import sympy
from .. import precision_control
from .. import optimizer_tools
from .. import kernel_basis_filter_tensor
from . import diff_utils
from . import vc_utils
__all__ = []

#%%
def _meshgen(mesh_bound, mesh_size, start=0, end=0):
    mesh_bound = array(mesh_bound, dtype=float64)
    mesh_size = array(mesh_size, dtype=int32)
    N = len(mesh_size)
    xyzshape = list(mesh_size+1+end-start)
    xyz = zeros([N,]+xyzshape)
    for i in range(N):
        seq = mesh_bound[0,i]+(mesh_bound[1,i]-mesh_bound[0,i])*arange(start, mesh_size[i]+1+end)/mesh_size[i]
        newsize = ones(N, dtype=int32)
        newsize[i] = xyzshape[i]
        seq = reshape(seq, newsize)
        xyz[i] = xyz[i]+seq
    perm = arange(1, N+2, dtype=int32)
    perm[N] = 0
    return transpose(xyz, axes=perm)

def meshgen(mesh_bound, mesh_size, boundary='Periodic', holdboundary=False, dtype=np.float64):
    """
    依据 mesh_bound, mesh_size 生成网格, 依据boundary, holdboundary决定包含上下是否被包含
    Args:
        mesh_bound: ndarray, R^m中各维度bound, shape = [2,m]
        mesh_size: ndarray, R^m中各坐标方向网格数, shape = [m,]
        boundary: 边值条件, Dirichlet, Neumann, Periodic
        holdboundary: 在 boundary 为 Dirichlet, Neumann 时有效
        dtype: np.float64, np.float32, or precision_control.TENSOR_PRECISION()
    """
    N = len(mesh_size)
    if boundary.upper() == 'PERIODIC':
        xyz = _meshgen(mesh_bound, mesh_size, start=0, end=-1)
    elif holdboundary:
        xyz = _meshgen(mesh_bound, mesh_size, start=0, end=0)
    else:
        xyz = _meshgen(mesh_bound, mesh_size, start=1, end=-1)
    if (dtype is np.float64) or (dtype is np.float32):
        xyz = xyz.astype(dtype)
    else:
        assert dtype is precision_control.TENSOR_PRECISION()
        xyz = tf.Variable(xyz, dtype=dtype, trainable=False)
    return xyz

#%%
class pdenet2d(object):
    def __init__(self, dt_block, *, dt_max=None, boundary, mesh_bound, mesh_size, extra_property=None):
        """
        Args:
            boundary: 边值条件, Dirichlet, Neumann, Periodic. 
            mesh_bound: ndarray, R^m中各维度bound, shape = [2,m]
            mesh_size: ndarray, R^m中各坐标方向网格数, shape = [m,]
        """
        self.dt_block = dt_block
        self.dt_max = dt_max
        self.mesh_bound = array(mesh_bound, dtype=float64)
        self.mesh_size = array(mesh_size, dtype=int32)
        self.boundary = boundary
        self.extra_property = {} if extra_property is None else extra_property
    def evolution(self, u0, dt, dt_max=None):
        """
        依据self.boundary对u0演化dt,若dt>dt_max需要对分裂dt, 演化系统基于self.dt_block
        """
        dt_max = (self.dt_max if dt_max is None else dt_max)
        dt_num = (1 if dt_max is None else ceil(dt/dt_max))
        dt_new = dt/dt_num
        u = u0
        while dt_num>0:
            u = self.dt_block(u, dt_new)
            dt_num = dt_num-1
        return u
    def traj(self, u0, dt, dt_num, dt_max=None):
        u = ndarray(dt_num+1, dtype=np.object)
        u[0] = u0
        for i in range(dt_num):
            u[i+1] = self.evolution(u[i], dt, dt_max)
        return u
    def downsample(self, u, mesh_size=None):
        """
        依据self.mesh_size,mesh_size及self.boundary对u进行下采样
        """
        mesh_size = (self.mesh_size if mesh_size is None else mesh_size)
        mesh_size = array(mesh_size, dtype=int32)
        assert linalg.norm(self.mesh_size%mesh_size) == 0
        stride = self.mesh_size//mesh_size
        if self.boundary.upper() == 'PERIODIC':
            return u[:,::stride[0],::stride[1]]
        else:
            return u[:,stride[0]-1::stride[0],stride[1]-1::stride[1]]

#%%
def _initgen_periodic(mesh_size):
    freq = 3
    dim = len(mesh_size)
    x = random.randn(*mesh_size)
    coe = fft.ifftn(x)
    freqs = random.randint(freq, 2*freq, size=[dim,])
    freqs = [10,10]
    for i in range(dim):
        perm = arange(dim, dtype=int32)
        perm[i] = 0
        perm[0] = i
        coe = coe.transpose(*perm)
        coe[freqs[i]+1:-freqs[i]] = 0
        coe = coe.transpose(*perm)
    x = fft.fft2(coe)
    assert linalg.norm(x.imag) < 1e-8
    x = x.real
    return x
def _initgen(mesh_size, boundary='Periodic'):
    x = _initgen_periodic(mesh_size)
    if boundary.upper() == 'DIRICHLET':
        dim = x.ndim
        for i in range(dim):
            y = arange(mesh_size[i])/mesh_size[i]
            y = y*(1-y)
            s = ones(dim, dtype=int32)
            s[i] = mesh_size[i]
            y = reshape(y, s)
            x = x*y
        for i in range(dim):
            perm = arange(dim, dtype=int32)
            perm[i] = 0
            perm[0] = i
            x = x.transpose(*perm)
            x = x[1:]
            x = x.transpose(*perm)
    return x
def initgen(batch_size, mesh_size, boundary='Dirichlet', dtype=np.float64):
    if boundary.upper() == 'PERIODIC':
        x = zeros([batch_size,*mesh_size])
    else:
        x = zeros([batch_size,*(array(mesh_size)-1)])
    for i in range(batch_size):
        x[i] = _initgen(mesh_size, boundary=boundary)
    x = x*1000
    if (dtype is np.float64) or (dtype is np.float32):
        x = x.astype(dtype)
    else:
        x = tf.Variable(x, trainable=False, dtype=precision_control.TENSOR_PRECISION())
    return x
#%%



