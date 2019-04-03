#!/usr/bin/env python
# -*- coding: utf-8 -*-
#%%
from numpy import *
import tensorflow as tf

#%%
def meshgen(mesh_bound, mesh_size):
    mesh_bound = array(mesh_bound, dtype=float64)
    mesh_size = array(mesh_size, dtype=int32)
    N = len(mesh_size)
    xyz = zeros([N,]+list(mesh_size))
    for i in range(N):
        seq = mesh_bound[0,i]+(mesh_bound[1,i]-mesh_bound[0,i])*arange(mesh_size[i])/mesh_size[i]
        newsize = ones(N, dtype=int32)
        newsize[i] = mesh_size[i]
        seq = reshape(seq, newsize)
        xyz[i] = xyz[i]+seq
    perm = arange(1, N+2, dtype=int32)
    perm[N] = 0
    return transpose(xyz, axes=perm)

#%%


