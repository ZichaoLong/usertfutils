#!/usr/bin/env python
# -*- coding: utf-8 -*-
#%%
import tensorflow as tf
from numpy import float32,float64
#__all__ = []
__all__ = ['set_precision', 'get_precision', 'TENSOR_PRECISION', 'NUMPY_PRECISION']
#%%
PRECISION = tf.float64
def set_precision(precision):
    global PRECISION
    assert precision in [tf.float32, tf.float64]
    PRECISION = precision
    return None
def get_precision():
    return PRECISION
def TENSOR_PRECISION():
    return PRECISION
def NUMPY_PRECISION():
    return (float32 if TENSOR_PRECISION() is tf.float32 else float64)
#%%


