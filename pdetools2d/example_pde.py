#!/usr/bin/env python
# -*- coding: utf-8 -*-
#%%

import numpy as np
from numpy import *
from numpy.fft import *
from improc_tools import circshift, wrap_filter2d
__all__ = ['meshgen', 'ConstCoeLinearPDE2d', 'VariantCoeLinearPDE2d', 
        'test_ConstCoeLinearPDE2d', 'test_VariantCoeLinearPDE2d']

#%%
def meshgen(mesh_bound, mesh_size):
    mesh_bound = array(mesh_bound)
    mesh_size = array(mesh_size)
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
class ConstCoeLinearPDE2d(object):
    def __init__(self, n=80, scales=5, a0=0, a1=(0.6,0.8), a2=(1,0,1), freq=5):
        assert n%scales == 0
        assert n>=80
        assert freq*2 < n
        assert (a2[0] >= 0 and a2[2] >= 0 and a2[0]*a2[2] >= (a2[1]**2)/2)
        x = random.randn(n,n)
        coe = ifft2(x)
        freq0 = random.randint(freq, 2*freq)
        freq1 = random.randint(freq, 2*freq)
        coe[freq0+1:-freq0] = 0
        coe[:,freq1+1:-freq1] = 0
        x = fft2(coe)
        assert linalg.norm(x.imag) < 1e-8
        x = x.real
        self.coe = coe
        self.x = x
        self.n = n
        self.dx = 2*pi/n
        self.scales = scales
        freq_shift_coe = zeros((n,))
        freq_shift_coe[:2*freq] = arange(2*freq)
        freq_shift_coe[:-2*freq:-1] = -arange(1, 2*freq)
        shift0_coe = reshape(freq_shift_coe, (n,1))
        shift1_coe = reshape(freq_shift_coe, (1,n))
        freq_shift_coe = (a0 - 1j*(a1[0]*shift0_coe + a1[1]*shift1_coe)-
                (a2[0]*shift0_coe**2 + a2[1]*shift0_coe*shift1_coe + a2[2]*shift1_coe**2)
                )
        self.shift_coe = shift1_coe
        self.freq_shift_coe = freq_shift_coe
        self.coordinate = zeros((n,n,2))
        self.coordinate[:,:,0] = repeat(reshape(2*pi*arange(n)/n, (n,1)), n, axis=1)
        self.coordinate[:,:,1] = self.coordinate[:,:,0].transpose()
        def b0(x):
            return zeros(x.shape[:-1])+a0
        def b10(x):
            return zeros(x.shape[:-1])+a1[0]
        def b01(x):
            return zeros(x.shape[:-1])+a1[1]
        def b02(x):
            return zeros(x.shape[:-1])+a2[2]
        def b11(x):
            return zeros(x.shape[:-1])+a2[1]
        def b20(x):
            return zeros(x.shape[:-1])+a2[0]
        def b00(x):
            return zeros(x.shape[:-1])
        self.a = ndarray([5,5], dtype=np.object)
        self.a[0,0] = b0
        self.a[0,1] = b01
        self.a[1,0] = b10
        self.a[0,2] = b02
        self.a[1,1] = b11
        self.a[2,0] = b20
        self.a[list(range(4)),list(range(3,-1,-1))] = b00
        self.a[list(range(5)),list(range(4,-1,-1))] = b00
    def state(self, ts):
        ts = reshape(array(ts), [-1,1,1])
        x = fft2(exp(
            ts*(
                reshape(self.freq_shift_coe, [1, self.n, self.n])
                )
            )*reshape(self.coe, [1,self.n,self.n])
            )
        assert linalg.norm(x.imag) < 1e-8
        x = squeeze(x.real)
        return x
    def __call__(self, ts, sample_stride=None):
        assert len(ts) > 1
        ts = reshape(array(ts), [-1,1,1])
        dom_size = self.n//self.scales
        sample_stride0 = (random.randint(1,self.scales+1) if sample_stride is None else sample_stride[0])
        lb0 = random.randint((self.scales-sample_stride0)*dom_size+sample_stride0)
        ub0 = lb0+sample_stride0*(dom_size-1)+1
        dy = sample_stride0*self.dx
        sample_stride1 = (random.randint(1,self.scales+1) if sample_stride is None else sample_stride[1])
        lb1 = random.randint((self.scales-sample_stride1)*dom_size+sample_stride1)
        ub1 = lb1+sample_stride1*(dom_size-1)+1
        dx = sample_stride1*self.dx
        return [
                self.state(ts)[:,lb0:ub0:sample_stride0,lb1:ub1:sample_stride1],
                squeeze(ts),
                dy,
                dx,
                self.coordinate[lb0:ub0:sample_stride0,lb1:ub1:sample_stride1]
                ]

def coe_modify(A, B, m):
    A[:m,:m] = B[:m,:m]
    A[:m,-m+1:] = B[:m,-m+1:]
    A[-m+1:,:m] = B[-m+1:,:m]
    A[-m+1:,-m+1:] = B[-m+1:,-m+1:]
    return
class VariantCoeLinearPDE2d(object):
    def __init__(self, n=80, scales=5, N=50, variant_coe_magnitude=1, a=None):
        assert N%2 == 0
        self.n = n
        self.dx = 2*pi/n
        self.scales = scales
        initial_state_generator = ConstCoeLinearPDE2d(n,scales,a0=0,a1=[0.6,0.8],a2=[0.3,0,0.3],freq=5)
        self.initial_state_generator = initial_state_generator
        self.coordinate = initial_state_generator.coordinate
        self.N = N
        freq_shift_coe = zeros((N,))
        freq_shift_coe[:N//2] = arange(N//2)
        freq_shift_coe[:-N//2-1:-1] = -arange(1, 1+N//2)
        self.K0 = reshape(freq_shift_coe, (N,1))
        self.K1 = reshape(freq_shift_coe, (1,N))
        def b0(x):
            return zeros(x.shape[:-1])
        def b10(x):
            y = reshape(x, [-1,2])
            return variant_coe_magnitude*0.5*reshape(cos(y[:,0])+y[:,1]*(2*pi-y[:,1])*sin(y[:,1]), x.shape[:-1])+0.6
            #return zeros(x.shape[:-1])
        def b01(x):
            y = reshape(x, [-1,2])
            return variant_coe_magnitude*2*reshape(cos(y[:,0])+sin(y[:,1]), x.shape[:-1])+0.8
            #return zeros(x.shape[:-1])
        def b20(x):
            return zeros(x.shape[:-1])+0.2
        def b11(x):
            return zeros(x.shape[:-1])
        def b02(x):
            return zeros(x.shape[:-1])+0.3
        def b00(x):
            return zeros(x.shape[:-1])
        self.a = ndarray([5,5], dtype=np.object)
        self.a[0,0] = b0
        self.a[0,1] = b01
        self.a[1,0] = b10
        self.a[0,2] = b02
        self.a[1,1] = b11
        self.a[2,0] = b20
        self.a[list(range(4)),list(range(3,-1,-1))] = b00
        self.a[list(range(5)),list(range(4,-1,-1))] = b00
        self.a_fourier_coe = ndarray([5,5], dtype=np.object)
        self.a_smooth = ndarray([5,5], dtype=np.object)

        xx = arange(0,2*pi,2*pi/N)
        yy = xx.copy()
        yy,xx = meshgrid(xx,yy)
        xx = expand_dims(xx, axis=-1)
        yy = expand_dims(yy, axis=-1)
        xy = concatenate([xx,yy], axis=2)
        m = N//2
        for k in range(3):
            for j in range(k+1):
                tmp_fourier = ifft2(self.a[j,k-j](xy))
                self.a_fourier_coe[j,k-j] = tmp_fourier
                tmp = zeros([m*3,m*3], dtype=np.complex128)
                coe_modify(tmp, tmp_fourier, m)
                self.a_smooth[j,k-j] = fft2(tmp).real
    def vc_conv(self, order, coe):
        N = self.N
        m = N//2
        vc_smooth = self.a_smooth[order[0], order[1]]
        tmp = zeros(vc_smooth.shape, dtype=np.complex128)
        coe_modify(tmp, coe, m)
        C_aug = ifft2(vc_smooth*fft2(tmp))
        C = zeros(coe.shape, dtype=np.complex128)
        coe_modify(C, C_aug, m)
        return C
    def rhs_fourier(self, L):
        rhsL = zeros(L.shape, dtype=np.complex128)
        rhsL += self.vc_conv([1,0], -1j*self.K0*L)
        rhsL += self.vc_conv([0,1], -1j*self.K1*L)
        rhsL += self.vc_conv([2,0], -self.K0**2*L)
        rhsL += self.vc_conv([1,1], -self.K0*self.K1*L)
        rhsL += self.vc_conv([0,2], -self.K1**2*L)
        return rhsL
    def state(self, ts):
        ts = reshape(array(ts), [-1,])
        x = zeros([ts.size,self.n,self.n])
        x[0] = self.initial_state_generator.state(ts[0:1])
        Y = zeros([self.N,self.N], dtype=np.complex128)
        m = self.N//2
        L = ifft2(x[0])
        coe_modify(Y, L, m)
        for i in range(ts.size-1):
            dt = ts[i+1]-ts[i]
            rhsL1 = self.rhs_fourier(Y)
            rhsL2 = self.rhs_fourier(Y+0.5*dt*rhsL1)
            rhsL3 = self.rhs_fourier(Y+0.5*dt*rhsL2)
            rhsL4 = self.rhs_fourier(Y+dt*rhsL3)

            Y = Y+(rhsL1+2*rhsL2+2*rhsL3+rhsL4)*dt/6
            coe_modify(L, Y, m)
            x_tmp = fft2(L)
            assert linalg.norm(x_tmp.imag) < 1e-10
            x[i+1] = x_tmp.real
            L = ifft2(x_tmp.real)
            coe_modify(Y, L, m)
        return squeeze(x)
    def __call__(self, ts, sample_stride=None):
        assert len(ts) > 1
        ts = reshape(array(ts), [-1,1,1])
        dom_size = self.n//self.scales
        sample_stride0 = (random.randint(1,self.scales+1) if sample_stride is None else sample_stride[0])
        lb0 = random.randint((self.scales-sample_stride0)*dom_size+sample_stride0)
        ub0 = lb0+sample_stride0*(dom_size-1)+1
        dy = sample_stride0*self.dx
        sample_stride1 = (random.randint(1,self.scales+1) if sample_stride is None else sample_stride[1])
        lb1 = random.randint((self.scales-sample_stride1)*dom_size+sample_stride1)
        ub1 = lb1+sample_stride1*(dom_size-1)+1
        dx = sample_stride1*self.dx
        return [
                self.state(ts)[:,lb0:ub0:sample_stride0,lb1:ub1:sample_stride1],
                squeeze(ts),
                dy,
                dx,
                self.coordinate[lb0:ub0:sample_stride0,lb1:ub1:sample_stride1]
                ]

#%%

class BurgersEquation2d(object):
    def __init__(self, n=20, scales=5, viscosity=1, lb=1, frequency=None):
        phi = Heat2d(n=n, scales=scales, viscosity=viscosity, frequency=frequency)
        ker = array([[0,0,0],[-1,0,1],[0,0,0]])
        pdx = wrap_filter2d(ker, method='origin')
        ker = array([[0,-1,0],[0,0,0],[0,1,0]])
        pdy = wrap_filter2d(ker, method='origin')
        assert lb > 0
        self.add_constant = lb-phi.state(0).min()
        self.viscosity = viscosity
        self.phi = phi
        self.pdx = pdx
        self.pdy = pdy
        self.dx = phi.dx
        self.scales = scales
        self.n = n
    def state(self, ts):
        ts = 2*self.viscosity*reshape(array(ts), [-1,1,1])
        phi = self.phi.state(ts)+self.add_constant
        phi = -(0.5/self.viscosity)**2*log(phi)
        if phi.ndim == 2:
            phi = phi[newaxis,:,:]
        u = []
        for i in range(phi.shape[0]):
            a = reshape(self.pdx(phi[i], mode='same', boundary='wrap'), [1,self.n,self.n])
            b = reshape(self.pdy(phi[i], mode='same', boundary='wrap'), [1,self.n,self.n])
            if i == 0:
                u.append(a)
                u.append(b)
            else:
                u[0] = concatenate([u[0],a], axis=0)
                u[1] = concatenate([u[1],b], axis=0)
        u[0] = squeeze(u[0])
        u[1] = squeeze(u[1])
        return u
    def __call__(self, dts, sample_stride=None):
        ts = ([0,] if len(dts) == 1 else [])+(list(cumsum(dts)))
        dom_size = self.n//self.scales
        sample_stride = (random.randint(1,self.scales+1) if sample_stride is None else sample_stride)
        lb = random.randint((self.scales-sample_stride)*dom_size+sample_stride)
        ub = lb+sample_stride*(dom_size-1)+1
        u = self.state(ts)
        return [u[0][:,lb:ub:sample_stride,lb:ub:sample_stride],u[1][:,lb:ub:sample_stride,lb:ub:sample_stride]]

class xzma(object):
    def __init__(self, init_select=0, scales=5, pdeNum=0):
        import scipy.io as sio
        n = 500
        self.n = n
        save_path = '/home/zlong/pde_data/pde'+str(pdeNum)
        DATA = sio.loadmat(save_path+'/u'+str(init_select)+'.mat')
        self.u = DATA['u_matrix'][:n,:n]
        self.u = transpose(self.u, [2,0,1])
        self.dt = (squeeze(DATA['dt']) if 'dt' in DATA else 0.05)
        self.scales = scales
        self.dx = 2*pi/n
        assert self.n%self.scales == 0
        assert self.n>=80
        self.coordinate = zeros((n,n,2))
        self.coordinate[:,:,0] = repeat(reshape(2*pi*arange(n)/n, (n,1)), n, axis=1)
        self.coordinate[:,:,1] = self.coordinate[:,:,0].transpose()
        if pdeNum == 0:
            def a0(x):
                return zeros(x.shape[:-1])
            def a10(x):
                t = arange(x.ndim, dtype=int32)
                x = transpose(x, (t[-1],*t[:-1]))
                return cos(x[0])
            def a11(x):
                t = arange(x.ndim, dtype=int32)
                x = transpose(x, (t[-1],*t[:-1]))
                return sin(x[1])
            def a20(x):
                return zeros(x.shape[:-1])+0.5
            def a21(x):
                return zeros(x.shape[:-1])
            def b00(x):
                return zeros(x.shape[:-1])
            a22 = a20
            a0 = [a0,]
            a1 = [a10,a11]
            a2 = [a20,a21,a22]
            self.a = [a0,a1,a2,[b00,]*4]
    def __call__(self, ts, sample_stride=None):
        assert len(ts) > 1
        ts = reshape(array(ts), [-1])
        if ts.dtype == float64:
            ts = (ts/self.dt).astype(int32)
        dom_size = self.n//self.scales
        sample_stride0 = (random.randint(1,self.scales+1) if sample_stride is None else sample_stride[0])
        lb0 = random.randint((self.scales-sample_stride0)*dom_size+sample_stride0)
        ub0 = lb0+sample_stride0*(dom_size-1)+1
        dy = sample_stride0*self.dx
        sample_stride1 = (random.randint(1,self.scales+1) if sample_stride is None else sample_stride[1])
        lb1 = random.randint((self.scales-sample_stride1)*dom_size+sample_stride1)
        ub1 = lb1+sample_stride1*(dom_size-1)+1
        dx = sample_stride1*self.dx
        return [
                self.u[ts,lb0:ub0:sample_stride0,lb1:ub1:sample_stride1],
                squeeze(ts)*self.dt,
                dy,
                dx,
                self.coordinate[lb0:ub0:sample_stride0,lb1:ub1:sample_stride1]
                ]

#%%

def test_ConstCoeLinearPDE2d():
    data = ConstCoeLinearPDE2d(n=200, scales=2, a0=0.5, a1=[0.6,0.8], a2=[0.1,0.1,0.3])
    dt = 2e-2
    n = 50
    u,ts,dy,dx,coordinate = data(cumsum([dt,]*n), sample_stride=[1,1])
    import matplotlib.pyplot as plt
    h = plt.figure()
    a = h.add_subplot(111)
    for i in range(n):
        b = a.imshow(u[i], cmap='jet')
        #a.set_ylim([coordinate[:,:,0].min(), coordinate[:,:,0].max()])
        #a.set_xlim([coordinate[:,:,1].min(), coordinate[:,:,1].max()])
        c = h.colorbar(b, ax=a)
        plt.pause(0.001)
        if i != n-1:
            c.remove()
def test_VariantCoeLinearPDE2d():
    data = VariantCoeLinearPDE2d(n=200, scales=1)
    dt_scale = 5
    dt = 1e-3*dt_scale
    n = int(700/dt_scale)
    u,ts,dy,dx,coordinate = data(array(cumsum([dt,]*n)), sample_stride=[1,1])
    import matplotlib.pyplot as plt
    h = plt.figure()
    a = h.add_subplot(111)
    for i in linspace(0,n-1,50, dtype=int32):
        b = a.imshow(u[i], cmap='jet')
        #a.set_ylim([coordinate[:,:,0].min(), coordinate[:,:,0].max()])
        #a.set_xlim([coordinate[:,:,1].min(), coordinate[:,:,1].max()])
        c = h.colorbar(b, ax=a)
        plt.pause(0.001)
        if i != n-1:
            c.remove()
#%%
def test_BurgersEquation2d():
    data = BurgersEquation2d(n=100, scales=1, viscosity=0.1, lb=1, frequency=[3,3])
    dt = 2e-2
    n = 100
    x = data([dt,]*n, sample_stride=1)
    import matplotlib.pyplot as plt
    h = plt.figure()
    a = []
    a.append(h.add_subplot(2,2,1))
    a.append(h.add_subplot(2,2,2))
    a.append(h.add_subplot(2,2,3))
    a.append(h.add_subplot(2,2,4))
    c = [0,1]
    for i in range(n):
        b = a[0].imshow(x[0][i], cmap='jet')
        c[0] = h.colorbar(b, ax=a[0])
        b = a[1].imshow(x[1][i], cmap='jet')
        c[1] = h.colorbar(b, ax=a[1])
        a[2].plot(x[0][i][50,:])
        a[3].plot(x[1][i][:,50])
        plt.pause(0.001)
        c[0].remove()
        c[1].remove()
        a[2].clear()
        a[3].clear()


#%%

