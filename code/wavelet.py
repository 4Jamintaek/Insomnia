#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 15:26:26 2017

@author: J & M





"""

import numpy as np
from scipy.signal import cwt
from scipy.signal import ricker
from scipy.signal import gausspulse
from scipy.signal import convolve
from scipy.signal import morlet
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

"""
# test 1
length = 526
width = length/10
a = morlet(length,w = 5.0,s=1.0,complete = True)
b = ricker (length,width)
t = np.linspace(-1,1,length)
plt.figure()
plt.plot(t,a)
plt.figure()
plt.plot(t,b)
"""
"""
# test 2
plt.close('all')
t = np.linspace(-1, 1, 200, endpoint=False)
sig1  = np.cos(2 * np.pi * 7 * t)
sig2 = gausspulse(t - 0.4, fc=2)
sig = sig1 + sig2
widths = np.arange(1, 31)
cwtmatr = cwt(sig, ricker, widths)
plt.figure()
plt.plot(t,sig1)
plt.figure()
plt.plot(t,sig2)
plt.figure()
plt.plot(t,sig)
plt.figure()
plt.imshow(cwtmatr, extent=[-1, 1, 1, 31], cmap='PRGn', aspect='auto',
           vmax=abs(cwtmatr).max(), vmin=-abs(cwtmatr).max())
plt.show()
"""

def morlcwt (data,M,w,s,complete,widths):
    
    """
    Continuous wavelet transform.
    Adapted from scipy and modified by Junwoo and Mintaek to use morlet wavelet 
    Performs a continuous wavelet transform on `data`,
    using the `wavelet` function. A CWT performs a convolution
    with `data` using the `wavelet` function, which is characterized
    by a width parameter and length parameter.
    
    Parameters
    ----------
    data : (N,) ndarray
        data on which to perform the transform.
    M : int
        Length of morlet waveform
    w : float, optional
        Omega0
    s : float, optional
        scaling factor, windowed from -s*2*pi to +s*2*pi. Default is 1.
    Returns
    -------
    cwt: (M, N) ndarray
        Will have shape of (len(widths), len(data)).
    """
    
    
    output = np.zeros([len(widths), len(data)])
    for ind, width in enumerate(widths):
        wavelet_data = morlet(M,w = w, s = s, complete = complete)
        output[ind, :] = convolve(data, wavelet_data,
                                  mode='same')
    return output


def fixedp(f,x0,tol=10e-5,maxiter=100):
    
 """ Fixed point algorithm """
 
 e = 1
 itr = 0
 xp = []
 while(e > tol and itr < maxiter):
  x = f(x0)      # fixed point equation
  e = np.linalg.norm(x0-x) # error at the current step
  x0 = x
  xp.append(x0)  # save the solution of the current step
  itr = itr + 1
 return x,xp

"""
f = lambda x : np.sqrt(x)
x_start = .5
xf,xp = fixedp(f,x_start)

x = np.linspace(0,2,100)
y = f(x)

plt.plot(x,y,xp,f(xp),'bo',
     x_start,f(x_start),'ro',xf,f(xf),'go',x,x,'k')
"""

plt.close('all')
t = np.linspace(-1, 1, 500, endpoint=False)
m = morlet(500,w = 5, s = 1, complete = True)
fig = plt.figure()
ax = fig.add_subplot(111,projection ='3d')
ax.plot(t,np.real(m),np.imag(m))


plt.figure()
plt.plot(t,np.real(m))