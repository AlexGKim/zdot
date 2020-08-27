#!/usr/bin/env python

import numpy
from scipy.integrate import quad
# import copy
import matplotlib.pyplot as plt
# from scipy.integrate import quad
# from scipy.interpolate import interp1d
# from astropy.cosmology import FlatLambdaCDM
# import pprint


def integrand(w,t, s,w0):
  return 1./numpy.sqrt(2*numpy.pi*s**2)*numpy.exp(-0.5*(w-w0)**2/s**2)*numpy.cos(w*t)**2


w0=1667.
sig=w0*50/3e5

t=numpy.arange(0,1e0,1e-5)



ans=0.5*(1+numpy.exp(-2*sig**2*t**2)*numpy.cos(2*t*w0))

plt.plot(t,ans)
plt.show()