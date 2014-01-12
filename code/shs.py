#!/usr/bin/env python

import numpy
import copy
import matplotlib.pyplot as plt
from scipy.integrate import quad
from astropy.cosmology import FlatLambdaCDM
import pprint

def sigmasinegamma_exact(_sigma0,_shs):
  #  try:
  sintheta=numpy.sin(_shs.theta)
  costheta=numpy.cos(_shs.theta)
  factor=_shs.moverd/_sigma0-sintheta

  if factor > 1:
    raise Exception()
  
  singamma= -factor*costheta+sintheta*numpy.sqrt(1-factor**2)
  ans= _sigma0*singamma
  return ans

def line(_sigma, _sigma0, _a0, _l02):
  return _a0/numpy.sqrt(2*numpy.pi*_l02)*numpy.exp(-(_sigma-_sigma0)**2/2/_l02)

def integrand(_sigma, _x, _sigma1, _a1, _l12, _shs):
  ans=line(_sigma,_sigma1,_a1,_l12)* numpy.cos(2*numpy.pi*_x*2*sigmasinegamma_exact(_sigma,_shs))
  return ans

def intensity(_x, _lines, _shs):
  #first term of the integrand
  ans=_lines.a1+_lines.a2

  #second term in the integrand
  width=5
  ans=ans+ quad(integrand,_shs.minsigma,_lines.sigma1+width*numpy.sqrt(_lines.l12),args=(_x,_lines.sigma1, _lines.a1, _lines.l12, _shs),limit=10000)[0]
  ans=ans+ quad(integrand,_lines.sigma2-width*numpy.sqrt(_lines.l22),_shs.maxsigma,args=(_x,_lines.sigma2, _lines.a2, _lines.l22, _shs),limit=10000)[0]
  if _lines.lambda3_0 is not None:
    ans=ans +_lines.a3
    ans=ans + quad(integrand,_lines.sigma3-width*numpy.sqrt(_lines.l32),_lines.sigma3+width*numpy.sqrt(_lines.l32),args=(_x,_lines.sigma3, _lines.a3, _lines.l32, _shs),limit=10000)[0]
 
  #background contribution
  if _lines.back !=0:
    if _x ==0:
      sky= _lines.back*(_shs.maxsigma-_shs.minsigma)
    else:
      eightpixtantheta=8*numpy.pi*_x*numpy.tan(_shs.theta)
      sky=_lines.back*((_shs.maxsigma-_shs.minsigma)+1/eightpixtantheta*(numpy.sin(eightpixtantheta*(_shs.maxsigma-_shs.littrow))-numpy.sin(eightpixtantheta*(_shs.minsigma-_shs.littrow))))
    ans=ans+sky

  #normalization for dx
  return ans / (2*_shs.xs[-1])

def counts(_lines,_shs):
  sig = []
  for x in _shs.xs:
    sig.append(intensity(x, _lines, _shs))
  sig=numpy.array(sig)*_shs.deltax*_shs.aperture*_shs.etime*_shs.eff
  return sig
#  return signal(_lines,_shs)*_shs.deltax*_shs.aperture*_shs.etime*_shs.eff

def plotspectrum(_lines):
  sigmas=numpy.arange(_lines.sigma1-5*numpy.sqrt(_lines.l12),_lines.sigma2+5*numpy.sqrt(_lines.l22),1)
  shit=[]
  for sig in sigmas:
    shit.append(line(sig,_lines.sigma1,_lines.a1,_lines.l12)+line(sig,_lines.sigma2,_lines.a2,_lines.l22))
  shit=numpy.array(shit)
  plt.clf()
  plt.plot(sigmas/100,shit)
  plt.xlabel('wavenumber (cm$^{-1}$)')
  plt.ylabel('Flux')
  plt.savefig('shsinput.eps')
  #  plt.show()

def plotcounts(_lines):
  plt.clf()
  nratios=[1/3.5,1/1.05,1,1.05,3.5]
  nratiostxt=["1/3.5","1/1.05","1","1.05","3.5"]
  #  for nratio in numpy.arange(1.2,12,20):
  ind=0
  for nratio in nratios:
    shs=SHS(_lines,nratio)
    print 2*(_lines.sigma1+_lines.sigma2-2*shs.littrow)*numpy.tan(shs.theta), 2*(_lines.sigma1-_lines.sigma2)*numpy.tan(shs.theta)
    s1=counts(lines,shs)
    s1=s1
    plt.plot(shs.xs/numpy.max(shs.xs),s1+ind*numpy.max(s1)*1.04,label='n='+nratiostxt[ind])
    plt.xlabel('x')
    plt.ylabel('counts per resolution element + offset')
    ind=ind+1
  plt.legend()
  plt.legend(prop={'size':10})
  plt.savefig('shscounts.pdf')
  
class Lines(object):
  hc=6.626e-27*3e8

  def __init__(self, plate, fiber, line):

    if (line == 'OIII'):
      self.lambda1_0=5006.843e-10 
      #self.lambda3_0=4958.911e-10
      #self.lambda2_0=4861.325e-10
      self.lambda2_0=4958.911e-10
      self.lambda3_0=None
      if (plate == 1720 and fiber == 459):
        self.z=0.217681
        self.deltav=15.9865/3e5
        self.a1_0=1630.04e-17
        self.a2_0=551.777e-17
        self.a3_0=453.1056e-17
        self.gal=6.65e-17 
        self.sky=8.71E-18 
        #if (plate == 1349 and fiber == 175):
        #self.z=0.124904
        #self.deltav=24.32/3e5
        #self.a1_0=5840.02e-17
        #self.a2_0=2059.42e-17
        #self.a3_0=1080.65356445e-17
        #self.gal=13.2e-17
        #self.sky=8.71E-18
      if (plate == 1349 and fiber == 175):
        self.z=0.124904
        self.deltav=10.0346/3e5
        self.a1_0=3341.699e-17
        self.a2_0=2059.42/5840.02 *3341.699e-17
        self.a3_0=1080.65356445e-17
        self.gal=17.2e-17
        self.sky=8.71E-18
      elif (plate == 1424 and fiber == 515):
        self.z=0.103444
        self.deltav=205.888/3e5
        self.a1_0=10266.7e-17
        self.a2_0=3436.22e-17
        self.gal=29e-17
        self.sky=8.71E-18
      elif (plate == 649 and fiber == 117):
        self.z=0.120763
        self.deltav=12.6515/3e5
        self.a1_0=1898.23e-17
        self.a2_0=639.61e-17
        self.gal=13.6E-17 
        self.sky=8.71E-18
      elif (plate == 585 and fiber == 22):
        self.z=0.5483138
        self.deltav=58.42435/3e5
        self.a1_0=1611.298e-17
        self.a2_0=563.9542e-17
        self.gal=4.6E-17 
        self.sky=8.71E-18 ##fix
      elif (plate == 1758 and fiber == 490):
        self.z=0.255127
        self.deltav=19.15351/3e5
        self.a1_0=1240.638e-17
        self.a2_0=434.2233e-17
        self.gal=10.6E-17 
        self.sky=4.61E-18  ##fix
      if (plate == 1268 and fiber == 318):
        self.z=0.1257486
        self.deltav=10.03548/3e5
        self.a1_0=3438.711e-17
        self.a2_0=1203.549e-17
        self.gal=(16.56708+15.54943)/2*1e-17
        self.sky=8.71E-18
      if (plate == 1935	 and fiber == 204):
        self.z=0.09838755
        self.deltav=10.03865/3e5
        self.a1_0=3547.454e-17
        self.a2_0=1241.609e-17
        self.gal=21e-17
        self.sky=8.71E-18
      if (plate == 1657	 and fiber == 483):
        self.z=0.2213398
        self.deltav=10.30226/3e5
        self.a1_0=1699.26e-17
        self.a2_0=594.7409e-17
        self.gal=(13.99286+12.74685)/2*1e-17
        self.sky=6.6e-18
      if (plate == 4794	 and fiber == 757):
        self.z=0.5601137
        self.deltav=26.45063/3e5
        self.a1_0=124.1981e-17
        self.a2_0=43.46934e-17
        self.gal=(2.72337+2.504938)/2*1e-17
        self.sky=10.6e-18 #
      if (plate == 1514	 and fiber == 137):
        self.z=0.318046
        self.deltav=10.00679/3e5
        self.a1_0=55.258061e-17
        self.a2_0=19.34032e-17
        self.gal=(14.41701+13.46722)/2*1e-17
        self.sky=10.6e-18 #
 
    #[OII] http://arxiv.org/pdf/1310.0615.pdf
    elif (line == 'OII'):
      self.lambda1_0=3729.875e-10
      self.lambda2_0=3727.092e-10
      self.lambda3_0=None
      if (plate == 1610 and fiber == 379):
        self.z=0.106548
        self.deltav=83.0663/3e5
        self.a1_0=1564.34e-17
        self.a2_0=1656.03e-17
        self.gal=44.1E-17 
        self.sky=4.61E-18
      elif (plate == 0766 and fiber == 492):
        self.z=0.0959272
        self.deltav=47.4223/3e5
        self.a1_0=980.253e-17
        self.a2_0=1223.54e-17
        self.gal=33.5E-17 
        self.sky=4.61E-18 
        #      elif (plate == 1349 and fiber == 175):
        #self.z=0.124904
        #self.deltav=24.32/3e5
        #self.a1_0=898.3423e-17
        #self.a2_0=903.752e-17
        #self.gal=22.14E-17 
        #self.sky=4.61E-18
      elif (plate == 1349 and fiber == 175): #352797
        self.z=0.124904
        self.deltav=10.07/3e5
        self.a1_0=1602.899e-17
        self.a2_0=864.0497e-17
        self.gal=20.9E-17 
        self.sky=4.61E-18
      elif (plate == 649 and fiber == 117):
        self.z=0.120763
        self.deltav=12.6515/3e5
        self.a1_0=695.0e-17
        self.a2_0=851.595e-17
        self.gal=20.55E-17 
        self.sky=4.61E-18
      elif (plate == 1758 and fiber == 490):
        self.z=0.255127
        self.deltav=20/3e5
        self.a1_0=599.7573e-17
        self.a2_0=493.4042e-17
        self.gal=12.7E-17 
        self.sky=4.61E-18  ##fix
      elif (plate == 1814 and fiber == 352):
        self.z=0.3754629
        self.deltav=35.5/3e5
        self.a1_0=395.2258e-17
        self.a2_0=484.433e-17
        self.gal=10.75E-17 
        self.sky=4.61E-18  ##fix
      elif (plate == 1268 and fiber == 318):
        self.z=0.1257486	
        self.deltav=(10.0547+10.03259)/2/3e5
        self.a1_0=609.7103e-17
        self.a2_0=686.3459e-17
        self.gal=(17.36104+16.66029)/2*1E-17 
        self.sky=4.61E-18  ##fix
      elif (plate == 1935 and fiber == 204):
        self.z=0.09838755	
        self.deltav=10.05437/3e5
        self.a1_0=1515.795e-17
        self.a2_0=786.8914e-17
        self.gal=18E-17 
        self.sky=4.61E-18  ##fix
      elif (plate == 1657 and fiber == 483):
        self.z=0.2213398	
        self.deltav=(10.35904+10.35904)/2/3e5
        self.a1_0=459.011e-17
        self.a2_0=476.065e-17
        self.gal=(14.72571+13.4333)/2*1E-17 
        self.sky=5.15E-18  ##fix
      elif (plate == 4794 and fiber == 757):
        self.z=0.5601137		
        self.deltav=(49.60205+42.17744)/2/3e5
        self.a1_0=305.8288e-17
        self.a2_0=207.7097e-17
        self.gal=(1.292684+1.193439)/2*1E-17 
        self.sky=7.15E-18  ##fix
      elif (plate == 1514 and fiber == 137):
        self.z=0.318046		
        self.deltav=(10.00539+10.0053)/2/3e5
        self.a1_0=125.847e-17
        self.a2_0=115.5477e-17
        self.gal=(8.154473+7.531853)/2*1E-17 
        self.sky=6.15E-18  ##fix

    if (line == 'OII&OIII'):
      self.lambda1_0=5006.843e-10 
      self.lambda3_0=None
      self.lambda2_0=3729.875e-10
      if (plate == 1720 and fiber == 459):
        self.z=0.217681
        self.deltav=15.9865/3e5
        self.a1_0=1630.04e-17
        self.a2_0=496.727e-17
        self.gal=6.65e-17
        self.sky=8.71E-18
    seeing = 1.
    self.aperture=numpy.pi*(seeing/2)**2

    self.setz(self.z)

  def setz(self,z):
    self.z=z
    lambda1=self.lambda1_0*(1+self.z)
    lambda2=self.lambda2_0*(1+self.z) 
    mnlambda=(lambda1+lambda2)/2
                                   
    self.sigma2=1/lambda2
    self.sigma1=1/lambda1

    self.l12=(self.sigma1*self.deltav)**2
    self.l22=(self.sigma2*self.deltav)**2

    self.a1=self.a1_0 * lambda1/Lines.hc  #erg/cm2/s to photons
    self.a2=self.a2_0 * lambda2/Lines.hc   #erg/cm2/s

    if self.lambda3_0 is not None:
      lambda3=self.lambda3_0*(1+self.z) 
      self.sigma3=1/lambda3
      self.l32=(self.sigma3*self.deltav)**2
      self.a3=self.a3_0 * lambda3/Lines.hc   #erg/cm2/s
    
    self.back=(self.gal+self.sky) * mnlambda /Lines.hc # for photons V=21.8
    
    self.back = self.back * 1e10 # Angstrom to m
    self.back = self.back*(mnlambda**2) #per wavelength to per wavenumber
    self.back = self.back*self.aperture


#A description of the hardware
class SHS(object):
  def __init__(self,_lines,n):
    self.littrow=(_lines.sigma1+_lines.sigma2)/2-(_lines.sigma1-_lines.sigma2)/2./n
    self.moverd=1.*(1200*1e3)
    #self.moverd=1.*(2000*1e3)
    self.theta = numpy.arcsin(self.moverd/2/self.littrow)

    # the ranges is defined by the width of the line
    fdecay =12*numpy.sqrt(_lines.l12)*numpy.tan(self.theta)
    xrange=numpy.abs(1/fdecay) 
    
    self.npt=1024
    SHS.setXRange(self,xrange)

    self.minsigma=_lines.sigma1-3*numpy.sqrt(_lines.l12)
    self.maxsigma=_lines.sigma2+3*numpy.sqrt(_lines.l22)

    self.aperture=numpy.pi*(10e2/2)**2 
    self.etime=3600.*8

    self.eff=.7

  def setXRange(self,xrange):
    self.deltax=1.00*xrange/self.npt
    self.xs=numpy.arange(0*xrange,1*xrange,self.deltax)

def oneratiopartials(_lines,nratio):
  _lines2=copy.copy(_lines)
  _lines2.setz(_lines2.z+1e-10)
  shs=SHS(_lines,nratio)
  #print shs.theta, sigmasinegamma_exact(_lines.sigma1,shs), sigmasinegamma_exact(_lines.sigma2,shs)
  sig=counts(_lines, shs)
  sig2=counts(_lines2, shs)
  partials=(sig2-sig)/1e-10
  return partials,sig


def uncertaintyvsnratio(_lines):
  binsize=0.1
  xax=numpy.append(1/numpy.arange(5,1,-binsize),numpy.arange(1,5.00001,binsize))
  #xax=10**numpy.arange(-3,1,.25)
  #xax=numpy.arange(.001,numpy.pi/4,.01)
  ans=[]
  for nratio in xax:
    partials = oneratiopartials(_lines,nratio)
    ans.append(numpy.sqrt(1/numpy.sum(partials[0]*partials[0]/partials[1])/2))
  ans=numpy.array(ans)
  plt.clf()
  plt.plot(xax,ans)
  plt.xlabel('n')
  plt.ylabel('z uncertainty')
  #  plt.xscale('log')
  plt.savefig('uncertaintyvsnratio.eps')


def fisherandplot(_lines):
  allpartials=None
  allsignals=None
  plt.clf()
  #for nratio in numpy.arange(.1,.5,1):
  for nratio in numpy.arange(1.5,9,10):
    shs=SHS(_lines,nratio)
    partials_ = oneratiopartials(_lines,nratio)
    if allpartials is None:
      allpartials = partials_[0]
      allsignals=partials_[1]
    else:
      allpartials=numpy.append(allpartials,partials_[0])
      allsignals=numpy.append(allsignals,partials_[1])
    plt.plot(shs.xs*100,partials_[0],label='n='+str(nratio))
  plt.legend()
  plt.xlabel('x (cm)')
  plt.ylabel('d(counts)/dz')
  plt.savefig('dsigdz.eps')
  fisher=numpy.sum(allpartials*allpartials/allsignals) * 2 #the extra 2 for the -x values
  return fisher

def twoline(plate, fiber):
  lines=Lines(plate,fiber,'OII')
  fisher=fisherandplot(lines)
  o2=1/numpy.sqrt(fisher)
  print o2,
  lines=Lines(plate,fiber,'OIII')
  fisher=fisherandplot(lines)
  o3=1/numpy.sqrt(fisher)
  print o3,
  print 1./numpy.sqrt(1/o2**2+1/o3**2)


lines=Lines(1268,318,'OIII')
plotcounts(lines)
shit


print 1935, 204, Lines(1935,204,'OII').z,
twoline(1935,204)
print 1268, 318, Lines(1268,318,'OII').z,
twoline(1268,318)
print 1657, 483, Lines(1657,483,'OII').z,
twoline(1657,483) #53520
print 1514, 137, Lines(1514,137,'OII').z,
twoline(1514,137)
print 4794, 757, Lines(4794,757,'OII').z,
twoline(4794,757)
#twoline(1349,175) not happy
print shit

#lines=Lines(1349,175,'OIII')
#lines=Lines(1349,175,'OII')
#lines=Lines(649,117,'OII')
#lines=Lines(649,117,'OIII')
#lines=Lines(585,22,'OIII')
#lines=Lines(1814,352,'OII') #one with only OII at high redshift
#lines=Lines(1758,490,'OII') #one at z=0.25
#lines=Lines(1758,490,'OIII')
#lines=Lines(1268,318,'OIII')
#lines=Lines(1268,318,'OII')
#lines=Lines(1935,204,'OIII')
#lines=Lines(1935,204,'OII')

#shs=SHS(lines,1.5)
#print shs.theta, shs.littrow, lines.sigma1,lines.sigma2
#print shit
#plotspectrum(lines)
#print shit

#plotcounts(lines)
#print shit
#uncertaintyvsnratio(lines)
#print shit

fisher=fisherandplot(lines)
print 1/numpy.sqrt(fisher)

#def gamma_exact(_sigma0,_shs):
#  ans=_shs.moverd/_sigma0
#  ans=ans-numpy.sin(_shs.theta)
  #  if numpy.abs(ans) > 1:
  #    raise Exception()
  #  ans=numpy.arcsin(ans)  
  #  return _shs.theta-ans

  #def sigmasinegamma_exact(_sigma0,_shs):
  #  try:
  #  ans = _sigma0*numpy.sin(gamma_exact(_sigma0,_shs))
    #  except:
    #  raise Exception()
    #  return ans

#def gamma(_sigma0,_shs):
#  return numpy.arcsin(2*(_sigma0-_shs.littrow)/_sigma0*numpy.tan(_shs.theta))

#def sigmasinegamma(_sigma0,_shs):
#  return 2*(_sigma0-_shs.littrow)*numpy.tan(_shs.theta)

#def signal(_lines, _shs):
#  sig = []
#  for x in _shs.xs:
#    sig.append(intensity(x, _lines, _shs))
#  sig=numpy.array(sig)
#  return sig
#def plotsignals(_lines):
#  plt.clf()
#  for nratio in numpy.arange(1.5,5,10):
#    shs=SHS(_lines,nratio)
#    s1=signal(lines,shs)
#    plt.plot(shs.xs*100,s1,label='n='+str(nratio))
#    plt.xlabel('x (cm)')
#    plt.ylabel('signal')
#  plt.legend()
#  plt.savefig('shssignal.eps')

