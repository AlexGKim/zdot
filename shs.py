#!/usr/bin/env python

import numpy
import copy
import matplotlib.pyplot as plt
from scipy.integrate import quad
from mpl_toolkits.mplot3d import Axes3D
from numpy.random import randn
from astropy.cosmology import FlatLambdaCDM

import pprint

def gamma_exact(_sigma0,_shs):
  ans=_shs.moverd/_sigma0
  ans=ans-numpy.sin(_shs.theta)
  #  if numpy.abs(ans) > 1:
  #    raise Exception()
  ans=numpy.arcsin(ans)  
  return _shs.theta-ans

def sigmasinegamma_exact(_sigma0,_shs):
  #  try:
  ans = _sigma0*numpy.sin(gamma_exact(_sigma0,_shs))
    #  except:
    #  raise Exception()
  return ans

def sigmasinegamma_exact2(_sigma0,_shs):
  #  try:
  sintheta=numpy.sin(_shs.theta)
  costheta=numpy.cos(_shs.theta)
  factor=_shs.moverd/_sigma0-sintheta

  if factor > 1:
    raise Exception()
  
  singamma= -factor*costheta+sintheta*numpy.sqrt(1-factor**2)
  ans= _sigma0*singamma
  return ans

#def gamma(_sigma0,_shs):
#  return numpy.arcsin(2*(_sigma0-_shs.littrow)/_sigma0*numpy.tan(_shs.theta))

#def sigmasinegamma(_sigma0,_shs):
#  return 2*(_sigma0-_shs.littrow)*numpy.tan(_shs.theta)

def line(_sigma, _sigma0, _a0, _l02):
  return _a0/numpy.sqrt(2*numpy.pi*_l02)*numpy.exp(-(_sigma-_sigma0)**2/2/_l02)

def integrand(_sigma, _x, _sigma1, _a1, _l12, _shs):
  ans=line(_sigma,_sigma1,_a1,_l12)* numpy.cos(2*numpy.pi*_x*2*sigmasinegamma_exact2(_sigma,_shs))
  return ans

def intensity(_x, _lines, _shs):
  #first term of the integrand
  ans=_lines.a1+_lines.a2

  #second term in the integrand
  #integrator doesn't like going too far so set the extreme width here
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

def signal(_lines, _shs):
  sig = []
  for x in _shs.xs:
    sig.append(intensity(x, _lines, _shs))
  sig=numpy.array(sig)
  return sig

def counts(_lines,_shs):
  return signal(_lines,_shs)*_shs.deltax*_shs.aperture*_shs.etime*_shs.eff

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

def plotsignals(_lines):
  plt.clf()
  for nratio in numpy.arange(1.5,5,10):
    shs=SHS(_lines,nratio)
    s1=signal(lines,shs)
    plt.plot(shs.xs*100,s1,label='n='+str(nratio))
    plt.xlabel('x (cm)')
    plt.ylabel('signal')
  plt.legend()
  plt.savefig('shssignal.eps')

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
  plt.savefig('shscounts.eps')
  
class Lines(object):
  hc=6.626e-27*3e8

  def __init__(self, plate, fiber, line):

    if (line == 'OIII'):
      self.lambda1_0=5006.843e-10 
      self.lambda3_0=4958.911e-10
      self.lambda2_0=4861.325e-10
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
      if (plate == 1349 and fiber == 175):
        self.z=0.124904
        self.deltav=24.32/3e5
        self.a1_0=5840.02e-17
        self.a2_0=2059.42e-17
        self.a3_0=1080.65356445e-17
        self.gal=13.2e-17
        self.sky=8.71E-18
      elif (plate == 1424 and fiber == 515):
        self.z=0.103444
        self.deltav=205.888/3e5
        self.a1_0=10266.7e-17
        self.a2_0=3436.22e-17
        self.gal=29e-17
        self.sky=8.71E-18
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
      elif (plate == 1349 and fiber == 175):
        self.z=0.124904
        self.deltav=24.32/3e5
        self.a1_0=898.3423e-17
        self.a2_0=903.752e-17
        self.gal=22.14E-17 
        self.sky=4.61E-18
      elif (plate == 649 and fiber == 117):
        self.z=0.120763
        self.deltav=12.6515/3e5
        self.a1_0=695.0e-17
        self.a2_0=851.595e-17
        self.gal=20.55E-17 
        self.sky=4.61E-18
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
    #print self.littrow,n
    #print _lines.sigma1+_lines.sigma2-2*self.littrow,  _lines.sigma1-_lines.sigma2
    #print shhit
    self.moverd=1.*(1200*1e3)
    #self.moverd=1.*(2000*1e3)
    self.theta = numpy.arcsin(self.moverd/2/self.littrow)

    # the ranges is defined by the width of the line
    fdecay =8*numpy.sqrt(_lines.l12)*numpy.tan(self.theta)
    xrange=numpy.abs(1/fdecay) 
    
    self.npt=1024
    SHS.setXRange(self,xrange)

    self.minsigma=_lines.sigma1-3*numpy.sqrt(_lines.l12)
    self.maxsigma=_lines.sigma2+3*numpy.sqrt(_lines.l22)

    self.aperture=numpy.pi*(10e2/2)**2 
    self.etime=3600.*8

    self.eff=.8

  def setXRange(self,xrange):
    self.deltax=1.00*xrange/self.npt
    self.xs=numpy.arange(0*xrange,1*xrange,self.deltax)

    #lines=Lines(1720,459,'OII&OIII')
    #lines=Lines(1720,459,'OIII')
lines=Lines(1349,175,'OIII')
    #lines=Lines(1610,379,'OII')
    #lines=Lines(649,117,'OII')

    #shs=SHS(lines,1)
    #print shs.theta, shs.littrow, lines.sigma1,lines.sigma2
    #print shit
  #plotspectrum(lines)
#print shit

#plotcounts(lines)
#print shit


def oneratiopartials(_lines,nratio):
  _lines2=copy.copy(_lines)
  _lines2.setz(_lines2.z+1e-10)
  shs=SHS(_lines,nratio)
  print shs.theta, sigmasinegamma_exact2(_lines.sigma1,shs), sigmasinegamma_exact2(_lines.sigma2,shs)
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

  #uncertaintyvsnratio(lines)
  #print shit

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

fisher=fisherandplot(lines)
print 1/numpy.sqrt(fisher)

print shit
def dbeatdz_theta():
  shs=SHS(lines,1)

  ans=[]
  ans2=[]
  x= numpy.arange(-numpy.pi/2,numpy.pi/2,.01)
  for theta in x:
    shs.theta=theta

    d1=dsigmasinegammadsigma(lines.sigma1,shs)/ lines.lambda1_0/(1+lines.z)**2
    d2=dsigmasinegammadsigma(lines.sigma2,shs)/lines.lambda2_0/(1+lines.z)**2
    
    ans.append(d1+d2)
    ans2.append(d1-d2)
  plt.clf()
  plt.plot(x,ans)
  plt.plot(x,ans2)
  plt.show()

def dbeatdz():
  shs=SHS(lines,1)

  x= numpy.arange(-numpy.pi/2,numpy.pi/2,.001)
  y= numpy.arange(600*1e3,2400*1e3,1e4)
  thetas=numpy.zeros((len(x),len(y)))
  moverds=numpy.zeros((len(x),len(y)))
  ans=numpy.zeros((len(x),len(y)))
  ans2=numpy.zeros((len(x),len(y)))
  for i in xrange(len(x)):
    for j in xrange(len(y)):
      shs.theta=x[i]
      shs.moverd=y[j]
      d1=dsigmasinegammadsigma(lines.sigma1,shs)/ lines.lambda1_0/(1+lines.z)**2
      d2=dsigmasinegammadsigma(lines.sigma2,shs)/lines.lambda2_0/(1+lines.z)**2
    
      ans[i,j]=(d1+d2)
      ans2[i,j]=(d1-d2)
      thetas[i,j]=shs.theta
      moverds[i,j]=shs.moverd
      
  ans[numpy.isnan(ans)]=0
  maxin=numpy.unravel_index(numpy.argmax(ans),ans.shape)
  print maxin,ans.shape,ans[maxin]
  print thetas[maxin],moverds[maxin],ans[maxin]
  plt.clf()

  fig = plt.figure()
  ax = fig.gca(projection='3d')

  ax.plot_surface(thetas,moverds,ans)
  plt.show()
  plt.plot(x,ans2)
  plt.show()

  #dbeatdz()

  #print shit

def foreric(_lines):
  allpartials=None
  allsignals=None
  plt.clf()
  #for nratio in numpy.arange(.1,.5,1):
  for nratio in numpy.arange(3,9,10):
    shs=SHS(_lines,nratio)
    partials_ = oneratiopartials_a(_lines,nratio)
    if allpartials is None:
      allpartials = partials_[0]
      allsignals=partials_[1]
    else:
      allpartials=numpy.append(allpartials,partials_[0])
      allsignals=numpy.append(allsignals,partials_[1])
    plt.plot(shs.xs*100,partials_[0],label='n='+str(nratio))
  plt.legend()
  plt.xlabel('x (cm)')
  plt.ylabel('dln(count)/dln(1+z)')
  plt.savefig('dcdln1plusz.eps')
  return fisher

#foreric(lines)
#print shit
def dbeatdz_moverd():
  shs=SHS(lines,1)

  ans=[]
  ans2=[]
  x=numpy.arange(lines.sigma1*(1+numpy.sqrt(2)),lines.sigma1*(1+numpy.sqrt(2))+100,100)
  for moverd in x:
    shs.moverd=moverd
    try:
      d1=dsigmasinegammadsigma(lines.sigma1,shs) * lines.sigma1_0/(1+lines.z)**2
      d2=dsigmasinegammadsigma(lines.sigma2,shs) * lines.sigma2_0/(1+lines.z)**2
    except:
      d1=0
      d2=0
    ans.append(d1+d2)
    ans2.append(d1-d2)
  ans=numpy.array(ans)
  ans2=numpy.array(ans2)
  plt.clf()
  plt.plot(x,ans)
  plt.plot(x,ans2)
  plt.show()
def oneratiopartials_a(_lines,nratio):
  _lines2=copy.copy(_lines)
  var=(1+_lines2.z)
  var=numpy.log(1+_lines2.z)+1e-10
  var=numpy.exp(var)-1
  _lines2.setz(var)
  shs=SHS(_lines,nratio)
  sig=counts(_lines, shs)
  sig2=counts(_lines2, shs)
  print (sigmasinegamma_exact2(_lines.sigma1,shs)+sigmasinegamma_exact2(_lines.sigma2,shs)),
  print (sigmasinegamma_exact2(_lines.sigma1,shs)-sigmasinegamma_exact2(_lines.sigma2,shs))
  print (sigmasinegamma_exact2(_lines2.sigma1,shs)+sigmasinegamma_exact2(_lines2.sigma2,shs)),
  print (sigmasinegamma_exact2(_lines2.sigma1,shs)-sigmasinegamma_exact2(_lines2.sigma2,shs))
  partials=numpy.log(sig2/sig)/1e-10
  return partials,sig

