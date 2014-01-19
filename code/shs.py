#!/usr/bin/env python

import numpy
import copy
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.interpolate import interp1d
from astropy.cosmology import FlatLambdaCDM
import pprint

def skyflux(_in):
  mag=numpy.array([22.08,22.81,21.79,21.19,19.85])
  wavelength=numpy.array([3650,4450,5510,6580,8080])*1e-10
  flux=numpy.array([6.36e-18,4.96e-18,6.75e-18,6.28e-18,1.16e-17])
  fn=interp1d(wavelength,flux)
  return fn(_in)
  
def sigmasinegamma_exact(_sigma0,_shs):
  #  try:
  sintheta=numpy.sin(_shs.theta)
  costheta=numpy.cos(_shs.theta)
  factor=_shs.moverd/_sigma0-sintheta

  if numpy.sum(factor > 1) != 0:
    raise Exception()
  
  singamma= -factor*costheta+sintheta*numpy.sqrt(1-factor**2)
  ans= _sigma0*singamma
  return ans

def line(_sigma, _sigma0, _a0, _l02):
  return _a0/numpy.sqrt(2*numpy.pi*_l02)*numpy.exp(-(_sigma-_sigma0)**2/2/_l02)

def integrand(_sigma, _x, _sigma1, _a1, _l12, _shs):
  ans=line(_sigma,_sigma1,_a1,_l12)* numpy.cos(2*numpy.pi*(_x*2*sigmasinegamma_exact(_sigma,_shs)+_shs.tau*_sigma*(1+_shs.phase)))
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

def ed_counts(_lines, _shs):
  ans=numpy.zeros((len(_shs.xs),len(_shs.sigmas)))
  for i in xrange(len(_shs.xs)):
    ans[i]=line(_shs.sigmas,_lines.sigma1,_lines.a1,_lines.l12)
    ans[i]=ans[i]+line(_shs.sigmas,_lines.sigma2,_lines.a2,_lines.l22)
    ans[i]=ans[i]+integrand(_shs.sigmas,_shs.xs[i],_lines.sigma1,_lines.a1,_lines.l12,_shs)
    ans[i]=ans[i]+integrand(_shs.sigmas,_shs.xs[i],_lines.sigma2,_lines.a2,_lines.l22,_shs)
    ans[i]=ans[i]+ _lines.back
    ans[i]=ans[i]*_shs.binwidths
  ans=ans*_shs.deltax*_shs.aperture*_shs.etime*_shs.eff/ (2*_shs.xs[-1])
  #  plt.imshow(numpy.log(ans),aspect='equal')
  #plt.show()
  return ans
  

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
      if (plate == 1349 and fiber == 175):
        self.z=0.124904
        self.deltav=10.0346/3e5
        self.a1_0=3341.699e-17
        self.a2_0=2059.42/5840.02 *3341.699e-17
        self.a3_0=1080.65356445e-17
        self.gal=17.2e-17
      elif (plate == 1424 and fiber == 515):
        self.z=0.103444
        self.deltav=205.888/3e5
        self.a1_0=10266.7e-17
        self.a2_0=3436.22e-17
        self.gal=29e-17
      elif (plate == 649 and fiber == 117):
        self.z=0.120763
        self.deltav=12.6515/3e5
        self.a1_0=1898.23e-17
        self.a2_0=639.61e-17
        self.gal=13.6E-17 
      elif (plate == 585 and fiber == 22):
        self.z=0.5483138
        self.deltav=58.42435/3e5
        self.a1_0=1611.298e-17
        self.a2_0=563.9542e-17
        self.gal=4.6E-17 
      elif (plate == 1758 and fiber == 490):
        self.z=0.255127
        self.deltav=19.15351/3e5
        self.a1_0=1240.638e-17
        self.a2_0=434.2233e-17
        self.gal=10.6E-17 
      if (plate == 1268 and fiber == 318):
        self.z=0.1257486
        self.deltav=10.03548/3e5
        self.a1_0=3315.32e-17 #3438.711e-17
        self.a2_0=1094.53e-17 #1203.549e-17
        self.gal=(14.7624+14.3708)/2*1e-17
      if (plate == 1935	 and fiber == 204):
        self.z=0.09838755
        self.deltav=10.03865/3e5
        self.a1_0=5934.72e-17 #3547.454e-17
        self.a2_0=1959.3e-17 #1241.609e-17
        self.gal=(19.3408+19.0352)/2*1e-17
      if (plate == 1657	 and fiber == 483):
        self.z=0.2213398
        self.deltav=10.30226/3e5
        self.a1_0=1839.82e-17 #1699.26e-17
        self.a2_0=607.401e-17 # 594.7409e-17
        self.gal=(11.0375+10.6736)/2*1e-17
      if (plate == 4794	 and fiber == 757):
        self.z=0.5601137
        self.deltav=26.45063/3e5
        self.a1_0=25.0992e-17 #124.1981e-17
        self.a2_0=8.28632e-17 #43.46934e-17
        self.gal=(1.54566+1.47933)/2*1e-17
      if (plate == 1514	 and fiber == 137):
        self.z=0.318046
        self.deltav=10.00679/3e5
        self.a1_0=116.704e-17#55.258061e-17
        self.a2_0=38.5289e-17#19.34032e-17
        self.gal=(10.9357+10.4893)/2*1e-17
      if (plate == 1036 and fiber == 584):
        self.mjd=52562
        self.z=0.1078989
        self.deltav=4.545177/3e5
        self.a1_0=105.064e-17
        self.a2_0=34.6861e-17
        self.gal=(4.7564+4.08471)/2*1e-17
      if (plate == 1073 and fiber == 225):
        self.mjd=52649
        self.z=0.2716023
        self.deltav=1.428721/3e5
        self.a1_0=113.574e-17
        self.a2_0=37.4957e-17
        self.gal=(3.88981+3.88917)/2*1e-17
      if (plate == 1523 and fiber == 602):
        self.mjd=52937
        self.z=0.08933221
        self.deltav=5.1353/3e5
        self.a1_0=213.148e-17
        self.a2_0=70.3692e-17
        self.gal=(1.875331+2.05596)/2*1e-17
      if (plate == 2959 and fiber == 354):
        self.mjd=54537
        self.z=0.1199353
        self.deltav=6.857259/3e5
        self.a1_0=33.6067e-17
        self.a2_0=11.095e-17
        self.gal=(0.165449+0.137032)/2*1e-17
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
      elif (plate == 0766 and fiber == 492):
        self.z=0.0959272
        self.deltav=47.4223/3e5
        self.a1_0=980.253e-17
        self.a2_0=1223.54e-17
        self.gal=33.5E-17 
      elif (plate == 1349 and fiber == 175): #352797
        self.z=0.124904
        self.deltav=10.07/3e5
        self.a1_0=1602.899e-17
        self.a2_0=864.0497e-17
        self.gal=20.9E-17 
      elif (plate == 649 and fiber == 117):
        self.z=0.120763
        self.deltav=12.6515/3e5
        self.a1_0=695.0e-17
        self.a2_0=851.595e-17
        self.gal=20.55E-17 
      elif (plate == 1758 and fiber == 490):
        self.z=0.255127
        self.deltav=20/3e5
        self.a1_0=599.7573e-17
        self.a2_0=493.4042e-17
        self.gal=12.7E-17 
      elif (plate == 1814 and fiber == 352):
        self.z=0.3754629
        self.deltav=35.5/3e5
        self.a1_0=395.2258e-17
        self.a2_0=484.433e-17
        self.gal=10.75E-17 
      elif (plate == 1268 and fiber == 318):
        self.z=0.1257486	
        self.deltav=(10.0547+10.03259)/2/3e5
        self.a1_0=629.109e-17 #686.3459e-17
        self.a2_0=589.096e-17 # 609.7103e-17
        self.gal=(17.6749+17.7632)/2*1E-17 
      elif (plate == 1935 and fiber == 204):
        self.z=0.09838755	
        self.deltav=10.05437/3e5
        self.a1_0=1276.21e-17 #1515.795e-17
        self.a2_0=1055.79e-17 #786.8914e-17
        self.gal=(23.8402+23.9842)/2*1E-17 
      elif (plate == 1657 and fiber == 483):
        self.z=0.2213398	
        self.deltav=(10.35904+10.35904)/2/3e5
        self.a1_0=427.779e-17 #459.011e-17
        self.a2_0=469.207e-17 #476.065e-17
        self.gal=(13.5476+13.5792)/2*1E-17 
      elif (plate == 4794 and fiber == 757):
        self.z=0.5601137		
        self.deltav=(49.60205+42.17744)/2/3e5
        self.a1_0=31.8402e-17 #305.8288e-17
        self.a2_0=20.2544e-17 #207.7097e-17
        self.gal=(0.819377+0.816817)/2*1E-17 
      elif (plate == 1514 and fiber == 137):
        self.z=0.318046		
        self.deltav=(10.00539+10.0053)/2/3e5
        self.a1_0=173.733e-17 #125.847e-17
        self.a2_0=120.246e-17 #115.5477e-17
        self.gal=(6.33959+6.35175)/2*1E-17 
      elif (plate == 1059 and fiber == 564):
        self.z=0.693329
	self.mjd=52592
        self.deltav=(9.460785+9.699522)/2/3e5
        self.a1_0=1943.71e-17
        self.a2_0=1.39517e-17
        self.gal=(1.46043+1.47121)/2*1E-17 
      elif (plate == 1073 and fiber == 225):
        self.z=0.2716023
	self.mjd=52649
        self.deltav=(84.61399+84.87743)/2/3e5
        self.a1_0=104.709e-17
        self.a2_0=83.7931e-17
        self.gal=(3.33375+3.39536)/2*1E-17 
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

    sky=skyflux(mnlambda)
    self.back=(self.gal+sky) * mnlambda /Lines.hc # for photons V=21.8
    
    self.back = self.back * 1e10 # Angstrom to m
    self.back = self.back*(mnlambda**2) #per wavelength to per wavenumber
    self.back = self.back*self.aperture


#A description of the hardware
class SHS(object):
  def __init__(self,_lines,n):
    #self.littrow=(_lines.sigma1+_lines.sigma2)/2-(_lines.sigma1-_lines.sigma2)/2./n
    self.littrow=(_lines.sigma1+_lines.sigma2)/2
    self.moverd=1.*(1200*1e3)
    #self.moverd=1.*(2000*1e3)
    self.theta = numpy.arcsin(self.moverd/2/self.littrow)
   #self.tau = 1./(_lines.sigma2-_lines.sigma1)
    self.tau=0
    self.phase = 0

    self.r=20000
    self.edge=0.001
    nmax=numpy.log(_lines.sigma2*(1.+self.edge)/_lines.sigma1/(1-self.edge))*self.r
    self.sigmas=(_lines.sigma1*(1-self.edge))*numpy.exp(numpy.arange(0,nmax)/self.r)
    self.binwidths = self.sigmas/self.r

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

class EDI(object):
  def __init__(self,_lines,taufactor,dz):
    self.r=20000
    self.edge=0.0005
    #mnsigma=(1/_lines.lambda1_0+1/_lines.lambda2_0)/2
    #dsigma=mnsigma*(1/(1+_lines.z)-1/(1+_lines.z+dz))    
    self.tau=1/(2.36*numpy.sqrt(_lines.l12)*2)
    
    self.aperture=numpy.pi*(10e2/2)**2 
    self.etime=3600.*8

    nmax=numpy.log(_lines.sigma2*(1.+self.edge)/_lines.sigma1/(1-self.edge))*self.r
    self.subres=100
    #self.nus=(_lines.sigma1)*(1-self.edge)*numpy.exp(numpy.arange(0,nmax)/self.r)
    self.finenus=(_lines.sigma1)*(1-self.edge)*numpy.exp(numpy.arange(0,nmax,1./self.subres)/self.r)
    self.nus=self.finenus[self.subres::self.subres]
    self.eff=.7

def edicounts(_lines,edi):
  tau=edi.tau
  
  s0=line(edi.finenus, _lines.sigma1, _lines.a1, _lines.l12)+line(edi.finenus, _lines.sigma2, _lines.a2, _lines.l22)
  s0=s0+_lines.back*edi.finenus/edi.r/edi.subres
  s0=s0*edi.aperture*edi.etime*edi.eff
  sw=[]
  plab=['0','$\pi/2$','$\pi$','$3\pi/2$']
  ind=0
  for phi in numpy.arange(0,2*numpy.pi-0.001,numpy.pi/2):
    fineans=.25*s0*(1+numpy.cos(2*numpy.pi*tau*edi.finenus+phi))
    kernel=numpy.zeros(edi.subres)+1
    fineans=numpy.convolve(fineans,kernel)
    ans=fineans[edi.subres::edi.subres]
    sw.append(ans)
    #    plt.plot(nus,sw[-1],label=plab[ind])
    #plt.xlim([_lines.sigma2*(1.-edge),_lines.sigma2*(1.+edge)])
    ind=ind+1
    #plt.legend()
    #plt.show()
  return sw,s0[edi.subres::edi.subres]

def edifisher(lines, zshift,taufactor):
  epsilon=zshift
  lines2=copy.copy(lines)
  lines2.setz(lines2.z+epsilon)
  edi=EDI(lines,taufactor,zshift)
  f1=edicounts(lines,edi)
  f2=edicounts(lines2,edi)
  deltas=[]
  #plt.clf()
  for i in xrange(4):
    deltas.append((f2[0][i]-f1[0][i])/epsilon)
    #plt.plot(f1[2],f1[0][i])
    #plt.savefig('edi_signal.pdf')
  fisher=0
  #plt.clf()
  for i in xrange(4):
    fisher=fisher + numpy.sum(deltas[i]*deltas[i]/f1[0][i])
    #plt.plot(f2[2],deltas[i])
    #plt.savefig('edi_delta.pdf')
  f10=f1[1]
  f20=f2[1]

  deltas0=(f20-f10)/epsilon
  fisher0= numpy.sum(deltas0*deltas0/f10)
  
  return numpy.sqrt(1/fisher), numpy.sqrt(1/fisher0), 1/numpy.sqrt(fisher-fisher0)

def shsedifisher(lines, zshift):
  epsilon=zshift
  lines2=copy.copy(lines)
  lines2.setz(lines2.z+epsilon)
  f1=[]
  f2=[]
  shs=SHS(lines,1.5)
  shs.tau=0
  f10=counts(lines,shs)
  f20=counts(lines2,shs)
  for phase in numpy.arange(0,1,.25):
    shs=SHS(lines,1.5)
    shs.phase=phase
    f1.append(.25*counts(lines,shs))
    #plt.plot(f1[-1])
    f2.append(.25*counts(lines2,shs))
    #plt.show()
  deltas=[]
  #plt.clf()
  for i in xrange(4):
    deltas.append((f2[i]-f1[i])/epsilon)
    #plt.plot(f1[2],f1[0][i])
    #plt.savefig('edi_signal.pdf')
  fisher=0
  #plt.clf()
  for i in xrange(4):
    fisher=fisher + numpy.sum(deltas[i]*deltas[i]/f1[i])
    #plt.plot(f2[2],deltas[i])
    #plt.savefig('edi_delta.pdf')
  fisher=fisher*2 #double range


  deltas0=(f20-f10)/epsilon
  fisher0= numpy.sum(deltas0*deltas0/f10)*2
  
  return numpy.sqrt(1/fisher), numpy.sqrt(1/fisher0), 1/numpy.sqrt(fisher-fisher0)

def editwooutput(plate,line):
  a3= edifisher(Lines(plate,line,'OIII'),1e-10,3.)
  a2= edifisher(Lines(plate,line,'OII'),1e-10,3.)
  print "{} &{} &${:5.1e}}}$ & ${:5.1e}}}$ & ${:5.1e}}}$ && ${:5.1e}}}$ & ${:5.1e}}}$ & ${:5.1e}}}$ && ${:5.1e}}}$\\\\".format(plate,line,a2[0],a2[1],a2[2],a3[0],a3[1],a3[2],1/numpy.sqrt(1/a2[0]**2+1/a3[0]**2))

def edioneoutput(plate,line,str):
  if str=='OIII':
    a3= edifisher(Lines(plate,line,'OIII'),1e-10,3.)
    a2=[0,0,0]
    last=a3[0]
  else:
    a2= edifisher(Lines(plate,line,'OII'),1e-10,3.)
    a3=[0,0,0]
    last=a2[0]
  print "{} &{} &${:5.1e}}}$ & ${:5.1e}}}$ & ${:5.1e}}}$ && ${:5.1e}}}$ & ${:5.1e}}}$ & ${:5.1e}}}$ && ${:5.1e}}}$\\\\".format(plate,line,a2[0],a2[1],a2[2],a3[0],a3[1],a3[2],last)

def table_edi():
  edioneoutput(1523,602,'OIII')
  editwooutput(1935,204)
  edioneoutput(1036,584,'OIII')
  edioneoutput(2959,354,'OIII')
  editwooutput(1268,318)
  editwooutput(1657,483)
  editwooutput(1073,225)
  editwooutput(1514,137)
  editwooutput(4794,757)
  edioneoutput(1059,564,'OII')
  #table_edi()


def shseditwooutput(plate,line):
  a3= shsedifisher(Lines(plate,line,'OIII'),1e-10)
  a2= shsedifisher(Lines(plate,line,'OII'),1e-10)
  print "{} &{} &${:5.1e}}}$ & ${:5.1e}}}$ & ${:5.1e}}}$ && ${:5.1e}}}$ & ${:5.1e}}}$ & ${:5.1e}}}$ && ${:5.1e}}}$\\\\".format(plate,line,a2[0],a2[1],a2[2],a3[0],a3[1],a3[2],1/numpy.sqrt(1/a2[0]**2+1/a3[0]**2))

def shsedioneoutput(plate,line,str):
  if str=='OIII':
    a3= shsedifisher(Lines(plate,line,'OIII'),1e-10)
    a2=[0,0,0]
    last=a3[0]
  else:
    a2= shsedifisher(Lines(plate,line,'OII'),1e-10)
    a3=[0,0,0]
    last=a2[0]
  print "{} &{} &${:5.1e}}}$ & ${:5.1e}}}$ & ${:5.1e}}}$ && ${:5.1e}}}$ & ${:5.1e}}}$ & ${:5.1e}}}$ && ${:5.1e}}}$\\\\".format(plate,line,a2[0],a2[1],a2[2],a3[0],a3[1],a3[2],last)

def table_shsedi():
  shsedioneoutput(1523,602,'OIII')
  shseditwooutput(1935,204)
  shsedioneoutput(1036,584,'OIII')
  shsedioneoutput(2959,354,'OIII')
  shseditwooutput(1268,318)
  shseditwooutput(1657,483)
  shseditwooutput(1073,225)
  shseditwooutput(1514,137)
  shseditwooutput(4794,757)
  shsedioneoutput(1059,564,'OII')
  #table_shsedi()

def oneratiopartials(_lines,nratio):
  _lines2=copy.copy(_lines)
  _lines2.setz(_lines2.z+1e-9)
  shs=SHS(_lines,nratio)
  #print shs.theta, sigmasinegamma_exact(_lines.sigma1,shs), sigmasinegamma_exact(_lines.sigma2,shs)
  sig=counts(_lines, shs)
  sig2=counts(_lines2, shs)
  partials=(sig2-sig)/1e-9
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
  #  plt.clf()
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
      #    plt.plot(shs.xs*100,partials_[0],label='n='+str(nratio))
      #plt.legend()
      #plt.xlabel('x (cm)')
      #plt.ylabel('d(counts)/dz')
      #plt.savefig('dsigdz.eps')
  fisher=numpy.sum(allpartials*allpartials/allsignals) * 2 #the extra 2 for the -x values
  return fisher

def twoline(plate, fiber):
  lines=Lines(plate,fiber,'OII')
  fisher=fisherandplot(lines)
  o2=1/numpy.sqrt(fisher)
  print '${:5.1e}}}$ &'.format(o2),
  lines=Lines(plate,fiber,'OIII')
  fisher=fisherandplot(lines)
  o3=1/numpy.sqrt(fisher)
  print '${:5.1e}}}$ &'.format(o3),
  print '${:5.1e}}}$ '.format(1./numpy.sqrt(1/o2**2+1/o3**2)), '\\\\'

def ed_twoline(plate, fiber):
  lines=Lines(plate,fiber,'OII')
  fisher=ed_fisher(lines)
  o2=1/numpy.sqrt(fisher)
  print '${:5.1e}}}$ &'.format(o2),
  lines=Lines(plate,fiber,'OIII')
  fisher=ed_fisher(lines)
  o3=1/numpy.sqrt(fisher)
  print '${:5.1e}}}$ &'.format(o3),
  print '${:5.1e}}}$ '.format(1./numpy.sqrt(1/o2**2+1/o3**2)), '\\\\'

def table():
  lines=Lines(1523,602,'OIII')
  fisher=fisherandplot(lines)
  o2=1/numpy.sqrt(fisher)
  print 1523, '&', 602,'&', lines.z,'&','&',
  print '${:5.1e}}}$ '.format(o2),'&','${:5.1e}}}$ \\\\'.format(o2)

  print 1935, '&', 204,'&', Lines(1935,204,'OII').z,'&',
  twoline(1935,204) #53387

  lines=Lines(1036,584,'OIII')
  fisher=fisherandplot(lines)
  o2=1/numpy.sqrt(fisher)
  print 1036, '&', 584,'&', lines.z,'&','&',
  print '${:5.1e}}}$ '.format(o2),'&','${:5.1e}}}$ \\\\'.format(o2)

  lines=Lines(2959,354,'OIII')
  fisher=fisherandplot(lines)
  o2=1/numpy.sqrt(fisher)
  print 2959, '&', 354,'&', lines.z,'&','&',
  print '${:5.1e}}}$ '.format(o2),'&','${:5.1e}}}$ \\\\'.format(o2)

  print 1268, '&', 318, '&', Lines(1268,318,'OII').z, '&',
  twoline(1268,318) #52933
  print 1657, '&', 483, '&', Lines(1657,483,'OII').z, '&',
  twoline(1657,483) #53520

  lines=Lines(1073,225,'OIII')
  fisher=fisherandplot(lines)
  o2=1/numpy.sqrt(fisher)
  print 1073, '&', 225,'&', lines.z,'&','&',
  print '${:5.1e}}}$ '.format(o2),'&','${:5.1e}}}$ \\\\'.format(o2)

  print 1073, '&', 225,  '&',Lines(1073,225,'OII').z, '&',
  twoline(1073,225) #55647     
  print 1514, '&', 137, '&', Lines(1514,137,'OII').z, '&',
  twoline(1514,137) #52931
  print 4794, '&', 757,  '&',Lines(4794,757,'OII').z, '&',
  twoline(4794,757) #55647     
  lines=Lines(1059,564,'OII')
  fisher=fisherandplot(lines)
  o2=1/numpy.sqrt(fisher)
  print 1059, '&', 564,'&', lines.z,'&',
  print '${:5.1e}}}$ &'.format(o2),'& &','${:5.1e}}}$ \\\\'.format(o2)

                    #twoline(1349,175) not happy
#table()

def ed_fisher(_lines):
  for nratio in numpy.arange(1.5,9,10):
    shs=SHS(_lines,nratio)
    _lines2=copy.copy(_lines)
    _lines2.setz(_lines2.z+1e-9)
    c1=ed_counts(_lines,shs)
    c2=ed_counts(_lines2,shs)
    #plt.plot(c1)
    #plt.show()
    partials_ = (c2-c1)/1e-9    
    fisher=numpy.sum(partials_*partials_/c1) * 2 #the extra 2 for the -x values
  return fisher

def ed_table():
  lines=Lines(1523,602,'OIII')
  fisher=ed_fisher(lines)
  o2=1/numpy.sqrt(fisher)
  print 1523, '&', 602,'&','&',
  print '${:5.1e}}}$ '.format(o2),'&','${:5.1e}}}$ \\\\'.format(o2)

  print 1935, '&', 204,'&',
  ed_twoline(1935,204) #53387

  lines=Lines(1036,584,'OIII')
  fisher=ed_fisher(lines)
  o2=1/numpy.sqrt(fisher)
  print 1036, '&', 584,'&','&',
  print '${:5.1e}}}$ '.format(o2),'&','${:5.1e}}}$ \\\\'.format(o2)

  lines=Lines(2959,354,'OIII')
  fisher=ed_fisher(lines)
  o2=1/numpy.sqrt(fisher)
  print 2959, '&', 354,'&','&',
  print '${:5.1e}}}$ '.format(o2),'&','${:5.1e}}}$ \\\\'.format(o2)

  print 1268, '&', 318, '&', 
  ed_twoline(1268,318) #52933
  print 1657, '&', 483, '&', 
  ed_twoline(1657,483) #53520

  print 1073, '&', 225,  '&',
  ed_twoline(1073,225) #55647     
  print 1514, '&', 137, '&', 
  ed_twoline(1514,137) #52931
  print 4794, '&', 757,  '&',
  ed_twoline(4794,757) #55647     
  lines=Lines(1059,564,'OII')
  fisher=ed_fisher(lines)
  o2=1/numpy.sqrt(fisher)
  print 1059, '&', 564,'&', 
  print '${:5.1e}}}$ &'.format(o2),'& &','${:5.1e}}}$ \\\\'.format(o2)

ed_table()
sh
lines=Lines(1268,318,'OII')
print 1/numpy.sqrt(ed_fisher(lines))
shit

def plotvelocity():
  plate=1268
  fiber=318
  vel=[]
  ans=[]
  for fact in numpy.arange(.5,2,.1):
    lines=Lines(plate,fiber,'OII')
    lines.deltav=lines.deltav*fact
    lines.setz(lines.z)
    fisher=fisherandplot(lines)
    o2=1/numpy.sqrt(fisher)
    lines=Lines(plate,fiber,'OIII')
    lines.deltav=lines.deltav*fact
    lines.setz(lines.z)
    fisher=fisherandplot(lines)
    o3=1/numpy.sqrt(fisher)
    vel.append(lines.deltav)
    ans.append(1./numpy.sqrt(1/o2**2+1/o3**2))
  vel=numpy.array(vel)*3e5
  ans=numpy.array(ans)
  plt.clf()
  plt.plot(vel,ans)
  plt.xlabel('$\Delta v$ (km s$^{-1}$)')
  plt.ylabel('$\sigma_z^{[OII]&[OIII]}$')
  plt.savefig('vdependence.pdf')

  #plotvelocity()
  #shit

def plotflux():
  plate=1268
  fiber=318
  vel=[]
  ans=[]
  for fact in numpy.arange(.5,10,.25):
    lines=Lines(plate,fiber,'OII')
    lines.a1_0=lines.a1_0*fact
    lines.a2_0=lines.a2_0*fact
    lines.setz(lines.z)
    fisher=fisherandplot(lines)
    o2=1/numpy.sqrt(fisher)
    lines=Lines(plate,fiber,'OIII')
    lines.a1_0=lines.a1_0*fact
    vel.append(lines.a1_0)
    lines.a2_0=lines.a2_0*fact
    lines.setz(lines.z)
    fisher=fisherandplot(lines)
    o3=1/numpy.sqrt(fisher)
    ans.append(1./numpy.sqrt(1/o2**2+1/o3**2))
  vel=numpy.array(vel)*3e5
  ans=numpy.array(ans)
  plt.clf()
  plt.plot(vel,ans)
  plt.xlabel('[OIII](5006) flux (erg cm$^{-1}$ s$^{-1}$)')
  plt.ylabel('$\sigma_z^{[OII]&[OIII]}$')
  plt.savefig('fdependence.pdf')

#plotflux()

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

