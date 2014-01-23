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
  return ans / (1*_shs.xs[-1])

def shs_counts(_lines,_shs):
  npix=3
  ans=numpy.zeros((npix,len(_shs.xs)))
  sig = []
  for x in _shs.xs:
    sig.append(intensity(x, _lines, _shs))
  sig=numpy.array(sig)*_shs.deltax*_shs.aperture*_shs.etime*_shs.eff
  for np in xrange(npix):
    ans[np]=sig/npix
  return ans

def shsphase_counts(_lines,_shs):
  shs=copy.copy(_shs)
  shs.etime=shs.etime/4
  ans=numpy.zeros((4,len(_shs.xs)))
  phases=[1,1.25,1.5,1.75]
  i=0
  for phase in phases:
    shs.phase=phase
    ans[i]=shs_counts(_lines,shs)
    i=i+1
  return ans

def edshs_counts(_lines, _shs):
  ans=numpy.zeros((len(_shs.xs),len(_shs.spectro.sigmas)))
  kernel=numpy.zeros(_shs.spectro.subres)+1
  for i in xrange(len(_shs.xs)):
    an=line(_shs.spectro.finesigmas,_lines.sigma1,_lines.a1,_lines.l12)
    an=an+line(_shs.spectro.finesigmas,_lines.sigma2,_lines.a2,_lines.l22)
    an=an+integrand(_shs.spectro.finesigmas,_shs.xs[i],_lines.sigma1,_lines.a1,_lines.l12,_shs)
    an=an+integrand(_shs.spectro.finesigmas,_shs.xs[i],_lines.sigma2,_lines.a2,_lines.l22,_shs)
    an=an+ _lines.back
    an=an*_shs.spectro.finebinwidths
    an=numpy.convolve(an,kernel,mode='same')
    ans[i]=an[_shs.spectro.subres::_shs.spectro.subres]
  ans=ans*_shs.deltax*_shs.aperture*_shs.etime*_shs.eff/ (1*_shs.xs[-1])
  return ans

def edi_counts(_lines,edi):
  npix=3
  tau=edi.tau
  s0=line(edi.spectro.finesigmas, _lines.sigma1, _lines.a1, _lines.l12)+line(edi.spectro.finesigmas, _lines.sigma2, _lines.a2, _lines.l22)
  s0=s0+_lines.back
  s0 =s0*edi.spectro.finebinwidths
  s0=s0*edi.aperture*edi.etime*edi.eff
  kernel=numpy.zeros(edi.spectro.subres)+1
  sw=numpy.zeros((4*npix,len(edi.spectro.sigmas)))
  ind=0
  for phi in numpy.arange(0,2*numpy.pi-0.001,numpy.pi/2):
    ans=.25*s0*(1+numpy.cos(2*numpy.pi*tau*edi.spectro.finesigmas+phi))
    ans=numpy.convolve(ans,kernel,mode='same')
    ans=ans[edi.spectro.subres::edi.spectro.subres]
    for np in xrange(npix):
      sw[ind*npix+np]=ans/npix
    ind=ind+1
  return sw

def spec_counts(_lines,edi):
  npix=3
  sw=numpy.zeros((npix,len(edi.spectro.sigmas)))
  tau=edi.tau
  s0=line(edi.spectro.finesigmas, _lines.sigma1, _lines.a1, _lines.l12)+line(edi.spectro.finesigmas, _lines.sigma2, _lines.a2, _lines.l22)
  s0=s0+_lines.back
  s0 =s0*edi.spectro.finebinwidths
  s0=s0*edi.aperture*edi.etime*edi.eff
  s0=s0[edi.spectro.subres::edi.spectro.subres]
  for np in xrange(npix):
    sw[np]=s0/npix
  return sw

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
      elif (plate == 1349 and fiber == 175):
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
      elif (plate == 1268 and fiber == 318):
        self.z=0.1257486
        self.deltav=10.03548/3e5
        self.a1_0=3315.32e-17 #3438.711e-17
        self.a2_0=1094.53e-17 #1203.549e-17
        self.gal=(14.7624+14.3708)/2*1e-17
      elif (plate == 1935	 and fiber == 204):
        self.z=0.09838755
        self.deltav=10.03865/3e5
        self.a1_0=5934.72e-17 #3547.454e-17
        self.a2_0=1959.3e-17 #1241.609e-17
        self.gal=(19.3408+19.0352)/2*1e-17
      elif (plate == 1657	 and fiber == 483):
        self.z=0.2213398
        self.deltav=10.30226/3e5
        self.a1_0=1839.82e-17 #1699.26e-17
        self.a2_0=607.401e-17 # 594.7409e-17
        self.gal=(11.0375+10.6736)/2*1e-17
      elif (plate == 4794	 and fiber == 757):
        self.z=0.5601137
        self.deltav=26.45063/3e5
        self.a1_0=25.0992e-17 #124.1981e-17
        self.a2_0=8.28632e-17 #43.46934e-17
        self.gal=(1.54566+1.47933)/2*1e-17
      elif (plate == 1514	 and fiber == 137):
        self.z=0.318046
        self.deltav=10.00679/3e5
        self.a1_0=116.704e-17#55.258061e-17
        self.a2_0=38.5289e-17#19.34032e-17
        self.gal=(10.9357+10.4893)/2*1e-17
      elif (plate == 1036 and fiber == 584):
        self.mjd=52562
        self.z=0.1078989
        self.deltav=4.545177/3e5
        self.a1_0=105.064e-17
        self.a2_0=34.6861e-17
        self.gal=(4.7564+4.08471)/2*1e-17
      elif (plate == 1073 and fiber == 225):
        self.mjd=52649
        self.z=0.2716023
        self.deltav=1.428721/3e5
        self.a1_0=113.574e-17
        self.a2_0=37.4957e-17
        self.gal=(3.88981+3.88917)/2*1e-17
      elif (plate == 1523 and fiber == 602):
        self.mjd=52937
        self.z=0.08933221
        self.deltav=5.1353/3e5
        self.a1_0=213.148e-17
        self.a2_0=70.3692e-17
        self.gal=(1.875331+2.05596)/2*1e-17
      elif (plate == 2959 and fiber == 354):
        self.mjd=54537
        self.z=0.1199353
        self.deltav=6.857259/3e5
        self.a1_0=33.6067e-17
        self.a2_0=11.095e-17
        self.gal=(0.165449+0.137032)/2*1e-17
      else:
        raise NameError('bad name')
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
      else:
        raise NameError('bad name')
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
  def __init__(self,_lines,(n,tau)):
    self.spectro = Spectrograph(_lines)
    self.ccd=CCD()
    self.littrow=(_lines.sigma1+_lines.sigma2)/2-(_lines.sigma1-_lines.sigma2)/2./n
    self.moverd=1.*(1200*1e3)
    self.theta = numpy.arcsin(self.moverd/2/self.littrow)
    if tau:
      self.tau = 1./(_lines.sigma2-_lines.sigma1)
    else:
      self.tau=0
    self.phase = 0

    # the ranges is defined by the width of the line
    fdecay =12*numpy.sqrt(_lines.l12)*numpy.tan(self.theta)
    xrange=numpy.abs(1/fdecay) 
    
    self.npt=1024
    SHS.setXRange(self,xrange)

    self.minsigma=_lines.sigma1-3*numpy.sqrt(_lines.l12)
    self.maxsigma=_lines.sigma2+3*numpy.sqrt(_lines.l22)

    self.aperture=numpy.pi*(10e2/2)**2 
    self.etime=3600.*8
    self.nexp=1
    self.eff=.7

  def setXRange(self,xrange):
    self.deltax=1.00*xrange/self.npt
    self.xs=numpy.arange(0*xrange,1*xrange,self.deltax)

class EDI(object):
  def __init__(self,_lines,dum=None):
    self.spectro = Spectrograph(_lines)
    self.ccd=CCD()
    #mnsigma=(1/_lines.lambda1_0+1/_lines.lambda2_0)/2
    #dsigma=mnsigma*(1/(1+_lines.z)-1/(1+_lines.z+dz))    
    self.tau=1/(2.36*numpy.sqrt(_lines.l12)*2)
    
    self.aperture=numpy.pi*(10e2/2)**2 
    self.etime=3600.*8
    self.nexp=4
    self.eff=.7

class Spectrograph(object):
  def __init__(self,_lines):
    self.r=40000
    self.edge=0.001
    self.subres=10
    nmax=numpy.log(_lines.sigma2*(1.+self.edge)/_lines.sigma1/(1-self.edge))*self.r
    self.finesigmas=(_lines.sigma1)*(1-self.edge)*numpy.exp(numpy.arange(0,nmax,1./self.subres)/self.r)
    self.sigmas=self.finesigmas[self.subres::self.subres]
    self.finebinwidths = self.finesigmas/self.r/self.subres
    self.binwidths = self.sigmas/self.r

class CCD(object):
  rn=2.
  dc=1/3600.
  maxtime=2*3600.


def fisher(lines, inst, _counts):
  epsilon=1e-9
  lines2=copy.copy(lines)
  lines2.setz(lines2.z+epsilon)
  f1=_counts(lines,inst)
  f2=_counts(lines2,inst)
  deltas=(f2-f1)/epsilon
  noise=f1+inst.nexp*inst.ccd.rn**2+inst.ccd.dc*inst.etime
  return numpy.sum(deltas*deltas/noise)

def galdata(plate, fiber, _inst, _counts,args=None, vfactor=1.):
  out=dict()
  out['fiber']=fiber
  try:
    lines=Lines(plate,fiber,'OII')
    z=lines.z
  except NameError:
    lines=Lines(plate,fiber,'OIII')
    z=lines.z
  out['z']=z
  cum=0
  try:
    lines=Lines(plate,fiber,'OII')
    lines.deltav=vfactor*lines.deltav
    lines.setz(lines.z)
    inst=_inst(lines,args)
    fish=fisher(lines,inst,_counts)
    o2=1/numpy.sqrt(fish)
    out['OII']=o2
    cum=1/o2**2
  except NameError:
    out['OII']='\\nodata'
  try:
    lines=Lines(plate,fiber,'OIII')
    lines.deltav=vfactor*lines.deltav
    lines.setz(lines.z)
    inst=_inst(lines,args)
    fish=fisher(lines,inst,_counts)
    o3=1/numpy.sqrt(fish)
    out['OIII']=o3
    cum=cum+1/o3**2
  except NameError:
    out['OIII']='\\nodata'
  out['OII&OIII']=1./numpy.sqrt(cum)
  return out

def gendata():
  names=['Convetional','EDI','SHS','EDSHS']
  insts=[EDI,EDI,SHS,SHS]
  counts=[spec_counts,edi_counts,shs_counts,edshs_counts]
  _args=[False,False,(1.5,False),(1.5,False)]

  gals=[(1523,602),(1935,204),(1036,584),(2959,354),(1268,318),(1657,483),(1073,225),(1514,137),(4794,757),(1059,564)]
  ans=dict()
  ind=0
  for name,inst,_counts,args in zip(names,insts,counts,_args):
    for plate,fiber in gals:
      if ind == 0:
        ans[(plate,fiber)]=dict()
      ans[(plate,fiber)][name]=galdata(plate,fiber,inst,_counts, args)
    ind=ind+1

  import pickle
  file = open('gendata.pkl', 'wb')
  pickle.dump(ans,file)
  file.close()

#gendata()

def table():
  names=['Convetional','EDI','SHS','EDSHS']
  file = open('gendata.pkl', 'rb')
  import pickle
  ans=pickle.load(file)
  file.close()
  keys=ans.keys()
  zs=[]
  for key in keys:
    zs.append(ans[key][names[0]]['z'])
  zs=numpy.array(zs)
  so=numpy.argsort(zs)
  
  for i in xrange(len(keys)):
    key=keys[so[i]]
    docomb=True
    an=ans[key]
    print "{} & {} ".format(key[0], key[1])
    if an[an.keys()[0]]['OII'] != '\\nodata':
      print '& OII' ,
      for nm in names:
        print '& ${:5.1e}}}$ '.format(an[nm]['OII']),
      print '\\\\'
    else:
      docomb=False

    if an[an.keys()[0]]['OIII'] != '\\nodata':
      print '& &OIII ',
      for nm in names:
        print '& ${:5.1e}}}$ '.format(an[nm]['OIII']),
      print '\\\\'
    else:
      docomb=False

    if docomb:
      print '& &OII\\&OIII ',
      for nm in names:
        print '& ${:5.1e}}}$ '.format(an[nm]['OII&OIII']),
      print '\\\\'
    print '\\tableline'
    
#table()

def widthplot():
  factors=numpy.arange(0.5,2,.25)
  names=['Convetional','EDI','SHS','EDSHS']
  insts=[EDI,EDI,SHS,SHS]
  counts=[spec_counts,edi_counts,shs_counts,edshs_counts]
  _args=[False,False,(1.5,False),(1.5,False)]

  plate,fiber =(1268,318)
  ind=0
  for name,inst,_counts,args in zip(names,insts,counts,_args):
    ans=[]
    for factor in factors:
      ans.append(galdata(plate,fiber,inst,_counts, args, vfactor=factor)['OII&OIII'])
    ans=numpy.array(ans)
    plt.plot(factors,ans,label=name)
  plt.legend()
  plt.show()
    
  #widthplot()

def linetable():
  gals=[(1523,602),(1935,204),(1036,584),(2959,354),(1268,318),(1657,483),(1073,225),(1514,137),(4794,757),(1059,564)]
  for plate,fiber in gals:
    print "{} & {} &".format(plate,fiber),
    try:
      lines=Lines(plate,fiber,'OII')
      z=lines.z
    except NameError:
      lines=Lines(plate,fiber,'OIII')
      z=lines.z
    print "{:6.3f}".format(z),
    try:
      lines=Lines(plate,fiber,'OII')
      print "&[OII] & ${:5.3f}$ &${:5.2e}}}$ &${:5.2e}}}$ &${:5.2e}}}$\\\\".format(lines.deltav*3e5, lines.a2_0, lines.a1_0, lines.gal)
      print "&&"
    except NameError:
      print ""
    try:
      lines=Lines(plate,fiber,'OIII')
      print "&[OIII]& ${:5.2f}$ &${:5.2e}}}$ &${:5.2e}}}$ &${:5.2e}}}$ \\\\ ".format(lines.deltav*3e5, lines.a2_0, lines.a1_0, lines.gal)
      print "\\tableline"
    except NameError:
      print "\\tableline"

def ediplot():
  plate=1268
  fiber=318
  lines=Lines(plate,fiber,'OIII')
  edi=EDI(lines)
  ans=edi_counts(lines,edi)
  plt.clf()
  plt.plot(edi.spectro.sigmas/100,ans[0],label="$\Delta\phi=0$")
  plt.plot(edi.spectro.sigmas/100,ans[3],label="$\Delta \phi=\pi/2$")
  plt.plot(edi.spectro.sigmas/100,ans[7],label="$\Delta \phi=\pi$")
  plt.plot(edi.spectro.sigmas/100,ans[11],label="$\Delta \phi=3\pi/2$")
  plt.xlabel('Wavenumber (cm$^{-1}$)')
  plt.ylabel('Counts per resolution element')
  plt.xlim((lines.sigma1*.9998/100,lines.sigma1*1.0002/100))
  plt.ticklabel_format(axis='x', useOffset=False)
  plt.legend()
  plt.savefig("/Users/akim/Work/zdot/paper/edi.pdf")

ediplot()

#linetable()
st

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

#shs=SHS(lines,1.5)
#print shs.theta, shs.littrow, lines.sigma1,lines.sigma2
#print shit
#plotspectrum(lines)
#print shit

#plotcounts(lines)
#print shit
#uncertaintyvsnratio(lines)
#print shit

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
  
