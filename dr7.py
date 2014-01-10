#!/usr/bin/env python

import pprint
import numpy
import matplotlib.pyplot as plt
from astropy.io import fits
#http://www.mpa-garching.mpg.de/SDSS/DR7/SDSS_line.html
hdulist = fits.open('gal_line_dr7_v5_2.fit.fit.fit.gz')
linedata=hdulist[1].data
#lOIII_4959_flux=linedata.field('OIII_4959_flux')
#lOIII_4959_cont=linedata.field('OIII_4959_cont')
#lOIII_5007_flux=linedata.field('OIII_5007_flux')
#lOIII_5007_cont=linedata.field('OIII_5007_cont')
#lOIII_5007_chisq=linedata.field('OIII_5007_chisq')

#lOII_3726_flux=linedata.field('OII_3726_flux')
#lOII_3726_cont=linedata.field('OII_3726_cont')
#lOII_3729_flux=linedata.field('OII_3729_flux')
#lOII_3729_cont=linedata.field('OII_3729_cont')
#lOII_3729_chisq=linedata.field('OII_3729_chisq')

HBeta_flux = linedata.field('H_beta_flux')

lpid=numpy.array(linedata.field('plateid'))
lfid=numpy.array(linedata.field('fiberid'))
linedata=None

print HBeta_flux[numpy.logical_and(lpid == 1349 , lfid==175)]
print sto
hdulist = fits.open('gal_info_dr7_v5_2.fit.fit.fit.gz')
galdata=hdulist[1].data
gv=galdata.field('v_disp')
gz=galdata.field('z')
gpid=galdata.field('plateid')
gfid=galdata.field('fiberid')
galdata=None

hdulist=None

#gv[gv==0] = 1000
#lOIII_5007_flux[lOIII_5007_flux > 1e8]=-1
lOII_3729_flux[lOII_3729_flux > 1e8]=-1
#vso = numpy.argsort(gv)
vso = numpy.argsort(lOII_3729_flux)
f = open('dr7_bright_OII.txt', 'w')
#for i in xrange(50000):
for i in xrange(-1,-50000,-1):
  plateid=gpid[vso[i]]
  fiberid=gfid[vso[i]]
  ind = numpy.logical_and(lpid == plateid , lfid == fiberid)
#  print gv[vso[i]], gz[vso[i]], lOIII_4959_flux[ind][0],lOIII_4959_cont[ind][0],lOIII_5007_flux[ind][0],lOIII_5007_cont[ind][0],plateid,fiberid
#  if  (lOIII_4959_flux[ind][0] > 100 and lOIII_4959_flux[ind][0] > 100 and gz[vso[i]] > 0.095):
#        f.write('{} {} {} {} {} {} {} {} {}\n'.format(gv[vso[i]], gz[vso[i]], lOIII_4959_flux[ind][0],lOIII_4959_cont[ind][0],lOIII_5007_flux[ind][0],lOIII_5007_cont[ind][0],lOIII_5007_chisq[ind][0],plateid,fiberid))
  if  gz[vso[i]] > 0.095:
        f.write('{} {} {} {} {} {} {} {} {}\n'.format(gv[vso[i]], gz[vso[i]], lOII_3726_flux[ind][0],lOII_3726_cont[ind][0],lOII_3729_flux[ind][0],lOII_3729_cont[ind][0],lOII_3729_chisq[ind][0],plateid,fiberid))
f.close()
