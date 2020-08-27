#!/usr/bin/env python

import numpy
import matplotlib.pyplot as plt
from astropy.cosmology import WMAP9 as cosmo
from astropy.constants import c, M_sun, G
from astropy import units as u




def thetadot_norm(zl, zs):
	dls = cosmo.angular_diameter_distance_z1z2(zl, zs)
	dl= cosmo.angular_diameter_distance(zl)
	ds = cosmo.angular_diameter_distance(zs)

	theta = numpy.sqrt(dls/dl/ds).cgs

	thetadot = 0.5*cosmo.H(0).cgs + (0.5/theta**2 *(cosmo.H(zs)/(1+zs)/ds - cosmo.H(zl)/(1+zl)/dl)).cgs
	print thetadot
	return thetadot.value

if __name__ == "__main__":
	for zs in numpy.arange(3.773,10,10):
		zls = numpy.arange(.986,2,10)
		ans=[]
		for zl in zls:
			ans.append(thetadot_norm(zl,zs))

		ans = numpy.array(ans)

		plt.plot(zls,ans,label=zs)

	plt.legend()
	plt.show()