import os
from os import path
#from mangle import *#cube_mangle
from astropy.io import fits
from glob import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
from astropy.stats import sigma_clipped_stats, sigma_clip
import astropy.units as u
from astropy.wcs import WCS
from copy import deepcopy


from scipy.ndimage.filters import convolve
from scipy.ndimage import gaussian_filter, label
from scipy.ndimage import shift
from copy import deepcopy

from astropy.coordinates import SkyCoord, Angle
from astropy import units as u
import pandas as pd
from astroquery.vizier import Vizier

from scipy.optimize import minimize
import astropy.table as at

from photutils import DAOStarFinder
from astropy.stats import sigma_clipped_stats

from scipy.interpolate import griddata

from joblib import Parallel, delayed
from scipy import signal

from .helpers import *
from .trail_psf import create_psf
from .cube_simulator import cube_simulator
import pysynphot as S

import warnings
warnings.filterwarnings("ignore") #Shhhhhhh, it's okay


class cuber():
	def __init__(self,file,trail=True,model_maglim=21,cal_maglim=18,savepath=None,
				 catalog=None,spec_catalog='ck',key_filter=None,ref_filter=None,
				 psf_profile='gaussian',psf_preference='data',
				 plot=True,run=True,verbose=True,numcores=5):
		self.file = file
		self.plot=plot
		self.cal_maglim = cal_maglim
		self.model_maglim = model_maglim
		self.cat = catalog
		self.numcores = numcores
		self.spec_cat = spec_catalog
		self.psf_profile = psf_profile.lower()
		self.psf_preference = psf_preference.lower()

		if (key_filter is None) & (catalog is None):
			self.key_filter = 'G'
			self.ref_filter = 'G_mag'
		else:
			self.key_filter = key_filter + '_mag'
		


		self.verbose = verbose
		
		if savepath is not None:
			self.savepath = savepath
		else:
			self.savepath = os.getcwd() + '/'
		
		if run:
			self.run_cube(trail=trail)


	def _check_dirs(self):
		"""
		Check that all reduction directories are constructed
		"""
		dirlist = ['spec_figs']
		for d in dirlist:
			if not os.path.isdir(self.savepath + d):
				os.mkdir(self.savepath + d)



	def _load_cube(self):
		self.hdu = fits.open(self.file)
		self.wcs = WCS(self.hdu[1])
		self.cube = self.hdu[1].data
		self.header0 = self.hdu[0].header
		self.header1 = self.hdu[1].header
		self.ra, self.dec, _ = self.wcs.all_pix2world(self.cube.shape[2]/2,self.cube.shape[1]/2,0,0)
		_,_,self.lam = self.wcs.all_pix2world(0,0,np.arange(0,len(self.cube)),0)

		self.lam *= 1e10 # convert to Angst

		self.image = np.nanmean(self.cube,axis=0)
		#self.image[np.isnan(self.image)] = 0
		self.bright_mask = self.image > np.nanpercentile(self.image,90)
		self.image = self.image - np.nanmedian(self.image[~self.bright_mask])

	def _get_cat(self):
		if self.cat is None:
			self.cat = get_gaia_region([self.ra],[self.dec],size=50)
			self.cat = self.cat.sort_values('G_mag')
			ind = self.cat.G_mag.values < self.model_maglim
			if self.verbose:
				print(f'Number of sources brighter than {self.model_maglim}: {sum(ind)}')
			if np.sum(ind) < 2:
				m = f'!!! No sources brighter than {self.model_maglim} !!! \nMaximum maglim is {np.min(self.cat.G_mag.values[3])}'
				raise ValueError(m)
			self.cat = self.cat.iloc[ind]
		else:
			if (self.ref_filter is None):
				keys = list(self.cat.keys())
				f = []
				for key in keys:
					if 'filt' in key:
						f += [key]
				m = f'No ref_filter defined, selecting first filter in catalog as ref_filter: {f[0]}'
				self.ref_filter = f[0]
		self._sort_cat_filts_mags()
		
	def _sort_cat_filts_mags(self,cat=None):
		#if cat is None:
		cat = self.cat 
		keys = list(cat.keys())
		filts = []
		mags = []
		if self.key_filter is None:
			for key in keys:
				if 'filt' in key:
					filts += [key]
					f = key.split('_')[0]
					mags += [f + '_mag']
		else:
			mags = [self.key_filter + '_mag']
			filts = [self.key_filter + '_filt']
		filts = cat[filts].iloc[0].values
		mags = cat[mags].values
		#if cat is None:
		self.filts = filts
		self.mags = mags
		#else:
		#	return filts, mags

	def _estimate_trail_angle(self,trail):
		fft = np.fft.fft2(self.bright_mask)
		fft = np.fft.fftshift(fft)
		magnitude_spectrum = np.abs(fft)
		power_spectrum = np.log10(np.square(magnitude_spectrum))

		peak_threshold = 0.8*np.nanmax(power_spectrum)
		peaks = np.where(power_spectrum >= peak_threshold)

		py = peaks[1] - np.median(peaks[0])
		px = -(peaks[0] - np.median(peaks[1]))
		grad, y0 = np.polyfit(px, py, 1)

		angle_radians = np.arctan2(grad,1) # 1 is the length of the x-axis
		# Convert the angle from radians to degrees
		angle_degrees = np.degrees(angle_radians)
		self.angle = angle_degrees

		if self.plot:
			xx = np.arange(min(-px),max(-px)+1)

			plt.figure()
			plt.imshow(power_spectrum, cmap='gray')
			plt.plot(peaks[0],peaks[1],'.',alpha=0.3)
			plt.plot(xx+np.mean(peaks[1]),xx*-grad+y0+np.mean(peaks[0]))


			# Show the image and the peak(s)
			plt.title('Directionality: {} degrees'.format(np.round(angle_degrees,0)))



	def _find_sources_DAO(self):
		blur = gaussian_filter(self.image,5)

		mean, med, std = sigma_clipped_stats(blur, sigma=3.0)

		daofind = DAOStarFinder(fwhm=10,theta = self.angle,ratio=.5, threshold=3*std,exclude_border=True)
		self._dao_s = daofind(blur - med)
		#if len(self._dao_s) < 2:

		if self.plot:
			plt.figure()
			plt.title('Sources found in image')
			plt.imshow(self.image,vmin=np.nanpercentile(self.image,16),vmax=np.nanpercentile(self.image,84),origin='lower')
			plt.plot(self._dao_s['xcentroid'],self._dao_s['ycentroid'],'C1.')

	def _find_sources_cluster(self):

		labeled, nr_objects = label(self.bright_mask > 0) 
		obj_size = []
		for i in range(nr_objects):
			obj_size += [np.sum(labeled==i)]
		obj_size = np.array(obj_size)
		targs = np.where((obj_size > 100) & (obj_size<1e4))[0]

		dy = np.zeros_like(targs)
		dx = np.zeros_like(targs)
		my = np.zeros_like(targs)
		mx = np.zeros_like(targs)
		sign = np.zeros_like(targs)
		for i in range(len(targs)):
			ty,tx = np.where(labeled == targs[i])
			#x,y = np.average(np.array([tx,ty]).T,weights=self.image[ty,tx],axis=0)
			my[i] = np.nanmedian(ty)
			mx[i] = np.nanmedian(tx)
			dy[i] = np.nanmax(ty) - np.nanmin(ty)
			dx[i] = np.nanmax(tx) - np.nanmin(tx)
			#if ty[np.argmin(tx)] > ty[np.argmax(tx)]:
			sign[i] = np.sign(ty[np.argmax(tx)] - ty[np.argmin(tx)])
			
		ind = ~sigma_clip(dx,sigma=2).mask & ~sigma_clip(dy,sigma=2).mask
		mx = mx[ind]; my = my[ind]; dx = dx[ind]; dy = dy[ind]; sign = sign[ind]
		ind = ((mx < self.image.shape[1] - np.max(dx)/2) & (mx > np.max(dx)/2) &
			   (my < self.image.shape[0] - np.max(dy)/2) & (my > np.max(dy)/2))
		mx = mx[ind]; my = my[ind]; dx = dx[ind]; dy = dy[ind]; sign = sign[ind]
		if len(mx) < 2:
			print(f'!!! Only {len(mx)} sources identified, coordinates will not fit!!!')
		self._dao_s = {'xcentroid':mx,'ycentroid':my}

		isox = np.sqrt(mx[:,np.newaxis] - mx[np.newaxis,:])
		isox[isox==0] = 900
		isox = np.nanmin(isox,axis=0)
		isoy = np.sqrt(my[:,np.newaxis] - my[np.newaxis,:])
		isoy[isoy==0] = 900
		isoy = np.nanmin(isoy,axis=0)
		iso = (isox > np.nanmedian(dx)) & (isoy > np.nanmedian(dy))
		#dx = dx[iso]; dy = dy[iso]; sign = sign[iso]
		buffer = np.nanmin([np.nanmedian(dy),np.nanmedian(dx)])
		if buffer < 20:
			buffer = 20

		self.y_length = int((np.nanmedian(dy)+buffer*0.5) / 2)
		self.x_length = int((np.nanmedian(dx)+buffer*0.5) / 2)
		self._trail_grad = np.nanmedian(dy/dx)
		self.angle = np.degrees(np.arctan2(self._trail_grad,1))
		if np.mean(sign) < 0:
			self.angle *= -1
		self.trail = np.nanmedian(np.sqrt(dy**2+dx**2))


	def _cat_fitter(self,ind):
		x0 = [0,0,0]
		x, y, _ = self.wcs.all_world2pix(self.cat.ra.values[:ind],self.cat.dec.values[:ind],0,0)

		catx = x; caty = y
		sourcex = self._dao_s['xcentroid']; sourcey = self._dao_s['ycentroid']
		bounds = [[-10,10],[-10,10],[0,np.pi/2]]
		res = minimize(minimize_dist,x0,args=(catx,caty,sourcex,sourcey,self.image),method='Nelder-Mead',bounds=bounds)

		xx = x + res.x[0]
		yy = y + res.x[1]
		cx = self.image.shape[1]/2; cy = self.image.shape[0]/2
		xx = cx + ((xx-cx)*np.cos(res.x[2])-(yy-cy)*np.sin(res.x[2]))
		yy = cy + ((xx-cx)*np.sin(res.x[2])+(yy-cy)*np.cos(res.x[2]))

		#ind = (xx > 0) & (xx < image.shape[1]) & (yy > 0) & (yy < image.shape[0])
		#xx = xx[ind]; yy = yy[ind]

		cut = min_dist(xx,yy,sourcex,sourcey) < 10
			#plt.figure()
			#plt.plot(sourcex,sourcey,'*',label='image')
			#plt.plot(x,y,'s',label='orig cat')
			#plt.plot(xx,yy,'+',label='shift cat')
			#plt.legend()
		res = minimize(minimize_dist,x0,args=(catx[cut],caty[cut],sourcex,sourcey,self.image),method='Nelder-Mead',bounds=bounds)
		if self.verbose:
			print('wcs shift: ',res.x)
		self.wcs_shift = res.x

	def _match_by_shift(self):
		labeled, nr_objects = label(self.bright_mask > 0) 
		obj_size = []
		for i in range(nr_objects):
			obj_size += [np.sum(labeled==i)]
		obj_size = np.array(obj_size)
		targs = np.where((obj_size > 100) & (obj_size<1e4))[0]
		ims = []
		for i in range(len(targs)):
			ims += [(labeled == targs[i])*1.0]
		ims = np.array(ims)
		ind = ~sigma_clip(np.sum(ims,axis=(1,2)),sigma=2).mask
		
		ix = self.cube.shape[2] /2
		iy = self.cube.shape[1] /2
		cims = deepcopy(ims[ind])
		for i in range(len(cims)):
			ty,tx = np.where(cims[i] > 0)
			my = np.nanmedian(ty)
			mx = np.nanmedian(tx)
			sx = ix - mx; sy = iy - my
			cims[i] = shift(cims[i],[sy,sx])
		im = (np.nansum(ims,axis=0) > 0) * 1.
		im[im==0] = np.nan
		s = (np.nanmedian(cims,axis=0) > 0.5) * 1.
		ty,tx = np.where(s > 0)
		s = s[min(ty)-3:max(ty)+4,min(tx)-3:max(tx)+4]
		
		x, y, _ = self.wcs.all_world2pix(self.cat.ra.values,self.cat.dec.values,0,0)
		# brute force it
		X,Y = np.meshgrid(np.arange(-100,100,10),np.arange(-100,100,10))
		positions = np.vstack([X.ravel(), Y.ravel(),X.ravel()*0]).T
		res = []
		for p in positions:
			guess = basic_image(p,x,y,im,s)
			res += [np.nansum(im - guess)]
		res = np.array(res)

		ind = np.argmin(res)
		x0 = positions[ind]
		bounds = [[x0[0]-10,x0[0]+10],[x0[1]-10,x0[1]+10],[0,np.pi/2]]
		res2 = minimize(minimize_cats,x0,args=(x,y,im,s),method='Nelder-Mead',bounds=bounds)
		if self.verbose:
			print('wcs shift: ',res2.x)

		self.wcs_shift = res2.x



	def _fit_DAO_to_cat(self,maxiter=5,method='shift'):
		if method.lower() == 'dist':
			safety = 0
			failed = True
			sourcenum = [1,2,3,4,5]
			while (safety < maxiter) & failed:
				ind = len(self._dao_s['xcentroid'])*sourcenum[safety] #self.cat['Gmag'].values < 20
				try:
					if self.verbose:
						print(f'Attempting to match sources, round {sourcenum[safety]}')
					self._cat_fitter(ind)
					failed = False
				except:
					m = 'Failed fitting sources, inceasing number of sources'
				safety += 1
			if failed:
				m = 'Could not match sources, send the data to Ryan!'
				ValueError(m)

		if method.lower() == 'shift':
			self._match_by_shift()


		

	def _transform_coords(self,plot=False):
		if plot is None:
			plot = self.plot

		x, y, _ = self.wcs.all_world2pix(self.cat.ra.values,self.cat.dec.values,0,0)

		xx,yy = transform_coords(x,y,self.wcs_shift,self.image)
		ys, xs = np.where(np.isfinite(self.image))
		d = np.sqrt((xx[:,np.newaxis] - xs[np.newaxis,:])**2 + (yy[:,np.newaxis] - ys[np.newaxis,:])**2)
		md = np.nanmin(d,axis=1)
		ind = md < (self.trail / 2)
		#ind = (xx >= 0 - self.x_length/2) & (xx < self.image.shape[1]) & (yy >= 0) & (yy < self.image.shape[0])

		#ind2 = np.isfinite(self.image[yy.astype(int),xx.astype(int)])

		self.cat['x'] = xx
		self.cat['y'] = yy
		self.cat = self.cat.iloc[ind]
		self.cat['xint'] = (self.cat['x'].values + 0.5).astype(int)
		self.cat['yint'] = (self.cat['y'].values + 0.5).astype(int)
		
		if plot:
			plt.figure()
			plt.title('Matching cube sources with catalogue')
			plt.imshow(self.image,vmin=np.nanpercentile(self.image,16),vmax=np.nanpercentile(self.image,85),cmap='gray',origin='lower')
			plt.plot(self._dao_s['xcentroid'],self._dao_s['ycentroid'],'o',ms=7,label='DAO')
			plt.plot(self.cat['x'],self.cat['y'],'C1*',label='Catalog')

	def _mag_isolation(self,dmag=3):
		mags = self.cat[self.ref_filter].values
		dmags = mags[:,np.newaxis] - mags[np.newaxis,:]
		ind = dmags > dmag
		d = np.sqrt((self.cat.x.values[:,np.newaxis] - self.cat.x.values[np.newaxis,:])**2 + 
					(self.cat.y.values[:,np.newaxis] - self.cat.y.values[np.newaxis,:])**2)
		d[ind] = 1e3
		d[d==0] = 1e3
		d = np.nanmin(d,axis=0)
		return d


	def _identify_cals(self):
		#ind = ((self.cat['x'].values + self.x_length/1.8 < self.image.shape[1]) & 
		#		(self.cat['y'].values + self.y_length/1.8 < self.image.shape[0]) & 
		#		(self.cat['x'].values - self.x_length/1.8 > 0) & 
		#		(self.cat['y'].values - self.y_length/1.8 > 0))
		ind = ((self.cat['x'].values.astype(int) < self.image.shape[1]) & 
				(self.cat['y'].values.astype(int) < self.image.shape[0]) & 
				(self.cat['x'].values.astype(int) > 0) & 
				(self.cat['y'].values.astype(int) > 0))

		d = self._mag_isolation()
		ind2 = d > 10
		ind3 = np.isfinite(self.image[self.cat['y'].values[ind].astype(int),self.cat['x'].values[ind].astype(int)])
		ind4 = self.cat[self.ref_filter].values <= self.cal_maglim

		ind[ind] = ind3
		self.cat['cal_source'] = 0
		ind = ind & ind2 & ind4
		self.cat['cal_source'].iloc[ind] = 1
		self.cals = self.cat.iloc[ind]
	
	def complex_isolation_cals(self,xdist=8):
	    xx = self.cat.xint.values; yy = self.cat.yint.values
	    ang = np.radians(self.angle)
	    cx = self.image.shape[1]/2; cy = self.image.shape[0]/2
	    xxx = cx + ((xx-cx)*np.cos(ang)-(yy-cy)*np.sin(ang))
	    yyy = cy + ((xx-cx)*np.sin(ang)+(yy-cy)*np.cos(ang))
	    
	    dmag = 2
	    mags = self.cat[self.ref_filter].values
	    dmags = mags[:,np.newaxis] - mags[np.newaxis,:]
	    ind = dmags > dmag

	    dy = abs(yyy[:,np.newaxis] - yyy[np.newaxis,:])
	    dy[dy==0] = 1e3
	    dy[ind] = 1e3

	    dx = abs(xxx[:,np.newaxis] - xxx[np.newaxis,:])
	    dx[dx==0] = 1e3
	    dx[ind] = 1e3
	    
	    iy,ix = np.where(dx > xdist)
	    dy[iy,ix] = 1e3
	    iy,ix = np.where(dx < xdist)
	    ii = dy[iy,ix] > self.trail*1.2
	    dx[iy[ii],ix[ii]] = 1e3
	    dy[iy[ii],ix[ii]] = 1e3
	    isoind = (np.nanmin(dy,axis=0) > self.trail*1.2) & (mags <self.cal_maglim)

	    ind = ((self.cat['x'].values.astype(int) < self.image.shape[1]) & 
				(self.cat['y'].values.astype(int) < self.image.shape[0]) & 
				(self.cat['x'].values.astype(int) > 0) & 
				(self.cat['y'].values.astype(int) > 0))

	    self.cat['cal_source'] = 0 
	    self.cat['cal_source'].iloc[isoind & ind] = 1
	    self.cals = self.cat.iloc[ind & isoind]
	    


	def _calc_angle(self):
		angles_r = []
		grads = []
		if len(self.cals) < 10:
			count = len(self.cals)
		else:
			count = 10
			
		for i in range(count):
			yint = self.cals.iloc[i]['yint']
			xint = self.cals.iloc[i]['xint']

			cut = self.image[yint-15:yint+15,xint-15:xint+15]
			mask = (cut > np.nanpercentile(cut,95)) * 1
			py,px = np.where(mask>0) 
			grad, y0 = np.polyfit(px, py, 1,w=cut[py,px])
			grads += [grad]
			angle_radians = np.arctan2(grad,1) # 1 is the length of the x-axis
			
			angles_r += [angle_radians]
		angles = np.array(angles_r)
		angle = np.nanmedian(angles)
		grad = np.nanmedian(np.array(grads))
		self._trail_grad = grad
		self.angle = angle

	def _calc_length(self):
		trails = []

		x = np.arange(0, self.image.shape[1])
		y = np.arange(0, self.image.shape[0])
		arr = np.ma.masked_invalid(self.image)
		xx, yy = np.meshgrid(x, y)
		x1 = xx[~arr.mask]
		y1 = yy[~arr.mask]
		newarr = arr[~arr.mask]

		for i in range(3):
			lx = np.arange(-11,12,2)
			testx = lx + self.cals['x'].iloc[i]
			testy = lx*self._trail_grad + self.cals['y'].iloc[i]
			dpix = np.sqrt((testx[0]-testx)**2+(testy[0]-testy)**2)
			estimate = griddata((x1, y1), newarr.ravel(),
								(testx,testy),method='linear')

			ind = np.where(np.nanpercentile(estimate[np.isfinite(estimate)],40) > estimate)[0]
			trail_length = max(np.diff(dpix[ind]))*1.2
			trails += [trail_length]
			
		self.trail = np.nanmedian(np.array(trails))
		self.y_length = abs(int(self.trail/2 * np.sin(self.angle)))
		self.x_length = abs(int(self.trail/2 * np.cos(self.angle)))
				
	def _isolate_cals(self):
		cals = self.cat.iloc[self.cat['cal_source'].values == 1]
		star_cuts, good = get_star_cuts(self.x_length,self.y_length,self.image,cals)
		self.cal_cuts = star_cuts 
		
		ind = cals[self.ref_filter].values < self.cal_maglim
		if np.sum(ind) < 1:
			m = f'{np.sum(ind)} targets above the mag lim, limit must be increased.\n Available mags: {cals[self.ref_filter].values}'
			raise ValueError(m)
		self.good_cals = good & ind

	def make_psf(self):
		self._isolate_cals()
		good = self.good_cals
		ct = self.cal_cuts[good]
		#ct -= np.nanmedian(ct,axis=(1,2))[:,np.newaxis,np.newaxis]
		
		psf = create_psf(self.x_length*2+1,self.y_length*2+1,self.angle,self.trail)

		params = []
		psfs = []
		shifts = []
		ind = len(ct)
		#if ind > 5:
		#	ind = 5
		'''for j in range(ind):
									psf.fit_psf(ct[j])
									psf.line()
									if self.psf_profile == 'moffat':
										params += [np.array([psf.alpha,psf.beta,psf.length,psf.angle])]
										shifts += [np.array([psf.source_x,psf.source_y])]
									elif self.psf_profile =='gaussian':
										params += [np.array([psf.stddev,psf.length,psf.angle])]
										shifts += [np.array([psf.source_x,psf.source_y])]
									psfs += [psf.longpsf]'''

		indo = np.arange(ind)
		params, shifts = zip(*Parallel(n_jobs=self.numcores)(delayed(parallel_psf_fit)(ct[i],psf,self.psf_profile) for i in indo))
		params = np.array(params)
		shifts = np.array(shifts)
		self.psf_param = np.nanmedian(params,axis=0)
		if self.psf_profile == 'moffat':
			self.psf = create_psf(x=self.x_length*2+1,y=self.y_length*2+1,alpha=self.psf_param[0],
					  			  beta=self.psf_param[1],length=self.psf_param[2],angle=self.psf_param[3],
					  			  psf_profile=self.psf_profile)
		elif self.psf_profile == 'gaussian':
			self.psf = create_psf(x=self.x_length*2+1,y=self.y_length*2+1,stddev=self.psf_param[0],length=self.psf_param[1],angle=self.psf_param[2],
					  			  psf_profile=self.psf_profile)
		self.psf.line()
		self._fine_psf_shift(shifts)
		self.complex_isolation_cals()
		self._isolate_cals()
		self.psf.make_data_psf(self.cal_cuts)
		self._check_psf_quality()

	def _check_psf_quality(self):
		diff = np.sum(abs(self.psf.data_psf-self.psf.longpsf))
		if (diff > 0.1) & (self.psf_preference=='data'):
			m = (f"!!! Large difference of {np.round(diff,2)} between model_psf and data_psf!!!\nUsing the data_psf, override by setting psf_preference='model'")
			print(m)
			self.psf_profile += ' data'

	def _fine_psf_shift(self,shifts,plot=None):
		if plot is None:
			plot = self.plot
		print('reshifting coords')
		sources = self.cat.iloc[self.cat.cal_source.values == 1].iloc[self.good_cals]
		catx = sources.x.values; caty = sources.y.values
		sourcex = catx + shifts[:,0]; sourcey = caty + shifts[:,1]
		bounds = [[-15,15],[-15,15],[0,np.pi/2]]
		x0 = [0,0,0]
		res = minimize(minimize_dist,x0,args=(catx,caty,sourcex,sourcey,self.image),method='Nelder-Mead',bounds=bounds)
		print('shift: ',res.x)

		xx,yy = transform_coords(self.cat.x.values,self.cat.y.values,res.x,self.image)
		self.cat['x'] = xx
		self.cat['y'] = yy
		self.cat['xint'] = (self.cat['x'].values + 0.5).astype(int)
		self.cat['yint'] = (self.cat['y'].values + 0.5).astype(int)
		if plot:
			plt.figure()
			plt.title('Matched cube')
			plt.imshow(self.image,vmin=np.nanpercentile(self.image,16),vmax=np.nanpercentile(self.image,85),cmap='gray',origin='lower')
			plt.plot(self.cat['x'],self.cat['y'],'C1*',label='Catalog')

	def calc_background(self):
		if 'data' in self.psf_profile:
			data = True
		else:
			data = False
		sim = cube_simulator(self.cube,self.psf,catalogue=self.cat,datapsf=data)
		psf_mask = (sim.all_psfs < np.nanpercentile(sim.all_psfs,70)) * 1.
		psf_mask[psf_mask ==0] = np.nan
		m = (self.cube < np.nanpercentile(psf_mask * self.cube,99,axis=(1,2))[:,np.newaxis,np.newaxis]) * 1.
		m[m==0] = np.nan
		
		bkg = np.nanmedian(psf_mask * self.cube * m,axis=(1,2))[:,np.newaxis,np.newaxis]
		self.bkgstd = (np.nanmedian(psf_mask * self.cube * m,axis=(1,2))).astype(int)
		self.bkg = bkg 
		self.cube -= bkg




	def cal_spec(self):
		cal_specs, residual, cands_off = get_specs(self.cals.iloc[self.good_cals],self.cube,self.x_length,self.y_length,self.psf_param,self.lam,num_cores=self.numcores)

		cal_model, cors, ebvs = spec_match(cal_specs,self.mags[self.good_cals],self.filts,model_type=self.spec_cat,num_cores=self.numcores)
		
		self.cal_specs = cal_specs	
		self.cal_models = cal_model
		self.cal_cor = cors
		self.cal_ebv = np.array(ebvs)
	
	def fit_spec_residual(self,order=3,corr_limit=95):
		funcs = []
		cors = deepcopy(self.cors)
		cors[self.cat['cal_source'].values == 0] = 0
		cors[(self.cat['cal_source'].values == 1)][~self.good_cals] = 0

		ind = np.argsort(self.cat[self.ref_filter].values)#
		conds = (self.cors[ind] > corr_limit) & (self.cat['cal_source'].values[ind])
		diff = []
		for i in ind[conds]:
			diff += [self.models[i].sample(self.specs[i].wave) / self.specs[i].flux]
		diff = np.nanmedian(np.array(diff),axis=0)
		fin = np.where(np.isfinite(diff))
		poly_param = np.polyfit(self.lam[fin],diff[fin],order)
		pf = np.polyval(poly_param,self.lam)

		self.flux_corr = pf#[:,np.newaxis,np.newaxis]
		self._rerun_model_fit()

	def _rerun_model_fit(self):
		for i in range(len(self.specs)):
			self.specs[i] = S.ArraySpectrum(wave=self.lam,flux=self.specs[i].flux * self.flux_corr,name = self.specs[i].name)


		model, cors, ebvs = spec_match(self.specs,self.mags,self.filts,model_type=self.spec_cat,num_cores=self.numcores)

		self.models = model
		self.cors = cors 
		self.ebvs = ebvs
		if self.plot:
			self.plot_specs()

	def all_spec(self):
		data_psf = None
		if 'data' in self.psf_profile:
			data_psf = self.psf.data_psf
		specs, residual, cat_off = get_specs(self.cat,self.cube,self.x_length,self.y_length,
											 self.psf_param,self.lam,num_cores=self.numcores,
											 data_psf=data_psf)
		self.cat = cat_off
		self.specs = specs
		self._sort_cat_filts_mags()

		model, cors, ebvs = spec_match(specs,self.mags,self.filts,model_type=self.spec_cat,num_cores=self.numcores)

		self.models = model
		self.cors = cors 
		self.ebvs = ebvs
		self.spec_res = residual
		#if self.plot:
		#	self.plot_specs()

	def plot_specs(self):
		self._check_dirs()
		specs = self.specs
		model = self.models
		for i in range(len(specs)):
			plt.figure()
			plt.title(f'{specs[i].name} cor = {np.round(self.cors[i],2)}')
			plt.plot(specs[i].wave,specs[i].flux * 1e16,label='IFU')
			plt.plot(model[i].wave,model[i].flux * 1e16,'--',label= model[i].name,alpha=0.5)
			
			plt.xlim(min(specs[i].wave)*0.9,max(specs[i].wave)*1.1)
			plt.ylabel(r'Flux $\left[\rm \times10^{-16}\; erg/s/cm^2/\AA\right]$')
			plt.xlabel(r'Wavelength ($\rm \AA$)')
			plt.legend(loc='upper left')
			plt.savefig(f'{self.savepath}spec_figs/{self.cat.id.iloc[i]}.png',dpi=300)
			#plt.close()
			#plt.cla()
			#plt.clf()

	def make_scene(self):
		if 'data' in self.psf_profile:
			data = True
		else:
			data = False
		scene = cube_simulator(self.cube,psf=self.psf,catalogue=self.cat,datapsf=data)
		flux = []
		for i in range(len(self.cat)):
			#f = downsample_spec(self.models[i],self.lam)
			f = self.models[i].sample(self.lam)
			flux += [f*1e20 / self.flux_corr]
		flux = np.array(flux)
		scene.make_scene(flux)
		self.scene = scene

	def difference(self):
		#self.diff = (self.cube * self.flux_corr[:,np.newaxis,np.newaxis]) - self.scene.sim
		self.diff = self.cube - self.scene.sim
		if self.plot:
			self.plot_diff()


	def plot_diff(self):
		ind = 1800
		image = self.cube[ind]#np.nanmedian(self.cube,axis=0)
		scene = self.scene.sim[ind]#np.nanmedian(self.scene.sim,axis=0)
		diff = self.diff[ind]#np.nanmedian(self.diff,axis=0)

		vmin = np.nanmean(self.bkg) #np.nanpercentile(image,10)
		vmax = np.nanmax(self.image) * 0.1 #np.nanpercentile(image,90)

		x = self.cat.xint + self.cat.x_offset
		y = self.cat.yint + self.cat.y_offset

		plt.figure(figsize=(12,4))
		plt.subplot(131)
		plt.title('IFU',fontsize=15)
		plt.imshow(image,origin='lower',vmin=vmin,vmax=vmax)
		plt.plot(x,y,'C1.')

		plt.subplot(132)
		plt.title('Scene',fontsize=15)
		plt.imshow(scene,origin='lower',vmin=vmin,vmax=vmax)
		plt.plot(x,y,'C1.')

		plt.subplot(133)
		plt.title('IFU - Scene',fontsize=15)
		plt.imshow(diff,origin='lower',vmin=vmin,vmax=vmax)
		plt.plot(x,y,'C1.')

		plt.tight_layout()
		name = self.savepath +  self.file.split('/')[-1].split('.fits')[0] + '_sub.png'
		plt.savefig(name,bbox_inches='tight',dpi=300)
		#plt.close()

	def _update_header(self):
		self.header0['DIFF'] = ('True', 'scene difference')
		self.header0['FlUX_COR'] = ('True', 'flux corrected')
		self.header0['PSF_PROF'] = (self.psf_profile,'PSF profile used')
		self.header1['DIFF'] = ('True', 'scene difference')
		self.header2 = deepcopy(self.hdu[2].header)
		self.header2['EXTNAME'] = ('MASK','this extension contains star mask')
		self.header2['NAXIS3'] = ('1','length of data axis 3')
		self.header2['HDUCLAS2'] = ('MASK','this extension contains star mask')
		self.header2['HDUCLAS3'] = ('PSF','the extension contains star PSFs')

		


	def _make_psf_bin(self):
		if 'moffat' in self.psf_profile:
			rec = np.rec.array([np.array([self.psf_param[0]]),np.array([self.psf_param[1]]),
								np.array([self.psf_param[2]]),np.array([self.psf_param[3]])],
								formats='float32,float32,float32,float32',
								names='alpha,beta,length,angle')
		elif 'gaussian' in self.psf_profile:
			rec = np.rec.array([np.array([self.psf_param[0]]),np.array([self.psf_param[1]]),
								np.array([self.psf_param[2]])],
								formats='float32,float32,float32',
								names='stddev,length,angle')

		hdu = fits.BinTableHDU(data=rec)
		hdu.header['HDUCLAS1'] = ('TABLE','Image data format')
		hdu.header['HDUCLAS2'] = ('PSF','this extension contains paf params')
		self._fits_psf = hdu





	def save_hdu(self):
		name = self.savepath +  self.file.split('/')[-1].split('.fits')[0] + '_diff.fits'
		self._update_header()
		self._make_psf_bin()
		phdu = fits.PrimaryHDU(data = None, header = self.header0)
		dhdu = fits.ImageHDU(data = self.diff, header = self.header1)
		mhdu = fits.ImageHDU(data = self.scene.all_psfs, header = self.header2)
		
		hdul = fits.HDUList([phdu,dhdu,self.hdu[2],mhdu,self._fits_psf])

		hdul.writeto(name,overwrite=True)

	def run_cube(self,trail=True):
		self._load_cube()
		self._get_cat()
		#self._estimate_trail_angle(trail)
		#self._find_sources_DAO()
		self._find_sources_cluster()
		self._fit_DAO_to_cat()
		self._transform_coords()		
		if self.verbose:
			print('Coords transformed')
		#self._identify_cals()
		self.complex_isolation_cals()
		self.make_psf()
		if self.verbose:
			print('Made PSF')
		self.calc_background()
		if self.verbose:
			print('Background subtracted')
		#self.cal_spec()
		self.all_spec()
		if self.verbose:
			print('Extracted spectra')
		self.fit_spec_residual()
		self.make_scene()
		if self.verbose:
			print('Made scene')
		self.difference()
		self.save_hdu()
		if self.verbose:
			print('Saved reduction')


