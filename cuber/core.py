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
	def __init__(self,file,trail=True,cal_maglim=18,savepath=None,plot=True,run=True,verbose=True):
		self.file = file
		self.plot=plot
		self.cal_maglim = cal_maglim
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

	def _get_cat(self):
		self.cat = get_gaia_region([self.ra],[self.dec],size=50)
		self.cat = self.cat.sort_values('Gmag')

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
		bounds = [[-100,100],[-100,100],[0,np.pi/2]]
		res = minimize(minimize_dist,x0,args=(catx,caty,sourcex,sourcey,self.image),method='Nelder-Mead',bounds=bounds)

		xx = x + res.x[0]
		yy = y + res.x[1]
		cx = self.image.shape[1]/2; cy = self.image.shape[0]/2
		xx = cx + ((xx-cx)*np.cos(res.x[2])-(yy-cy)*np.sin(res.x[2]))
		yy = cy + ((xx-cx)*np.sin(res.x[2])+(yy-cy)*np.cos(res.x[2]))

		#ind = (xx > 0) & (xx < image.shape[1]) & (yy > 0) & (yy < image.shape[0])
		#xx = xx[ind]; yy = yy[ind]

		cut = min_dist(xx,yy,sourcex,sourcey) < 10
		if self.verbose:
			print('round 1: ',res.x)
			#plt.figure()
			#plt.plot(sourcex,sourcey,'*',label='image')
			#plt.plot(x,y,'s',label='orig cat')
			#plt.plot(xx,yy,'+',label='shift cat')
			#plt.legend()
		res = minimize(minimize_dist,x0,args=(catx[cut],caty[cut],sourcex,sourcey,self.image),method='Nelder-Mead',bounds=bounds)
		if self.verbose:
			print('round 2: ',res.x)
		self.wcs_shift = res.x


	def _fit_DAO_to_cat(self,maxiter=5):
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


		

	def _transform_coords(self):

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
		
		if self.plot:
			plt.figure()
			plt.title('Matching DAO sources with catalogue')
			plt.imshow(self.image,vmin=np.nanpercentile(self.image,16),vmax=np.nanpercentile(self.image,85),cmap='gray',origin='lower')
			plt.plot(self._dao_s['xcentroid'],self._dao_s['ycentroid'],'o',ms=7,label='DAO')
			plt.plot(self.cat['x'],self.cat['y'],'C1*',label='Gaia')

	def _identify_cals(self):
		#ind = ((self.cat['x'].values + self.x_length/1.8 < self.image.shape[1]) & 
		#		(self.cat['y'].values + self.y_length/1.8 < self.image.shape[0]) & 
		#		(self.cat['x'].values - self.x_length/1.8 > 0) & 
		#		(self.cat['y'].values - self.y_length/1.8 > 0))
		ind = ((self.cat['x'].values.astype(int) < self.image.shape[1]) & 
				(self.cat['y'].values.astype(int) < self.image.shape[0]) & 
				(self.cat['x'].values.astype(int) > 0) & 
				(self.cat['y'].values.astype(int) > 0))

		d = np.sqrt((self.cat.x.values[:,np.newaxis] - self.cat.x.values[np.newaxis,:])**2 + 
					(self.cat.y.values[:,np.newaxis] - self.cat.y.values[np.newaxis,:])**2)
		d[d==0] = 900
		d = np.nanmin(d,axis=0)
		ind2 = d > 10
		ind3 = np.isfinite(self.image[self.cat['y'].values[ind].astype(int),self.cat['x'].values[ind].astype(int)])

		ind[ind] = ind3
		self.cat['cal_source'] = 0
		ind = ind & ind2
		self.cat['cal_source'].iloc[ind] = 1
		self.cals = self.cat.iloc[ind]
		

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
		star_cuts, good = get_star_cuts(self.x_length,self.y_length,self.image,self.cals)
		self.cal_cuts = star_cuts 
		
		ind = self.cals.Gmag.values < self.cal_maglim
		if np.sum(ind) < 1:
			m = f'{np.sum(ind)} targets above the mag lim, limit must be increased.\n Available mags: {self.cals.Gmag.values}'
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
		ind = len(ct)
		if ind > 5:
			ind = 5
		for j in range(ind):
			psf.fit_psf(ct[j])
			psf.line()
			params += [np.array([psf.alpha,psf.beta,psf.length,psf.angle])]
			psfs += [psf.longpsf]
		params = np.array(params)
		self.psf_param = np.nanmedian(params,axis=0)
		self.psf = create_psf(x=self.x_length*2+1,y=self.y_length*2+1,alpha=self.psf_param[0],
				  			  beta=self.psf_param[1],length=self.psf_param[2],angle=self.psf_param[3])
		self.psf.line()

	def calc_background(self):
		sim = cube_simulator(self.cube,self.psf,catalogue=self.cat)
		psf_mask = sim.all_psfs < np.percentile(sim.all_psfs,70)
		bkg = np.nanmedian(psf_mask * self.cube,axis=(1,2))[:,np.newaxis,np.newaxis]
		self.bkg = bkg 
		self.cube -= bkg



	def cal_spec(self):
		cal_specs, residual, cands_off = get_specs(self.cals.iloc[self.good_cals],self.cube,self.x_length,self.y_length,self.psf_param,self.lam)

		cal_model, cors, ebvs = spec_match(cal_specs,self.cals.iloc[self.good_cals],model_type='ck')
		
		self.cal_specs = cal_specs	
		self.cal_models = cal_model
		self.cal_cor = cors
		self.cal_ebv = np.array(ebvs)
	
	def fit_spec_residual(self,order=3,corr_limit=95):
		funcs = []
		cors = deepcopy(self.cors)
		cors[self.cat['cal_source'].values == 0] = 0
		cors[(self.cat['cal_source'].values == 1)][~self.good_cals] = 0

		ind = np.argsort(self.cat['Gmag'].values)#(self.cors > corr_limit) & (self.cat['cal_source'].values)
		diff = []
		for i in ind[:3]:
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


		model, cors, ebvs = spec_match(self.specs,self.cat,model_type='ck')

		self.models = model
		self.cors = cors 
		self.ebvs = ebvs

	def all_spec(self):
		specs, residual, cat_off = get_specs(self.cat,self.cube,self.x_length,self.y_length,self.psf_param,self.lam)
		self.cat = cat_off
		self.specs = specs

		model, cors, ebvs = spec_match(specs,self.cat,model_type='ck')

		self.models = model
		self.cors = cors 
		self.ebvs = ebvs
		self.spec_res = residual
		if self.plot:
			self.plot_specs()

	def plot_specs(self):
		self._check_dirs()
		specs = self.specs
		model = self.models
		for i in range(len(specs)):
			plt.figure()
			plt.title(f'{specs[i].name} cor = {np.round(self.cors[i],2)}')
			plt.plot(specs[i].wave,specs[i].flux,label='MUSE')
			plt.plot(model[i].wave,model[i].flux,'--',label='ck ' + model[i].name)
			
			plt.xlim(min(specs[i].wave)*0.9,max(specs[i].wave)*1.1)
			plt.ylabel(r'Flux [$\rm erg/s/cm^2/\AA$]')
			plt.xlabel(r'Wavelength ($\rm \AA$)')
			plt.legend()
			plt.savefig(f'{self.savepath}spec_figs/{self.cat.Source.iloc[i]}.pdf')
			plt.close()

	def make_scene(self):
		scene = cube_simulator(self.cube,psf=self.psf,catalogue=self.cat)
		flux = []
		for i in range(len(self.cat)):
			flux += [self.models[i].sample(self.lam)*1e20]
		scene.make_scene(flux)
		self.scene = scene

	def difference(self):
		#self.diff = (self.cube * self.flux_corr[:,np.newaxis,np.newaxis]) - self.scene.sim
		self.diff = (self.cube * self.flux_corr[:,np.newaxis,np.newaxis]) - self.scene.sim
		if self.plot:
			self.plot_diff()


	def plot_diff(self):
		ind = 1800
		image = self.cube[ind]#np.nanmedian(self.cube,axis=0)
		scene = self.scene.sim[ind]#np.nanmedian(self.scene.sim,axis=0)
		diff = self.diff[ind]#np.nanmedian(self.diff,axis=0)

		vmin = np.nanpercentile(image,10)
		vmax = np.nanpercentile(image,90)

		x = self.cat.xint + self.cat.x_offset
		y = self.cat.yint + self.cat.y_offset

		plt.figure(figsize=(12,4))
		plt.subplot(131)
		plt.title('MUSE',fontsize=15)
		plt.imshow(image,origin='lower',vmin=vmin,vmax=vmax)
		plt.plot(x,y,'C1.')

		plt.subplot(132)
		plt.title('Scene',fontsize=15)
		plt.imshow(scene,origin='lower',vmin=vmin,vmax=vmax)
		plt.plot(x,y,'C1.')

		plt.subplot(133)
		plt.title('MUSE - Scene',fontsize=15)
		plt.imshow(diff,origin='lower',vmin=vmin,vmax=vmax)
		plt.plot(x,y,'C1.')

		plt.tight_layout()
		name = self.savepath +  self.file.split('/')[-1].split('.fits')[0] + '_sub.pdf'
		plt.savefig(name,bbox_inches='tight')

	def _update_header(self):
		self.header0['DIFF'] = ('True', 'scene difference')
		self.header0['FlUX_COR'] = ('True', 'flux corrected')
		self.header1['DIFF'] = ('True', 'scene difference')
		self.header2 = deepcopy(self.hdu[2].header)
		self.header2['EXTNAME'] = ('MASK','this extension contains star mask')
		self.header2['NAXIS3'] = ('1','length of data axis 3')
		self.header2['HDUCLAS2'] = ('MASK','this extension contains star mask')
		self.header2['HDUCLAS3'] = ('PSF','the extension contains star PSFs')

		


	def _make_psf_bin(self):
		rec = np.rec.array([np.array([self.psf_param[0]]),np.array([self.psf_param[1]]),
							np.array([self.psf_param[2]]),np.array([self.psf_param[3]])],
							formats='float32,float32,float32,float32',
							names='alpha,beta,length,angle')
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
		self._identify_cals()
		#self._calc_angle()
		#self._calc_length()
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


