import numpy as np
from joblib import Parallel, delayed
from scipy.optimize import minimize
from .trail_psf import create_psf
from scipy import signal

class PSF_photom():
	def __init__(self,data,cat,psf_params):
		self._set_data(data)
		#self._set_source_dims(source_dims)
		self._set_psf(psf_params)
		self.cat = cat


	def _set_psf(self,psf_params):
		self.psf_params = psf_params
		if self.cube:
			x=self.data.shape[2];y=self.data.shape[1]
		else:
			x=self.data.shape[1];y=self.data.shape[0]

		self.psf = create_psf(x=x,y=y,alpha=psf_param[0],
							  beta=psf_param[1],length=psf_param[2],angle=psf_param[3])
		self.psf.line()

	def _set_data(data):
		self.data = data
		if len(data.shape) == 2:
			self.cube = False
		else:
			self.cube = True

	def _set_source_dims(self,source_dims):
		try:
			length = len(source_dims)
			if length == 2:
				self.x_length = source_dims[0]
				self.y_length = source_dims[1]
			else:
				self.x_length = source_dims; self.y_length = source_dims
		except:
			self.x_length = source_dims; self.y_length = source_dims

	def _make_masks(self,limit=70):
		im = np.zeros((len(self.cat),self.image.shape[0],self.image.shape[1]))
		for i in range(len(self.cat)):
			im[i,self.cat.yint.values[i],self.cat.xint.values[i]] = 1
		kernel = (self.psf.longpsf > np.percentile(self.psf.longpsf,85))*1#np.ones((2*self.y_length,2*self.x_length))
		im = signal.fftconvolve(im,kernel[np.newaxis,:,:],mode='same')
		im = im > 0.1
		self.source_masks = im

	def group_sources(self,limit=85):
		im = np.zeros_like(self.image)
		im[self.cat.yint.values,self.cat.xint.values] = 1
		kernel = (self.psf.longpsf > np.percentile(self.psf.longpsf,limit))*1
		im = signal.fftconvolve(im,kernel,mode='same')

		im = im > 0.1
		labeled, nr_objects = label(im)
		self.cat['group'] = 0
		for i in range(len(self.cat)):
			self.cat['group'].iloc[i] = labeled[self.cat.yint.values[i],self.cat.xint.values[i]]
			
	
	def _groups_mask(self,ind):
		if np.sum(ind) > 1:
			mask = (np.nansum(self.source_masks[ind],axis=0) > 0) * 1.0
		else:
			mask = self.source_masks[ind] * 1.0
		mask[mask == 0] = np.nan

		return mask

	def _estimate_f0(self,data,ind):
		ind = np.where(ind)[0]
		f0 = []
		for i in ind:
			m = self.source_masks[i]
			f0 += [np.nansum(m * data)]
		f0 = np.array(f0)
		return f0

	def make_psf_image(self,x0,data,sources):
		f0 = x0[:len(sources)]; px = x0[len(sources):len(sources)*2] ; py = x0[2*len(sources):]
		tripys = []
		xs = sources.xint.values
		ys = sources.yint.values
		f = []
		t = create_psf(x=data.shape[1],y=data.shape[0],alpha=self.psf_param[0],
					   beta=self.psf_param[1],length=self.psf_param[2],angle=self.psf_param[3])
		for i in range(len(f0)):
			t.line(shiftx=xs[i]+px[i],shifty=ys[i]+py[i])
			f += [t.linepsf * f0[i]]
		f = np.array(f)
		f = np.nansum(f,axis=0)
		return f 

	def _group_minimizer(self,x0,data,sources):
		f = self.make_psf_image(x0,data,sources)
		res = np.nansum(abs(data - f))
		return res



	def group_fit(self,data,ind):
		sources = self.cat.iloc[ind]
		mask = self._groups_mask(ind)
		f0 = self._estimate_f0(data,ind)
		pos0 = np.append(sources['xint'].values,sources['yint'].values,axis=0)
		guess = np.append(f0,pos0,axis=0)
		f0_bounds = np.array([f0*0,f0*2])
		pos_bounds = np.zeros_like(f0_bounds)
		pos_bounds[:,0] = -5; pos_bounds[:,1] = 5
		bounds = np.append(f0_bounds,pos_bounds,axis=0)
		bounds = np.append(bounds,pos_bounds,axis=0)

		data_masked = data * mask

		res = minimize(self._group_minimizer,guess,args=(data_masked,sources),method='Nelder-Mead',bounds=bounds)

		flux = res.x[:len(sources)]; xp = res.x[len(sources):len(sources)*2]; yp = res.x[len(sources)*2:len(sources)*3]

		res = data - self.make_psf_image(res.x,data,sources)

		return flux, xp, yp, res



	def group_psf(self):
		groups = self.cat.group.unique()
		fluxes = {}
		residual = np.zeros_like(self.data)
		for group in groups:
			ind = self.cat.groups.values == group
			sources = self.cat.iloc[ind]
			if np.sum(ind) > 1:
				mask = (np.nansum(self.source_masks[ind],axis=0) > 0) * 1.0
			else:
				mask = self.source_masks[ind] * 1.0
			mask[mask == 0] = np.nan


			if self.cube:
				f, xp, yp, res = zip(*Parallel(n_jobs=num_cores)(delayed(self.group_fit)(image,ind) for image in self.data))
				flux = np.array(f).T; posx = np.array(xp); posy = np.array(yp); res = np.array(res)
				posx = np.nanmean(posx,axis=0); posy = np.nanmean(posy,axis=0)

			else:
				flux, posx, posy, res = self.group_fit(self.data,ind)

			self.cat['x'].iloc[ind] = posx; self.cat['y'].iloc[ind] = posy
			for i in range(len(sources)):
				fluxes[sources.iloc[i].id] = flux[i]
			residual = residual + res

		self.fluxes = fluxes 
		self.residual = residual




