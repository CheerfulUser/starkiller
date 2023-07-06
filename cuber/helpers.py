import pandas as pd 
import numpy as np

import pysynphot as S
from extinction import fitzpatrick99, apply

from joblib import Parallel, delayed

import multiprocessing
from .trail_psf import create_psf
from .mangle import *

from scipy.optimize import minimize
from scipy.stats import pearsonr
from scipy.signal import savgol_filter
from glob import glob
import astropy.table as at
from astropy.coordinates import SkyCoord, Angle

from scipy import signal

from astroquery.vizier import Vizier

import os
package_directory = os.path.dirname(os.path.abspath(__file__)) + '/'

def get_gaia_region(ra,dec,size=0.4, magnitude_limit = 21):
    """
    Get the coordinates and mag of all gaia sources in the field of view.

    -------
    Inputs-
    -------
        tpf 				class 	target pixel file lightkurve class
        magnitude_limit 	float 	cutoff for Gaia sources
        Offset 				int 	offset for the boundary 

    --------
    Outputs-
    --------
        coords 	array	coordinates of sources
        Gmag 	array 	Gmags of sources
    """
    c1 = SkyCoord(ra, dec, unit='deg')
    Vizier.ROW_LIMIT = -1

    result = Vizier.query_region(c1, catalog=["I/355/gaiadr3"],
                                     radius=Angle(size, "arcsec"),column_filters={'Gmag':f'<{magnitude_limit}'})

    keys = ['objID','RAJ2000','DEJ2000','e_RAJ2000','e_DEJ2000','gmag','e_gmag','gKmag','e_gKmag','rmag',
            'e_rmag','rKmag','e_rKmag','imag','e_imag','iKmag','e_iKmag','zmag','e_zmag','zKmag','e_zKmag',
            'ymag','e_ymag','yKmag','e_yKmag','tmag','gaiaid','gaiamag','gaiadist','gaiadist_u','gaiadist_l',
            'row','col']


    no_targets_found_message = ValueError('Either no sources were found in the query region '
                                          'or Vizier is unavailable')
    if result is None:
        raise no_targets_found_message
    elif len(result) == 0:
        raise no_targets_found_message


    result = result['I/355/gaiadr3'].to_pandas()
    result = result.rename(columns={'RA_ICRS':'ra','DE_ICRS':'dec','Gmag':'G_mag','Source':'id'})
    result['G_filt'] = 'GAIA/GAIA3.G'

    return result


def min_dist(x1,y1,x2,y2):
    dx = x1[:,np.newaxis] - x2[np.newaxis,:]
    dy = y1[:,np.newaxis] - y2[np.newaxis,:]
    d = np.sqrt(dx**2 + dy**2)
    md = np.nanmin(d,axis=1)
    return md

def minimize_dist(offset,x1,y1,x2,y2,image):
    x,y = transform_coords(x1,y1,offset,image)

    ind = (x > 0) & (x < image.shape[1]) & (y > 0) & (y < image.shape[0])
    x = x[ind]; y = y[ind]
    mdist = min_dist(x2,y2,x,y)
    return np.nanmean(mdist)

def basic_image(offset,x,y,image,kernel):
    x,y = transform_coords(x,y,offset,image)
    
    cut = (0<=x) & (x < image.shape[1]) & (0<=y) & (y < image.shape[0])
    x = x[cut]; y = y[cut]
    guess = np.zeros_like(image)
    guess[y.astype(int),x.astype(int)] = 1
    guess = (signal.fftconvolve(guess,kernel,mode='same') > 0.1) * 1.0
    return guess

def minimize_cats(offset,x,y,image,kernel):
    guess = basic_image(offset,x,y,image,kernel)
    res = np.nansum(image - guess)
    return res



def transform_coords(x,y,param,image):
    xx = x + param[0]
    yy = y + param[1]
    cx = image.shape[1]/2; cy = image.shape[0]/2
    xx = cx + ((xx-cx)*np.cos(param[2])-(yy-cy)*np.sin(param[2]))
    yy = cy + ((xx-cx)*np.sin(param[2])+(yy-cy)*np.cos(param[2]))
    ind = (xx > 0) & (xx < image.shape[1]) & (yy > 0) & (yy < image.shape[0])
    #xx = xx[ind]; yy = yy[ind]
    return xx, yy

def get_star_cuts(x_length,y_length,image,cat,norm=False):
    pad = np.nanmax([y_length,x_length])*2
    image = np.pad(image,pad) 
    image[image==0] = np.nan
    x = cat['xint'].values + pad; y = cat['yint'].values + pad
    star_cuts = []
    good = []

    for i in range(len(cat)):
        c = image[y[i]-y_length:y[i]+y_length+1,x[i]-x_length:x[i]+x_length+1]
        
        my,mx = np.where(np.nanmax(c) == c)
        star_cuts += [c]
        if ((mx > 5) & (mx < c.shape[1]-5) & (my > 5) & (my < c.shape[0]-5)).any():
            good += [True]
        else:
            good += [False]
    star_cuts = np.array(star_cuts)
    good = np.array(good)
    if norm:
        star_cuts = star_cuts / np.nansum(star_cuts,axis=(1,2))[:,np.newaxis,np.newaxis]
    return star_cuts, good
    
def replace_cut(x_length,y_length,image,cuts,cat):
    pad = np.nanmax([y_length,x_length])*2
    image = np.pad(image,pad) 
    image[image==0] = np.nan
    x = cat['xint'].values + pad; y = cat['yint'].values + pad
    star_cuts = []
    good = []
    replace = deepcopy(image)
    for i in range(len(cat)):
        replace[y[i]-y_length:y[i]+y_length+1,x[i]-x_length:x[i]+x_length+1] = cuts[i]
    replace = replace[pad:-pad,pad:-pad]
    return replace

def replace_cube(cube,cut,x_length,y_length,cat):
    replace = deepcopy(cube)
    for i in range(len(cube)):
        replace[i] = replace_cut(x_length,y_length,cut[i],cat)
    return replace


def psf_spec(cube,psf_param,num_cores=5):
    if len(psf_param) > 3:
        trip = create_psf(x=cube.shape[2],y=cube.shape[1],alpha=psf_param[0],
                          beta=psf_param[1],length=psf_param[2],angle=psf_param[3])
    else:
        trip = create_psf(x=cube.shape[2],y=cube.shape[1],stddev=psf_param[0],
                          length=psf_param[1],angle=psf_param[2])
    trip.fit_pos(np.nanmean(cube,axis=0))
    xoff = trip.source_x; yoff = trip.source_y
    
    
    #f,r = zip(*Parallel(n_jobs=num_cores)(delayed(trip.psf_flux)(image) for image in cube))
    flux = []
    res = []
    for image in cube:
        f,r = trip.psf_flux(image)
        flux += [f]
        res += [r]
    
    flux = np.array(flux)
    flux[flux<0] = 0
    residual = np.array(res)
    xoff = np.nanmedian(np.array(xoff)); yoff = np.nanmedian(np.array(yoff))
    return flux,residual, xoff, yoff

def cube_cutout(cube,cat,x_length,y_length):
    cuts = []
    for i in range(len(cube)):
        cut,good = get_star_cuts(x_length,y_length,cube[i],cat)
        cuts += [cut]
    cuts = np.array(cuts)
    cuts = np.swapaxes(cuts,0,1)
    return cuts

#def iter_sub_spec(cube,cat,x_length,y_length,psf_params):
 #   sub = deepcopy(cube)
  #  for i in range(len(cat)):






def get_specs(cat,cube,x_length,y_length,psf_params,lam,num_cores):

    cuts = cube_cutout(cube,cat,x_length,y_length)

    ind = np.arange(0,len(cuts)+1)
    #num_cores = multiprocessing.cpu_count() - 3
    specs = []
    residual = []
    #cat['x_offset'] = 0
    #cat['y_offset'] = 0
    sub_cube = deepcopy(cube)
    flux, res, xoff, yoff = zip(*Parallel(n_jobs=num_cores)(delayed(psf_spec)(cut,psf_params) for cut in cuts))
    #for cut in cuts:

    residual = np.array(res)
    cat['x_offset'] = xoff
    cat['y_offset'] = yoff
    cat['x'] = cat['xint'].values + xoff
    cat['y'] = cat['yint'].values + yoff
    for i in range(len(cuts)):
        #flux, res, xoff, yoff = psf_spec(cuts[i],psf_params,num_cores=num_cores)
        spec = S.ArraySpectrum(lam,flux[i]*1e-20,fluxunits='flam',name=cat.iloc[i].id)
        specs += [spec]

    return specs, residual, cat
        
def downsample_spec(spec,target_lam):
    diff = np.gradient(target_lam)
    step = np.min(diff)/10
    lam_grid = np.arange(target_lam[0],target_lam[-1]+step,step)
    int_f = spec.sample(lam_grid)
    flux = np.zeros_like(target_lam)

    for i in range(len(diff)):
        low = target_lam[i] - diff[i]/2
        high = target_lam[i] + diff[i]/2
        ind = (lam_grid >= low) & (lam_grid <= high)
        l = lam_grid[ind]
        f = int_f[ind]
        flux[i] = np.trapz(f, x=l)
    return flux

def _spec_compare(mod_files,lam,flux):
    cors = []
    for i in range(len(mod_files)):
        mod = at.Table.read(mod_files[i], format='ascii')
        mod_spec = S.ArraySpectrum(mod['wave'].value,mod['flux'].value,fluxunits='flam')
        f = downsample_spec(mod_spec,lam)
        ff = savgol_filter(f,101,1)
        corr = pearsonr(flux,ff)[0]
        cors += [corr]
    cors = np.array(cors)
    return cors, f

def _compare_catalog(model_files,lam,flux):
    model_files.sort()
    cors, flux = _spec_compare(model_files,lam,flux)
    ind = np.argmax(cors)
    name = model_files[ind].split('/')[-1].split('.dat')[0]
    model = at.Table.read(model_files[ind], format='ascii')
    if '_a' in name:
        name = name.split('_a')[0]
    model = S.ArraySpectrum(wave=model['wave'].value,
                                flux=model['flux'].value,fluxunits='flam',name=name)
    return model, cors[ind]

def match_spec_to_model(spec,catalog='ck+'):
    lam = spec.wave
    flux = spec.flux
    flux = savgol_filter(flux,101,1)
    if 'ck' in catalog.lower():
        path = f'{package_directory}data/ck04_stsci/*'
        model_files = glob(path)
        model, cor = _compare_catalog(model_files,lam,flux)
        
        if '+' in catalog:
            print('ck ',model.name)
            print('comparing precise models')
            temp = int(model.name.split('_')[-2])
            if temp <= 8000:
                path = f'{package_directory}data/t02_st/*'
                model_files = np.array(glob(path))
                
                temps = np.array([int(x.split('/')[-1][1:5]) for x in model_files])
                ul = temp + 500; ll = temp - 500
                if temp <= 3600:
                    ll = 2500
                ind = (temps >= ll) & (temps <= ul)

                model2, cor2 = _compare_catalog(model_files[ind],lam,flux)
                if cor2 > cor:
                    model = model2

            elif temp >= 15000:
                path = f'{package_directory}data/griddl-ob-i-line/*'
                model_files = glob(path)
                model2, cor2 = _compare_catalog(model_files,lam,flux)
                if cor2 > cor:
                    model = model2
            print('precise ',model.name)

    elif catalog.lower() == 'eso':
        path = f'{package_directory}data/eso_spec/*'
        model, cor = _compare_catalog(path,lam,flux)

    return model, cor



def ebv_minimiser(ebv,model,spec,Rv=3.1):
    ext = S.ArraySpectrum(model.wave, 
                    apply(fitzpatrick99(model.wave.astype('double'),ebv*Rv,Rv),model.flux))
    interp = ext.sample(spec.wave)
    corr = pearsonr(savgol_filter(spec.flux,101,1),savgol_filter(interp,101,1))[0]
    res = -corr
    
    return res

def fit_extinction(model,spec,Rv = 3.1):
    lam = spec.wave
    ebv0 = 0
    bounds = [[0,1]]
    res = minimize(ebv_minimiser, ebv0,args=(model,spec,Rv),method='Nelder-Mead',bounds=bounds)
    ebv = res.x[0]
    ext = S.ArraySpectrum(model.wave, 
                    apply(fitzpatrick99(model.wave.astype('double'),ebv*Rv,Rv),model.flux),name=model.name + ' ebv=' + str(np.round(ebv,3)))
    return ext,ebv


def _norm_spec(spec,mag,pbs):
    keys = pbs.keys()
    if len(keys) > 1:
        norm = spec_mangle(spec,mag,pbs,name=spec.name)
    else:
        norm = my_norm(spec,pbs,mag,name=spec.name)
    return norm

def _par_spec_fit(spec,pbs,mag,model_type):
    model,cor = match_spec_to_model(spec,model_type)
    model2 = _norm_spec(model,mag,pbs)

    ext, ebv = fit_extinction(model2,spec)

    ext = _norm_spec(ext,mag,pbs)

    return ext, cor, ebv



def spec_match(specs,mags,filters,model_type='ck+',num_cores=5):

    svo_bp=filters
    pbs = load_pbs(svo_bp,0,'AB',SVO=True)
    cal_model, cors, ebvs = zip(*Parallel(n_jobs=num_cores)(delayed(_par_spec_fit)(specs[i],pbs,mags[i],model_type) for i in range(len(specs))))
    '''for i in range(len(specs)):
                    model,cor = match_spec_to_model(specs[i],model_type)
                    model2 = my_norm(model,pbs,np.array([cat.Gmag.values[i]]),name=model.name)
                    ext, ebv = fit_extinction(model2,specs[i])
                    ext = my_norm(ext,pbs,np.array([cat.Gmag.values[i]]),name=ext.name)
                    cors += [cor]
                    cal_model += [model2]
                    ebvs += [ebv]'''
    ebvs = np.array(ebvs)
    cors = np.array(cors)
    return cal_model, cors, ebvs

