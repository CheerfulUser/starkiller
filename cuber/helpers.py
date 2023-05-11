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
    result = result.rename(columns={'RA_ICRS':'ra','DE_ICRS':'dec'})
    return result


def min_dist(x1,y1,x2,y2):
    dx = x1[:,np.newaxis] - x2[np.newaxis,:]
    dy = y1[:,np.newaxis] - y2[np.newaxis,:]
    d = np.sqrt(dx**2 + dy**2)
    md = np.nanmin(d,axis=1)
    return md

def minimize_dist(offset,x1,y1,x2,y2,image):
    cx = image.shape[1]/2; cy = image.shape[0]/2
    x = x1 + offset[0]
    y = y1 + offset[1]
    x = cx + ((x-cx)*np.cos(offset[2])-(y-cy)*np.sin(offset[2]))
    y = cy + ((x-cx)*np.sin(offset[2])+(y-cy)*np.cos(offset[2]))

    ind = (x > 0) & (x < image.shape[1]) & (y > 0) & (y < image.shape[0])
    x = x[ind]; y = y[ind]
    mdist = min_dist(x2,y2,x,y)
    return np.nanmean(mdist)

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
    

def psf_spec(cube,psf_param,num_cores=5):
    trip = create_psf(x=cube.shape[2],y=cube.shape[1],alpha=psf_param[0],
                      beta=psf_param[1],length=psf_param[2],angle=psf_param[3])
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

def get_specs(cat,cube,x_length,y_length,psf_params,lam):

    cuts = []
    for i in range(len(cube)):
        cut,good = get_star_cuts(x_length,y_length,cube[i],cat)
        cuts += [cut]
    cuts = np.array(cuts)
    cuts = np.swapaxes(cuts,0,1)

    ind = np.arange(0,len(cuts)+1)
    num_cores = multiprocessing.cpu_count() - 3
    specs = []
    residual = []
    #cat['x_offset'] = 0
    #cat['y_offset'] = 0
    flux, res, xoff, yoff = zip(*Parallel(n_jobs=num_cores)(delayed(psf_spec)(cut,psf_params) for cut in cuts))
    residual = np.array(res)
    cat['x_offset'] = xoff
    cat['y_offset'] = yoff
    for i in range(len(cuts)):
        #flux, res, xoff, yoff = psf_spec(cuts[i],psf_params,num_cores=num_cores)
        spec = S.ArraySpectrum(lam,flux[i]*1e-20,fluxunits='flam',name=cat.iloc[i].Source)
        specs += [spec]

    return specs, residual, cat
        


def match_spec_to_model(spec,model='eso'):
    lam = spec.wave
    flux = spec.flux
    flux = savgol_filter(flux,101,1)
    if model.lower() == 'ck':
        cks = glob(f'{package_directory}data/ck_spec/*')
        cks.sort()
        cors = []
        for i in range(len(cks)):
            ck = at.Table.read(cks[i], format='ascii')
            ck_spec = S.ArraySpectrum(ck['wave'].value,ck['flux'].value,fluxunits='flam')
            interp = ck_spec.sample(lam)
            corr = pearsonr(flux,interp)[0]
            cors += [corr]
        cors = np.array(cors)
        ind = np.argmax(cors)
        name = cks[ind].split('/')[-1].split('.dat')[0]
        ck = at.Table.read(cks[ind], format='ascii')
        model = S.ArraySpectrum(wave=ck['wave'].value,
                                    flux=ck['flux'].value,fluxunits='flam',name=name)
    
    elif model.lower() == 'eso':
        esos = glob(f'{package_directory}data/eso_spec/*')
        esos.sort()
        cors = []
        for i in range(len(esos)):
            eso = at.Table.read(esos[i], format='ascii')
            eso = S.ArraySpectrum(eso['col1'].value,eso['col2'].value,fluxunits='flam')
            interp = eso.sample(lam)
            corr = pearsonr(flux,interp)[0]
            cors += [corr]
        cors = np.array(cors)
        ind = np.argmax(cors)
        name = esos[ind].split('/')[-1].split('.dat')[0]
        eso = at.Table.read(esos[ind], format='ascii')
        model = S.ArraySpectrum(wave=eso['col1'].value,
                                    flux=eso['col2'].value,fluxunits='flam',name=name)
    cor = cors[ind]
    return model, cor



def ebv_minimiser(ebv,model,spec,Rv=3.1):
    ext = S.ArraySpectrum(model.wave, 
                    apply(fitzpatrick99(model.wave.astype('double'),ebv*Rv,Rv),model.flux))
    interp = ext.sample(spec.wave)
    corr = pearsonr(spec.flux,interp)[0]
    res = abs(1 - 1/corr)
    return np.exp(res)

def fit_extinction(model,spec,Rv = 3.1):
    lam = spec.wave
    ebv0 = 0
    bounds = [[0,1]]
    res = minimize(ebv_minimiser, ebv0,args=(model,spec,Rv),method='Nelder-Mead',bounds=bounds)
    ebv = res.x[0]
    ext = S.ArraySpectrum(model.wave, 
                    apply(fitzpatrick99(model.wave.astype('double'),ebv*Rv,Rv),model.flux),name=model.name + ' ebv=' + str(np.round(ebv,3)))
    return ext,ebv

def spec_match(specs,cat,model_type='ck'):
    cors = []
    cal_model = []
    svo_bp=['GAIA/GAIA3.G']
    ebvs = []
    pbs = load_pbs(svo_bp,0,'AB',SVO=True)

    for i in range(len(specs)):
        model,cor = match_spec_to_model(specs[i],model_type)
        model2 = my_norm(model,pbs,np.array([cat.Gmag.values[i]]),name=model.name)
        ext, ebv = fit_extinction(model2,specs[i])
        ext = my_norm(ext,pbs,np.array([cat.Gmag.values[i]]),name=ext.name)
        cors += [cor]
        cal_model += [model2]
        ebvs += [ebv]
    ebvs = np.array(ebvs)
    cors = np.array(cors)
    return cal_model, cors, ebvs

