import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

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
from astropy.stats import sigma_clipped_stats

from scipy import signal
from scipy.interpolate import interp1d

from astroquery.vizier import Vizier
from tqdm.notebook import trange, tqdm

import os
package_directory = os.path.dirname(os.path.abspath(__file__)) + '/'

def get_gaia_region(ra,dec,size=0.4, magnitude_limit = 21):
    """
    Get the coordinates and mag of all gaia sources in the field of view from the I/355/gaiadr3 catalog on Vizier.

    Parameters:
    ----------
        ra : float
            RA of the search area
        dec : float 
            Dec of the search field
        size : float
            Area of the cone search in arcsec
        magnitude_limit : float
            cutoff for Gaia sources
    
    Returns:
    --------
    result : pandas dataframe
        Dataframe containing the results of the Vizier query. 
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
    result['G_mag'] += 0.118 # Correct Gaia G from Vega to AB magnitude system

    return result

def _get_skymapper(ra,dec,rad = 0.4,magnitude_limit=25,star_lim=0.8):
    query = f'https://skymapper.anu.edu.au/sm-cone/public/query?RA={ra}&DEC={dec}&SR={rad}&RESPONSEFORMAT=CSV'
    sm = pd.read_csv(query)
    if len(sm) > 0:
        keep = ['object_id','raj2000','dej2000','u_psf', 'e_u_psf',
                'v_psf', 'e_v_psf','g_psf', 'e_g_psf','r_psf', 
                'e_r_psf','i_psf', 'e_i_psf','z_psf', 'e_z_psf','class_star']
        sm = sm[keep]
        sm = sm.rename(columns={'raj2000':'ra','dej2000':'dec','u_psf':'u_mag',
                                'v_psf':'v_mag','g_psf':'g_mag','r_psf':'r_mag',
                                'i_psf':'i_mag','z_psf':'z_mag',
                                'e_u_psf':'e_umag','e_v_psf':'e_vmag','e_g_psf':'e_gmag',
                                'e_r_psf':'e_rmag','e_i_psf':'e_imag','e_z_psf':'e_zmag'})
        sm['id'] = deepcopy(sm['object_id'].values)
        sm = sm.loc[sm.r_mag <= magnitude_limit]
        sm = sm.loc[sm.class_star >= star_lim]

        sm['r_filt'] = 'SkyMapper/SkyMapper.r'
        sm['g_filt'] = 'SkyMapper/SkyMapper.g'
        sm['i_filt'] = 'SkyMapper/SkyMapper.i'
        sm['z_filt'] = 'SkyMapper/SkyMapper.z'
    else:
        sm = None
    return sm


def min_dist(x1,y1,x2,y2):
    """
    Calculate the minimum distances between an 2 groups of points.

    Parameters
    ----------
    x1 : numpy ndarray
        x positions from group 1
    y1 : numpy ndarray
        y positions from group 1
    x2 : numpy ndarray
        x positions from group 2
    y2 : numpy ndarray
        y positions from group 2

    Returns
    -------
    md : numpy ndarray
        Array containing the smallest distances between the two groups for each entry.
    """
    dx = x1[:,np.newaxis] - x2[np.newaxis,:]
    dy = y1[:,np.newaxis] - y2[np.newaxis,:]
    d = np.sqrt(dx**2 + dy**2)
    md = np.nanmin(d,axis=1)
    return md

def minimize_dist(offset,x1,y1,x2,y2,image):
    """
    Minimizing function to find the minimum distance between 2 sets of xy coordinates.

    Parameters 
    ----------
    offset : offset : numpy ndarray, list
        Parameters used in the coordinate shift: [0] is the x shify; [1] is the y shift; [3] is the rotation
    x1 : numpy ndarray
        x positions from group 1
    y1 : numpy ndarray
        y positions from group 1
    x2 : numpy ndarray
        x positions from group 2
    y2 : numpy ndarray
        y positions from group 2
    image : numpy ndarray
        Target image, used to get the image dimensions correct.

    Returns
    -------
    mean_mdist : float
        Mean distance for between all sources in groups 1 and 2.


    """
    x,y = transform_coords(x1,y1,offset,image)

    ind = (x > 0) & (x < image.shape[1]) & (y > 0) & (y < image.shape[0])
    x = x[ind]; y = y[ind]
    mdist = min_dist(x2,y2,x,y)
    mean_mdist = np.nanmean(mdist)
    return mean_mdist

def basic_image(offset,x,y,image,kernel):
    """
    Create a simplistic catalog image from input positions and a convolution kernel. Sources are added to the array then convolved with the kernel.

    Parameters
    ----------
    offset : numpy ndarray, list
        Parameters used in the coordinate shift: [0] is the x shify; [1] is the y shift; [3] is the rotation
    x : numpy ndarray 
        x positions of the catalog sources 
    y : numpy ndarray 
        y positions of the catalog sources 
    image : numpy ndarray
        Target image, used to get the image dimensions correct.
    kernel : numpy ndarray
        Convolution kernel to create the simple image from the catalog sources.
    """
    x,y = transform_coords(x,y,offset,image)
    
    cut = (0<=x) & (x < image.shape[1]) & (0<=y) & (y < image.shape[0])
    x = x[cut]; y = y[cut]
    guess = np.zeros_like(image)
    guess[y.astype(int),x.astype(int)] = 1
    guess = (signal.fftconvolve(guess,kernel,mode='same') > 0.1) * 1.0
    return guess

def minimize_cats(offset,x,y,image,kernel):
    """
    Simplistic minimising function for finding an alignment between an image and a catalog by subtracting a convolved source image from the target image.

    Parameters
    ----------
    offset : numpy ndarray, list
        Parameters used in the coordinate shift: [0] is the x shify; [1] is the y shift; [3] is the rotation
    x : numpy ndarray
        x coordinates of the sources 
    y : numpy ndarray
        y coordinates of the sources 
    image : numpy ndarray 
        Boolean image to compare the catalog positions to.
    kernel : numpy ndarray 
        Basic kernel which is used to create a basic image from the x y catalog positions.

    Returns
    -------
    res : float
        Residual of the difference
    """
    guess = basic_image(offset,x,y,image,kernel)
    res = np.nansum(image - guess)
    return res



def transform_coords(x,y,param,image):
    """
    Transforms the input coordinates by x/y shifts and a rotation around the image center.

    Parameters
    ----------
    x : numpy ndarray
        x positions 
    y : numpy ndarray
        y positions 
    param : numpy ndarray, list 
        Parameters used in the coordinate shift: [0] is the x shify; [1] is the y shift; [3] is the rotation
    image : numpy ndarray
        Image that the coordinates are being applied to, this is only used to get the center of rotation.

    Returns
    -------
    xx : numpy ndarray
        new x positions 
    yy : numpy ndarray
        new y positions 
    """
    xx = x + param[0]
    yy = y + param[1]
    cx = image.shape[1]/2; cy = image.shape[0]/2
    xx = cx + ((xx-cx)*np.cos(param[2])-(yy-cy)*np.sin(param[2]))
    yy = cy + ((xx-cx)*np.sin(param[2])+(yy-cy)*np.cos(param[2]))
    ind = (xx > 0) & (xx < image.shape[1]) & (yy > 0) & (yy < image.shape[0])
    #xx = xx[ind]; yy = yy[ind]
    return xx, yy

def get_star_cuts(x_length,y_length,image,cat,norm=False):
    """
    Create image cutouts of sources based on xy positions from the input catalog.

    Parameters
    ----------
    x_length : int
        Length of the x dimension of the cutout
    y_length : int
        Length of the y dimension of the cutout 
    image : numpy ndarray 
        Image to make the cutout from 
    cat : pandas DataFrame 
        Catalog of sources containing the xy pixel position
    norm : Bool
        Option to normalise the cutouts.

    Returns
    -------
    star_cuts : numpy ndarray
        Cutouts of the sources defined in the catalog 
    good : numpy ndarray
        Array of boolean entries defining if sources are good or not. Bad sources are close to the edge.
    """
    pad = np.nanmax([y_length,x_length])*2
    image = np.pad(image,pad) 
    image[image==0] = np.nan
    try:
        x = cat['xint'].values + pad; y = cat['yint'].values + pad
    except:
        x = [cat['xint'] + pad]; y = [cat['yint'] + pad]
    star_cuts = []
    good = []

    for i in range(len(x)):
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
    """
    Attempt to repace the source in the cutout (dont think this is used.)

    Parameters
    ----------
    x_length : int
        Length of the x dimension of the cutout
    y_length : int
        Length of the y dimension of the cutout 
    image : numpy ndarray 
        Image to make the cutout from 
    cuts : numpy ndarray 
        Array of cutouts in the image to be placed back in.
    cat : pandas DataFrame
        Catalog of sources containing their coordinates.

    Retrurns
    --------
    replace : numpy ndarray
        Image where the cutouts have been replaced into it.

    """
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
    """
    Not used
    """
    replace = deepcopy(cube)
    for i in range(len(cube)):
        replace[i] = replace_cut(x_length,y_length,cut[i],cat)
    return replace


def psf_spec(cube,psf,data_psf=None,fitpos=True):
    """
    Create a PSF and fit it to the input cube. Then calculate the flux of the source in the image.

    Parameters
    ----------
    cube : numpy ndarray
        Image to calculate the flux of.
    psf_param : numpy ndarray
        Parameters used to create the PSF. [0] is the FWHM; [1] is the length of the PSF; [2] is the angle of the PSF.
    data_psf : numpy ndarray
        PSF to use in the fitting process. The default is None, which means a PSF will be created from the input image.

    Returns
    -------
    flux : float
        Flux of the source in the image.
    residual : numpy ndarray
        Residual of the PSF fit to the image.
    xoff : float
        X offset of the source from the PSF fit.
    yoff : float
        Y offset of the source from the PSF fit.
    """
    #if data_psf is None:
    #   psf_tuning = '' 
    #else:
    #   psf_tuning = ' data'
    #if len(psf_param) > 3:
    #   psf = create_psf(x=cube.shape[2],y=cube.shape[1],alpha=psf_param[0],
    #             beta=psf_param[1],length=psf_param[2],angle=psf_param[3],
    #             psf_profile='moffat'+psf_tuning)
    #   l = psf_param[2]
    #else:
    #   psf = create_psf(x=cube.shape[2],y=cube.shape[1],stddev=psf_param[0],
    #             length=psf_param[1],angle=psf_param[2],
    #             psf_profile='gaussian'+psf_tuning)
       
    l = psf.length
    if l < 2:
        r = 0.5
    else:
        r = 0.05 * l
        if r > 2:
            r = 2
    if fitpos:
        psf.fit_pos(np.nanmean(cube,axis=0),range=r)
        xoff = psf.source_x; yoff = psf.source_y
    else:
        xoff = 0; yoff = 0
    
    
    #flux,res = zip(*Parallel(n_jobs=num_cores)(delayed(psf.psf_flux)(image) for image in cube))
    flux = []
    res = []
    #psf.data_psf = data_psf
    for image in cube:
        f,r = psf.psf_flux(image)
        flux += [f]
        res += [r]
    
    flux = np.array(flux)
    flux[flux<0] = 0
    residual = np.array(res)
    xoff = np.nanmedian(np.array(xoff)); yoff = np.nanmedian(np.array(yoff))
    return flux,residual, xoff, yoff

def cube_cutout(cube,cat,x_length,y_length):
    """
    Create cutouts of sources from the input cube.
    """
    cuts = []
    for i in range(len(cube)):
        cut,good = get_star_cuts(x_length,y_length,cube[i],cat)
        cuts += [cut]
    cuts = np.array(cuts)
    cuts = np.swapaxes(cuts,0,1)
    return cuts


def get_specs(cat,cube,x_length,y_length,psf,lam,num_cores,data_psf=None,fitpos=True):
    """
    Get the spectra of sources in the input cube.

    Parameters
    ----------
    cat : pandas DataFrame
        Catalog of sources containing their coordinates.
    cube : numpy ndarray
        Input image containing the sources.
    x_length : int
        Length of the x dimension of the cutout
    y_length : int
        Length of the y dimension of the cutout 
    psf_params : numpy ndarray
        Parameters used to create the PSF. [0] is the FWHM; [1] is the length of the PSF; [2] is the angle of the PSF.
    lam : numpy ndarray
        Wavelength array of the input cube.
    num_cores : int
        Number of cores to use in the fitting process.
    data_psf : numpy ndarray
        PSF to use in the fitting process. The default is None, which means a PSF will be created from the input image.

    Returns
    -------
    specs : list
        List of spectra for each source in the input catalog.
    residual : numpy ndarray
        Residual of the PSF fit to the image.
    cat : pandas DataFrame
        Catalog of sources containing their coordinates and offsets from the PSF fit.
    """
    cuts = cube_cutout(cube,cat,x_length,y_length)
    ind = np.arange(0,len(cuts)+1)
    #num_cores = multiprocessing.cpu_count() - 3
    specs = []
    residual = []
    sub_cube = deepcopy(cube)
    flux, res, xoff, yoff = zip(*Parallel(n_jobs=num_cores,verbose=0)(delayed(psf_spec)(cut,psf,data_psf,fitpos) for cut in cuts))
    #flux = np.zeros(len(cuts)); res = np.zeros(len(cuts))
    #xoff = np.zeros(len(cuts)); yoff = np.zeros(len(cuts))
    #for i in range(len(cuts)):
    #   f, r, xo, yo = psf_spec(cuts[i],psf_params)
    #   flux[i] = f; res[i] = r; xoff[i] = xo; yoff[i] = yo

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


def psf_spec2(cube,psf,data_psf=None):
    """
    Create a PSF and fit it to the input cube. Then calculate the flux of the source in the image.

    Parameters
    ----------
    cube : numpy ndarray
        Image to calculate the flux of.
    psf_param : numpy ndarray
        Parameters used to create the PSF. [0] is the FWHM; [1] is the length of the PSF; [2] is the angle of the PSF.
    data_psf : numpy ndarray
        PSF to use in the fitting process. The default is None, which means a PSF will be created from the input image.

    Returns
    -------
    flux : float
        Flux of the source in the image.
    residual : numpy ndarray
        Residual of the PSF fit to the image.
    xoff : float
        X offset of the source from the PSF fit.
    yoff : float
        Y offset of the source from the PSF fit.
    """
    #if data_psf is None:
    #   psf_tuning = '' 
    #else:
    #   psf_tuning = ' data'
    #if len(psf_param) > 3:
    #   psf = create_psf(x=cube.shape[2],y=cube.shape[1],alpha=psf_param[0],
    #             beta=psf_param[1],length=psf_param[2],angle=psf_param[3],
    #             psf_profile='moffat'+psf_tuning)
    #   l = psf_param[2]
    #else:
    #   psf = create_psf(x=cube.shape[2],y=cube.shape[1],stddev=psf_param[0],
    #             length=psf_param[1],angle=psf_param[2],
    #             psf_profile='gaussian'+psf_tuning)
    l = psf.length
    if l < 5:
        r = 1
    else:
        r = 1
    psf.fit_pos(np.nanmean(cube,axis=0),range=r)
    xoff = psf.source_x; yoff = psf.source_y
    
    
    #flux,res = zip(*Parallel(n_jobs=num_cores)(delayed(psf.psf_flux)(image) for image in cube))
    flux = []
    res = []
    psf.data_psf = data_psf
    for image in cube:
        f,r = psf.psf_flux(image)
        flux += [f]
        res += [r]
    
    flux = np.array(flux)
    flux[flux<0] = 0
    residual = np.array(res)
    xoff = np.nanmedian(np.array(xoff)); yoff = np.nanmedian(np.array(yoff))
    return flux,residual, xoff, yoff

def _smooth_spec(spec,sigma=10,repeats=3,smooth_bin=51):
    flux = deepcopy(spec.flux)
    y = deepcopy(spec.flux)
    med = np.nanmedian(y)
    if med > 0:
        for repeat in range(repeats):
            med = np.nanmedian(y)
            y /= med
            x = spec.wave
            dy_dx = np.abs(np.gradient(y, x))
            m, med, std = sigma_clipped_stats(dy_dx)

            # Define a threshold for the gradient (you may need to tune this)
            gradient_threshold = med + sigma * std
            # Identify where the gradient exceeds the threshold
            sharp_feature_mask = (dy_dx > gradient_threshold) | np.isnan(dy_dx)
            y[sharp_feature_mask] = np.nan
            finite = np.isfinite(y) & np.isfinite(x)

            x_valid = x[~sharp_feature_mask & finite]  # x values where sharp features are not present
            y_valid = flux[~sharp_feature_mask & finite]  # corresponding y values
            if len(x_valid) > 10:
                # Clip or smooth the spectrum at sharp features (using Gaussian smoothing as an example)
                interp_func = interp1d(x_valid, y_valid, kind='linear', fill_value="nan",bounds_error=False)
            else:
                print('bad')
                return S.ArraySpectrum(x,y * 0,fluxunits='flam',name=str(spec.name)+'_smooth')
        # Apply the interpolation to fill the masked sharp feature regions
        y_filled = flux.copy()
        y_filled[sharp_feature_mask | ~finite] = interp_func(x[sharp_feature_mask | ~finite])
        y_filled = y_filled

        y_smooth = savgol_filter(y_filled,smooth_bin,1)

        s = S.ArraySpectrum(x,y_smooth,fluxunits='flam',name=str(spec.name)+'_smooth')
    else:
        s = deepcopy(spec)
    return s
        
def downsample_spec(spec,target_lam):
    """
    Downsample a spectrum to a new wavelength grid.

    Parameters
    ----------
    spec : pysynphot.spectrum.ArraySpectrum
        Input spectrum
    target_lam : numpy ndarray
        New wavelength grid to downsample to.

    Returns
    -------
    flux : numpy ndarray
        Downsampled spectrum.
    """
    diff = np.gradient(target_lam)
    step = np.min(diff)/10
    lam_grid = np.arange(target_lam[0],target_lam[-1]+step,step)
    int_f = spec.sample(lam_grid)
    flux = np.zeros_like(target_lam)

    for i in range(len(diff)):
        low = target_lam[i] - diff[i]*2
        high = target_lam[i] + diff[i]*2
        ind = (lam_grid >= low) & (lam_grid <= high)
        #l = lam_grid[ind]
        #f = int_f[ind]
        #flux[i] = np.trapz(f, x=l)
        flux[i] = np.nanmedian(int_f[ind])
    #flux[0] = int_f[0]; flux[-1] = int_f[-1]
    #flux = savgol_filter(flux,101,1)
    return flux

def _spec_compare(mod_files,lam,flux):
    """
    Compare a spectrum to a set of model spectra.

    Parameters
    ----------
    mod_files : list
        List of model spectra to compare to.
    lam : numpy ndarray
        Wavelength array of the input spectrum.
    flux : numpy ndarray
        Flux array of the input spectrum.

    Returns
    -------
    cors : numpy ndarray
        Array of correlation coefficients for each model spectrum.
    f : numpy ndarray
        Downsampled input spectrum.
    """
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
    """
    Compare an input spectrum to a set of model spectra.

    Parameters
    ----------
    model_files : list
        List of model spectra to compare to.
    lam : numpy ndarray
        Wavelength array of the input spectrum.
    flux : numpy ndarray
        Flux array of the input spectrum.

    Returns
    -------
    model : pysynphot.spectrum.ArraySpectrum
        Best fitting model spectrum.
    cor : float
        Correlation coefficient of the best fitting model spectrum.
    """
    model_files.sort()
    cors, flux = _spec_compare(model_files,lam,flux)
    ind = np.argmax(cors)
    name = model_files[ind].split('/')[-1].split('.dat')[0]
    model = at.Table.read(model_files[ind], format='ascii')
    if '_a' in name:
        name = name.split('_a')[0]
    model = S.ArraySpectrum(wave=model['wave'].value,#lam_vac2air(model['wave'].value),
                                flux=model['flux'].value,fluxunits='flam',name=name)
    return model, cors[ind]

def match_spec_to_model(specs,mags,pbs,catalog='ck+',num_cores=-1):
    """
    Match an input spectrum to a model spectrum.

    Parameters
    ----------
    spec : pysynphot.spectrum.ArraySpectrum
        Input spectrum
    catalog : str
        Name of the model catalog to compare to. The default is 'ck+'.

    Returns
    -------
    model : pysynphot.spectrum.ArraySpectrum
        Best fitting model spectrum.
    cor : float
        Correlation coefficient of the best fitting model spectrum.
    redshift : float
        Redshift of the input spectrum.
    """
    specs = np.array(specs)
    #lam = spec.wave
    #flux = spec.flux
    #flux = savgol_filter(flux,101,1)
    if 'ck' in catalog.lower():
        path = f'{package_directory}data/ck04_stsci/*'
        #path = f'{package_directory}data/munari05/*'
        model_files = glob(path)
        #model, cor = _compare_catalog(model_files,lam,flux)
        models, cor, ebvs = _match_obs_to_model(specs,model_files,mags,pbs,num_cores=num_cores)

        if '+' in catalog:
            high_t = []
            low_t = []
            low_temps = []
            high_temps = []
            for i in range(len(models)):
                model = models[i]
                temp = int(model.name.split('_')[-2])
                if temp <= 7000:
                    low_t += [i]
                    low_temps += [temp]
                elif temp >= 15000:
                    high_t += [i]
                    high_temps += [temp]
            high_t = np.array(high_t)
            low_t = np.array(low_t)
            high_temps = np.array(high_temps)
            low_temps = np.array(low_temps)

            if len(low_t) > 0:
                path = f'{package_directory}data/marcs-t02/*'
                model_files = np.array(glob(path))
                temps = np.array([int(x.split('/')[-1][1:5]) for x in model_files])
                ul = np.max(low_temps) + 500; ll = np.min(low_temps) - 500
                if np.min(low_temps) <= 3600:
                    ll = 2500
                ind = (temps >= ll) & (temps <= ul)
                model_files = model_files[ind]
                model2, cor2, ebv = _match_obs_to_model(specs[low_t],model_files,mags[low_t],pbs,num_cores=num_cores)
                ind = cor2 > cor[low_t]
                #models[low_t][ind] = model2[ind]
                #cor[low_t][ind] = cor2[ind]
                #ebvs[low_t][ind] = ebv[ind]
                models[low_t] = model2
                cor[low_t] = cor2
                ebvs[low_t] = ebv

            if len(high_t) > 0:
                path = f'{package_directory}data/griddl-ob-i-line/*'
                model_files = np.array(glob(path))
                model2, cor2, ebv = _match_obs_to_model(specs[high_t],model_files,mags[high_t],pbs,num_cores=num_cores)
                ind = cor2 > cor[high_t]
                #models[high_t][ind] = model2[ind]
                #cor[high_t][ind] = cor2[ind]
                #ebvs[high_t][ind] = ebv[ind]
                models[high_t] = model2
                cor[high_t] = cor2
                ebvs[high_t] = ebv

    elif catalog.lower() == 'eso':
        path = f'{package_directory}data/eso_spec/*'
        model_files = glob(path)
        models, cor, ebvs = _match_obs_to_model(specs,model_files,mags,pbs,num_cores=num_cores)#_compare_catalog(path,lam,flux)
    else:
        m = '!! No valid model spectral catalog selected !!'
        raise ValueError(m)
    redshift = []
    for spec in specs:
        shift,zerr, _ = calc_redshift(spec)
        redshift += [shift]

    #redshift = 0
    #model = model.redshift(-redshift)

    return models, cor, redshift, ebvs



def ebv_minimiser(ebv,model,spec,Rv=3.1):
    """
    Minimising function for fitting extinction to a spectrum.

    Parameters
    ----------
    ebv : float
        E(B-V) value to fit.
    model : pysynphot.spectrum.ArraySpectrum
        Model spectrum to apply the extinction to.
    spec : pysynphot.spectrum.ArraySpectrum
        Input spectrum to fit the extinction to.
    Rv : float
        Rv value to use in the extinction fit. The default is 3.1.

    Returns
    -------
    res : float
        Negative correlation coefficient of the model spectrum and the input spectrum.
    """
    ext = S.ArraySpectrum(model.wave, 
                    apply(fitzpatrick99(model.wave.astype('double'),ebv*Rv,Rv),model.flux))
    interp = ext.sample(spec.wave)
    corr = pearsonr(savgol_filter(spec.flux,101,1),savgol_filter(interp,101,1))[0]
    res = np.log(1-corr)
    
    return res

def fit_extinction(model,spec,Rv = 3.1):
    """
    Find the best extinction value to maximuze the correlation coefficient between the input spectrum and the model spectrum.

    Parameters
    ----------
    model : pysynphot.spectrum.ArraySpectrum
        Model spectrum to apply the extinction to.
    spec : pysynphot.spectrum.ArraySpectrum
        Input spectrum to fit the extinction to.
    Rv : float
        Rv value to use in the extinction fit. The default is 3.1.

    Returns
    -------
    ext : pysynphot.spectrum.ArraySpectrum
        Model spectrum with the extinction applied.
    ebv : float
        E(B-V) value that maximizes the correlation coefficient between the input spectrum and the model spectrum.
    """
    lam = spec.wave
    ebv0 = 0
    bounds = [[0,10]]
    res = minimize(ebv_minimiser, ebv0,args=(model,spec,Rv),method='Nelder-Mead',bounds=bounds)
    ebv = res.x[0]
    ext = S.ArraySpectrum(model.wave, 
                    apply(fitzpatrick99(model.wave.astype('double'),ebv*Rv,Rv),model.flux),name=model.name + ' ebv=' + str(np.round(ebv,3)))
    return ext,ebv


def _norm_spec(spec,mag,pbs):
    """
    Normalise the spectrum according to the input catalog magnitudes. If more than 1 filter is present, then the spectrum is mangled to best match all inputs.

    Parameters
    ----------
    spec : pysynphot.spectrum.ArraySpectrum
        Input spectrum
    mag : numpy ndarray
        Magnitude of the input spectrum
    pbs : numpy ndarray
        Photometric bandpasses corresponding to the input magnitudes.

    Returns
    -------
    norm : pysynphot.spectrum.ArraySpectrum
        Normalised spectrum.
    """
    keys = pbs.keys()
    if len(keys) > 1:
        norm = spec_mangle(spec,mag,pbs,name=spec.name)
    else:
        norm = my_norm(spec,pbs,mag,name=spec.name)
    return norm

def _par_spec_fit(spec,pbs,mag,model_type):
    """
    Parallel function for fitting models to spectra.

    Parameters
    ----------
    spec : pysynphot.spectrum.ArraySpectrum
        Input spectrum
    pbs : numpy ndarray
        Photometric bandpasses corresponding to the input magnitudes.
    mag : numpy ndarray
        Magnitude of the input spectrum
    model_type : str
        Name of the model catalog to compare to.

    Returns
    -------
    ext : pysynphot.spectrum.ArraySpectrum
        Model spectrum with the extinction applied.
    cor : float
        Correlation coefficient of the best fitting model spectrum.
    ebv : float
        E(B-V) value that maximizes the correlation coefficient between the input spectrum and the model spectrum.
    redshift : float
        Redshift of the input spectrum.
    """
    model,cor,redshift = match_spec_to_model(spec,model_type,mags,pbs)
    model2 = _norm_spec(model,mag,pbs)

    ext, ebv = fit_extinction(model2,spec)

    ext = _norm_spec(ext,mag,pbs)

    return ext, cor, ebv, redshift



def spec_match(specs,mags,filters,model_type='ck+',num_cores=5):
    """
    Match a set of input spectra to model spectra.

    Parameters
    ----------
    specs : list
        List of input spectra.
    mags : numpy ndarray
        Magnitudes of the input spectra.
    filters : numpy ndarray
        Photometric bandpasses corresponding to the input magnitudes.
    model_type : str
        Name of the model catalog to compare to. The default is 'ck+'.
    num_cores : int
        Number of cores to use in the fitting process. The default is 5.

    Returns
    -------
    cal_model : list
        List of model spectra with extinction applied.
    cors : numpy ndarray
        Array of correlation coefficients for each model spectrum.
    ebvs : numpy ndarray
        Array of E(B-V) values for each model spectrum.
    redshift : numpy ndarray
        Array of redshifts for each input spectrum.
    """

    svo_bp=filters
    pbs = load_pbs(svo_bp,0,'AB',SVO=True)
    model,cors,redshift,ebvs = match_spec_to_model(specs,mags,pbs,model_type,num_cores=num_cores)
    #cal_model, cors, ebvs, redshift = zip(*Parallel(n_jobs=num_cores)(delayed(_par_spec_fit)(specs[i],pbs,mags[i],model_type) for i in range(len(specs))))
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
    return model, cors, ebvs, redshift

def parallel_psf_fit(image,psf,psf_profile):
    """
    Parallel function for fitting a PSF to an image.

    Parameters
    ----------
    image : numpy ndarray
        Input image
    psf : trail_psf.create_psf
        PSF to fit to the image.
    psf_profile : str
        Name of the PSF profile to use.

    Returns
    -------
    params : numpy ndarray
        PSF fit parameters.
    shifts : numpy ndarray
        PSF fit offsets.
    """
    #xlims = image.shape[1] / 3
    #ylims = image.shape[0] / 3
    #if xlims < 10: xlims=10
    #if ylims < 10: ylims=10

    psf.fit_psf(image)#,limx=xlims,limy=ylims)
    psf.generate_line_psf()
    if 'moffat' in psf_profile:
        params = np.array([psf.alpha,psf.beta,psf.length,psf.angle])
        shifts = np.array([psf.source_x,psf.source_y])
    elif 'gaussian' in psf_profile:
        params = np.array([psf.stddev,psf.length,psf.angle])
        shifts = np.array([psf.source_x,psf.source_y])
    return params, shifts



def lam_vac2air(lam):
    """
    Convert vacuum wavelength to air wavelength.

    Parameters
    ----------
    lam : numpy ndarray
        Vacuum wavelength to convert.

    Returns
    -------
    air : numpy ndarray
        Air wavelength.
    """
    air = lam / (1.0 + 2.735182e-4 + 131.4182 / lam**2 + 2.76249E8 / lam**4)

    return air

def _has_len(obj):
    return hasattr(obj, '__len__')

def _create_model_grid(model,e,Rv=3.1):
    ext = S.ArraySpectrum(model.wave, 
                        apply(fitzpatrick99(model.wave.astype('double'),e*Rv,Rv),
                              model.flux),name=model.name + ' ebv=' + str(e))
    ext = S.ArraySpectrum(wave=ext.wave,
                        flux=ext.flux/np.nanmedian(ext.flux),fluxunits='flam',name=ext.name)
    return ext


def _model_grid(model_file,target_lam=None,max_ext=4,num_cores=1):
    extinctions = np.arange(0,max_ext,0.01)
    extinctions = np.round(extinctions,3)
    name = model_file.split('/')[-1].split('.dat')[0]
    model = at.Table.read(model_file, format='ascii')
    if '_a' in name:
        name = name.split('_a')[0]
    model = S.ArraySpectrum(wave=model['wave'].value,#lam_vac2air(model['wave'].value),
                            flux=model['flux'].value,fluxunits='flam',name=name)
    if target_lam is not None:
        model = S.ArraySpectrum(wave=target_lam,#lam_vac2air(model['wave'].value),
                                flux=downsample_spec(model,target_lam),fluxunits='flam',name=name)
    if num_cores == 1:
        exts = []
        for i in range(len(extinctions)):
            exts += [_create_model_grid(model,extinctions[i])]
    else:
        exts = Parallel(n_jobs=num_cores)(delayed(_create_model_grid)(model,extinctions[i]) for i in range(len(extinctions)))
    return exts

def _calc_cor(spec,model_fluxes,num_cores=-1):
    finite = np.isfinite(spec.flux)
    coeff = np.array([pearsonr(spec.flux[finite],m[finite])[0] for m in model_fluxes])
    coeff[coeff<0] = 0
    coeff[~np.isfinite(coeff)] = 0
    return coeff#[:,0]

def _refactoring(model):
    flux = model.flux
    ext = float(model.name.split('=')[-1])
    return flux, ext


def _parallel_match(spec,model_flux):
    cors = _calc_cor(spec,model_flux)
    ind = np.argmax(cors)
    cor = cors[ind]
    return cor, ind 

def _match_obs_to_model(obs_spec,model_files,mags,pbs,num_cores=-1):
    if not _has_len(obs_spec):
        obs_spec = [obs_spec]
    if not _has_len(mags):
        mags = [mags]

    for i in range(len(obs_spec)):
        obs_spec[i] = _smooth_spec(obs_spec[i])

    model_grid = Parallel(n_jobs=num_cores)(delayed(_model_grid)(model_files[i],target_lam=obs_spec[0].wave) for i in range(len(model_files)))
    model_grid = np.array(model_grid).flatten()
    model_flux, model_ebv = zip(*[_refactoring(m) for m in model_grid])
    model_flux = np.array(model_flux); model_ebv = np.array(model_ebv)
    cor, ind = zip(*Parallel(n_jobs=num_cores)(delayed(_parallel_match)(spec, model_flux) for spec in obs_spec))
    cor = np.array(cor); ind = np.array(ind)
    model_spec = model_grid[ind]
    ebv = model_ebv[ind]
    for i in range(len(model_spec)):
        model_spec[i] = my_norm(model_spec[i],pbs,mags[i],name=model_spec[i].name)

    return model_spec, cor, ebv
    

def calc_redshift(spec):
    """
    Calculate the redshift of an input spectrum by fitting Gaussians to notable absorption lines.

    Lines used:
        Hb
        NaD
        Ha
        CaII triplet (a b c)

    Parameters
    ----------
    spec : pysynphot.spectrum.ArraySpectrum
        Input spectrum

    Returns
    -------
    shift : float
        Redshift of the spectrum.
    fitted : dict
        Dictionary containing the fit results for each absorption line.
    """
    from astropy.modeling import models,fitting
    from astropy import modeling
    lines = {'Hb':np.array([4861.323]),
         'NaD':np.array([5889.951, 5895.924]),
         'Ha':np.array([6562.797]),
         'CaII_a':np.array([8498.020]),
         'CaII_b':np.array([8542.088]),
         'CaII_c':np.array([8662.138])
            }
    fitted = {}
    for line in lines.keys():
        fits = []
        wave = []
        flux = []
        em = lines[line]
        cont = []
        dip = []
        for i in range(len(em)):
            cont += [np.nanmedian(spec.sample(np.arange(em[i]+20,em[i]+40)))]
            dip += [spec.sample(em[i])/cont[i] - 1]

        mod = []
        if line == 'NaD':
            g_init = (models.Const1D(1) + models.Gaussian1D(amplitude=(dip[0]), mean=em[0], stddev=1)
                      + models.Gaussian1D(amplitude=(dip[1]), mean=em[1], stddev=1))
            mod = g_init
            ind = (spec.wave > em[0]-20) & (spec.wave < em[1]+20)
            wave = spec.wave[ind]
            flux = spec.flux[ind]/cont[0]
        else:
            for i in range(len(em)):
                g_init = (models.Const1D(1) + models.Gaussian1D(amplitude=(dip[i]), mean=em[i], stddev=1))
                mod = g_init
                ind = (spec.wave > em[0]-20) & (spec.wave < em[0]+20)
                wave = spec.wave[ind]
                flux =  spec.flux[ind]/cont[i]


        fit_mod = fitting.LevMarLSQFitter(calc_uncertainties=True)

        finite = np.isfinite(flux)
        if np.nansum(finite) > 10:
            g = fit_mod(mod, wave[finite], flux[finite])
            covariance = fit_mod.fit_info['param_cov']
            if line == 'NaD':
                diff = np.mean(np.array([g.mean_1.value/em[0]-1,g.mean_2.value/em[1]-1]))
                if covariance is not None:
                    error = np.sqrt(np.diag(fit_mod.fit_info['param_cov']))
                    pos_er = abs(np.nanmean(np.array([error[2] / (g.mean_1.value-em[0]),error[5] / (g.mean_2.value-em[1])])))
                    amp_er = abs(np.nanmean(error[[1,4]]) / np.nanmean([g.amplitude_1.value,g.amplitude_2.value]))
                else:
                    pos_er = 10; amp_er = 10

            else:
                diff = g.mean_1 / em[0] - 1
                if covariance is not None:
                    error = np.sqrt(np.diag(fit_mod.fit_info['param_cov']))
                    pos_er = abs(error[2] / (g.mean_1.value-em[0]))
                    amp_er = abs(error[1] / g.amplitude_1.value)
                else:
                    pos_er = 10; amp_er = 10
            m = g(wave)
            cor = pearsonr(m[finite],flux[finite]).correlation

            if (g.amplitude_1.value < 0) & (amp_er < 0.1) & (pos_er < 0.2):
                good = True
            else:
                good = False
        else:
            g = None
            cor = 0
            diff = np.nan
            good = False
            pos_er = np.nan
        fitted[line] = {'fit':g,'wave':wave,'flux':flux,'shift':diff,'cor':cor,'quality':good,'error':abs(pos_er*diff)}

    shift = []
    er = []
    for line in lines.keys():   
        if fitted[line]['quality']:
            shift += [fitted[line]['shift']]
            er += [fitted[line]['error']]
    er = np.array(er)
    shift = np.nansum(shift/er**2)/np.nansum(1/er**2) #np.average(shift,weights=abs(1/er))
    er = np.sqrt(1/np.nansum(1/er**2))
    if np.isnan(shift):
        shift = 0
        er = 0
    
    return shift, er, fitted


def _calc_val(val):
    val = str(int(np.round(val,0)))
    if '-' in val:
        val = r'$-$' + val[1:]
    return val

def plot_z_shifts(spec):
    """
    Plot the input spectrum with the notable absorption lines marked and the best fitting model overlaid.

    Parameters
    ----------
    spec : pysynphot.spectrum.ArraySpectrum
        Input spectrum
    """
    from astropy.constants import c
    light = c.to('km/s').value
    fig_width_pt = 240.0  # Get this from LaTeX using \showthe\columnwidth
    inches_per_pt = 1.0/72.27       # Convert pt to inches
    golden_mean = (np.sqrt(5)-1.0)/2.0   # Aesthetic ratio
    fig_width = fig_width_pt*inches_per_pt  # width in inches
    shift,error,fitted = calc_redshift(spec)
    fig, axs = plt.subplot_mosaic('''
                                   ABC
                                   DEF
                                   ''',
                                  figsize=(1.5*fig_width*2,1*fig_width*1.5))


    axs['A'].set_title(r'H$\beta$')
    axs['A'].plot(fitted['Hb']['wave'],fitted['Hb']['flux'],label='Spectrum')
    mod = fitted['Hb']['fit'](fitted['Hb']['wave'])
    axs['A'].plot(fitted['Hb']['wave'],mod,'--',label='Model')
    axs['A'].legend(loc=4)
    axs['A'].set_ylabel('Normalised flux',fontsize=12)
    val = _calc_val(fitted['Hb']['shift']*light)

    axs['A'].annotate(r'$v=$'+val+r'$\pm$'
                       + str(int(np.round(fitted['Hb']['error']*light,0)))+' km/s',
                      (.02,.04), xycoords='axes fraction',fontsize=14)
    axs['A'].set_ylim(np.min(fitted['Hb']['flux'])*.9,np.max(fitted['Hb']['flux'])+.05)

    axs['B'].set_title(r'H$\alpha$')
    axs['B'].plot(fitted['Ha']['wave'],fitted['Ha']['flux'])
    mod = fitted['Ha']['fit'](fitted['Ha']['wave'])
    axs['B'].plot(fitted['Ha']['wave'],mod,'--')
    val = _calc_val(fitted['Ha']['shift']*light)
    axs['B'].annotate(r'$v=$'+val+r'$\pm$'
                       + str(int(np.round(fitted['Ha']['error']*light,0)))+' km/s',
                      (.02,.04), xycoords='axes fraction',fontsize=14)
    axs['B'].set_ylim(np.min(fitted['Ha']['flux'])*.9,np.max(fitted['Ha']['flux'])+.05)

    axs['C'].set_title(r'Na D')
    axs['C'].plot(fitted['NaD']['wave'],fitted['NaD']['flux'])
    mod = fitted['NaD']['fit'](fitted['NaD']['wave'])
    axs['C'].plot(fitted['NaD']['wave'],mod,'--')
    val = _calc_val(fitted['NaD']['shift']*light)
    axs['C'].annotate(r'$v=$'+val+r'$\pm$'
                       + str(int(np.round(fitted['NaD']['error']*light,0)))+' km/s',
                      (.02,.04), xycoords='axes fraction',fontsize=14)
    axs['C'].set_ylim(np.min(fitted['NaD']['flux'])*.9,np.max(fitted['NaD']['flux'])+.05)

    axs['D'].set_title(r'Ca II triplet $a$')
    axs['D'].plot(fitted['CaII_a']['wave'],fitted['CaII_a']['flux'])
    mod = fitted['CaII_a']['fit'](fitted['CaII_a']['wave'])
    axs['D'].plot(fitted['CaII_a']['wave'],mod,'--')
    axs['D'].set_xlabel(r'Wavelength ($\rm \AA$)',fontsize=12)
    axs['D'].set_ylabel('Normalised flux',fontsize=12)
    val = _calc_val(fitted['CaII_a']['shift']*light)
    axs['D'].annotate(r'$v=$'+val+r'$\pm$'
                       + str(int(np.round(fitted['CaII_a']['error']*light,0)))+' km/s',
                      (.02,.04), xycoords='axes fraction',fontsize=14)
    axs['D'].set_ylim(np.min(fitted['CaII_a']['flux'])*.9,np.max(fitted['CaII_a']['flux'])+.05)

    axs['E'].set_title(r'Ca II triplet $b$')
    axs['E'].plot(fitted['CaII_b']['wave'],fitted['CaII_b']['flux'])
    mod = fitted['CaII_b']['fit'](fitted['CaII_b']['wave'])
    axs['E'].plot(fitted['CaII_b']['wave'],mod,'--')
    axs['E'].set_xlabel(r'Wavelength ($\rm \AA$)',fontsize=12)
    val = _calc_val(fitted['CaII_b']['shift']*light)
    axs['E'].annotate(r'$v=$'+val+r'$\pm$'
                       + str(int(np.round(fitted['CaII_b']['error']*light,0)))+' km/s',
                      (.02,.04), xycoords='axes fraction',fontsize=14)
    axs['E'].set_ylim(np.min(fitted['CaII_b']['flux'])*.9,np.max(fitted['CaII_b']['flux'])+.05)

    axs['F'].set_title(r'Ca II triplet $c$')
    axs['F'].plot(fitted['CaII_c']['wave'],fitted['CaII_c']['flux'])
    mod = fitted['CaII_c']['fit'](fitted['CaII_c']['wave'])
    axs['F'].plot(fitted['CaII_c']['wave'],mod,'--')
    axs['F'].set_xlabel(r'Wavelength ($\rm \AA$)',fontsize=12)
    val = _calc_val(fitted['CaII_c']['shift']*light)
    axs['F'].annotate(r'$v=$'+val+r'$\pm$'
                       + str(int(np.round(fitted['CaII_c']['error']*light,0)))+' km/s',
                      (.02,.04), xycoords='axes fraction',fontsize=14)
    axs['F'].set_ylim(np.min(fitted['CaII_c']['flux'])*.9,np.max(fitted['CaII_c']['flux'])+.05)

    plt.tight_layout()


from astropy.stats import SigmaClip
from photutils.background import Background2D, MedianBackground


def __bkg_calc(data,box_size,filter_size,sigma_clip,bkg_estimator,mask=None):
    if mask is None:
        bkg = Background2D(data, (box_size,box_size), filter_size=(filter_size,filter_size),
                           sigma_clip=sigma_clip, bkg_estimator=bkg_estimator,fill_value=0.0)
    else:
        bkg = Background2D(data, (box_size,box_size), filter_size=(filter_size,filter_size),
                           sigma_clip=sigma_clip, bkg_estimator=bkg_estimator,coverage_mask=mask,fill_value=0.0)
    background = bkg.background
    return background

def _calc_bkg(data,mask=None,box_size=6,filter_size=7,num_cores=-1):
    sigma_clip = SigmaClip(sigma=3.0)
    bkg_estimator = MedianBackground()
    background = np.zeros_like(data)
    if np.isfinite(data).any():
        if len(data.shape) < 3:
            background = __bkg_calc(data,box_size,filter_size,sigma_clip,bkg_estimator,mask)
        else:
            b = Parallel(n_jobs=num_cores,verbose=0)(delayed(__bkg_calc)(d,box_size,filter_size,sigma_clip,bkg_estimator,mask) for d in data)
            background = np.array(b)
    return background


def bin_spec(spec,bin_factor=100):
    wave = spec.wave
    flux = spec.flux
    if bin_factor > 1:
        l = int((wave[1] - wave[0]) * bin_factor)
        w = np.arange(wave[0],wave[-1],l)
        f = np.zeros_like(w)
        for i in range(len(w)-1):
            if i == len(w) - 1:
                ind = (wave >= w[i]) & (wave < l[-1])
            else:
                ind = (wave >= w[i]) & (wave < w[i+1])
            f[i] = np.nansum(flux[ind])
    else:
        w = wave
        f = flux
    s = np.array([w[:-1],f[:-1]])
    
    return s
        

from AstroColour.AstroColour import RGB

def make_false_colour(cube, savePath:str, saveName:str, scaleNorm:str="linear"):
    """
    Using AstroColour by zgl12 to false colour a cube. 
    Inputs:
    ------
        cube

        savePath:str, required
            The path to save the image to
        saveName:str, required
            The name of the image (without a file extension)
        scaleNorm: str, optional
            The Norm to scale the images by, default "Linear". Other options include "log", "sinh", "asinh"
    
    Outputs:
    --------
        colour: ndarray
            the scaled slices of the cube that can then be used for reploting
    """
    thirds = int(cube[0]/3)

    blueCube = cube[:thirds,:,:] 
    greenCube = cube[thirds:2*thirds,:,:] 
    redCube = cube[2*thirds:,:,:] 

    blueIm = np.nansum(blueCube, axis=0)
    greenIm = np.nansum(greenCube, axis=0)
    redIm = np.nansum(redCube, axis=0)

    imList = [redIm, greenIm,blueIm]

    rgb = RGB(
        imList,
        colours=['red', 'green', 'blue'],
        intensities=[0.55, 1, 0.6],
        uppers=[99, 99, 99],
        lowers=[60, 50, 50],
        save=True,
        save_name=saveName,
        save_folder=savePath,
        gamma=1.5,
        norm=scaleNorm,
        min_separation=29,
        star_size=5,
        epsf_plot=False,
        epsf=False,
        manual_override=20
    )
    colour = rgb.plot()
    return colour
