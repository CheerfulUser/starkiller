import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd 
from starkiller import starkiller
from glob import glob
from starkiller import spec_match, load_pbs, my_norm,downsample_spec
import pysynphot as S
import astropy.table as at
from extinction import fitzpatrick99, apply
from scipy.stats import pearsonr
from tqdm import trange, tqdm
from joblib import Parallel, delayed

def _has_len(obj):
    return hasattr(obj, '__len__')

def _parallel_model_grid(model,e,Rv=3.1):
    ext = S.ArraySpectrum(model.wave, 
                        apply(fitzpatrick99(model.wave.astype('double'),e*Rv,Rv),
                              model.flux),name=model.name + ' ebv=' + str(e))
    ext = S.ArraySpectrum(wave=ext.wave,
                        flux=ext.flux/np.nanmedian(ext.flux),fluxunits='flam',name=ext.name)
    return ext


def _model_grid(model_file,target_lam=None,max_ext=4,num_cores=10):
    extinctions = np.arange(0,max_ext,0.01)
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
        for i in trange(len(extinctions)):
            exts += [_parallel_model_grid(model,extinctions[i])]
    else:
        exts = Parallel(n_jobs=num_cores)(delayed(_parallel_model_grid)(model,extinctions[i]) for i in range(len(extinctions)))
    return exts

def _parallel_cor(spec,model_fluxes,num_cores=10):
    #flux = savgol_filter(flux,101,1)
    #coeff = Parallel(n_jobs=num_cores)(delayed(pearsonr)(spec.flux,model_fluxes[i]) for i in trange(len(model_fluxes),desc='Correlation'))
    coeff = Parallel(n_jobs=num_cores)(delayed(pearsonr)(spec.flux,model_fluxes[i]) for i in range(len(model_fluxes)))
    coeff = np.array(coeff)
    coeff[coeff<0] = 0
    coeff[~np.isfinite(coeff)] = 0
    return coeff[:,0]

def _parallel_refactoring(model):
    flux = model.flux
    ext = float(model.name.split('=')[-1])
    return flux, ext

def _match_obs_to_model(obs_spec,model_files,mags,pbs,num_cores=10):
    if not _has_len(obs_spec):
        obs_spec = [obs_spec]
    if not _has_len(mags):
        mags = [mags]    
    #for i in range(len(obs_spec)):
    #    flux = obs_spec[i].flux
    #    flux = savgol_filter(flux,101,1)
    #    obs_spec[i] = S.ArraySpectrum(wave=obs_spec[i].wave,#lam_vac2air(model['wave'].value),
    #                                  flux=flux,fluxunits='flam',name=obs_spec[i].name)

    model_grid = []
    for i in trange(len(model_files),desc='Model grid'):
        model_grid += [_model_grid(model_files[i],target_lam=obs_spec[0].wave)]
    model_grid = np.array(model_grid).flatten()
    model_flux, model_ebv = zip(*Parallel(n_jobs=num_cores)(delayed(_parallel_refactoring)(model_grid[i]) for i in trange(len(model_grid),desc='Refactor')))
    model_spec = []
    cor = []
    ebv = []
    for i in range(len(obs_spec)):
        spec = obs_spec[i]
        cors = _parallel_cor(spec,model_flux)
        ind = np.argmax(cors)
        cor += [cors[ind]]
        model = my_norm(model_grid[ind],pbs,mags[i],name=model_grid[ind].name)
        model_spec += [model]
        ebv += [model_ebv[ind]]
    return model_spec, cor, ebv


ck_files = glob('../starkiller/data/ck04_stsci/*')
svo_bp = 'GAIA/GAIA0.G'
pbs = load_pbs(svo_bp,0,'AB',SVO=True)
testfile = glob('../starkiller/data/marcs-t02/*')

test_ids = np.arange(len(ck_files))[::4]
Rv = 3.1
test_ebvs = []
models = []
for i in test_ids:
    model = at.Table.read(ck_files[i], format='ascii')
    name = ck_files[i].split('/')[-1].split('.dat')[0]

    if '_a' in name:
            name = name.split('_a')[0]
    model = S.ArraySpectrum(wave=model['wave'].value,#lam_vac2air(model['wave'].value),
                            flux=model['flux'].value,fluxunits='flam',name=name)
    ebvs = np.random.rand(20) * 4
    test_ebvs += [ebvs]
    for e in ebvs:
        test = S.ArraySpectrum(model.wave, 
                                apply(fitzpatrick99(model.wave.astype('double'),e*Rv,Rv),
                                      model.flux),name=model.name + ' ebv=' + str(np.round(e,3)))
        test = my_norm(test,pbs,10,name=test.name)
        test = S.ArraySpectrum(wave=test.wave,#lam_vac2air(model['wave'].value),
                                    flux=test.flux/np.nanmedian(test.flux),fluxunits='flam',name=test.name)#my_norm(test,pbs,10,name=test.name)
        test = S.ArraySpectrum(wave=test.wave,#lam_vac2air(model['wave'].value),
                                    flux=test.flux + np.random.rand(len(test.flux))*0.1,fluxunits='flam',name=test.name)#my_norm(test,pbs,10,name=test.name)

        models += [test]

print('Made the models')

spec, cor, e = _match_obs_to_model(models,ck_files,np.ones_like(ebvs)*10,pbs)

mod_names = []
spec_names = []
for i in range(len(spec)):
    spec_names += [spec[i].name]
    mod_names += [models[i].name]
spec_names = np.array(spec_names)
mod_names = np.array(mod_names)

result = pd.DataFrame(columns=['Model','Matched','model_ebv','matched_ebv'],data=np.array([mod_names,spec_names,test_ebvs,e]).T)

result.to_csv('extinction_recovery_test.csv',index=False)
