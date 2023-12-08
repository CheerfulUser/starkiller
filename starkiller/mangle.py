from astroquery.svo_fps import SvoFps
import numpy as np
import pysynphot as S
#from thief.mangle_spectrum import mangle_spectrum2
import astropy.table as at
from collections import OrderedDict
import os
from copy import deepcopy
import warnings
package_dir = os.path.dirname(os.path.abspath(__file__)) + '/'


def spec_mangle(spec,mags,pbs,name):
	"""
	Mangle model spectra to input magnitudes and filters. Filters must be given as their SVO designation.
	"""
	#pbs = load_pbs(svo_bp,0,'AB',SVO=True)
	mangled = []
	#for i in range(len(specs)):
	flux = mangle_spectrum2(spec.wave,spec.flux,pbs,mags)
	mangled = S.ArraySpectrum(wave=spec['wave'],
                                   flux=flux,fluxunits='flam',name=name)

	return mangled


def get_pb_zpt(pb, reference='AB', model_mag=0):
	"""
	Determines a passband zeropoint for synthetic photometry, given a reference
	standard and its model magnitude in the passband

	Parameters
	----------
	pb : :py:class:`pysynphot.ArrayBandpass` or :py:class:`pysynphot.obsbandpass.ObsModeBandpass`
		The passband data.
		Must have ``dtype=[('wave', '<f8'), ('throughput', '<f8')]``
	reference : str, optional
		The name of the reference spectrophotometric standard to use to determine the passband zeropoint.
		'AB' or 'Vega' (default: 'AB')
	model_mag : float, optional
		The magnitude of the reference spectrophotometric standard in the passband.
		default = None

	Returns
	-------
	pbzp : float
		passband AB zeropoint

	Raises
	------
	RuntimeError
		If the passband and reference standard do not overlap
	ValueError
		If the reference model_mag is invalid

	See Also
	--------
	:py:func:`source_synphot.passband.synflux`
	:py:func:`source_synphot.passband.synphot`
	"""

	# setup the photometric system by defining the standard and corresponding magnitude system
	if reference.lower() not in ('vega', 'ab'):
		message = 'Do not understand mag system reference standard {}. Must be AB or Vega'.format(reference)
		raise RuntimeError(message)
	if reference.lower() == 'ab':
		refspec = S.FlatSpectrum(3631, fluxunits='jy')
		mag_type= 'abmag'
	else:
		refspec	= S.Vega
		mag_type= 'vegamag'

	refspec.convert('flam')

	if model_mag is None:
		ob = S.Observation(refspec, pb)
		model_mag = ob.effstim(mag_type)

	synphot_mag = synphot(refspec, pb, zp=0.)
	#print(synphot_mag)
	#print(model_mag)
	outzp = model_mag - synphot_mag
	return outzp

def synflux(spec, pb):
	"""
	Compute the synthetic flux of spectrum ``spec`` through passband ``pb``

	Parameters
	----------
	spec : :py:class:`pysynphot.ArraySpectrum`
		The spectrum. Must have ``dtype=[('wave', '<f8'), ('flux', '<f8')]``
	pb : :py:class:`pysynphot.ArrayBandpass` or :py:class:`pysynphot.obsbandpass.ObsModeBandpass`
		The passband data.
		Must have ``dtype=[('wave', '<f8'), ('throughput', '<f8')]``

	Returns
	-------
	flux : float
		The normalized flux of the spectrum through the passband

	Notes
	-----
		The passband is assumed to be dimensionless photon transmission
		efficiency.

		Routine is intended to be a mch faster implementation of
		:py:meth:`pysynphot.observation.Observation.effstim`, since it is called over and
		over by the samplers as a function of model parameters.

		Uses :py:func:`numpy.trapz` for interpolation.
	"""
	overlap = pb.check_overlap(spec)
	#if overlap == 'none':
	#	return np.nan
	#elif overlap == 'partial':
	#	pass
	#	if pb.check_sig(spec):
	#		pass
	#	else:
	#		return np.nan
	#else:
	#	pass
	flux = spec.sample(pb.wave)
	n = np.trapz(flux*pb.wave*pb.throughput, pb.wave)
	d = np.trapz(pb.wave*pb.throughput, pb.wave)
	#print(n,d)
	out = n/d
	return out


def synphot(spec, pb, zp=0.):
	"""
	Compute the synthetic magnitude of spectrum ``spec`` through passband ``pb``

	Parameters
	----------
	spec : :py:class:`pysynphot.ArraySpectrum`
		The spectrum. Must have ``dtype=[('wave', '<f8'), ('flux', '<f8')]``
	pb : :py:class:`pysynphot.ArrayBandpass` or :py:class:`pysynphot.obsbandpass.ObsModeBandpass`
		The passband transmission.
	zp : float, optional
		The zeropoint to apply to the synthetic flux

	Returns
	-------
	mag : float
		The synthetic magnitude of the spectrum through the passband

	See Also
	--------
	:py:func:`source_synphot.passband.synflux`
	"""
	flux = synflux(spec, pb)
	m = -2.5*np.log10(flux) + zp
	return m

def load_pbs(pbnames, model_mags, model='AB',SVO = False):
	"""
	Loads passbands, and calibrates their zeropoints so that ``model`` has
	magnitude ``model_mags`` through them.

	Parameters
	----------
	pbnames : array-like
		The passband names. Passed to :py:func:`source_synphot.io.read_passband`
	model_mags : array-like
		The magnitudes of ``model`` in the passbands ``pbnames``
	model : str, optional
		The reference model for the passband. Either ``'AB'`` or ``'Vega'``.
		The same reference model is used for all the passbands. If the
		passbands have different standards they are calibrated to, then call
		the function twice, and concatenate the output.

	Returns
	-------
	pbs : dict
		The dictionary of passband transmissions and zeropoints, such that
		:py:func:`synphot` of ``model`` through passband ``pbnames`` returns
		magnitude ``model_mags``.

	Raises
	------
	ValueError
		If the number of ``model_mags`` does not match the number of passbands
		in ``pbnames``

	Notes
	-----
		The passband zeropoint computed here is what number must be used with
		:py:func:`synphot` and ``model``. It is not what observers think of as
		the zeropoint, which invariable encapsulates the telescope collecting
		area.

	See Also
	--------
	:py:func:`source_synphot.io.read_passband`
	:py:func:`source_synphot.passband.get_pb_zpt`
	"""

	pbs = OrderedDict()
	if np.isscalar(pbnames):
		pbnames = np.array(pbnames, ndmin=1)
	else:
		pbnames = np.array(pbnames).flatten()
	npb = len(pbnames)

	if np.isscalar(model_mags):
		model_mags = np.repeat(model_mags, npb)
	else:
		model_mags = np.array(model_mags).flatten()

	if len(model_mags) != npb:
		message = 'Model mags and pbnames must be 1-D arrays with the same shape'
		raise ValueError(message)

	for i, pbmag in enumerate(zip(pbnames, model_mags)):
		pb, mag = pbmag
		if SVO:
			filt = SvoFps.get_transmission_data(pb)  
			thispb = S.ArrayBandpass(filt['Wavelength'].data.data, filt['Transmission'].data.data, waveunits='Angstrom', name=pb)
		else:
			try:
				#thispb, _ = io.read_passband(pb)
				print('No thanks')
			except Exception as e:
				print(e)
				message = 'Passband {} not loaded'.format(pb)
				#warnings.warn(message, RuntimeWarning)
				continue
			
		thispbzp = get_pb_zpt(thispb, model_mag=mag, reference=model)
		pbs[pb] = thispb, thispbzp

	return pbs

def Syn_mag(pbs,spec):
    mag = {}
    for pb in pbs:
        if spec is not None:
            syn_mag = synphot(spec,pbs[pb][0],zp=pbs[pb][1])
        else:
            syn_mag = np.nan
        mag[pb] = syn_mag
        
    return mag

def my_norm(Spec,pbs,mag,name=None):
    spec = deepcopy(Spec)
    m = Syn_mag(pbs,spec)
    m=m[list(pbs.keys())[0]]
    factor = 10**(-2/5*(mag-m))
    spec = S.ArraySpectrum(wave=spec.wave,waveunits=spec.waveunits,
                                   flux=spec.flux*factor,fluxunits=spec.fluxunits,name=name)
    return spec
    

