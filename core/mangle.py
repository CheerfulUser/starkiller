from astroquery.svo_fps import SvoFps
import numpy as np
import pysynphot as S

bp = SvoFps.get_transmission_data('GAIA/GAIA3.Gbp')
G = SvoFps.get_transmission_data('GAIA/GAIA3.G')
rp = SvoFps.get_transmission_data('GAIA/GAIA3.Grp')

mags = [d['gMeanPSFMag'].iloc[i],d['rMeanPSFMag'].iloc[i],d['iMeanPSFMag'].iloc[i]]
flux = mangle_spectrum2(spec.wave,spec.flux,pb_ps1,mags)


def gaia_mangle(spec_files,mags,svo_bp=['GAIA/GAIA3.Gbp','GAIA/GAIA3.G','GAIA/GAIA3.Grp']):
	
	pbs = load_pbs(svo_filters,0,'AB',SVO=True)


model_mags = 0.
magmodel = 'AB'
pbs = source_synphot.passband.load_pbs(pbnames, model_mags, magmodel)
pb_ps1 = source_synphot.passband.load_pbs(pb_ps1, model_mags, magmodel)


def get_pb_zpt(pb, reference='AB', model_mag=None):
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
			print('SVO')
			thispb = SvoFps.get_transmission_data(pb)  
		else:
			try:
				thispb, _ = io.read_passband(pb)
			except Exception as e:
				print(e)
				message = 'Passband {} not loaded'.format(pb)
				warnings.warn(message, RuntimeWarning)
				continue
			
		thispbzp = get_pb_zpt(thispb, model_mag=mag, reference=model)
		pbs[pb] = thispb, thispbzp

	return pbs