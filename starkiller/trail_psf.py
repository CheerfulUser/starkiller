import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, least_squares
from scipy import signal
from skimage.util.shape import view_as_windows
from copy import deepcopy
from astropy.modeling.functional_models import Gaussian2D
from scipy.ndimage import shift
from scipy.interpolate import griddata

def downSample2d(arr,sf):
    """
    Downsample an array by a factor of sf.

    Parameters
    ----------
    arr : 2d numpy array
        Input array to be downsampled.
    sf : int
        Downsample factor.

    Returns
    -------
    arr_ds : 2d numpy array
        Downsampled array.
    """
    isf2 = 1.0/(sf*sf)
    (A,B) = arr.shape
    windows = view_as_windows(arr, (sf,sf), step = sf)
    return windows.sum(3).sum(2)*isf2

class create_psf():
    def __init__(self,x,y,angle,length,alpha=3.8,beta=4.765,stddev=2,
                 repFact=10,verbose=False,psf_profile='gaussian'):
        """
        Create a PSF object that can be used for both non-siderial and sidrial tracking.
        This class is built from the TRIPPY package.

        Parameters
        ----------
        x : int or 1d numpy array
            x dimension of the PSF
        y : int or 1d numpy array
            y dimension of the PSF
        angle : float
            Angle of motion of the object in degrees.
        length : float
            Length of the exposure in the same units as the rate.
        alpha : float
            Alpha parameter of the moffat profile.
        beta : float
            Beta parameter of the moffat profile.
        stddev : float
            Standard deviation of the gaussian profile.
        repFact : int
            Replication factor for the PSF. PSF will be downsampled by this factor.
        verbose : bool
            Verbose output.
        psf_profile : str
            Profile of the PSF. Options are 'gaussian' or 'moffat'.

        Returns
        -------
        psf : object
            PSF object.

        Examples
        --------
        >>> psf = create_psf(100,100,angle=10,length=10,repFact=10)
        >>> psf.fit_psf(image)
        >>> psf.psf_fig()
        """
        self.A=None
        self.alpha=alpha
        self.beta=beta
        self.stddev = stddev
        self.chi=None
        self.rate = None
        self.angle = deepcopy(angle)
        self.angle_o = deepcopy(angle)
        self.length = deepcopy(length)
        self.length_o = deepcopy(length)
        self.source_x = 0
        self.source_y = 0
        self._set_psf_profile(psf_profile)

        if type(x)!=type(np.ones(1)):
            self.x=np.arange(x)+0.5
            self.y=np.arange(y)+0.5
        elif len(x)==1:
            self.x=np.arange(x)+0.5
            self.y=np.arange(y)+0.5
        else:
            self.x=x*1.0+0.5
            self.y=y*1.0+0.5
        self.cent=np.array([len(self.y)/2.,len(self.x)/2.])
        self.centx=self.cent[1]
        self.centy=self.cent[0]
        self.repFact=repFact

        self.psf=np.ones([len(self.y),len(self.x)]).astype('float')

        self.inds=np.zeros((len(self.y),len(self.x),2)).astype('int')
        for ii in range(len(self.y)):
            self.inds[ii,:,1]=np.arange(len(self.x))
        for ii in range(len(self.x)):
            self.inds[:,ii,0]=np.arange(len(self.y))

        self.coords=self.inds+np.array([0.5,0.5])
        self.r=np.sqrt(np.sum((self.coords-self.cent)**2,axis=2))


        self.X=np.arange(len(self.x)*self.repFact)/float(self.repFact)+0.5/self.repFact
        self.Y=np.arange(len(self.y)*self.repFact)/float(self.repFact)+0.5/self.repFact
        self.XX, self.YY = np.meshgrid(self.X-self.centx,self.Y-self.centy)
        self.Inds=np.zeros((len(self.y)*self.repFact,len(self.x)*self.repFact,2)).astype('int')
        for ii in range(len(self.y)*self.repFact):
            self.Inds[ii,:,1]=np.arange(len(self.x)*self.repFact)
        for ii in range(len(self.x)*self.repFact):
            self.Inds[:,ii,0]=np.arange(len(self.y)*self.repFact)
        self.Coords=(self.Inds+np.array([0.5,0.5]))/float(self.repFact)

        self.R=np.sqrt(np.sum((self.Coords-self.cent)**2,axis=2))


        self.PSF=self.moffat(self.R)
        self.PSF/=np.sum(self.PSF)
        self.psf=downSample2d(self.PSF,self.repFact)

        self.fullPSF=None
        self.fullpsf=None


        self.shape=self.psf.shape

        self.aperCorrFunc=None
        self.aperCorrs=None
        self.aperCorrRadii=None
        self.lineAperCorrFunc=None
        self.lineAperCorrs=None
        self.lineAperCorrRadii=None

        self.verbose=verbose
        self.fitted=False

        self.lookupTable=None
        self.lookupF=None
        self.lookupR=None
        #self.rDist=None
        #self.fDist=None

        self.line2d=None
        #self.longPSF=None
        #self.longpsf=None

        self.bgNoise=None


        #from fitting a psf to a source
        self.model=None
        self.residual=None

        self.psfStars=None


    def _set_psf_profile(self,prof):
        """
        Set the PSF profile.

        Parameters
        ----------
        prof : str
            Profile of the PSF. Options are 'gaussian' or 'moffat'.

        """
        options = ['gaussian','moffat']
        if (options[0] in prof.lower()) | (options[1] in prof.lower()):
            self.psf_profile = prof
        else:
            m = f'{prof} is not an accepted option. Please choose from: {options}'
            raise ValueError(m)

    def moffat(self,rad):
        """
        Return a moffat profile evaluated at the radii in the input numpy array.

        Parameters
        ----------
        rad : 1d numpy array
            Radii at which to evaluate the moffat profile.

        Returns
        -------
        moffat : 1d numpy array
            Moffat profile evaluated at the input radii.
        """

        #normalized flux profile return 1.-(1.+(rad/self.alpha)**2)**(1.-self.beta)
        a2=self.alpha*self.alpha
        return (self.beta-1)/(np.pi*a2)*(1.+(rad/self.alpha)**2)**(-self.beta)

    def gauss2d(self,x,y):
        """
        Return a 2d gaussian profile evaluated at the input x and y coordinates.

        Parameters
        ----------
        x : 2d numpy array
            x coordinates at which to evaluate the gaussian profile.
        y : 2d numpy array
            y coordinates at which to evaluate the gaussian profile.

        Returns
        -------
        g : 2d numpy array
            2d gaussian profile evaluated at the input coordinates.
        """
        g2d = Gaussian2D(x_stddev=self.stddev,y_stddev=self.stddev)
        g = g2d(x,y)
        return g


    def generate_line_psf(self,angle=None,length=None,shiftx=0,shifty=0, verbose=False):
        """
        Generate a PSF for a given angle and length.

        Parameters
        ----------
        angle : float
            Angle of the object relative to the x axis
        length : float
            Length of trail.
        shiftx : float
            Shift in the x direction.
        shifty : float
            Shift in the y direction.
        verbose : bool
            Verbose output.
        """

        if angle is not None:
            self.angle=angle
        if length is not None:
            self.length=length
        angr = self.angle*np.pi/180.


        self.line2d=self.PSF*0.0
        centx_s = self.centx + shiftx
        centy_s = self.centy + shifty
        w=np.where(np.abs(self.X-centx_s) < np.cos(angr)*(self.length/2.))
        if len(w[0])>0:
            x=self.X[w]*1.0
            y=np.tan(angr)*(x-centx_s) + centy_s
            X=(x*self.repFact).astype('int')
            Y=(y*self.repFact).astype('int')
            ind = (X >= 0) & (X < self.line2d.shape[1]) & (Y >= 0) & (Y < self.line2d.shape[0])
            self.line2d[Y[ind],X[ind]]=1.0

            w=np.where(self.line2d>0)
            yl,yh=np.min(w[0]),np.max(w[0])
            xl,xh=np.min(w[1]),np.max(w[1])
        
            #self.line2d=self.line2d[yl:yh+1,xl:xh+1]

        else:
            self.line2d=np.array([[1.0]])

        #if line_width > 1:
        #   k = np.zeros((int(line_width),int(line_width)))
        #   self.line2d = signal.fftconvolve(k,self.line2d,mode='same')
        #   self.line2d[self.line2d > 0] = 1
        #self.longPSF=signal.convolve2d(self.moffProf,self.line2d,mode='same')
        if 'gaussian' in self.psf_profile:
            self.profile = self.gauss2d(self.XX,self.YY)
        elif 'moffat' in self.psf_profile:
            self.profile = self.moffat(self.R-np.min(self.R))

        self.longPSF = signal.fftconvolve(self.profile,self.line2d,mode='same')
        self.longPSF *= np.sum(self.profile)/np.sum(self.longPSF)
        self.longPSF /= np.nansum(self.longPSF)
        self.longpsf = downSample2d(self.longPSF,self.repFact)
        self.longpsf /= np.nansum(self.longpsf)

    def psf_fig(self):
        """
        Plot the PSF.
        """
        plt.figure()
        plt.imshow(self.longpsf,origin='lower')
        plt.colorbar()

    def minimizer(self,coeff,image):
        """
        Minimizer function for fitting a PSF to an image.

        Parameters
        ----------
        coeff : 1d numpy array
            Parameters for the minimizer.
        image : 2d numpy array
            Image to fit the PSF to.

        Returns
        -------
        residual : float
            Residual of the minimizer.
        """
        #print(coeff)
        if  'moffat' in self.psf_profile:
            self.alpha = coeff[0]
            self.beta = coeff[1]
            self.length = coeff[2]
            self.angle = coeff[3]
            self.source_x = coeff[4]
            self.source_y = coeff[5]
            #floor = coeff[6]
        elif 'gaussian' in self.psf_profile:
            self.stddev = coeff[0]
            self.length = coeff[1]
            self.angle = coeff[2]
            self.source_x = coeff[3]
            self.source_y = coeff[4]
            #floor = coeff[5]
        self.generate_line_psf(shiftx = self.source_x, shifty = self.source_y)
        psf = self.longpsf / np.nansum(self.longpsf) # type: ignore
        
        diff = abs(image - psf)
        residual = np.nansum(diff)
        #self.residual = residual
        return np.exp(residual)


    def fit_psf(self,image,limx=10,limy=10):
        """
        Fit a PSF profile to the input image.

        Parameters
        ----------
        image : numpy array
            image of target to fit the PSF to 
        limx : float
            limit for fitting in the x shift 
        limy : float
            limit for fitting in the x shift 
        """
        

        image -= np.nanmedian(image)
        normimage = image / np.nansum(image)
        anglebs = [self.angle_o*0.6,self.angle_o*1.4]

        if self.psf_profile == 'moffat':
            coeff = [self.alpha,self.beta,self.length,self.angle,0,0]
            lims = [[0.1,100],[1,100],[self.length_o*0.6,self.length_o*1.4],
                    [np.min(anglebs),np.max(anglebs)],[-limx,limx],[-limy,limy]]
        elif self.psf_profile == 'gaussian':
            coeff = [self.stddev,self.length,self.angle,0,0]
            lims = [[0.1,20],[self.length_o*0.6,self.length_o*1.4],
                    [np.min(anglebs),np.max(anglebs)],[-limx,limx],[-limy,limy]]
        else:
            m = 'Incorrect psf_profile, please select from moffat or gaussian.'
            raise ValueError(m)
        #lims = [[-100,100],[-100,100],[5,20],[-80,80],[-limx,limx],[-limy,limy]]
        #res = least_squares(self.minimizer,coeff,args=normimage,method='trf',x_scale=0.1)
        res = minimize(self.minimizer, coeff, args=normimage, method='Powell',bounds=lims)
                        #jac = '2-point',options={'finite_diff_rel_step':0.1})
        self.psf_fit = res

    def make_data_psf(self,data_cuts):
        """
        Create a PSF using the calibration stars. This does a better job for longer exposures with scintilation.

        Parameters
        ----------
        data_cuts : numpy array 
            cuts of the image that contain the target stars PSF
        """
        psf_mask = (self.longpsf > 1e-6) * 1
        data_cuts *= psf_mask[np.newaxis,:,:]
        data_cuts[np.isnan(data_cuts)] = 0.0
        medcuts = data_cuts / np.nanmax(data_cuts,axis=(1,2))[:,np.newaxis,np.newaxis]
        sm = np.zeros_like(medcuts)
        for i in range(len(medcuts)):
            self.fit_pos(medcuts[i])
            sm[i] = shift(medcuts[i],[-self.source_y,-self.source_x],mode='nearest')
        data_psf = np.nanmedian(sm,axis=0)    
        self.data_psf = data_psf / np.nansum(data_psf)
        self.data_psf[self.data_psf < 1e-6] = 0

        x = np.arange(0,self.data_psf.shape[1])
        y = np.arange(0,self.data_psf.shape[0])
        xx, yy = np.meshgrid(x, y)
        XX, YY = np.meshgrid(self.X-0.5,self.Y-0.5)

        estimate = griddata((xx.ravel(), yy.ravel()), self.data_psf.ravel(),
                            (XX.ravel(),YY.ravel()),method='linear',fill_value=0)

        estimate = estimate.reshape(len(self.Y),len(self.X))
        estimate[estimate < 1e-5] = 0
        self.data_PSF = estimate
        self.source_x = 0; self.source_y = 0
        self.generate_line_psf()




    def minimize_pos(self,coeff,image):
        """
        Minimizer function for fitting the position of the source to an image.

        Parameters
        ----------
        coeff : 1d numpy array
            Parameters for the minimizer.
        image : 2d numpy array
            Image to fit the position of the source to.

        Returns
        -------
        residual : float
            Residual of the minimizer.
        """
        self.generate_line_psf(shiftx = coeff[0], shifty = coeff[1])
        psf = self.longpsf / np.nansum(self.longpsf)
        diff = abs(image - psf)
        residual = np.nansum(diff)
        return np.exp(residual)
        
    def fit_pos(self,image,range=5):
        """
        Fit the position of the source to the input image.

        Parameters
        ----------
        image : 2d numpy array
            Image to fit the position of the source to.
        range : float
            Range to fit the position of the source over.

        Returns
        -------
        source_x : float
            Source x position.
        source_y : float
            Source y position.
        
        """
        normimage = image / np.nansum(image)
        coeff = [0,0]
        lims = [[-range,range],[-range,range]]
        res = minimize(self.minimize_pos,coeff, args=normimage, method='Powell',bounds=lims)
        self.source_x = res.x[0]
        self.source_y = res.x[1]

    
        
    def minimize_psf_flux(self,coeff,image):
        """
        Minimizer function for fitting the flux of the PSF to an image.

        Parameters
        ----------
        coeff : 1d numpy array
            Parameters for the minimizer.
        image : 2d numpy array
            Image to fit the PSF to.

        Returns
        -------
        residual : float
            Residual of the minimizer.
        """ 
        res = np.nansum(abs(image - self.longpsf*coeff[0]))
        return res

    def psf_flux(self,image,output = True):
        """
        Fit the flux of the PSF to the input image.

        Parameters
        ----------
        image : 2d numpy array
            Image to fit the PSF to.

        Returns
        -------
        flux : float
            Flux of the PSF.
        image_residual : 2d numpy array
            Residual image after subtracting the PSF from the input image.
        """
        if self.longpsf is None:
            self.generate_line_psf()
        mask = np.zeros_like(self.longpsf)
        mask[self.longpsf > np.nanpercentile(self.longpsf,70)] = 1 # type: ignore
        f0 = np.nansum(image*mask)
        bkg = np.nanmedian(image[~mask.astype(bool)])
        image = image - bkg
        if 'data' in self.psf_profile:
            if self.data_psf is not None:
                self.longpsf = shift(self.data_psf,[self.source_y,self.source_x],mode='nearest')
            else:
                raise ValueError('No data psf defined! Use the make_data_psf function first.')
        else:
            self.generate_line_psf(shiftx=self.source_x,shifty=self.source_y)
        res = minimize(self.minimize_psf_flux,f0,args=(image),method='Nelder-Mead')
        self.flux = res.x[0]
        self.image_residual = image - self.longpsf*self.flux
        #if output:
        return self.flux, self.image_residual



