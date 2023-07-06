import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, least_squares
from scipy import signal
from skimage.util.shape import view_as_windows
from copy import deepcopy
from astropy.modeling.functional_models import Gaussian2D

def downSample2d(arr,sf):
    isf2 = 1.0/(sf*sf)
    (A,B) = arr.shape
    windows = view_as_windows(arr, (sf,sf), step = sf)
    return windows.sum(3).sum(2)*isf2

class create_psf():
    def __init__(self,x,y,angle,length,alpha=3.8,beta=4.765,stddev=2,repFact=10,verbose=False,
                 psf_profile='gaussian'):
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
        self.longPSF=None
        self.longpsf=None

        self.bgNoise=None


        #from fitting a psf to a source
        self.model=None
        self.residual=None

        self.psfStars=None


    def _set_psf_profile(self,prof):
        options = ['gaussian','moffat']
        if prof.lower() in options:
            self.psf_profile = prof
        else:
            m = f'{prof} is not an accepted option. Please choose from: {options}'
            raise ValueError(m)

    def moffat(self,rad):
        """
        Return a moffat profile evaluated at the radii in the input numpy array.
        """

        #normalized flux profile return 1.-(1.+(rad/self.alpha)**2)**(1.-self.beta)
        a2=self.alpha*self.alpha
        return (self.beta-1)/(np.pi*a2)*(1.+(rad/self.alpha)**2)**(-self.beta)

    def gauss2d(self,x,y):
        g2d = Gaussian2D(x_stddev=self.stddev,y_stddev=self.stddev)
        g = g2d(x,y)
        return g


    def line(self,angle=None,length=None,shiftx=0,shifty=0, verbose=False):
        """
        Compute the TSF given input rate of motion, angle of motion, length of exposure, and pixelScale.

        Units choice is irrelevant, as long as they are all the same! eg. rate in "/hr, and dt in hr.
        Angle is in degrees +-90 from horizontal.

        display=True to see the TSF

        useLookupTable=True to use the lookupTable. OTherwise pure moffat is used.
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
        if self.psf_profile == 'gaussian':
            self.profile = self.gauss2d(self.XX,self.YY)
        elif self.psf_profile == 'modffat':
            self.profile = self.moffat(self.R-np.min(self.R))

        self.longPSF = signal.fftconvolve(self.profile,self.line2d,mode='same')
        self.longPSF *= np.sum(self.profile)/np.sum(self.longPSF)
        self.longPSF /= np.nansum(self.longPSF)
        self.longpsf = downSample2d(self.longPSF,self.repFact)
        self.longpsf /= np.nansum(self.longpsf)

    def psf_fig(self):
        plt.figure()
        plt.imshow(self.longpsf,origin='lower')
        plt.colorbar()

    def minimizer(self,coeff,image):
        #print(coeff)
        if self.psf_profile == 'moffat':
            self.alpha = coeff[0]
            self.beta = coeff[1]
            self.length = coeff[2]
            self.angle = coeff[3]
            self.source_x = coeff[4]
            self.source_y = coeff[5]
        elif self.psf_profile == 'gaussian':
            self.stddev = coeff[0]
            self.length = coeff[1]
            self.angle = coeff[2]
            self.source_x = coeff[3]
            self.source_y = coeff[4]

        self.line(shiftx = self.source_x, shifty = self.source_y)
        psf = self.longpsf / np.nansum(self.longpsf)

        diff = abs(image - psf)
        residual = np.nansum(diff)
        #self.residual = residual
        return np.exp(residual)

 

    def fit_psf(self,image,limx=10,limy=10):
        image -= np.nanmedian(image)
        normimage = image / np.nansum(image)
        anglebs = [self.angle_o*0.6,self.angle_o*1.4]

        if self.psf_profile == 'mofatt':
            coeff = [self.alpha,self.beta,self.length,self.angle,0,0]
            lims = [[0.1,100],[1,100],[self.length_o*0.6,self.length_o*1.4],
                    [np.min(anglebs),np.max(anglebs)],[-limx,limx],[-limy,limy]]
        elif self.psf_profile == 'gaussian':
            coeff = [self.stddev,self.length,self.angle,0,0]
            lims = [[1,20],[self.length_o*0.6,self.length_o*1.4],
                    [np.min(anglebs),np.max(anglebs)],[-limx,limx],[-limy,limy]]
        #lims = [[-100,100],[-100,100],[5,20],[-80,80],[-limx,limx],[-limy,limy]]
        #res = least_squares(self.minimizer,coeff,args=normimage,method='trf',x_scale=0.1)
        res = minimize(self.minimizer, coeff, args=normimage, method='Powell',bounds=lims)
                        #jac = '2-point',options={'finite_diff_rel_step':0.1})
        self.psf_fit = res

    def minimize_pos(self,coeff,image):
        self.line(shiftx = coeff[0], shifty = coeff[1])
        psf = self.longpsf / np.nansum(self.longpsf)
        diff = abs(image - psf)
        residual = np.nansum(diff)
        return np.exp(residual)
        
    def fit_pos(self,image):
        normimage = image / np.nansum(image)
        coeff = [0,0]
        lims = [[-2,2],[-2,2]]
        res = minimize(self.minimize_pos,coeff, args=normimage, method='Powell',bounds=lims)
        self.source_x = res.x[0]
        self.source_y = res.x[1]

        
        
        
    def minimize_psf_flux(self,coeff,image):
        res = np.nansum(abs(image - self.longpsf*coeff[0]))
        return res

    def psf_flux(self,image,output = True):
        if self.longpsf is None:
            self.line()
        mask = np.zeros_like(self.longpsf)
        mask[self.longpsf > np.nanpercentile(self.longpsf,70)] = 1
        f0 = np.nansum(image*mask)
        bkg = np.nanmedian(image[~mask.astype(bool)])
        image = image - bkg
        self.line(shiftx=self.source_x,shifty=self.source_y)
        res = minimize(self.minimize_psf_flux,f0,args=(image),method='Nelder-Mead')
        self.flux = res.x[0]
        self.image_residual = image - self.longpsf*self.flux
        #if output:
        return self.flux, self.image_residual



