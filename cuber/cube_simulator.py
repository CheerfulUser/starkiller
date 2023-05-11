from skimage.util.shape import view_as_windows
from scipy import signal
import numpy as np 


def downSample2d(arr,sf):
    isf2 = 1.0/(sf*sf)
    (A,B) = arr.shape
    windows = view_as_windows(arr, (sf,sf), step = sf)
    return windows.sum(3).sum(2)*isf2

class cube_simulator():
    def __init__(self,cube,psf=None,catalogue=None,repFact=10,padding=20):
        self.xdim = cube.shape[2] + 2*padding
        self.ydim = cube.shape[1] + 2*padding
        self.sim = np.zeros_like(cube)
        self.psf = psf
        self.cat = catalogue
        self.repFact = repFact
        self.padding = padding

        self._make_super_sample()
        self._create_seeds()
    
    def _stack_median(self):
        return np.nanmedian(self.cube,axis=0)
    
    def _cat_fluxes(self,catalogue=None):
        if catalogue is not None:
            self.cat = catalogue
        counts = 10**(2/5*(self.cat['Gmag'].values + 48.6)) * 1e20 # cgs zp, and MUSE offset
        self.cat['counts'] = counts
    
    def _make_super_sample(self):
        self.X = np.arange((self.xdim)*self.repFact)/float(self.repFact)+0.5/self.repFact - (self.padding+0.5)
        self.Y = np.arange((self.ydim)*self.repFact)/float(self.repFact)+0.5/self.repFact - (self.padding+0.5)
        self.Sim = np.zeros((len(self.Y),len(self.X)))
        
    def _create_seeds(self):
        seeds = []
        Seeds = []
        x = self.cat.xint.values + self.cat.x_offset.values #+ self.padding
        y = self.cat.yint.values + self.cat.y_offset.values #+ self.padding
        for i in range(len(self.cat)):
            xind = np.argmin(abs(self.X - x[i]))
            yind = np.argmin(abs(self.Y - y[i]))
            s = np.zeros_like(self.Sim) 
            s[yind,xind] = 1
            s = signal.fftconvolve(s,self.psf.longPSF,mode='same')
            #Seeds += [s]
            s = downSample2d(s,self.repFact)
            s = s / np.nansum(s)
            seeds += [s]
        #self.Seeds = np.array(Seeds)
        self.seeds = np.array(seeds)
        
        # remove buffer
        #self.Seeds = self.Seeds[:,self.padding*self.repFact:-self.padding*self.repFact,self.padding*self.repFact:-self.padding*self.repFact]
        self.seeds = self.seeds[:,self.padding:-self.padding,self.padding:-self.padding]
        self.all_psfs = np.nansum(self.seeds,axis=0)
        
        
        
    def make_scene(self,flux):

        for i in range(len(flux)):
            seed = self.seeds[i]
            self.sim += seed[np.newaxis,:,:] * flux[i][:,np.newaxis,np.newaxis]

        
                              
                              
                              