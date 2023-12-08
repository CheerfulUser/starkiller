from skimage.util.shape import view_as_windows
from scipy import signal
import numpy as np 
import pandas as pd


def downSample2d(arr,sf):
    #isf2 = 1.0/(sf*sf) # Removed this factor since we don't want scaling
    (A,B) = arr.shape
    windows = view_as_windows(arr, (sf,sf), step = sf)
    return windows.sum(3).sum(2)#*isf2

class cube_simulator():
    """
    Simulates a spectral datacube from an inout catalogue and PSF. The cube is made at higher resolution then downsampled.
    
    Atributes
    ---------


    """
    def __init__(self,cube,psf=None,catalogue=None,repFact=10,padding=20,datapsf=False):
        """
        Parameters
        ----------
        cube : array
            Datacube to simulate, this is used for dimensions only.
        psf : psf class
            PSF class containing the PSF fit to the data cube. This is used to insert objects into the simulated cube.
        catalogue : pd.dataframe
            Dataframe containing the catalogue positions of all sources to simulate.
        repFact : int
            Replication factor controls the supersampling and downsampling.
        datapsf : bool
            Option to force the selection of the datapsf from the psf class.

        """
        self.xdim = cube.shape[2] + 2*padding
        self.ydim = cube.shape[1] + 2*padding
        self.sim = np.zeros_like(cube)
        self.weights = np.zeros_like(cube)
        self.psf = psf
        self.cat = catalogue
        self.repFact = repFact
        self.padding = padding

        if datapsf:
            self.psf.longPSF = self.psf.data_PSF
            print('Using the data PSF')

        self._make_super_sample()
        self._create_seeds()
    
    def _stack_median(self):
        """
        Creates a median of the inout data cube
        """
        return np.nanmedian(self.cube,axis=0)
    
    def _cat_fluxes(self,catalogue=None):
        """
        Calculates the fluxes of the catalogue sources, assuming a zeropoint of 25.
        """
        if catalogue is not None:
            self.cat = catalogue
        counts = 10**(2/5*(self.cat['Gmag'].values + 25)) #* 1e20 # cgs zp, and MUSE offset
        self.cat['counts'] = counts
    
    def _make_super_sample(self):
        """
        Create the super sampled X, Y arrays and Sim image, determined by the repFact.
        """
        self.X = np.arange((self.xdim)*self.repFact)/float(self.repFact)+0.5/self.repFact - (self.padding+0.5)
        self.Y = np.arange((self.ydim)*self.repFact)/float(self.repFact)+0.5/self.repFact - (self.padding+0.5)
        self.Sim = np.zeros((len(self.Y),len(self.X)))
        
    def _create_seeds(self):
        """
        Create the "PSF seed" images from which the cube will be constructed.
        """
        seeds = []
        Seeds = []
        x = self.cat.x.values #+ self.padding
        y = self.cat.y.values #+ self.padding
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
        
    def mag_image(self):
        """
        Creates an image scaled by the catalogue flux values
        """
        self._cat_fluxes()
        image = np.zeros_like(self.seeds[0])
        for i in range(len(self.seeds)):
            image += self.seeds[i] * self.cat['counts'].iloc[i]
        self.image = image
    
    def make_scene(self,flux):
        """
        Creates the scene which consists of the simulated sources in the data cube.

        Parameters
        ----------
        flux : np.array
            Array of fluxes for each spectral dimension in the data cube.
        """
        for i in range(len(flux)):
            seed = self.seeds[i]
            self.sim += seed[np.newaxis,:,:] * flux[i][:,np.newaxis,np.newaxis]

    def make_weights(self,weights):
        """
        Create a weighted image based on the seeds. 

        Parameters
        ----------
        weights : np.array
            Array containing the weights of the sources to created a weighted image using the PSF.
        """
        for i in range(len(weights)):
            seed = self.seeds[i]
            seed = (seed > np.percentile(seed,10)) * 1.0
            self.weights += seed[np.newaxis,:,:] * weights[i][:,np.newaxis,np.newaxis]

        
                              
                              
                              