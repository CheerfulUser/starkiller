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
    Simulates a spectral datacube from an inout catalog and PSF. The cube is made at higher resolution then downsampled.
    
    Atributes
    ---------
    xdim : int
        Dimension of the output cube in the x direction
    ydim : int
        Dimension of the output cube in the y direction
    sim : np.array
        The simulated data cube
    weights : np.array
        The simulated weighted image
    psf : psf class
        The PSF class containing the PSF fit to the data cube
    cat : pd.dataframe
        The catalog containing the positions of all sources to simulate
    repFact : int
        Replication factor controls the supersampling and downsampling
    padding : int
        Number of pixels to pad the simulated cube
    X : np.array
        The X axis of the simulated cube
    Y : np.array
        The Y axis of the simulated cube
    Sim : np.array
        The simulated cube at higher resolution then downsampled
    seeds : np.array
        The PSF seed images used to create the simulated cube
    all_psfs : np.array
        The sum of all the PSF seed images
    image : np.array
        Median stack of the input data cube



    """
    def __init__(self,cube,psf=None,catalog=None,repFact=10,padding=20,datapsf=False,satellite=None):
        """
        Parameters
        ----------
        cube : array
            Datacube to simulate, this is used for dimensions only.
        psf : psf class
            PSF class containing the PSF fit to the data cube. This is used to insert objects into the simulated cube.
        catalog : pd.dataframe
            Dataframe containing the catalog positions of all sources to simulate.
        repFact : int
            Replication factor controls the supersampling and downsampling.
        padding : int
            Value to pad the sides of the cube by to capture sources centered outside the original image bounds.
        datapsf : bool
            Option to force the selection of the datapsf from the psf class.
        satellite : satkiller
            Class information for modelling the satellite streaks. 


        """
        self.xdim = cube.shape[2] + 2*padding
        self.ydim = cube.shape[1] + 2*padding
        self.sim = np.zeros_like(cube)
        self.weights = np.zeros_like(cube)
        self.psf = psf
        self.cat = catalog
        self.repFact = repFact
        self.padding = padding
        self.satellite = satellite

        if datapsf:
            self.psf.longPSF = self.psf.data_PSF
            print('Using the data PSF')

        self._make_super_sample()
        self._create_seeds()
        if self.satellite is not None:
            self._create_satellite_seeds()
    
    def _stack_median(self):
        """
        Creates a median of the input data cube
        """
        return np.nanmedian(self.cube,axis=0)
    
    def _cat_fluxes(self,catalog=None):
        """
        Calculates the fluxes of the catalog sources, assuming a zeropoint of 25.
        """
        if catalog is not None:
            self.cat = catalog
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

    def _create_satellite_seeds(self):
        seeds = []
        x = self.satellite.satcat.x.values
        y = self.satellite.satcat.y.values
        for i in range(len(x)):
            xind = np.argmin(abs(self.X - x[i]))
            yind = np.argmin(abs(self.Y - y[i]))
            s = np.zeros_like(self.Sim) 
            s[yind,xind] = 1
            psf = self.satellite.sat_psfs[i]
            s = signal.fftconvolve(s,psf.longPSF,mode='same')
            s = downSample2d(s,self.repFact)
            s = s / np.nansum(s)
            seeds += [s]
        #self.Seeds = np.array(Seeds)
        self.satellite_seeds = np.array(seeds)
        
        # remove buffer
        self.satellite_seeds = self.satellite_seeds[:,self.padding:-self.padding,self.padding:-self.padding]

        
    def mag_image(self):
        """
        Creates an image scaled by the catalog flux values
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
        if self.satellite is not None:
            for i in range(len(self.satellite_seeds)):
                seed = self.satellite_seeds[i]
                self.sim += seed[np.newaxis,:,:] * self.satellite.sat_fluxes[i][:,np.newaxis,np.newaxis]

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

        
                              
                              
                              