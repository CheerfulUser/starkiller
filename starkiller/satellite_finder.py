import numpy as np
import cv2
from copy import deepcopy
from astropy.stats import sigma_clipped_stats
from .trail_psf import create_psf
from .helpers import *


import matplotlib.pyplot as plt
import pandas as pd
from scipy import ndimage
from scipy.signal import find_peaks


class sat_killer():
    def __init__(self,cube,psf,wavelength=None,sat_thickness=17,sat_sigma=10,
                 savename=None,y_close=5,angle_close=2,dist_close=10,num_cores=5,run=True):
        self.cube = cube
        self.star_psf = psf
        self.thickness = sat_thickness
        self.savename = savename
        #self.y_close = y_close
        #self.angle_close = angle_close
        self.num_cores = num_cores
        self.wavelength = wavelength
        self.streak_coef = []

        if run:
            self._make_image()
            #* (ble) Looks like these are also done in __detection_funcs()
            self._set_threshold(sat_sigma)
            self._dilate()
            self._edges()
            self._lines()
            #* ^ 

            self.__detection_funcs(sat_sigma)
            if len(self.streak_coef) > 0:
                self.make_mask()
                self._detected()
                if self.sat_num > 0:
                    self.make_satellite_psf()
                    self._fit_spec()

    def _make_image(self):
        #* how rri did it, which was struggling to catch streaks
        # image = np.nanmedian(self.cube,axis=0)
        # self.image = image - np.nanmedian(image)

        #*ble, to make it more like a quicklook, which we know works for streak detection
        image = np.nanmedian(self.cube, axis = 0)

        #*Sets `quicklook-like` bounds, and applies them to the image
        vmin = np.nanpercentile(image, 1).round(2) 
        vmax = np.nanpercentile(image, 98).round(2)
        image[image>=vmax] = vmax
        image[image<=vmin] = vmin

        image = image-vmin #*should make lowest Val 0
        image = 255*(image/(vmax-vmin)) #*should make the range (0,255), like an 8-bit image
        image = np.where(np.isfinite(image), image, 0) #*turns any nans into 0s, to keep image finite everywhere

        self.image = image

        #*Check to make sure the image was made when a line could be seen. 
        # fig,  ax = plt.subplots()
        # ax.imshow(image, origin="lower", cmap="grey")
        # if self.savename is not None:
        #     fig.savefig(f"{self.savename}sat_image.png")
        # else:
        #     fig.savefig("./sat_image.png")

    
    def _set_threshold(self,sigma):
        mean, med, std = sigma_clipped_stats(self.image)
        self.threshold = mean + sigma*std  #! This can easily be > max of image
        #! with default sat_sigma = 10, a std of ~20 can blow past the 255 upper bound with a modest mean around 50.
        #! If std is >25, it is going to zero the gray image out. 
        #! (ble)   

    def _detected(self):
        if len(self.streak_coef) > 0:
            self.satellite = True
        else:
            self.satellite = False
        self.sat_num = len(self.streak_coef)

    def _dilate(self):
        # set all values below the threshold to zero
        arr = deepcopy(self.image)
        arr[arr < self.threshold] = 0

        # create a structuring element for morphological dilation
        kernel = np.ones((9, 9))

        # dilate the array
        dilated = cv2.dilate(arr, kernel, iterations=1)
        dilated[dilated<10]  = 0
        # set all non-zero values in the dilated array that are not connected to other non-zero values to zero
        arr[(arr != 0) & (dilated == 0)] = 0 #? This is never used? (ble)

        d = (dilated > 0) * 1
        self.gray = (d*255/np.max(d)).astype('uint8')
        
    def _edges(self):
        low_threshold = 50
        high_threshold = 150
        self.edges = cv2.Canny(self.gray, low_threshold, high_threshold)

    def _lines(self):
        lines = cv2.HoughLinesP(self.edges, # Input edge image
                                1, # Distance resolution in pixels
                                np.pi/180, # Angle resolution in radians
                                threshold=100, # Min number of votes for valid line
                                minLineLength=100, # Min allowed length of line
                                maxLineGap=50 # Max allowed gap between line for joining them
                                )
        if lines is not None:
            good = []
            for i in range(len(lines)):
                line = lines[i]
                x1, y1, x2, y2 = line[0]
                if x1 == x2:
                    x2 += 0.1
                # calculate the angle of the line
                angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
                #if abs(angle) < 85:
                good += [i]
            good = np.array(good,dtype=int)

            self.lines = lines[good]
            coefs = []
            for line in self.lines:
                x1, y1, x2, y2 = line[0]
                coefs += [np.polyfit([x1,x2], [y1,y2], 1)]

            self.streak_coef = np.array(coefs)
        else:
            self.lines = []
    
    def _match_lines(self,close=30,minlines=1):
        x = np.arange(0,self.image.shape[1],0.5)
        yy = []
        for c in self.streak_coef:
            yy += [x*c[0]+c[1]]
        yy = np.array(yy)

        xx = np.zeros_like(yy)
        xx[:] = x
        
        
        ind = (yy<0) | (yy > self.image.shape[0])
        yy[ind] = np.nan
        
        d = np.sqrt((yy[:,np.newaxis,:,np.newaxis] - yy[np.newaxis,:,np.newaxis,:])**2 
                    + (xx[:,np.newaxis,:,np.newaxis] - xx[np.newaxis,:,np.newaxis,:])**2)
        d[d==0] = np.nan

        dmatrix = np.nanmean(np.nanmin(d,axis=2),axis=2)
        np.fill_diagonal(dmatrix,0)
        d_bool = dmatrix < close
        d_bool = np.unique(d_bool,axis=0)
        ind = np.sum(d_bool,axis=1) > minlines
        d_bool = d_bool[ind]

        n_coefs =[]
        for i in d_bool:
            n_coefs += [np.nanmean(self.streak_coef[i],axis=0)]
        ## Unsure if this is needed
        #ind = np.sum(d_bool,axis=1) <= minlines
        #for i in d_bool:
        #    n_coefs += [self.streak_coef[i]]
        n_coefs = np.array(n_coefs)
        self.streak_coef = n_coefs
        if len(n_coefs) > 0:
            self._find_center()

    def __match_lines(self,y_close=None,angle_close=None,dist_close=None):
        if len(self.lines ) > 1:
            if y_close is None:
                y_close = self.y_close
            if angle_close is None:
                angle_close = self.angle_close
            if dist_close is None:
                dist_close = self.dist_close
                
            coefs = []
            xs = []
            ys = []
            for line in self.lines:
                x1, y1, x2, y2 = line[0]
                xs += [[x1,x2]]
                ys += [[y1,y2]]
                coefs += [np.polyfit([x1,x2], [y1,y2], 1)]
            coefs = np.array(coefs)
            xs = np.array(xs); ys = np.array(ys)
            xx = np.arange(np.min(xs),np.max(xs),1)
            yy = []
            for c in coefs:
                yy += [xx*c[0]+c[1]]
            yy = np.array(yy)


            new_coefs = []

            used = np.zeros(len(yy))
            for i in range(len(yy)):
                if not used[i]:
                    diff = abs(yy[np.newaxis,i] - yy)
                    diff = np.nanmedian(diff,axis=1)
                    ind = diff < y_close
                    new_coefs += [np.nanmean(coefs[ind],axis=0)]
                    used[ind] = 1

            new_coefs = np.array(new_coefs)
            yy = []
            for c in new_coefs:
                yy += [xx*c[0]+c[1]]
            yy = np.array(yy)
            n_coefs = []
            distdiff = np.sqrt(((new_coefs[1][np.newaxis,:] * new_coefs[0][np.newaxis,:] - new_coefs[1][:,np.newaxis] * new_coefs[0][:,np.newaxis])/(new_coefs[0][:,np.newaxis]*new_coefs[0][np.newaxis,0]+1)))
            used = np.zeros(len(yy))
            angle = np.arctan(new_coefs[:,0])
            for i in range(len(yy)):
                if not used[i]:
                    diff = abs(angle[i] - angle)
                    distdiff = np.sqrt((yy[i]-yy)**2 + (xx,))
                    ind = (diff < angle_close) & (distdiff < dist_close)
                    n_coefs += [np.nanmean(new_coefs[ind],axis=0)]
                    used[ind] = 1
            n_coefs = np.array(n_coefs)

            self.streak_coef = n_coefs
            
            self._find_center()

    def __lc_variation_test(self,variation_frac=0.2):
        """
        Checks for variation along the streak, and requires it to be lower than a bound (variation_frac) (rri/ble)

        Parameters
        ----------
            variation_frac: float, optional
                The maximum allowed variation along a candiate streak. Should be 0<variation_frac<1, default 0.2

        Returns
        -------
            Nothing, but changes self.streak_coef based on the results of the check  
        """
        
        good = []
        for c in self.streak_coef:
            xx = np.arange(0,self.image.shape[1],1)
            yy = xx*c[0] + c[1]
            yy = (yy+0.5).astype(int)
            ind = (yy > 0) & (yy < self.image.shape[0])
            lc = self.image[yy[ind],xx[ind]] #! if image angled, the streak can be to short, and therefore the median is low, but a largeish fraction is still streak (ble) 

            lc = lc[np.where(lc!=2)]  #*This should fix the above, as 2 is never (well, it shouldn't be) anywhere in frame (ble)

            mean, med, std = sigma_clipped_stats(lc)
            if std == 0:
                std = 1
            if std > 30:
                std = 30
            var = abs(lc - med)
            frac = np.sum(var >= 3*std) / len(lc)
            # print(f"frac is {frac}")  
            if frac < variation_frac:
                good += [True]
            else:
                good += [False]
        self.streak_coef = self.streak_coef[good]

    def __lc_stars_vetting(self,sigma=2):
        """
        Checks for strings of stars. The streaks median value must be more than the image median plus sigma* the image std. (rri/ble)

            Parameters
            ----------

            sigma: float, optional
                The number of standard deviations above the median the streak should satisfy. Default 2. 

            Returns
            -------
                Nothing, but modifies self.streak_coef
        """
        
        good = []
        for c in self.streak_coef:
            xx = np.arange(0,self.image.shape[1],1)
            yy = xx*c[0] + c[1]
            yy = (yy+0.5).astype(int)
            ind = (yy > 0) & (yy < self.image.shape[0])
            lc = self.image[yy[ind],xx[ind]]
            lc = lc[np.where(lc!=2)] #* Same as in __lc_varitaion_test()  (ble)
            mean, med, std = sigma_clipped_stats(lc)
            pmean, pmed, pstd = sigma_clipped_stats(self.image,maxiters=20)
            cond = med > pmed + sigma*pstd
            if cond:
                good += [True]
            else:
                good += [False]
        self.streak_coef = self.streak_coef[good]
        


    def plot_lines(self):
        plt.figure()
        plt.imshow(self.image,origin='lower',cmap='gray',vmin=np.nanpercentile(self.image,16),vmax=np.nanpercentile(self.image,95))
        #for line in self.lines:
        #   x1, y1, x2, y2 = line[0]
        #   plt.plot([x1,x2],[y1,y2],'C1')
        counter = 1
        for c in self.streak_coef:
            xx = np.arange(0,self.image.shape[1],0.5)
            yy = xx*c[0] + c[1]
            plt.plot(xx,yy,f'C{counter}--',label=f'Sat {counter}')
            counter += 1
        plt.legend()
        plt.ylim(-0.5,self.image.shape[0]+0.5) #*y is 0 in shape (ble)
        
    def _find_center(self):
        """Finds the centers of the detected streaks (rri/ble)"""
        
        centers = []
        lengths = []
        angles = []
        cut_dims = []
        for c in self.streak_coef:
            xx = np.arange(0,self.image.shape[1],0.5)
            yy = xx*c[0] + c[1]
            ind = (yy >= 0) & (yy <= self.image.shape[0]) #*ble added = to inequality, so lines on edges were detected. They should get droped later. 
            yy = yy[ind]; xx = xx[ind]
            centers += [[np.nanmean(xx),np.nanmean(yy)]]
            lengths += [np.sqrt((xx[0]-xx[-1])**2 + (yy[0] - yy[-1])**2)]
            angles += [np.arctan2(yy[-1] - yy[0], xx[-1] - xx[0]) * 180 / np.pi]
            cut_dims += [[abs(xx[0] - xx[-1])/2,abs(yy[0]-yy[-1])/2]]
        self.centers = np.array(centers)
        self.lengths = np.array(lengths)
        self.angles = np.array(angles)
        self.cut_dims = (np.array(cut_dims) * 1.5).astype(int)
        
        satcat = pd.DataFrame([])
        satcat['xint'] = self.centers[:,0].astype(int)
        satcat['yint'] = self.centers[:,1].astype(int)
        self.satcat = satcat
        
    
    def make_mask(self,thickness=None):
        if thickness is None:
            thickness = self.thickness
        # create a black image with the same size as the input image
        mask = np.zeros_like(self.image)

        # loop through the consolidated lines
        sat_mask = []
        for c in self.streak_coef:
            xx = np.arange(0,self.image.shape[1],0.5)
            yy = xx*c[0] + c[1]
            ind = (yy > 0) & (yy < self.image.shape[0])
            yy = yy[ind]; xx = xx[ind]
            
            tmp = cv2.line(mask, (int(xx[0]), int(yy[0])), (int(xx[-1]), int(yy[-1])), (255, 255, 255), thickness=thickness)
            tmp[tmp > 0] = 1
            sat_mask += [tmp]

        # The pixels that are covered by the line will have a value of 8 in the mask
        # You can use this mask to extract the pixels that belong to the lines
        sat_mask = np.array(sat_mask).astype(int)
        if len(sat_mask) == 0:
            sat_mask = mask
        
        self.mask = sat_mask

        tmp = np.nansum(self.mask,axis=0)
        tmp[tmp > 0] = 1
        self.total_mask = tmp.astype(int)
        
    def make_satellite_psf(self):
        self.sat_psfs = []
        for i in range(self.sat_num):
            if 'gaussian' in self.star_psf.psf_profile:
                self.sat_psfs += [create_psf(self.cut_dims[i,0]*2+1,self.cut_dims[i,1]*2+1,angle = self.angles[i],
                                           length = self.lengths[i],stddev=self.star_psf.stddev)]
            elif 'moffat' in self.star_psf.psf_profile:
                self.sat_psfs += [create_psf(self.cut_dims[i,0]*2+1,self.cut_dims[i,1]*2+1,angle = self.angles[i],
                                           length = self.lengths[i],alpha=self.star_psf.alpha,beta=self.star_psf.beta)]
        
            
            
    def _fit_spec(self):
        sat_fluxes = []
        self.satcat['x'] = 0; self.satcat['y'] = 0
        self.satcat['xoff'] = 0; self.satcat['yoff'] = 0;
        for i in range(self.sat_num):
            cut = cube_cutout(self.cube,self.satcat.iloc[i],self.cut_dims[i,0],self.cut_dims[i,1])[0]
            psf = self.sat_psfs[i]
            psf.fit_pos(np.nanmean(cut,axis=0),range=5)
            xoff = psf.source_x; yoff = psf.source_y

            flux,res = zip(*Parallel(n_jobs=self.num_cores)(delayed(psf.psf_flux)(image) for image in cut))
            sat_fluxes += [np.array(flux)]
            
            self.satcat['x'].iloc[i] = self.satcat['xint'].values[i] + xoff
            self.satcat['y'].iloc[i] = self.satcat['yint'].values[i] + yoff
            self.satcat['xoff'] = xoff
            self.satcat['yoff'] = yoff
        self.sat_fluxes = np.array(sat_fluxes) #* 1e-20

    def spatial_specs(self,spacing=2,plot=True):

        if 'gaussian' in self.star_psf.psf_profile:
            fwhm = 2.355 * self.star_psf.stddev
            width = int(fwhm*2)
            psf = create_psf(width*2+1,width*2+1,angle=0,length=1,stddev=self.star_psf.stddev,psf_profile='gaussian')
        elif 'moffat' in self.star_psf.psf_profile:
            fwhm = 2*self.star_psf.alpha * np.sqrt(2**(1/self.star_psf.beta)-1)
            width = int(fwhm*4)
            psf = create_psf(width*2+1,width*2+1,angle=0,length=1,alpha=self.star_psf.alpha,beta=self.star_psf.beta,psf_profile='moffat')
        gap = fwhm * spacing
        self._fwhm = fwhm
        psf.generate_line_psf()

        spatial_specs = []
        for i in range(len(self.streak_coef)):
            c = self.streak_coef[i]
            xx = np.arange(0,self.image.shape[1],gap/(1+c[0]**2)**(1/2))
            
            yy = (xx*c[0] + c[1]) + self.satcat['yoff'].values[i]
            ind = (yy > 0) & (yy < self.image.shape[0]) 
            yy = yy[ind]; xx = xx[ind] + self.satcat['xoff'].values[i]

            linecat = pd.DataFrame([])
            linecat['y'] = yy; linecat['x'] = xx
            linecat['yint'] = (yy+.5).astype(int); linecat['xint'] = (xx+.5).astype(int)
            linecat['id'] = 'Sat: ' + str(i) #+ ' (' +linecat['xint'].values.astype(str)+',' + linecat['yint'].values.astype(str) + ')'
            if self.wavelength is None:
                lam = np.arange(0,len(self.cube)+1)
            else:
                lam = self.wavelength
            specs, residual, cat_off = get_specs(linecat,self.cube,width,width,
                                                 psf,lam,num_cores=self.num_cores,fitpos=False)
            linecat['spec'] = 0
            for i in range(len(specs)):
                linecat['spec'].iloc[i] = specs[i]
            spatial_specs += [linecat]

        self.spatial_specs = spatial_specs

        if plot:
            self.plot_spatial_specs()


    def streak_in_image_dims(self, reqLenIn:int=10):
        """
        Checks to see if the streak in inside the the image dimensions (ble)

        Parameters
        ----------
            reqLenIn: int, optional
                The number of points of the streak inside the image, default 10. Its a magic number 
        """
        oldCoefs = self.streak_coef
        toDrop = []
        if len(oldCoefs)>0:
            for i, coef in enumerate(oldCoefs):
                shape = self.image.shape #! Had shapes the wrong way around. 1 is x, 0 is y. 
                xs = np.linspace(0,shape[1],4*shape[0])
                ys = coef[0]*xs +coef[1]
                
                inFrame = ys[np.nonzero((ys>=0) & (ys<=shape[0]))] #making sure that the line detected is actually in the image
                if len(inFrame)<reqLenIn: 
                    #TODO remove steak
                    toDrop.append(i)
                    print(f"Should remove {coef} as it isn't in the frame\n")
        
        newCoefs = np.delete(oldCoefs, toDrop, axis=0)
        self.streak_coef = newCoefs



    def scan_for_parallel_streaks(self, interestWidth:int,peakWidth:int = 1, plotting:bool=False, diagnosing:bool= False, saving:bool=False):
        """Rotates and Scans horizontally for streaks (ble)
        
        Parameters
        ----------
            interestWidth: int, required
            The search width around a Hough transform found peak to scan for other peaks. 

            peakWidth: int, optional
                The FWHM of the assumed Gaussian streaks. It is to be used to check the distance between peaks.
                Default 1, suggested to be ~5

            plotting: bool, optional
                Decide to plot the outputs by setting True. Default False. Some of the on image plots are made elsewhere, so should probably remain False.
            
            diagnosing: bool, optional
                Plot even more checks of the functionallity. Default False. These are not plots made elsewhere, but also not helpful when they are working as intended.
            
            saving: bool optional
                If the figures should be saved
        
        Returns
        -------
            Nothing, but changes self.streak_coef
        """

        oldStreakCoefs = self.streak_coef
        xLen = self.image.shape[1]
        if len(oldStreakCoefs) == 0:
            #don't need to scan if there is nothing there.
            return
        
        print(f"Scanning file {self.savename}")
        print(f"Old streak coefiecents are: \n     {oldStreakCoefs}")

       

        newStreakCoefs = []
        
        for sNum,streak_coef in enumerate(oldStreakCoefs):
            #basic streak properties
            m = streak_coef[0]
            c = streak_coef[1]
            theta = np.arctan(m) 
            
            #rotate image
            rotIm = ndimage.rotate(self.image, np.degrees(theta), cval=np.nan)
            rotIm = rotIm.astype(np.float64)
            rotIm[rotIm<=2] = np.nan
            
            #Project found streak to rotated axis
            offset = np.sin(theta)*xLen
            if theta <0:
                offset=0

            cPrime = int(round(c *np.cos(theta) +offset,0))
            
            #Scanning along rows
            medVals = np.nanmedian(np.where(rotIm>2, rotIm, np.nan), axis=1) #* now only taking values in field (2 is off-sky black in quicklooks.)



            sigVals = np.nanstd(rotIm, axis=1)
            #Stats for whole axis
            mean, med, sig = sigma_clipped_stats(medVals)
        
            # #only looking close to streak
            # ofInterest = medVals[cPrime-interestWidth:cPrime+interestWidth] #!could be <0 or >len, so need to fix
            #*fix with min/max vals
            minVal = cPrime-interestWidth
            maxVal = cPrime +interestWidth

            if minVal<0:
                minVal=0
            if maxVal>= len(medVals):
                maxVal = len(medVals)-1

            #only looking close to streak
            medOfInterest = medVals[minVal:maxVal]
            sigOfInterest = sigVals[minVal:maxVal]

            #* uses peakWidth as FWHM of peaks 
            pPrime , _ = find_peaks(medOfInterest, height=mean + 5*sig, distance=peakWidth) 

            minMed = np.nanmin(medOfInterest)
            sideshift = 10 #*Magic number
            oiLen = len(medOfInterest)
            toDrop = []
            # print("any to drop?")
            for i, p in enumerate(pPrime):  #! can go wrong, if Image is angled, and streak close to corner. Cause the med value of the nearby row will be at min, due to >1/2 of px being not part of the frame.  
            #* Sort of fixed with sig vals
            #! but lc variation drops them still. 
            #* Have now fixed the lc variation. A higher Variation fraction allow more satellites through (also more junk, but there was always going to have to be validation by eye)
                lowShift = p-sideshift
                highShift = p+sideshift
                if lowShift<0 or highShift >=oiLen:
                    # print(f"Should drop {p} at {i}, out of range")
                    toDrop.append(i) #as outside of the image 
                    continue
                
                downMed = medOfInterest[p-sideshift]
                downSig = round(sigOfInterest[p-sideshift],0)

                upMed = medOfInterest[p+sideshift]
                upSig = round(sigOfInterest[p+sideshift])

                if np.isfinite(upMed) and np.isfinite(downMed):
                    #* This does still need doing, as strips can be wide enough
                    downMed = int(medOfInterest[p-sideshift])
                    downSig = int(round(sigOfInterest[p-sideshift],0))

                    upMed = int(medOfInterest[p+sideshift])
                    upSig = int(round(sigOfInterest[p+sideshift]))

                    if (downMed ==minMed and downSig == 0) or (upMed==minMed and upSig==0):
                        # print(f"Should drop {p} at {i}, too small")
                        toDrop.append(i)
                else:
                    toDrop.append(i)

            pPrime = np.delete(np.array(pPrime), toDrop)

            pPrime += minVal #gets into correct 0 for coords


            newIntercepts = ((pPrime-offset) / np.cos(theta))
            print(f"Streak {streak_coef} rotation and scan complete. \nFound {newIntercepts} as new Intercepts for the gradient {m}")
            
            for p in newIntercepts:
                newStreakCoefs.append([m,p])
            
            if plotting and diagnosing:

                fig2, ax2 = plt.subplots()
                ax2.imshow(rotIm, origin="lower")

                fig3, ax3 = plt.subplots(figsize=(12,6))
                ax3.plot(medVals)
                # ax2.set(xlim=(cPrime-60,cPrime +60))
                ax3.axhline(mean + 5*sig, label=f"Detection lower limit, $\mu + 5\sigma$", ls=":", c="r")
                ax3.vlines(np.round(oldStreakCoefs[:,1] *np.cos(theta) +offset,0).astype(int),0,np.nanmax(medVals)+10, label="Detected Satellites", colors="k")
                # ax3.vlines(pPrime,0,255)
                ax3.set(xlabel="Row", ylabel ="Median Intensity")
                ax3.legend()

                fig4, ax4 = plt.subplots(figsize=(12,6))
                ax4.plot(np.arange(len(medOfInterest))-cPrime+minVal, medOfInterest)
                ax4.vlines([pPrime-cPrime], 0,np.nanmax(medOfInterest)+10, colors="g", label="Found Peaks")
                ax4.axvline(0, c="k", label="Hough Transfrom Detection")
                ax4.set(xlabel="Rows From Detection", ylabel ="Median Intensity")
                ax4.legend()

                if saving and self.savename is not None:
                    # sName = fileName.split("/")[-1].split(".")[:-1]
                    # sName = ".".join(sName)
                    fig3.savefig(f"{self.savename}_MedVals{sNum}.png")
                    fig4.savefig(f"{self.savename}_MedPeakFound{sNum}.png")

        newStreakCoefs = np.array(newStreakCoefs)


        if len(newStreakCoefs.shape) >1: #So single streaks aren't indexed wrong.
            duplicates = []
            #* This gets some of the double ups from lines being not joined properly. 
            for i, c in enumerate(newStreakCoefs[:,1]):
                for j, c2 in enumerate(newStreakCoefs[:,1]):
                    if np.abs(c-c2) < peakWidth and i>j:
                        duplicates.append(i) 

            newStreakCoefs = np.delete(newStreakCoefs, duplicates, axis=0) #removes doubleups

        if plotting:
            fig, ax = plt.subplots(figsize=(10,10))
            ax.imshow(self.image, cmap="gray")
            xx = np.arange(xLen)
            
            for i, streak in enumerate(oldStreakCoefs):
                oldM= streak[0]
                oldC= streak[1]
                ax.plot(xx, oldM*xx + oldC, c=f"C{i+1}", ls="--",lw=5, label=f"Sat {i+1}")
            ax.set(ylim=(0,self.image.shape[0]))
            ax.legend()

            fig5, ax5 = plt.subplots(figsize=(10,10))
            ax5.imshow(self.image, cmap="gray")

            for i, streak in enumerate(newStreakCoefs):
                newC  = streak[1] 
                ax5.plot(xx, m*xx+newC,c=f"C{i+1}",ls="--",lw=5, label=f"New Sat {i+1}")

            ax5.set(ylim=(0,self.image.shape[0]))
            ax5.legend()

            if saving and self.savename is not None:
                # sName = fileName.split("/")[-1].split(".")[:-1]
                # sName = ".".join(sName)
                fig.savefig(f"{self.savename}_OldSats.png")
                fig5.savefig(f"{self.savename}_NewSats.png")

        print(f"\n The New Streak Coefs are \n {newStreakCoefs}\n")
        self.streak_coef = newStreakCoefs #* set class variable to what was found inside method, as is standard here
        
        self.streak_in_image_dims()

        if len(self.streak_coef) > 0:  
            #This was at the end of matchlines, when new coeffs were found, so doing it again here for consistency.
            #TODO Check if it needs to be done again.
            self._find_center() 


    def __detection_funcs(self,threshold:float):
        """
        Calls the functions needed for satellite detection in order. The image thershold is set, then the image is dilated to this threshold. Edges and lines are detected with Canny detection and a Hough transform. These lines are then matched if there are multiple of them. If there have been streaks detected, the image is scanned for parallel streaks that have been grouped together as one, and these streaks are then validated for low variation and stellar strings. Streaks that survive these checks are then plotted onto the image. (rri/ble)
        
        Parameters
        ----------
            threshold: float, required 
                This value is used as the number of standard deviations (sigma) that is added to the mean of the image. This sum is then set as self.threshold 
        
        Returns
        -------
            Nothing, but sets many self. variable as it goes, importantly self.streak_coef (2,n ndarray[Floats]) and self.satellite (bool)
        """
        
        self._set_threshold(threshold)
        self._dilate()
        self._edges()
        self._lines()
        if len(self.lines) > 1:
            self._match_lines(close=5,minlines=0)            
            self._match_lines(close=60,minlines=1)
        if len(self.streak_coef) > 0:           
            self.scan_for_parallel_streaks(interestWidth=100, peakWidth=5, diagnosing=False, plotting=False)            
            self.__lc_variation_test(variation_frac=0.3) #* trial with higher frac was sucessful
            #* now consistent. No extra _tpl combined quicklooks without the same streak in a _pst single cube one
            self.__lc_stars_vetting()  #! would throw out sats 
             #*Had to change this to
            if len(self.streak_coef) > 0:
                self.make_mask()
                self.plot_lines()
                if self.savename is not None:
                    plt.savefig(self.savename)
        self._detected()



    def quick_detection(self,image,threshold:float=3,savename:str|None=None):
        """
        Runs the satellite detection on just an image, without needing a full datacube. If no satellites are detected, it runs again at a higher threshold (rri)

        Parameters
        ----------
            image: Arraylike, required
                The image that is potentially contaminated with satellite streaks
            
            threshold: float, optional
                This value is used as the number of standard deviations (sigma) that is added to the mean of the image. This sum is then set as self.threshold. Default is 3 
            
            savename str | None, optional
                The path/to/file/name to save any plots made during the run. Default is None, needs to be given if saving figures is desired.
        
        Returns
        -------
            Nothing
        """
        self.image = image
        self.savename = savename
        self.__detection_funcs(threshold)
        if (threshold < 10) & (self.sat_num == 0):
            self.__detection_funcs(10)

        


    def plot_spatial_specs(self,bin_size=1):
        for linecat in self.spatial_specs:
            cmap = plt.get_cmap('viridis')
            norm = plt.Normalize(0, len(linecat))
            colours = cmap(norm(np.arange(0,len(linecat)+1)))
            plt.figure(figsize=(8,4))
            plt.subplot(121)
            vmin = np.nanpercentile(self.image,5)
            vmax = np.nanpercentile(self.image,99)
            plt.imshow(self.image,origin='lower',cmap='gray_r',alpha=0.4,vmin=vmin,vmax=vmax)
            plt.scatter(linecat['xint'].values,linecat['yint'].values,s=40, facecolors='none', edgecolors=colours)

            plt.subplot(122)
            for i in range(len(linecat)):
                spec = linecat['spec'].iloc[i]
                s = bin_spec(spec,bin_size)
                flux = s[1]; wave = s[0]
                nf = flux/np.nanmedian(flux)

                if np.nanstd(nf) < 1:
                    plt.plot(wave,flux/np.nanmedian(flux)+(i+1)*2,c=colours[i],alpha=0.6)#1-i/len(linecat))
            plt.ylabel('Normalised counts + offset',fontsize=12)
            plt.xlabel(r'Wavelength ($\rm \AA$)',fontsize=12)
            #col = plt.colorbar()
            #col.ax.set_ylabel('Sequence',fontsize=12)
            plt.tight_layout()
            plt.savefig('satellite_trail.png')


    def save_specs(self,savepath='.',spatial=False):
        for i in range(len(self.sat_fluxes)):
            save = np.array([self.wavelength,self.sat_fluxes[i]]).T
            # np.save(savepath + f'sat_{i+1}.png',save) #* Doesn't need the .png on the end
            np.save(savepath + f'sat_{i+1}',save) 



