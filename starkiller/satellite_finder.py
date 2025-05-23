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
            self._set_threshold(sat_sigma)
            self._dilate()
            self._edges()
            self._lines()
            self.__detection_funcs(sat_sigma)
            if len(self.streak_coef) > 0:
                self.make_mask()
                self._detected()
                if self.sat_num > 0:
                    self.make_satellite_psf()
                    self._fit_spec()

    def _make_image(self):
        image = np.nanmedian(self.cube,axis=0)
        self.image = image - np.nanmedian(image)
    
    
    def _set_threshold(self,sigma):
        mean, med, std = sigma_clipped_stats(self.image)
        self.threshold = mean + sigma*std

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
        arr[(arr != 0) & (dilated == 0)] = 0

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
                    ind = (diff < angle_close) & (ydiff < dist_close)
                    n_coefs += [np.nanmean(new_coefs[ind],axis=0)]
                    used[ind] = 1
            n_coefs = np.array(n_coefs)

            self.streak_coef = n_coefs
            
            self._find_center()

    def __lc_variation_test(self,variation_frac=0.2):
        good = []
        for c in self.streak_coef:
            xx = np.arange(0,self.image.shape[1],1)
            yy = xx*c[0] + c[1]
            yy = (yy+0.5).astype(int)
            ind = (yy > 0) & (yy < self.image.shape[0])
            lc = self.image[yy[ind],xx[ind]]
            mean, med, std = sigma_clipped_stats(lc)
            if std == 0:
                std = 1
            if std > 30:
                std = 30
            var = abs(lc - med)
            frac = np.sum(var >= 3*std) / len(lc)
            if frac < variation_frac:
                good += [True]
            else:
                good += [False]
        self.streak_coef = self.streak_coef[good]

    def __lc_stars_vetting(self,sigma=2):
        good = []
        for c in self.streak_coef:
            xx = np.arange(0,self.image.shape[1],1)
            yy = xx*c[0] + c[1]
            yy = (yy+0.5).astype(int)
            ind = (yy > 0) & (yy < self.image.shape[0])
            lc = self.image[yy[ind],xx[ind]]
            
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
        plt.ylim(-0.5,self.image.shape[1]+0.5)
        
    def _find_center(self):
        centers = []
        lengths = []
        angles = []
        cut_dims = []
        for c in self.streak_coef:
            xx = np.arange(0,self.image.shape[1],0.5)
            yy = xx*c[0] + c[1]
            ind = (yy > 0) & (yy < self.image.shape[0])
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
                self.sat_psfs += [create_psf(self.cut_dims[0,0]*2+1,self.cut_dims[0,1]*2+1,angle = self.angles[0],
                                           length = self.lengths[0],stddev=self.star_psf.stddev)]
            elif 'moffat' in self.star_psf.psf_profile:
                self.sat_psfs += [create_psf(self.cut_dims[0,0]*2+1,self.cut_dims[0,1]*2+1,angle = self.angles[0],
                                           length = self.lengths[0],alpha=self.star_psf.alpha,beta=self.star_psf.beta)]
        
            
            
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


    def scan_for_parallel_streaks(self, interestWidth:int,peakWidth:int = 1, plotting:bool=False, diagnosing:bool= False, saving:bool=False):
        """Rotates and Scans horizontally for streaks (ble61)
        Inputs:
        -------
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
        """
        print(f"Scanning file {self.savename}")
        oldStreakCoefs = self.streak_coef
        xLen = self.image.shape[1]

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
            rotIm[rotIm==0] = np.nan
            
            #Project found streak to rotated axis
            offset = np.sin(theta)*xLen
            if theta <0:
                offset=0

            cPrime = int(round(c *np.cos(theta) +offset,0))
            
            #Scanning along rows
            medVals = np.nanmedian(rotIm, axis=1)
            
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
            ofInterest = medVals[minVal:maxVal]

            #* uses peakWidth as FWHM of peaks 
            pPrime , _ = find_peaks(ofInterest, height=mean + 5*sig, distance=peakWidth) 
    
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
                ax3.vlines(np.round(oldStreakCoefs[:,1] *np.cos(theta) +offset,0).astype(int),0,np.max(medVals)+10, label="Detected Satellites", colors="k")
                # ax3.vlines(pPrime,0,255)
                ax3.set(xlabel="Row", ylabel ="Median Intensity")
                ax3.legend()

                fig4, ax4 = plt.subplots(figsize=(12,6))
                ax4.plot(np.arange(len(ofInterest))-cPrime+minVal, ofInterest)
                ax4.vlines([pPrime-cPrime], 0,np.max(ofInterest)+10, colors="g", label="Found Peaks")
                ax4.axvline(0, c="k", label="Hough Transfrom Detection")
                ax4.set(xlabel="Rows From Detection", ylabel ="Median Intensity")
                ax4.legend()

                if saving and self.savename is not None:
                    # sName = fileName.split("/")[-1].split(".")[:-1]
                    # sName = ".".join(sName)
                    fig3.savefig(f"{self.savename}_MedVals{sNum}.png")
                    fig4.savefig(f"{self.savename}_MedPeakFound{sNum}.png")

        newStreakCoefs = np.array(newStreakCoefs)
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
        if len(newStreakCoefs) > 0:  
            #This was at the end of matchlines, when new coeffs were found, so doing it again here for consistency.
            #TODO Check if it needs to be done again.
            self._find_center() 


    def __detection_funcs(self,threshold):
        self._set_threshold(threshold)
        self._dilate()
        self._edges()
        self._lines()
        if len(self.lines) > 1:
            self._match_lines(close=5,minlines=0)
            self._match_lines(close=60,minlines=1)
            self.scan_for_parallel_streaks(interestWidth=100, peakWidth=5)
            self.__lc_variation_test()
            self.__lc_stars_vetting()
            if len(self.streak_coef) > 0:
                self.make_mask()
                self.plot_lines()
                if self.savename is not None:
                    plt.savefig(self.savename)
        self._detected()



    def quick_detection(self,image,threshold=3,savename=None):
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
            np.save(savepath + f'sat_{i+1}.png',save)



