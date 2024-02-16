import numpy as np
import cv2
from copy import deepcopy
from astropy.stats import sigma_clipped_stats
from .trail_psf import create_psf
from .helpers import *


import matplotlib.pyplot as plt
import pandas as pd


class sat_killer():
    def __init__(self,cube,psf,wavelength=None,sat_thickness=17,sat_sigma=15,y_close=5,angle_close=2,num_cores=5,run=True):
        self.cube = cube
        self.star_psf = psf
        self._make_image()
        self._set_threshold(sat_sigma)
        self.thickness = sat_thickness
        self.y_close = y_close
        self.angle_close = angle_close
        self.num_cores = num_cores
        self.wavelength = wavelength

        if run:
            self._dilate()
            self._edges()
            self._lines()
            self._match_lines()
            self._find_center()
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
        else:
            self.lines = []
    
    

    def _match_lines(self,y_close=None,angle_close=None):
        if y_close is None:
            y_close = self.y_close
        if angle_close is None:
            angle_close = self.angle_close
            
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

        used = np.zeros(len(yy))
        angle = np.arctan(new_coefs[:,0])
        for i in range(len(yy)):
            if not used[i]:
                diff = abs(angle[i] - angle)
                ind = diff < angle_close
                n_coefs += [np.nanmean(new_coefs[ind],axis=0)]
                used[ind] = 1
        n_coefs = np.array(n_coefs)

        self.streak_coef = n_coefs
        
        self._find_center()

    def plot_lines(self):
        plt.figure()
        plt.imshow(self.image,origin='lower',vmax=10,vmin=0)
        #for line in self.lines:
        #    x1, y1, x2, y2 = line[0]
        #    plt.plot([x1,x2],[y1,y2],'C1')
        for c in self.streak_coef:
            xx = np.arange(0,self.image.shape[1],0.5)
            yy = xx*c[0] + c[1]
            plt.plot(xx,yy,'C1')
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
            
            self.satcat['x'].iloc[i] = self.satcat['xint'].values + xoff
            self.satcat['y'].iloc[i] = self.satcat['yint'].values + yoff
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


    def plot_spatial_specs(self):
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
                nf = spec.flux/np.nanmedian(spec.flux)

                if np.nanstd(nf) < 1:
                    plt.plot(spec.wave,spec.flux/np.nanmedian(spec.flux)+(i+1)*2,c=colours[i],alpha=0.6)#1-i/len(linecat))
            plt.ylabel('Normalised counts + offset',fontsize=12)
            plt.xlabel(r'Wavelength ($\rm \AA$)',fontsize=12)
            #col = plt.colorbar()
            #col.ax.set_ylabel('Sequence',fontsize=12)
            plt.tight_layout()
            plt.savefig('satellite_trail.png')












            