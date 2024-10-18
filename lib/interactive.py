import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
try:
    from ipywidgets import interact
except:
    print('unable to load ipywidgets')

class InteractiveMask():
    """
    Interactive mask tool for 2D or 3D images. The tool allows the user to draw a mask on the image by 
    clicking and dragging the mouse. The mask can be saved and used for further analysis. The tool also
    allows the user to set a threshold on the histogram of the image to create a binary mask. The tool 
    is implemented using matplotlib and numpy.
    Parameters:
    images: 2D or 3D numpy array of images
    reduction_mode: str, default='std'
        The mode to reduce the 3D images along the third axis. Available modes are 'std', 'mean', 'max', 'min'.
    mask_alpha: float, default=0.3
        The alpha value for the mask overlay on the image plot.
    additional parameters are passed to the plt.subplots() function.
    Returns:
    InteractiveMask object
    
    Methods:
    getMask(): Return the mask.
    getNanMask(): Return the mask with nan values.
    getManualMask(): Return the manual mask.
    getThresholdMask(): Return the threshold mask.
    getReducedImage(): Return the reduced image.
    getRoi(): Return the region of interest mask.
    getThreshold(): Return the threshold values.
    getRoiThreshold(): Return the roi threshold values.
    getResult(): Return the combined mask. (Obsolete, use getMask())

    Example:
        interactive_mask = InteractiveMask(xrd_map, reduction_mode='std', figsize=(8, 6))
        mask = interactive_mask.getMask()
        nan_mask = interactive_mask.getNanMask()

    """
    def __init__(self, images, reduction_mode='std', mask_alpha=0.3, *args, **kwargs):
        self.reduction_mode = reduction_mode
        self.is_darkmode = not plt.rcParams['axes.facecolor'] == 'white'
        if self.is_darkmode:
            self.line_color='w'
        else:
            self.line_color='k'

        if len(images.shape) == 3:
            # add an additional third subplot for the ROI above the image plot
            self.fig, axes = plt.subplots(2, 2, gridspec_kw={'height_ratios': [1, 6], 'width_ratios': [6, 1]},*args, **kwargs)
            self.ax = {'image': axes[1, 0], 'hist': axes[1, 1], 'roi': axes[0, 0]}
            # delete the unused axis
            axes[0, 1].remove()
            self.roi = np.ones(images.shape[2], dtype=bool)
            self._initRoi(images)
        else:
            self.fig, axes = plt.subplots(1, 2, gridspec_kw={'width_ratios': [6, 1]},*args, **kwargs)
            self.ax = {'image': axes[0], 'hist': axes[1]}
        self.canvas = self.fig.canvas

        # check the dimension of the images
        if len(images.shape) == 2:
            reduction_mode = 'none'
        elif len(images.shape) < 2:
            raise ValueError('The image(s) must be at least 2D')
        # ensure the images are at least 3D
        self.images = np.atleast_3d(images)
        # reduce the images along the third axis
        self.image = self.reduce_images(self.images, mode=reduction_mode)
        self.manual_mask = np.zeros(self.image.shape, dtype=int)
        self.threshold_mask = np.zeros(self.image.shape, dtype=int)

        self.mask_alpha = mask_alpha
        
        # create the mask as the boolean combination of the threshold mask and the manual mask
        self.mask = self.threshold_mask + self.manual_mask
        
        self._initImage()

        # create the histogram plot with swapped axis
        self._initHist()
        self.updateHist()
        self.updateThreshold(self.ax['hist'].get_ylim()[-1], 'red')
        
        # add a discrete text label below the image plot
        notification = self.ax['image'].text(0.0, -0.05, 'Untoggle zoom/pan in the toolbar to use the mask tools',
                              horizontalalignment='left', verticalalignment='center', 
                              transform=self.ax['image'].transAxes,
                              fontsize=8, color='red')
        notification.set_visible(False)


        self.canvas.mpl_connect('motion_notify_event', self.ondrag)
        self.canvas.mpl_connect('button_press_event', self.onclick)

    def _initImage(self):
        # create the image plot
        self.im = self.ax['image'].imshow(self.image, cmap='Greys_r')
        # create the mask overlay
        self.mask_overlay = self.ax['image'].imshow(self.mask,
                                                    cmap=self.get_red_to_transparent(),
                                                    alpha=self.mask_alpha,
                                                    vmin=0,
                                                    vmax=1)
        self.ax['image'].set_xticks([])
        self.ax['image'].set_yticks([])
        self.ax['image'].set_title('Interactiv mask tool')
    
    def _initHist(self):
        # create the histogram plot with swapped axis
        self.hist = self.ax['hist'].plot([0],[0], color=self.line_color)[0]
        # remove ticks
        self.ax['hist'].set_xticks([])
        self.ax['hist'].set_yticks([])

        # add red and blue horizontal moveable lines to indicate the thresholds
        self.threshold_line_red = self.ax['hist'].axhline(0, color='r', lw=2)
        self.threshold_line_blue = self.ax['hist'].axhline(0, color='b', lw=2)

    def _initRoi(self,images):
        # create the roi plot
        y_mean = np.nanmean(images,axis=(0,1))
        self.roi_plot = self.ax['roi'].plot(y_mean, color=self.line_color)[0]
        self.ax['roi'].set_xticks([])
        self.ax['roi'].set_yticks([])
        # set the margins
        self.ax['roi'].margins(x=0.02, y=0.05)

        # add red and blue horizontal moveable lines to indicate the roi
        self.roi_line_red = self.ax['roi'].axvline(y_mean.shape[0], color='r', lw=2)
        self.roi_line_blue = self.ax['roi'].axvline(0, color='b', lw=2)

    def updateThreshold(self, threshold, color):
        # self.threshold_line.set_ydata([threshold, threshold])
        if color == 'blue':
            self.threshold_line_blue.set_ydata([threshold, threshold])
        elif color == 'red':
            self.threshold_line_red.set_ydata([threshold, threshold])
        # get the upper and lower thresholds from the red and blue lines
        upper = max(self.threshold_line_red.get_ydata()[0], self.threshold_line_blue.get_ydata()[0])
        lower = min(self.threshold_line_red.get_ydata()[0], self.threshold_line_blue.get_ydata()[0])
        self.threshold_mask = (self.image > lower).astype(int) * (self.image < upper).astype(int)
        self.updateMask()
        plt.draw()

    def updateRoi(self, roi, color):
        if color == 'blue':
            self.roi_line_blue.set_xdata([roi, roi])
        elif color == 'red':
            self.roi_line_red.set_xdata([roi, roi])
        self.roi = np.logical_and(np.arange(self.roi.shape[0]) > min(self.roi_line_red.get_xdata()[0], self.roi_line_blue.get_xdata()[0]),
                                  np.arange(self.roi.shape[0]) < max(self.roi_line_red.get_xdata()[0], self.roi_line_blue.get_xdata()[0]))
        self.image = self.reduce_images(self.images, mode=self.reduction_mode)
        # update the image plot with the new image and set the clim
        self.im.set_data(self.image)
        self.im.set_clim(vmin=np.nanmin(self.image), vmax=np.nanmax(self.image))
        self.updateHist()
        plt.draw()

    def updateMask(self):
        self.mask = np.clip(self.threshold_mask + self.manual_mask, 0, 1)
        self.mask_overlay.set_data(self.mask)

    def updateHist(self):
        # calculate the histogram of the image
        cen, hist = self.calcHistogram()
        self.hist.set_data(hist, cen)
        # autoscale the histogram plot by first calculating the margins
        x_margin = np.abs(max(hist)-min(hist))*0.05
        y_margin = np.abs(cen[-1]-cen[0])*0.02
        # set the limits of the histogram plot
        self.ax['hist'].set_ylim(cen[0]-y_margin, cen[-1]+y_margin)
        self.ax['hist'].set_xlim(min(hist)-x_margin, max(hist)+x_margin)
        # update the threshold lines to be half way data limits and the margin
        self.updateThreshold(cen[0]-y_margin/2, 'blue')
        self.updateThreshold(cen[-1]+y_margin/2, 'red')
        # self.updateThreshold(self.ax['hist'].get_ylim()[-1], 'red')

    def reduce_images(self, images, mode='std'):
        """
        reduce the images along the third axis using the specified mode.
        modes: 'std', 'mean', 'max', 'min'        
        """
        if mode == 'std':
            return np.nanstd(images[:,:,self.roi], axis=2)
        elif mode == 'mean':
            return np.nanmean(images[:,:,self.roi], axis=2)
        elif mode == 'max':
            return np.nanmax(images[:,:,self.roi], axis=2)
        elif mode == 'min':
            return np.nanmin(images[:,:,self.roi], axis=2)
        elif mode == 'none':
            return images
        else:
            raise ValueError('Invalid mode')

    def calcHistogram(self):
        """calculate the histogram of the image, return the center and the histogram"""
        hist, bins = np.histogram(self.image[~np.isnan(self.image)], bins=512, density=True)
        cen = (bins[:-1] + bins[1:]) / 2
        return cen, hist

    def ondrag(self, event):
        if self.canvas.toolbar.mode == '':
            self.ax['image'].texts[0].set_visible(False)
            if event.inaxes == self.ax['image'] and event.button in [1, 3]:
                x, y = round(event.xdata), round(event.ydata)
                if event.button == 1:
                    self.manual_mask[y, x] = 1
                elif event.button == 3:
                    self.manual_mask[y, x] = -1
                self.updateMask()
                plt.draw()
        else:
            self.ax['image'].texts[0].set_visible(True)
            plt.draw()

    def onclick(self, event):
        if self.canvas.toolbar.mode == '':
            self.ax['image'].texts[0].set_visible(False)
            if event.inaxes == self.ax['image']:
                x, y = round(event.xdata), round(event.ydata)
                if event.button == 1:
                    self.manual_mask[y, x] = 1
                elif event.button == 3:
                    self.manual_mask[y, x] = -1
                self.updateMask()
                plt.draw()
            # check if the click is in the histogram plot
            elif event.inaxes == self.ax['hist']:
                threshold = event.ydata
                if event.button == 1:
                    self.updateThreshold(threshold,'blue')
                elif event.button == 3:
                    self.updateThreshold(threshold,'red')
                #self.threshold_line.set_visible(True)
                #self.updateThreshold(event.ydata)
                plt.draw()
            # check if the click is in the roi plot
            elif 'roi' in self.ax and event.inaxes == self.ax['roi']:
                threshold = round(event.xdata)
                if event.button == 1:
                    self.updateRoi(threshold, 'blue')
                elif event.button == 3:
                    self.updateRoi(threshold, 'red')
                plt.draw()
        else:
            self.ax['image'].texts[0].set_visible(True)
            plt.draw()

    def get_red_to_transparent(self):
        """Return the red to transparent colormap"""
        # Define the color transition
        cdict = {
            'red':   [(0.0, 1.0, 1.0),  # Red at the start
                    (1.0, 1.0, 1.0)], # Red at the end
            'green': [(0.0, 0.0, 0.0),  # No green at the start
                    (1.0, 0.0, 0.0)], # No green at the end
            'blue':  [(0.0, 0.0, 0.0),  # No blue at the start
                    (1.0, 0.0, 0.0)], # No blue at the end
            'alpha': [(0.0, 1.0, 1.0),  # Fully opaque at the start
                    (1.0, 0.0, 0.0)]  # Fully transparent at the end
        }

        # Create the colormap
        red_to_transparent = LinearSegmentedColormap('RedToTransparent', cdict)
        return red_to_transparent
    
    def getResult(self):
        """
        OBSOLETE - USE getMask()
        Return combined mask"""
        return self.mask
    
    def getManualMask(self):
        """Return manual mask"""
        return self.manual_mask
    
    def getThresholdMask(self):
        """Return threshold mask"""
        return self.threshold_mask
    
    def getMask(self):
        """Return mask"""
        return self.mask
    
    def getNanMask(self):
        """Return nan mask"""
        nan_mask = self.mask.astype(float)
        nan_mask[nan_mask<1.] = np.nan
        return nan_mask
    
    def getRoi(self):
        """Return roi"""
        return self.roi
    
    def getThreshold(self):
        """Return threshold"""
        return self.threshold_line_red.get_ydata()[0], self.threshold_line_blue.get_ydata()[0]
    
    def getRoiThreshold(self):
        """Return roi threshold"""
        return self.roi_line_red.get_xdata()[0], self.roi_line_blue.get_xdata()[0]
    
    def getReducedImage(self):
        """Return the reduced image"""
        return self.image


def interactiveImageHist(im,ignore_zero=False):
    """
    Plot an interactive image with histogram to easily adjust lower and upper thresholds
        Parameters
            im          - Image as an (n,m) numpy array
            ignore_zero - Ignore pixel values less than or equal to zero (default False)
    """
    def plotImageHistogram(im,ignore_zero=False):
        """
        Plot an image with histogram
            Parameters
                im          - Image as an (n,m) numpy array
                ignore_zero - Ignore pixel values less than or equal to zero (default False)
            Return
                fig         - matplotlib.figure
                cm          - matplotlib.image
                ax0         - matplotlib.axes (image axis)
                ax1         - matplotlib.axes (histogram axis)
        """
        if ignore_zero:
            # remove negative and nan pixels
            _im = im[im>0]
        else:
            # remove nan pixels
            _im = im[~np.isnan(im)]
        # generate bin edges and centers
        edges = np.linspace(np.min(_im),np.max(_im),256)
        cen = edges[:-1]+np.mean(np.diff(edges))/2
        # create histogram
        val, edges = np.histogram(_im,bins=edges,density=True)
        vmax = cen[np.argmin(np.abs(np.diff(val[np.argmax(val):])))]
        
        # estimate appropriate figure aspect ratio
        #im_aspect = im.shape[0]/im.shape[1]
        width_ratios=[10,1]
        #ax_aspect = 1+(width_ratios[1]/np.sum(width_ratios))
        #fig_aspect = im_aspect*ax_aspect
    
        #fig = plt.figure(figsize=(5*fig_aspect,5))
        fig = plt.figure()
        # initialize grid and subplot with different size-ratios
        grid = plt.GridSpec(1,2,width_ratios=width_ratios) #rows,columns
        ax0, ax1 = [fig.add_subplot(gr) for gr in grid]
        # set tick parameters
        ax0.set_xticks([])
        ax0.set_yticks([])
        ax1.set_xticks([])
        ax1.yaxis.tick_right()
        # plot image
        cm = ax0.imshow(im,vmax=vmax)
        vmin,vmax = cm.get_clim()
        # plot histogram
        ax1.plot(val,cen)
        ax1.set_ylim(vmin,vmax)
        fig.tight_layout()
        return fig, cm, ax0, ax1

    # initialize the figure
    fig, cm, ax0, ax1 = plotImageHistogram(im,ignore_zero=ignore_zero)

    # make a simple function to update the displayed image 
    def update_clim(vmin=0,vmax=100,log=False):
        """function for updating the image color limit. Called by ipywidgets.interact()"""
        norm = 'log' if log else 'linear'
        cm.set_norm(norm)
        ax1.set_yscale(norm)
        cm.set_clim(vmin,vmax)
        ax1.set_ylim(vmin,vmax)
        #ax0.set_title(f'vmin: {vmin:.2E}   vmax:{vmax:.2E}',
        #              loc='left',
        #              fontdict={'fontsize':'small'})
        #return f'vmin: {vmin:.3E}   vmax:{vmax:.3E}'
        return f'{vmin:.3E}   {vmax:.3E}'
    # find vmin/vmax range and step size
    vmin,vmax = np.nanmin(im),np.nanmax(im)
    step = (vmax-vmin)/1000
    # start interactive widget
    inter = interact(update_clim,
                     vmin=(vmin,vmax,step),
                     vmax=(vmin,vmax,step),
                     log=False)
    return inter.widget