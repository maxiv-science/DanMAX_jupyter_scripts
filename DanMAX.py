import os
import h5py
import glob
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as sci_op
from azint import AzimuthalIntegrator
import ipywidgets as ipyw
import IPython
import fabio


def getCurrentProposal():
    """Return current proposal number and visit number"""
    idx = os.getcwd().split('/').index('danmax')
    proposal, visit =  os.getcwd().split('/')[idx+1:idx+3]
    return proposal, visit

def getLatestScan(scan_type='any',proposal='',visit='',require_integrated=False):
    """
    Return the path to the latest /raw/*/*.h5 scan for the provided proposal and visit.
    Defaults to the current proposal directory of proposal and visit are not specified.
    
    Use scan_type (str) to specify which scan type to search for, i.e. 'timescan', 'dscan', 'ascan', etc.
    
    Use require_integrated = True to ensure that the returned scan has a valid integrated .h5 file.
    """
    if not proposal or not visit:
        proposal, visit = getCurrentProposal()
        #print(proposal, visit)
    files = sorted(glob.glob(f'/data/visitors/danmax/{proposal}/{visit}/raw/**/*.h5', recursive=True), key = os.path.basename, reverse=True)

    for file in files:
        if not 'pilatus.h5' in file and not '_falconx.h5' in file:
            # find a valid raw .h5 file
            try:
                with h5py.File(file,'r') as f:
                    if scan_type != 'any':
                        title = f['entry/title/'][()].decode()
                        if scan_type in title:
                            pass
                        else:
                            raise OSError
                    else:
                        pass
                # check if a valid integrated .h5 file exists 
                try:
                    afile = file.replace('raw', 'process/azint').split('.')[0]+'_pilatus_integrated.h5'
                    with h5py.File(afile,'r') as f:
                        pass
                    #scan_id = os.path.basename(file).split('.h5')[0]
                except OSError as err:
                    if require_integrated:
                        raise OSError
                    print('Unable to find a valid integrated file for the latest scan')    
                if scan_type != 'any':
                    print(f'Latest valid {scan_type}:\n',file)
                else:
                    print('Latest valid scan:\n',file)
                return file
            except OSError as err:
                pass
    print(f"No scan of type '{scan_type}' found in '/data/visitors/danmax/{proposal}/{visit}/raw/'")
    return None

def getAzintFname(fname):
    """Return the expected file path for a provided /raw/*/*.h5 file path"""
    try:
        afname = fname.replace('raw', 'process/azint').split('.')[0]+'_pilatus_integrated.h5'
        with h5py.File(afname,'r') as f:
            pass
        return afname
    except OSError as err:
        print(err.strerror)

def getMetaData(fname,custom_keys=[],relative=True):
    """
    Return dictionary of selected meta data. Return {key:None} if key is not available.
    Use custom_keys to provide a list of custom keys for additional parameters. Should include the full .h5 path.
    Ex. custom_keys = ['entry/instrument/hex_x/value','entry/instrument/hex_y/value']
    Default keys:
        I0
        time
        temp
        energy
        
    relative: Bool - Toggle wheteher to return data relative to the specific scan (True) or as absolute values (False). Default: True
    """
    if fname.startswith('scan-'):
        fname = findScan(fname)
        
    data = {'I0':None,
            'time':None,
            'temp':None,
            'energy':None}
    with h5py.File(fname,'r') as f:
        try:
            I0 = f['entry/instrument/albaem-xrd_ch2/data'][:]
            if relative:
                I0 /= I0.max()
            data['I0'] = I0
        except KeyError:
            print('No I0 available')
        try:
            time = f['entry/instrument/pcap_trigts/data'][:]
            if relative:
                time -= time[0]
            data['time'] = time
        except KeyError:
            print('No timestamp available')
        try:
            data['temp'] = f['entry/instrument/lakeshore_tempC/data'][:]
        except KeyError:
            print('No lakeshore temperature available')
        try:
            data['temp'] = f['entry/instrument/cryo_temp/data'][:]
        except KeyError:
            print('No cryo stream temperature available')
        try:
            data['energy'] = f['entry/instrument/hdcm_E/data'][:]
        except KeyError:
            print('No cryo stream temperature available')
        for key in custom_keys:
            try:
                data[key] = f[key][:]
            except KeyError:
                data[key] = None
                print(f'{key} not available')
    return data

def getMetaDic(fname):
    """Return dictionary of available meta data, reusing the .h5 dictionary keys."""
    data = {}
    with h5py.File(fname,'r') as f:
        for key in f['/entry/instrument/'].keys():
             if key != 'pilatus' and key != 'start_positioners':
                for k in f['/entry/instrument/'][key].keys():
                    data[key] = f['/entry/instrument/'][key][k][:]
    return data

def findAllScans(descending=True):
    """Return a sorted list of all scans in the current visit"""
    proposal, visit = getCurrentProposal()
    files = sorted(glob.glob(f'/data/visitors/danmax/{proposal}/{visit}/raw/**/*.h5', recursive=True), key = os.path.basename, reverse=descending)
    return [f for f in files if not ('pilatus.h5' in f or '_falconx.h5' in f)]


def findScan(scan_id):
    """Return the path of a specified scan number"""
    if type(scan_id) == int:
        scan_id = f'scan-{scan_id:04d}'
    elif type(scan_id) == str:
        scan_id = 'scan-'+scan_id.strip().split('scan-')[-1][:4]

    for sc in findAllScans():
        if scan_id in sc:
            return sc
    print('Unable to find {} in {}/{}'.format(scan_id,*getCurrentProposal()))
    
    
def getScanType(fname):
    """Return the scan type based on the .h5 scan title"""
    if fname.startswith('scan-'):
        fname = findScan(fname)
    with h5py.File(fname,'r') as f:
        try:
            scan_type = f['entry/title/'][()].decode()
            # clean up special characters
            scan_type = scan_type.replace('(',' ').replace(')',' ').strip("'").replace(',','')
            #print(scan_type)
            return scan_type
        except KeyError:
            print('No entry title available')

def getExposureTime(fname):
    """Return the exposure time in seconds as determined from the scan type"""
    scan_type = getScanType(fname)
    if 'timescan' in scan_type:
        exposure = scan_type.split()[2]
    elif 'ascan' in scan_type:
        exposure = scan_type.split()[5]
    elif 'dscan' in scan_type:
        exposure = scan_type.split()[5]
    elif 'mesh' in scan_type:
        exposure = scan_type.split()[9]
    return float(exposure)

def getScan_id(fname):
    """Return the scan_id from a full file path"""
    return 'scan-'+fname.strip().split('scan-')[-1][:4]
# def getVmax(im, percentile=99., bins=1000):
    # """
    # Return vmax (float) corresponding to the largest value
    # within a defined percentile of the cumulative histogram
    # """
    # limit = percentile/100
    # # close any open pyplot instance
    # plt.close()
    # # get image histogram
    # h = plt.hist(im[im>0],bins=bins, density=True, cumulative=True)
    # vargmax = np.argmax(h[0][h[0]<limit])
    # vmax = h[1][vargmax]
    # plt.close()
    # return vmax
    
def getVmax(im):
    """
    Return vmax (int) corresponding to the first pixel value with zero counts
    after the maximum value
    """
    h = np.histogram(im,bins=int(np.max(im)) ,range=(1,int(np.max(im))+1), density=False)
    first_zero = np.argmin(h[0][np.argmax(h[0]):])+np.argmax(h[0])
    return first_zero

def averageLargeScan(fname):
    """
    Return the average image of large scans
    Iteratively sum frames in a .h5 file without loading the full scan in the memory.
    """
    with h5py.File(fname, 'r') as fh:
        no_of_frames = fh['/entry/instrument/pilatus/data'].shape[0]
        print(f'{no_of_frames} frames in scan')
        for i in range(no_of_frames):
            if i<1:
                im = fh['/entry/instrument/pilatus/data'][0]
                im_max = np.max(im)
            else:
                im += fh['/entry/instrument/pilatus/data'][i]
                im_max = max(im_max,np.max(im))
            print(f'Progress: {i/(no_of_frames+1)*100:.1f}%',end='\r')
    print()
    exposure = getExposureTime(fname)
    print(f'Highest count rate in one frame: {im_max/exposure:,.2f} cps')
    im = im/no_of_frames
    return im
    
def getAverageImage(fname='latest'):
    """Return the average image of a scan - Default is the latest scan in the current folder"""
    if fname.lower() == 'latest':
        fname = getLatestScan()
    with h5py.File(fname, 'r') as fh:
        no_of_frames = fh['/entry/instrument/pilatus/data'].shape[0]
        if no_of_frames < 1000:
            im = fh['/entry/instrument/pilatus/data'][:]
            print(f'{no_of_frames} frames in scan')
            exposure = getExposureTime(fname)
            print(f'Highest count rate in one frame: {np.max(im)/exposure:,.2f} cps')
            return np.mean(im,axis=0)
    im = averageLargeScan(fname)
    return im

def getHottestPixel(fname):
    """Return the count rate (per second) for the hottest pixel in a scan"""
    # read data form the .h5 file
    with h5py.File(fname) as fh:
        max_counts = fh['/entry/instrument/pilatus/data'][:].max()
    exposure = getExposureTime(fname)
    cps = max_counts/exposure
    print(f'Highest count rate in one frame: {cps:,.2f} cps')
    return cps

        
def singlePeakFit(x,y):
    """
    Performe a single peak gaussian fit
    Return: amplitude, position, FWHM, background, y_calc
    """
    # Peak profile - in this case gaussian
    def gauss(x,amp,x0,fwhm,bgr):
        sigma = fwhm/(2*np.sqrt(2*np.log(2)))
        return amp*np.exp(-(x-x0)**2/(2*sigma**2))+bgr

    # initial position guess
    pos = x[np.argmax(y)]

    # initial background guess
    bgr = (y[0]+y[-1])/2
    
    # initial amplitude guess
    amp = np.nanmax(y)-bgr
    
    # Guess for sigma based on estimate of FWHM from array
    fwhm = np.abs(pos-x[np.argmin(np.abs(y-(amp/2)))])
    
    # Assume convergence before fit
    convergence = True

    # Fit the peak
    try:
        popt,pcov = sci_op.curve_fit(gauss,
                                     x,
                                     y,
                                     p0=[amp,pos,fwhm,bgr])
    except RuntimeError:
        print('sum(X) fit did not converge!')
        convergence = False
        popt = [None]*4

    amp,pos,sig,bgr = popt
    fwhm = 2*np.sqrt(2*np.log(2))*sig
    y_calc = gauss(x,amp,pos,fwhm,bgr)
    
    return amp,pos,fwhm,bgr,y_calc

def sampleDisplacementCorrection(tth,sdd,d):
    """
    Two theta correction for sample displacement along the beam (flat detector)
    Benjamin S. Hulbert and Waltraud M. Kriven - DOI: 10.1107/S1600576722011360
    Parameters:
        tth  - array - two theta values in degrees
        sdd  - float - sample-to-detector-distance
        d    - flaot - sample displacement, positive towards the detector
    Return:
        corr - array - angular correction in degrees such that 2th_corr = 2th-corr
    """
    corr = np.arctan(d*np.sin(2*tth*np.pi/180)/(2*(sdd-d*np.sin(tth*np.pi/180)**2)))*180/np.pi
    return corr


def integrateFile(fname, config,embed_meta_data=False):
    """
    DanMAX integration function
    Uses the python implementation of MATFRAIA to azimuthally integrate a /raw/SAMPLE/_pilatus.h5 file
    and save the integrated data in a corresponding /process/azint/SAMPLE/_pilatus_integrated.h5 file.
    Parameters:
        
        fname     : Absolute path for the master h5 file
        config = {
            poni_file          : Absolute path for the poni file 
            mask               : Absolute path for the mask file in .npy format 
            radial_bins        : Number of radial bins
            azimuthal_bins     : Number of azimuthal bins - See note below 
            unit               : "q" or "2th" 
            n_splitting        : Number of sub-pixel splitting used. The actual number of sub-pixels is N^2.
            polarization_factor: Polarization factor, should be very close to (but slightly lower than) 1.
            }
    More information about the integration can be found here:
    https://wiki.maxiv.lu.se/index.php/DanMAX:Pilatus_2M_CdTe#Live_azimuthal_integration_GUI
    
    ## Note on azimuthal bins ##
    
    The azimuthal integration direction is clockwise with origin in the inboard horizontal axis ("3 o'clock"). The azimuthal bins
    can be specified in several ways:
    int Number of bins out of the full 360° azimuthal range.
    np.linspace() or np.arange() Array of equidistant interval values in °, e.g np.arange(180,361,1) gives 180 1° bins from
    180°-360° with bin centers at [180.5, 181.5, .., 359.5]. 
    numpy array Array of arbitrary bin interval values in °, e.g. np.array([0,180,260,280,360]) will result in four bins of
    varying size: [0°-180°], [180°-260°], [260°-280°], and [280°-360°].
    
    NB: The intervals should be of ascending order. This is easily achieved with np.sort(). The integration does not wrap around
    360°, however %360 can be used to mitigate that, provided that zero is one of the bin bounds.
    
    Example:
    # make an interval that is symmetric around the vertical axis
    step, pad = 5,30
    bin_bounds = np.arange(180.-pad,360.+pad+step,step)%360
    bin_bounds = np.append(bin_bounds,360)
    bin_bounds.sort()
    
    """
    # set locked config parameters
    config=config.copy()
    config['error_model'] = None
    config['pixel_size'] = 172.0e-6
    config['shape'] = (1679, 1475)
    # read the mask file
    if type(config['mask'])==str:
        mask_fname = config['mask']
        if mask_fname.endswith('.npy'):
            config['mask'] = np.load(mask_fname)
        else:
            config['mask'] = fabio.open(mask_fname).data 
    
    # initialize the AzimuthalIntegrator
    ai = AzimuthalIntegrator(**config)
    
    
    # HDF5 dataset entry path
    dset_name = '/entry/instrument/pilatus/data'

    # read the meta data form the .h5 master file
    meta = getMetaDic(fname)
    # prepared output path and filename
    output_fname = fname.replace('raw', 'process/azint').replace('.h5','_pilatus_integrated.h5')
    output_folder = os.path.split(output_fname)[0]
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    
    # write to output file
    with h5py.File(output_fname,'w') as fi:
        # write the integration configuration information to the output file
        fi.create_dataset(ai.unit, data=ai.radial_axis)
        with open(config['poni_file'], 'r') as poni:
            p = fi.create_dataset('poni_file', data=poni.read())
            p.attrs['filename'] = config['poni_file']
        fi.create_dataset('mask_file', data=mask_fname)
        polarization_factor = config['polarization_factor'] 
        data = polarization_factor if polarization_factor is not None else 0
        fi.create_dataset('polarization_factor', data=data)
        
        if embed_meta_data:
            # write meta data to the _pilatus_integrated.h5 file
            for key in meta.keys():
                fi.create_dataset(key,data=meta[key])
        
        # read the detector images from the _pilatus.h5 file
        with h5py.File(fname.replace('.h5','_pilatus.h5'),'r') as fp:
            images = fp[dset_name]
            # display progress bar
            fname_widget = ipyw.Text(value = 'Integrating file: %s' %os.path.split(fname)[1])
            progress = ipyw.IntProgress(min=0, max=len(images))
            IPython.display.display(fname_widget)
            IPython.display.display(progress,display_id='prog_bar')

            # prepare the HDF5 dataset for the integrated data
            shape = (len(images), *ai.output_shape)
            I_dset = fi.create_dataset('I', shape=shape, dtype=np.float32)
            if ai.error_model == 'poisson':
                sigma_dset = fi.create_dataset('sigma', shape=shape, dtype=np.float32)

            for i, img in enumerate(images):
                if i % 10 == 0:
                    progress.value = i
                I, sigma = ai.integrate(img)
                I_dset[i] = I
                if sigma is not None:
                    sigma_dset[i] = sigma
        progress.value = i+1

def getMotorSteps(fname):
    """
    Return motor name(s), nominal positions, and registred (unique) positions for a given scan.
        Return list of lists [[motor_name_1,nominal,registred], ...]
    """
    
    dic = getMetaDic(fname)
    scan_type = getScanType(fname).lower().split()
    motors = [s for s in scan_type if s.islower()][1:]
    motor_steps = []
    for motor in motors:
        # get the nominal motor position from the macro title
        start, stop, steps = [scan_type[i+1:i+4] for i,s in enumerate(scan_type) if motor in s][0]
        nominal_pos = np.linspace(float(start),float(stop),int(steps)+1)
        # get the logged motor position
        motor_entry_id = [key for key in dic.keys() if motor in key][0]
        motor_pos = dic[motor_entry_id]
        motor_pos = np.unique(motor_pos)
        # compare nominal and actual motor positions
        if not np.all(nominal_pos == motor_pos):
            print(f'The nominal and registred motor positions for {motor} do not match!')
        motor_steps.append([motor,nominal_pos,motor_pos])
    
    return motor_steps