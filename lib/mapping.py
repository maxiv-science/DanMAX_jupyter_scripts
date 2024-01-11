import sys
sys.path.append('../')
import DanMAX as DM
import numpy as np
import h5py
import os
import matplotlib.pyplot as plt
from matplotlib.colors import rgb_to_hsv
from IPython.utils import io
from texture import colorCylinderAnnotation


def combineMaps(rh,
                gs=None,
                bv=None,
                hsv=False,
                scale_place=np.s_[10:12,10:30,:],
                normalize=False,
                minmax = [[0.,1.],[0.,1.],[0.,1.]]):
    """ Returns a combination of three maps to plot as an rgb map.
    inputs parameters: 
        rh (red or hue) range zero to one
        gs (green or saturation) range zero to one
        bv (blue or value) range zero to one
        hsv boolean, if true the combined map will be converted from rgb to hsv.
        scale_place : a numpy slice where there should be white for a scale bar
        normalize: boolan, if true will normalize to within the minmax variable
        minmax
    """

    if gs == None:
        gs = np.zeros(rh.shape)
    if  bv == None:
        bv = np.zeros(rh.shape)
        
    
        
    scale_place = np.s_[scale_place[0][0]:scale_place[0][1],scale_place[1][0]:scale_place[1][1],:]
    cm = np.nan(rh.shape[0],rh.shape[1],3)
    if normalize:
        for dim in range(cm.shape[2]):
            cmap = cm[:,:,dim]
            cmap[cmap<minmax[dim][0]] = minmax[dim][0]
            cmap[cmap>minmax[dim][1]] = minmax[dim][1]
            cmap -= minmax[dim][0]
            cmap /=np.max(cmap)
            cm[:,:,dim] = cmap
    
    cm[:,:,0] = rh
    cm[:,:,1] = gs
    cm[:,:,2] = bv
    cm[np.isnan(cm)] = 0
    cm[scale_place] = 1
    if hsv:
        cm = rgb_to_hsv(cm)
    return cm
    
    
    
        
def makeMap(x, y, actualXsteps, nominalYsteps, signal):
    """
    Return a map 
    """
    xmin, xmax = np.amin(x), np.amax(x)
    ymin, ymax = np.amin(y), np.amax(y)
    xi = np.linspace(xmin, xmax, 2*actualXsteps)
    yi = np.linspace(ymin, ymax, 2*nominalYsteps)
    XI, YI = np.meshgrid(xi, yi)
    ZI = griddata((x.reshape(-1), y.reshape(-1)), signal.reshape(-1), (XI, YI), 'nearest')
    return ZI

def getXRFFitFilename(scans,proposal=None,visit=None, base_folder= None,channel=0):

    proposal,visit = DM.getCurrentProposal(proposal,visit)
    if base_folder == None:
        session_path = f'/data/visitors/danmax/{proposal}/{visit}/'
    else:
        session_path = base_folder
    if len(scans) > 1:
        scan_name=f'scan_{scans[0]:04}_to_{scans[-1]:04}'
    else:
        scan_name=f'scan_{scans[0]:04}'

    fname = DM.findScan(int(scans[0]), proposal=proposal, visit=visit)
    idx = fname.split('/').index('danmax')
    sample_name = fname.split('/')[idx + 4]
    
    xrf_out_dir = f'{session_path}process/xrf_fit/{sample_name}/{scan_name}/'
    xrf_file_name = f'fitted_elements_{scan_name}_{channel}'
    return xrf_out_dir, xrf_file_name


   
def stitchScans(scans,
                XRF=True,
                XRD=True,
                XRD_cake=False,
                xrd_range=None,
                azi_range=None,
                xrf_calibration=[0.14478, 0.01280573],
                map_type=np.float32,
                normI0=True,
                proposal=None,
                visit=None,
                ):
    """Returns stitched XRF and XRD maps of multiple scans or a single scan

    Output parameters:
    -snitch: dictionary holding the following or nothing based on input:
        -x_map x-coordinates for the map (motor positions)
        -y_map y-coordinates for the map (motor positions)
        -xrd_map Diffraction intensity
        -cake_map map of XRD cake intensity
        -xrf_map XRF maps of all the elements.
        -x_xrd 2-theta or q for diffraction
        -energy list of energies
        -Emax energy used for the scan
        -Q Boolian, if true x_xrd is in q otherwise it is  2theta
        -I0_map a map of I0

    xrf_map, energy, Emax are only returned if the input XRF == True (default)
    xrd_map, x_xrd, Q are only returned if the input XRD == True (default)
    cake_map is only returned if inputs XRD and XRD_cake are True (not default)

    Input parameters:
    -scans: list of scans that needs to be stitched
    -XRF: import XRF data (default True)
    -XRD: import XRD data (default True)
    -XRD_cake: import XRD cake data (default True)
    -xrd_range: list/tuple of min/max scattering angle to load, None for whole dataset (default None)
    -azi_range: list/tuple of min/max azimuthal angle to load, None for whole dataset (default None)
    -xrf_calibration: Calibration parameters for the rayspec_detector:
        (calibration is np.range(4096)*xrf_calibration[1]-xrf_calibration[0])
        (default (0.14478, 0.01280573))
    -map_type: output map datatype, reduce for less memory footprint (default np.float32)
    -normI0 Boolean: If true I0 will be normalized after stitching. (default True)
    -proposal: select another proposal for testing. (default None)
    -visit: select another visit for testing default. (default None)
    """

    fname = DM.findScan(int(scans[0]), proposal=proposal, visit=visit)
    # This dictionary will get them stitches
    snitch = {'x_map': None,
              'y_map': None,
              'xrd_map': None,
              'cake_map': None,
              'xrf_map': None,
              'x_xrd': None,
              'energy': None,
              'Emax': None,
              'I0_map': None,
              'Q': None,
              }

    with h5py.File(fname, 'r') as f:
        if XRF:
            # import falcon x data
            Emax = f['/entry/instrument/pilatus/energy'][()]*10**-3  # keV
            # Energy calibration (Conversion of chanels to energy)
            channels = np.arange(4096)
            if len(xrf_calibration) == 2:
                energy = channels*xrf_calibration[1]+xrf_calibration[0]
            if len(xrf_calibration) == 3:
                energy = ((channels**2)*xrf_calibration[2] +
                          channels*xrf_calibration[1] +
                          xrf_calibration[0])
        try:
            snake = f['/entry/snake'][()]
        except KeyError:
            snake = False

    for i, scan in enumerate(scans):
        print(f'scan-{scan} - {i+1} of {len(scans)}', end='\r')
        # print statements are suppressed within the following context
        with io.capture_output() as _:
            fname = DM.findScan(int(scan), proposal=proposal, visit=visit)
            # import falcon x data
            if XRF:
                with h5py.File(fname, 'r') as f:
                    S = f['/entry/instrument/falconx/data'][:]
                S = S[:, energy < Emax*1.1]
                snitch['Emax'] = Emax
                snitch['energy'] = energy
            if XRD:
                # import azimuthally integrated data
                aname = DM.getAzintFname(fname)
                data = DM.getAzintData(aname,
                                       xrd_range = xrd_range,
                                       azi_range = azi_range)
                if i == 0:
                    if data['q'] is not None:
                        x_xrd = data['q']
                        snitch['Q'] = True
                    else:
                        x_xrd = data['tth']
                        snitch['Q'] = False
                xrd = data['I']

                if XRD_cake:
                    if i == 0:
                        snitch['azi_edge'] = data['azi_edge']
                        snitch['azi'] = data['azi']
                    cake = data['cake'][:]
                del data

            # import meta data and normalize to I0
            meta = DM.getMetaData(fname, relative=False)
            I0 = meta['I0']

            # get data shape, from either XRF or XRD data
            # To ensure to have it no matter which one is selected
            if XRF:
                S = S.T.T.astype(map_type)
                data_shape = S.shape[0]
            if XRD:
                xrd = xrd.T.T.astype(map_type)
                data_shape = xrd.shape[0]
                if XRD_cake:
                    cake = cake.astype(map_type)

            # get the motor names, nominal and registred positions
            # Returns list of lists [[motor_name_1,nominal,registred], ...]
            M1, M2 = DM.getMotorSteps(fname)
            y = M1[2]  # registered motor position for the fast motor
            x = M2[2]  # registered motor position for the slow motor

            # if the length of nominal and registered positions do not match
            if len(M1[1]) != len(M2[2]):
                x = np.repeat(x, len(y)/(len(x)))

            # get the shape of the map from the nominal positions
            map_shape = (len(M2[1]), len(M1[1]))
            #  Check if that shape fits with the actual shape of the data
            #  This will not always be the case, then unpack the shape
            if XRF or XRD:
                if np.prod(map_shape) != np.prod(data_shape):
                    map_shape = (map_shape[0]-1, map_shape[1]-1)

            # Check if I0 exists
            if I0 is None:
                I0 = np.ones(map_shape)
            else:
                I0 = I0.reshape(map_shape)
            # reshape x and y grids to the map dimensions
            xx_new = x
            yy_new = y
            I0_new = I0

            # Reshape data to map dimensions
            xx_new = xx_new.reshape((map_shape))
            yy_new = yy_new.reshape((map_shape))
            if XRF:
                xrf_map_new = S.reshape((map_shape[0],
                                         map_shape[1],
                                         S.shape[-1]))
            if XRD:
                xrd_map_new = xrd.reshape((map_shape[0],
                                           map_shape[1],
                                           xrd.shape[-1]))
                if XRD_cake:
                    cake_map_new = cake.reshape((map_shape[0],
                                                 map_shape[1],
                                                 cake.shape[-2],
                                                 cake.shape[-1]))

            if i < 1:
                # If its the first iteration, create temporary variables
                # Then find the overlap
                xx = xx_new
                yy = yy_new
                I0_s = I0_new
                if XRF:
                    xrf_map = xrf_map_new
                if XRD:
                    xrd_map = xrd_map_new
                    if XRD_cake:
                        cake_map = cake_map_new
                # get the number of overlapping indices
                step_size = np.mean(np.diff(xx, axis=0))
            else:
                # For later iterations, find the overlap, later loaded data.
                overlap = round(np.mean(xx[-1]-xx_new[0])/step_size)+1
                xx[-overlap:, :] = xx[-overlap:, :]
                yy[-overlap:, :] = yy[-overlap:, :]
                I0_s[-overlap:, :] = I0_s[-overlap:, :]

                # append the new values
                xx = np.append(xx, xx_new[overlap:], axis=0)
                yy = np.append(yy, yy_new[overlap:], axis=0)
                I0_s = np.append(I0_s, I0_new[overlap:], axis=0)
                if XRF:
                    xrf_map = np.append(xrf_map,
                                        xrf_map_new[overlap:, :, :],
                                        axis=0)
                if XRD:
                    xrd_map = np.append(xrd_map,
                                        xrd_map_new[overlap:, :, :],
                                        axis=0)
                    if XRD_cake:
                        cake_map = np.append(cake_map,
                                             cake_map_new[overlap:, :, :],
                                             axis=0)

    if snake:
        print('Treating data as snake maps --- flipping every other line')
        # if data are measured as "snake" format, flip every second line
        if XRF:
            xrf_map[1::2, :, :] = xrf_map[1::2, ::-1, :]
            snitch['xrf_map'] = xrf_map
        if XRD:
            xrd_map[1::2, :, :] = xrd_map[1::2, ::-1, :]
            if XRD_cake:
                cake_map[1::2, :, :] = cake_map[1::2, ::-1, :]

        yy[1::2, :] = yy[1::2, ::-1]
        I0_s[1::2, :] = I0_s[1::2, ::-1]

    if normI0:
        I0_s /= np.max(I0_s)

    snitch['I0_map'] = I0_s
    snitch['x_map'] = xx
    snitch['y_map'] = yy
    if XRD:
        x_xrd = x_xrd[np.min(xrd_map > 0, axis=(0, 1))]
        snitch['x_xrd'] = x_xrd
        if XRD_cake:
            cake_map = cake_map[:, :, :, np.min(xrd_map > 0, axis=(0, 1))]
            snitch['cake_map'] = cake_map
        xrd_map = xrd_map[:, :, np.min(xrd_map > 0, axis=(0, 1))]
        snitch['xrd_map'] = xrd_map

    # Giving out the stitches!
    return snitch
