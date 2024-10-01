import sys
import os
import h5py
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from IPython.utils import io
from matplotlib.colors import rgb_to_hsv
from matplotlib.colors import hsv_to_rgb
sys.path.append('../')
import DanMAX as DM


def combineMaps(rh,
                gs=None,
                bv=None,
                hsv=False,
                scale_place=[[10,12],[10,30]],
                normalize=False,
                minmax=[[0., 1.], [0., 1.], [0., 1.]]):
    """ Returns a combination of three maps to plot as an rgb map.
    inputs parameters:
        rh (red or hue) range zero to one
        gs (green or saturation) range zero to one
        bv (blue or value) range zero to one
        hsv boolean, if true the combined map will be go from rgb to hsv.
        scale_place : a numpy slice where there should be white for a scale bar
        normalize: boolan, if true will normalize to within the minmax variable
        minmax
    """

    if gs is None:
        gs = np.zeros(rh.shape)
    if bv is None:
        bv = np.zeros(rh.shape)

    scale_place = np.s_[
                    scale_place[0][0]:scale_place[0][1],
                    scale_place[1][0]:scale_place[1][1],
                    :
                    ]

    cm = np.nan*np.ones((rh.shape[0], rh.shape[1], 3))
    if normalize:
        for dim in range(cm.shape[2]):
            cmap = cm[:, :, dim]
            cmap[cmap < minmax[dim][0]] = minmax[dim][0]
            cmap[cmap > minmax[dim][1]] = minmax[dim][1]
            cmap -= minmax[dim][0]
            cmap /= np.max(cmap)
            cm[:, :, dim] = cmap

    cm[:, :, 0] = rh
    cm[:, :, 1] = gs
    cm[:, :, 2] = bv
    cm[np.isnan(cm)] = 0
    if hsv:
        cm = hsv_to_rgb(cm)
    cm[scale_place] = 1
    return cm


def getXRFFitFilename(
        scans,
        proposal_type=None,
        beamline=None,
        proposal=None,
        visit=None,
        base_folder=None,
        channel=0
        ):

    proposal, visit = DM.getCurrentProposal(proposal, visit)
    proposal_type, beamline = DM.getCurrentProposalType(proposal_type, beamline)

    if base_folder is None:
        session_path = f'/data/{proposal_type}/{beamline}/{proposal}/{visit}/'
    else:
        session_path = base_folder

    if len(scans) > 1:
        scan_name = f'scan_{scans[0]:04}_to_{scans[-1]:04}'
    else:
        scan_name = f'scan_{scans[0]:04}'

    fname = DM.findScan(int(scans[0]), proposal=proposal, visit=visit)
    idx = fname.split('/').index('danmax')
    sample_name = fname.split('/')[idx + 4]

    xrf_out_dir = f'{session_path}/process/xrf_fit/{sample_name}/{scan_name}/'
    xrf_file_name = f'fitted_elements_{scan_name}_{channel}.h5'

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
    -xrd_range: list/tuple of min/max scattering angle to load, None for whole
        dataset (default None)
    -azi_range: list/tuple of min/max azimuthal angle to load, None for whole
        dataset (default None)
    -xrf_calibration: Calibration parameters for the rayspec_detector:
        (calibration is np.range(4096)*xrf_calibration[1]-xrf_calibration[0])
        (default (0.14478, 0.01280573))
    -map_type: output map datatype, reduce for less memory footprint
        (default np.float32)
    -normI0 Boolean: If true I0 will be normalized after stitching.
        (default True)
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
              'azi': None,
              'azi_exge': None,
              'energy': None,
              'Emax': None,
              'I0_map': None,
              'Q': None,
              }

    with h5py.File(fname, 'r') as f:
        if XRF:
            # import falcon x data
            start_positioners = '/entry/instrument/start_positioners'
            Emax = min(
                    abs(f[f'{start_positioners}/hdcm_energy'][()]),
                    abs(f[f'{start_positioners}/mlm_energy'][()])
                    )
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
                    S = f['/entry/instrument/xspress3-dtc-2d/data'][:]
                S = S.squeeze()
                S = S[:, energy < Emax*1.1]
                snitch['Emax'] = Emax
                snitch['energy'] = energy
            if XRD:
                # import azimuthally integrated data
                aname = DM.getAzintFname(fname)
                data = DM.getAzintData(
                        aname,
                        xrd_range=xrd_range,
                        azi_range=azi_range
                        )
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
                    map_shape = (map_shape[0], map_shape[1]-1)
                if np.prod(map_shape) != np.prod(data_shape):
                    map_shape = (map_shape[0]-1, map_shape[1])

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
        if len(xrd_map.shape) == 3:
            x_xrd = x_xrd[np.min(xrd_map > 0, axis=(0, 1))]
        elif len(xrd_map.shape) == 4:
            x_xrd = x_xrd[np.min(xrd_map > 0, axis=(0, 1, 2))]
        snitch['x_xrd'] = x_xrd
        if XRD_cake:
            cake_map = cake_map[:, :, :, np.min(xrd_map > 0, axis=(0, 1))]
            snitch['cake_map'] = cake_map
        xrd_map = xrd_map[:, :, np.min(xrd_map > 0, axis=(0, 1))]
        snitch['xrd_map'] = xrd_map

    if XRF:
        snitch['xrf_map'] = xrf_map

    # Giving out the stitches!
    return snitch


def setup_h5_file(
        filename: str,
        attributes: dict,
        groups: list = None,
        ) -> None:

    with h5py.File(filename, 'w') as af:
        for group in attributes.keys():
            if groups is None or group in groups:
                h5group = af.create_group(group)
                for attr in attributes[group]:
                    h5group.attrs[attr[0]] = attr[1]


def setup_h5_softlinks(
        filename: str,
        links: dict,
        ) -> None:

    with h5py.File(filename, 'a') as af:
        for group in links.keys():
            for key in links[group].keys():
                af[links[group][key]] = h5py.SoftLink(key)

def copy_h5_linking(
        filename: str,
        links: dict,
        ) -> None:

    with h5py.File(filename, 'a') as af:
        for group in links.keys():
            for key in links[group].keys():
                af[links[group][key]] = af[key]

def transpose_order(
        shape: npt.ArrayLike,
        trans: bool
        ) -> npt.ArrayLike:
    '''
     Finds the permutation to put order as -1 1 0 2 3 4 ...

     parameters:
        Shape is the shape of the matrix to transpose
        trans (bool) is whether or not to transpose the matrix
     returns
        trans_order order of dimensions after transposing
    '''

    if trans and len(shape) > 2:
        trans_order = [len(shape)-1, 1, 0]
        for i in range(2, len(shape)-1):
            trans_order.append(i)
    else:
        trans_order = [i for i in range(len(shape))]

    return trans_order


def q_to_unit(q: bool) -> list:
    if q:
        unit = ['q', 'A-1']
    else:
        unit = ['tth', 'degrees']

    return unit


def save_maps(
        maps: dict,
        scans: list,
        transpose_data: bool = True,
        proposal_type: str = None,
        beamline: int = None,
        proposal: int = None,
        visit: int = None,
        ) -> None:
    '''
    Save shaped maps containing XRD or XRF data.
    Either from mapping, or XRDCT.

    Parameters:
        maps: (required) dictionary with the scan information, fields:
            - x_map: (required) map of x values
            - y_map: (required) map of y values
            - Q: (required) unit of xrd integration
            - xrd_map: map of 1d xrd data
            - cake_map: map of 2d xrd data
            - xrf_map: map of xrf data
            - x_xrd: diffraction signal x-values
            - azi: azimuthal values for 2d XRD data
            - energy: list of energies for XRF data
            - Emax: maximum energy for XRF data
            - I0_map: map of I0 values
        scans: list of scans used for the map
        transpose_data: Transpose large dimentional data for fast viewing
        proposal_type: Type of the proposal
        beamline: Beamline of the experiment
        proposal: Proposal number
        visit: Visit number
    Returns: None
    '''

    group_measurement = 'entry/measurement'
    group_scans = 'scan_list'
    group_xrd1d = 'entry/dataxrd1d'
    group_xrd2d = 'entry/dataxrd2d'
    group_xrf = 'entry/dataxrf'

    h5string = h5py.string_dtype()

    x_xrd = q_to_unit(maps['Q'])

    snitch_keys = {
            'x_map': f'{group_measurement}/x_map',
            'y_map': f'{group_measurement}/y_map',
            'azi': f'{group_measurement}/azi',
            'x': f'{group_measurement}/x',
            'y': f'{group_measurement}/y',
            'xrd_map': f'{group_xrd1d}/xrd',
            'cake_map': f'{group_xrd2d}/xrd',
            'xrf_map': f'{group_xrf}/xrf',
            'x_xrd': f'{group_measurement}/{x_xrd[0]}',
            'energy': f'{group_measurement}/energy',
            'Emax': f'{group_measurement}/Emax',
            'I0_map': f'{group_measurement}/I0_map',
            'Q': f'{group_measurement}/Q',
            }

    soft_links = {}
    groups = [group_measurement]
    if 'xrd_map' in maps and not maps['xrd_map'] is None:
        groups.append(group_xrd1d)
        soft_links[group_xrd1d] = {
                snitch_keys['x']: f'{group_xrd1d}/x',
                snitch_keys['y']: f'{group_xrd1d}/y',
                snitch_keys['x_xrd']: f'{group_xrd1d}/{x_xrd[0]}',
                }
    if 'cake_map' in maps and not maps['cake_map'] is None:
        groups.append(group_xrd2d)
        soft_links[group_xrd2d] = {
                snitch_keys['x']: f'{group_xrd2d}/x',
                snitch_keys['y']: f'{group_xrd2d}/y',
                snitch_keys['x_xrd']: f'{group_xrd1d}/{x_xrd[0]}',
                snitch_keys['azi']: f'{group_xrd2d}/azi',
                }
    if 'xrf_map' in maps and not maps['xrf_map'] is None:
        groups.append(group_xrf)
        soft_links[group_xrf] = {
                snitch_keys['x']: f'{group_xrf}/x',
                snitch_keys['y']: f'{group_xrf}/y',
                snitch_keys['energy']: f'{group_xrf}/energy',
                }

    maps['x'] = np.mean(maps['x_map'], axis=1)
    maps['y'] = np.mean(maps['y_map'], axis=0)

    attributes = {
            'entry': [
                ['NX_class', 'NXentry'],
                ],
            group_xrd1d: [
                ['NX_class', 'NXdata'],
                ['interpretation', 'image'],
                ['signal', 'xrd'],
                ['axes', np.array(['x', 'y', x_xrd[0]], dtype=h5string)],
                ],
            group_xrd2d: [
                ['NX_class', 'NXdata'],
                ['interpretation', 'image'],
                ['signal', 'xrd'],
                ['axes', np.array(['x', 'y', 'azi', x_xrd[0]], dtype=h5string)],
                ],
            group_xrf: [
                ['NX_class', 'NXdata'],
                ['interpretation', 'image'],
                ['signal', 'xrf'],
                ['axes', np.array(['x', 'y', 'energy'], dtype=h5string)],
                ],
            group_measurement: [
                ['NX_class', 'NXprocess'],
                ],
            }
    units = {
            'energy': 'keV',
            'x_xrd': x_xrd[1],
            'q': 'A-1',
            'x': 'mm',
            'y': 'mm',
            }

    for group in attributes.keys():
        for attr in attributes[group]:
            if attr[0] == 'axes':
                dims = np.ones([3*i for i in range(len(attr[1]))]).shape
                trans_order = transpose_order(
                                dims,
                                transpose_data)
                attr[1] = attr[1][trans_order]

    folder_name = os.path.dirname(
            DM.findScan(
                scans[0],
                proposal_type=proposal_type,
                beamline=beamline,
                proposal=proposal,
                visit=visit)).replace(
                    'raw',
                    'process/_maps')

    if not os.path.isdir(folder_name):
        os.makedirs(folder_name)

    if isinstance(scans[0], str):
        scan_nrs = [int(scans[0][-7:-3]),
                    int(scans[-1][-7:-3])]
    else:
        scan_nrs = [scans[0], scans[-1]]

    file_name = os.path.join(
            folder_name,
            f'scan_{scan_nrs[0]}-{scan_nrs[1]}.h5'
            )
    scan_filenames = np.array(
            [DM.findScan(
                scan,
                proposal=proposal,
                visit=visit
                ) for scan in scans],
            dtype=h5py.special_dtype(vlen=str)
            )

    setup_h5_file(file_name, attributes, groups=groups)

    with h5py.File(file_name, 'a') as sf:
        sf.create_dataset(
                group_scans,
                data=scan_filenames,
                dtype=h5py.special_dtype(vlen=str))

        for key in maps.keys():

            if not maps[key] is None:
                if not isinstance(maps[key], np.ndarray):
                    sf.create_dataset(
                            snitch_keys[key],
                            data=maps[key]
                            )
                    continue

                trans_order = transpose_order(
                                maps[key].shape,
                                transpose_data
                                )

                if x_map.dtype == np.float64:
                    savetype = np.float32
                else:
                    savetype = maps[key].dtype
                
                sf.create_dataset(
                        snitch_keys[key],
                        data=np.transpose(
                            maps[key],
                            trans_order
                            ).astype(savetype)
                        )
                if key in units.keys():
                    sf[snitch_keys[key]].attrs['unit'] = units[key]

    copy_h5_linking(file_name, soft_links)

def load_maps(fname):
    """
    Load xrd maps saved by save_maps
    
    NEEDS TO BE EXPANDED
    """
    group_xrd1d = 'entry/dataxrd1d'
    group_measurement = 'entry/measurement'
    
    maps = {'x_map':group_measurement+'/x_map',
            'y_map':group_measurement+'/y_map',
            'xrd_map':group_xrd1d+'/xrd',
            'x_xrd':[group_xrd1d+'/tth',group_xrd1d+'/q'],
           }
    
    
    with h5py.File(fname) as f:
        Q = f[group_measurement+'/Q'][()]
        for key in maps:
            if key == 'x_xrd':
                maps[key] = f[maps[key][0]][:]
                if Q:
                    maps[key] = f[maps[key][1]][:]
            else:
                maps[key] = f[maps[key]][:]
        maps['Q']=Q
    return maps


def getXRDctMap(fname,xrd_range=None):
    """
    PARTIALLY OBSOLETE
    Return the xrd ct map of the specified file as a dictionary of maps
    Parameters
        fname     - master .h5 file or path to _recon.h5 file
        xrd_range - list/tuple of min/max scattering angle to load, None for whole
                    dataset (default None)
    Return
        maps - dictionary: {'x_map':np.array,'y_map':np.array,'xrd_map':np.array,'x_xrd':np.array,'Q':bool}
    """
    # find reconstructed file name
    rname = fname.replace('raw', 'process/xrd_ct').replace('.h5', '_recon.h5')
    
    # load the reconstructed data
    with h5py.File(rname,'r') as f:
        recon = f['/reconstructed/gridrec'][:] # (n, m, radial)
        if 'q' in f.keys():
            Q = True
            x = f['q'][:]
        else:
            Q = False
            x = f['2th'][:]
        if 'micrometer_per_px' in f.keys():
            um_per_px = f['micrometer_per_px'][()]
        else:
            um_per_px = np.nan
    # generate x/y_map in px
    x_map, y_map = np.mgrid[0:recon.shape[0],0:recon.shape[1]]
    # convert to mm
    if not np.isnan(um_per_px):
        x_map = x_map*um_per_px*1e-3 # mm
        y_map = y_map*um_per_px*1e-3 # mm

    if xrd_range is None:
        roi = x==x
    else:
        roi = (x>xrd_range[0]) & (x<xrd_range[1])
    
    maps = {'x_map': x_map,
            'y_map': y_map,
            'xrd_map': recon[:,:,roi],
            'x_xrd': x[roi],
            'Q': Q,
            }
    return maps
