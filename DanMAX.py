
f"""Methods for notebooks at the DanMAX beamline
"""

version = '3.7.0'

#use_dark_mode = True
import os
import h5py
import glob
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from cycler import cycler
import scipy.optimize as sci_op
#import scipy.constants as sci_const
from IPython.utils import io
from datetime import datetime
try:
    #from ipywidgets import interact
    from pyFAI import geometry
    import pytz
except:
    print('Change to HDF5 server to load all modules')

# Importing DanMAX libraries for specialized functionality
# Done in try statements to catch different jupyter environments
try:
    from lib.misc import *
except:
    print('Unable to load lib/misc.py')
try:
    import lib.mapping as mapping
except:
    print('Unable to load lib/mapping.py')
try:
    import lib.texture as texture
except:
    print('Unable to load lib/texture.py')
try:
    import lib.integration as integration
except:
    print('Unable to load lib/integration.py')
try:
    import lib.parallel as parallel
except:
    print('Unable to load lib/parallel.py')
try:
    import lib.fitting as fitting
except:
    print('Unable to load lib/fitting.py')
try:
    import lib.archiver_danmax as archiver
except:
    print('Unable to load lib/archiver_danmax.py')
try:
    from lib.interactive import InteractiveMask, interactiveImageHist
except:
    print('Unable to load lib/interactive.py')
    raise
    
def pi(engineer=False):
    if engineer:
        return 3.
    return np.pi

def getTimeStamps(ts):
    """Convert an array of absolute utc timestamps to an array of readable string timestamps (yyyy-mm-dd HH:MM:SS.ffffff)"""
    timezone = pytz.timezone('Europe/Stockholm')
    if np.isscalar(ts):
        ts = np.array([ts])
    # set offset (daylight saving) based on first index
    offsetTZ = timezone.localize(datetime.utcfromtimestamp(ts[0])).utcoffset()
    ts = np.array([f'{datetime.utcfromtimestamp(t)+offsetTZ}' for t in ts])
    return ts

def getCurrentProposal(proposal=None, visit=None):
    """Return current proposal number and visit number
    If proposal/visit is provided it will pass it back"""
    
    if proposal != None and visit != None:
        return proposal, visit

    cwd = os.getcwd().replace('gpfs/offline1','data')
    idx = cwd.split('/').index('danmax')
    proposal_new, visit_new =  cwd.split('/')[idx+1:idx+3]
    if proposal == None:
        proposal = proposal_new
    if visit == None:
        visit = visit_new
    return proposal, visit

def getCurrentProposalType(proposal_type=None, beamline=None):
    """Return current proposal type and beamline
    If proposal_type/beamline is provided it will pass it back"""
    
    if proposal_type != None and beamline != None:
        return proposal_type, beamline
    cwd = os.getcwd().replace('gpfs/offline1','data')
    idx = cwd.split('/').index('data')
    proposal_type, beamline =  cwd.split('/')[idx+1:idx+3]
    return proposal_type, beamline

def getProposalScans(proposal_type=None, beamline=None,proposal=None, visit=None):
    """Get scan directory for the current proposal
    """
    proposal_type, beamline = getCurrentProposalType(proposal_type, beamline)
    proposal, visit = getCurrentProposal(proposal, visit)
    
    process_folder = f'/data/{proposal_type}/{beamline}/{proposal}/{visit}/process'
    dictionary_file = f'{process_folder}/sample_dictionary.json'

    if not os.path.isfile(dictionary_file):
        print(f'There is no scan_dictionary, run the dictionary creator first!!!')
        return None

    with open(dictionary_file,'r') as df:
        return json.load(df)

def saveProposalScans(scan_dictionary,proposal_type=None, beamline=None,proposal=None, visit=None):
    """save scan directory for the current proposal
    Takes a dictionary and saves it as a json file
    """

    proposal_type, beamline = getCurrentProposalType(proposal_type, beamline)
    proposal, visit = getCurrentProposal(proposal, visit)
    
    process_folder = f'/data/{proposal_type}/{beamline}/{proposal}/{visit}/process'
    
    if not os.path.isdir(process_folder):
        os.makedirss(process_folder)
        os.chmod(process_folder,0o770)

    dictionary_file = f'{process_folder}/sample_dictionary.json'
    with open(dictionary_file,'w') as df:
        json.dump(scan_dictionary, df)

    
def getLatestScan(scan_type='any',require_integrated=False,proposal=None,visit=None,is_parallel=False,proposal_type='visitors',beamline='danmax'):
    """
    Return the path to the latest /raw/*/*.h5 scan for the provided proposal and visit.
    Defaults to the current proposal directory of proposal and visit are not specified.
    
    Use scan_type (str) to specify which scan type to search for, i.e. 'timescan', 'dscan', 'ascan', etc.
    
    Use require_integrated = True to ensure that the returned scan has a valid integrated .h5 file.
    """
    proposal, visit = getCurrentProposal(proposal,visit)
    proposal_type, beamline = getCurrentProposalType(proposal_type, beamline)
    if is_parallel:
        return parallel.findAllParallel(proposal=None,visit=None)[-1]
    #print(proposal, visit)
    files = sorted(glob.glob(f'/data/{proposal_type}/{beamline}/{proposal}/{visit}/raw/**/*.h5', recursive=True), key = os.path.getctime, reverse=True)
    files = [f for f in files if not ('pilatus.h5' in f or '_falconx.h5' in f or '_orca.h5' in f or '_dxchange.h5' in f)]
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
    print(f"No scan of type '{scan_type}' found in '/data/{proposal_type}/{beamline}/{proposal}/{visit}/raw/'")
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

def getMetaData(fname,custom_keys={},relative=True,proposal=None,visit=None):
    """
    Return dictionary of selected meta data. Return {key:None} if key is not available.
    Use custom_keys to provide a dictionary of custom keys for additional parameters, where 
    the key will be inherited by the returned dictionary and the value is the full .h5 path.
    Example: 
        custom_keys = {'hex_x':'entry/instrument/hex_x/value',
                       'hex_y':'entry/instrument/hex_y/value',
                       }
    
    Default keys:
        I0
        time
        temp
        energy
        
    relative: Bool - Toggle wheteher to return data relative to the specific scan (True) or as absolute values (False). Default: True
    """
    if fname.startswith('scan-'):
        fname = findScan(fname,proposal,visit)

    if 'master.h5' in fname:
        fname = fname.replace('raw', 'process/azint').replace('.h5','_meta.h5')
        
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
                # check if value is a scalar
                if len(f[custom_keys[key]].shape)==0:
                    data[key] = f[custom_keys[key]][()]
                else:
                    data[key] = f[custom_keys[key]][:]
            except KeyError:
                data[key] = None
                print(f'{key} not available')
    return data

def getMetaDic(fname):
    """Return dictionary of available meta data, reusing the .h5 dictionary keys."""
    if 'master.h5' in fname:
        fname = fname.replace('raw', 'process/azint').replace('.h5','_meta.h5')
    data = {}
    with h5py.File(fname,'r') as f:
        for key in f['/entry/instrument/'].keys():
             if key != 'pilatus' and key != 'start_positioners':
                for k in f['/entry/instrument/'][key].keys():
                    data[key] = f['/entry/instrument/'][key][k][:]
    return data


def appendScans(scans,
                xrd_range=None,
                azi_range=None,
                proposal=None,
                visit=None,):
    """
    Return appended arrays of the integrated diffraction data and meta data for several scans
        Parameters
            scans - list of scan numbers
            xrd_range - list/tuple of min/max in scattering direction
            azi_range - list/tuple of min/max in azimuthal direction
            proposal - int, proposal to load from
            vist - int, visit to load from
        Return
            data - dictionary
            meta - dictionary
    """
    for i,scan in enumerate(scans):
        fname = findScan(scan,proposal=proposal,visit=visit)
        aname = getAzintFname(fname)
        metadic = getMetaDic(fname)
        ts = metadic['pcap_trigts']
        
        datadic = getAzintData(aname,xrd_range=xrd_range,azi_range=azi_range,proposal=proposal,visit=visit)
        
        if len(ts) != datadic['I'].shape[0]:
            print(f'Missing metadata in {fname}')
            datadic['I'] = datadic['I'][:len(ts)]
            if type(datadic['cake']) != type(None):
                datadic['cake'] = datadic['cake'][:len(ts)]
        if i<1:
            data = datadic.copy()
            meta = metadic.copy()
        else:
            for key in ['I','cake']:
                if type(datadic[key]) != type(None):
                    data[key] = np.append(data[key],datadic[key],axis=0)
            meta = {key:np.append(meta[key],metadic[key]) for key in metadic}
    return data, meta


def findAllScans(scan_type='any',descending=True,proposal=None,visit=None,proposal_type='visitors',beamline='danmax'):
    """
    Return a sorted list of all scans in the current visit
    Use scan_type (str) to specify which scan type to search for, i.e. 'timescan', 'dscan', 'ascan', etc.
    """
    proposal, visit = getCurrentProposal(proposal,visit)
    proposal_type, beamline = getCurrentProposalType(proposal_type, beamline)
    files = sorted(glob.glob(f'/data/{proposal_type}/{beamline}/{proposal}/{visit}/raw/**/*.h5', recursive=True), key = os.path.basename, reverse=descending)
    files = [f for f in files if not ('pilatus.h5' in f or '_falconx.h5' in f or '_orca.h5' in f or '_dxchange.h5' in f or '_xspress3-dtc-2d' in f)]
    if scan_type != 'any':
        files = [f for f in files if scan_type in getScanType(f)]
    if len(files)<1:
        print(f"No scans of type '{scan_type}' found in '/data/{proposal_type}/{beamline}/{proposal}/{visit}/raw/'")    
    return files


def findScan(scan_id=None,proposal=None,visit=None,is_parallel=False,proposal_type='visitors',beamline='danmax'):
    """Return the path of a specified scan number. If no scan number is specified, return latest scan"""
    if not is_parallel:
        if scan_id == None:
            return getLatestScan()
        elif type(scan_id) == int:
            scan_id = f'scan-{scan_id:04d}'
        elif type(scan_id) == str:
            scan_id = 'scan-'+scan_id.strip().split('scan-')[-1][:4]

        for sc in findAllScans(proposal=proposal,visit=visit,proposal_type='visitors',beamline='danmax'):
            if scan_id in sc:
                return sc
        print('Unable to find {} in {}/{}'.format(scan_id,*getCurrentProposal(proposal,visit)))

    else:
        scans = parallel.findAllParallel(proposal=None,visit=None)
        if scan_id is None:
            index = -1
        elif type(scan_id) == int:
            index = scan_id
        elif type(scan_id) == str:
            if scan_id in scans:
                return scan_id
            print('Unable to find {} in {}/{}'.format(scan_id,*getCurrentProposal(proposal,visit)))
        return scans[index]
    
    
def getScanType(fname,proposal=None,visit=None):
    """Return the scan type based on the .h5 scan title"""
    if fname.startswith('scan-'):
        fname = findScan(fname,proposal=proposal,visit=visit)
    with h5py.File(fname,'r') as f:
        try:
            scan_type = f['entry/title/'][()].decode()
            # clean up special characters
            scan_type = scan_type.replace('(',' ').replace(')',' ').strip("'").replace(',','')
            #print(scan_type)
            return scan_type
        except KeyError:
            print('No entry title available')
            return 'None'

def getExposureTime(fname,proposal=None,visit=None):
    """Return the exposure time in seconds as determined from the scan type"""
    scan_type = getScanType(fname,proposal=proposal,visit=visit)
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
    if 'master.h5' in fname:
        return fname.split('raw/')[-1].split('/master')[0]
    return 'scan-'+fname.strip().split('scan-')[-1][:4]

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
    
def getAverageImage(fname='latest',proposal=None,visit=None):
    """Return the average image of a scan - Default is the latest scan in the current folder"""
    if fname.lower() == 'latest':
        fname = getLatestScan(proposal,visit)
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

def getMotorSteps(fname,proposal=None,visit=None):
    """
    Return motor name(s), nominal positions, and registred positions for a given scan.
        Return list of lists [[motor_name_1,nominal,registred], ...]
    """
    # read meta data dictionary
    dic = getMetaDic(fname)
    # get the scan command containing motor names, exposure, latency, and boolean flags
    # as a list of entries
    scan_type = getScanType(fname).split()
    # remove scan type and all entries without letters
    motors = [s for s in scan_type if s.lower().islower()][1:]
    # remove boolean flags
    motors = [m for m in motors if not 'false' in m.lower() and not 'true' in m.lower()]
    motor_steps = []
    #print(motors)
    for motor in motors:
        # get the nominal motor position from the macro title
        #print(motor)
        start, stop, steps = [scan_type[i+1:i+4] for i,s in enumerate(scan_type) if motor in s][0]
        nominal_pos = np.linspace(float(start),float(stop),int(steps)+1)
        # get the logged motor position
        motor_entry_id = [key for key in dic.keys() if motor in key][0]
        motor_pos = dic[motor_entry_id]
        #motor_pos = np.unique(motor_pos)
        # compare nominal and actual motor positions
        if not len(nominal_pos)==len(motor_pos) or not np.all(nominal_pos == motor_pos):
            print(f'The nominal and registred motor positions for {motor} do not match!')
        motor_steps.append([motor,nominal_pos,motor_pos])
    
    return motor_steps
    
def getPixelCoords(pname,danmax_convention=True,corners=False):
    """
    Return the pixel coordinates in meter for a given PONI configuration
    
    Parameters:
        pname             - PONI file path
        danmax_convention - (default=True) return the coordinates in the danmax laboratory convention,
                            otherwise use the default PyFAI convention
        corners           - (default=False) If True, return the coordinates of the pixel corners rather then the centers
    Return:
        xyz - (danmax) numpy array of shape [(x,y,z),h,w] - Index [:,0,0] corresponds to the top left corner of the detector image
        OR
        xyz - (danmax,corners) numpy array of shape [(x,y,z),h+1,w+1] - Index [:,0,0] corresponds to the top left corner of the detector image
        OR
        xyz - (PyFAI) numpy array of shape [h,w,(z,y,x)] *see help(pyFAI.geometry.core.Geometry.position_array)
    """
    # read PONI and get xyz positions (in m)
    poni = geometry.core.ponifile.PoniFile(pname).as_dict()
    geo = geometry.core.Geometry(**{key:poni[key] for key in ['dist', 'poni1', 'poni2', 'rot1', 'rot2', 'rot3','detector','wavelength']}) 
    xyz = geo.position_array(corners=corners)
    if danmax_convention:
        if not corners:
            # convert to danmax convention NOTE: Might need to be flipped/rotated
            # [h,w,(z,y,x)] --> [(x,y,z),h,w] 
            xyz = np.transpose(xyz,axes=(2,0,1))[[2,1,0],:,:]
            xyz[[0,1]] *= -1
        else:
            # [h,w,4,(z,y,x)] --> [(x,y,z),(ul,ll,ur,lr),h,w] 
            xyz_c = np.transpose(xyz,axes=(3,2,0,1))[[2,1,0],:,:,:]
            xyz_c[[0,1]] *= -1

            #[(x,y,z),h+1,w+1] 
            xyz = np.full((3,xyz_c.shape[2]+1,xyz_c.shape[3]+1)
                        ,0.)
            # upper left
            xyz[:,:-1,:-1] = xyz_c[:,0,:,:]
            # lower left
            xyz[:,-1,:-1] = xyz_c[:,1,-1,:]
            # upper rigth
            xyz[:,:-1,-1] = xyz_c[:,2,:,-1]
            # lower right
            xyz[:,-1,-1] = xyz_c[:,3,-1,-1]
    return xyz

def getAzintData(fname,
                 get_meta = False,
                 xrd_range = None,
                 azi_range = None,
                 proposal = None,
                 visit = None):
    '''
        Return azimuthally integrated data for a specified file path. File name can be either the /raw/**/master.h5 or the /process/azint/**/*_pilatus_integrated_master.h5
        Dictionary entries for missing or irrelevant fields are set to None
        get_meta makes the function return the integration settings as a dictionary
            parameters:
                fname - string
                get_meta - bool flag (default = False)
                xrd_range - list/tuple of lower and upper radial scattering axis limit (default = None)
                azi_range - list/tuple of lower and upper azimuthal scattering axis limit(default = None)
                proposal - int (default = None)
                visit - int (default = None)
            return:
                data - dictionary
                meta - dictionary
    '''
    # get the azimuthally integrated filename from provided file name,
    if type(fname) == int:
        fname = findscan(fname,proposal=proposal,visit=visit)
    if '/raw/' in fname:
        aname = getAzintFname(fname)
    else:
        aname = fname
        fname = aname.replace('process/azint','raw').replace('_pilatus_integrated.h5','.h5')
    # define output dictionary
    data = {
        'I': None,
        'cake': None,
        'q': None,
        'tth': None,
        'azi': None,
        'q_edge': None,
        'tth_edge': None,
        'azi_edge': None,
        'I_error': None,
        'cake_error': None,
        }
    
    # define h5 data group names
    group_1D = 'entry/data1d'
    group_2D = 'entry/data2d'
    group_meta = 'entry/azint'
    
    # define key reference dictionary
    data_keys = {
        'I'         : f'{group_1D}/I',
        'cake'      : f'{group_2D}/cake',
        'q'         : f'{group_1D}/q',
        'tth'       : f'{group_1D}/2th',
        'azi'       : f'{group_2D}/azi',
        'q_edge'    : f'{group_1D}/q_edges',
        'tth_edge'  : f'{group_1D}/tth_edges',
        'azi_edge'  : f'{group_2D}/azi_edges',
        'I_error'   : f'{group_1D}/I_error',
        'cake_error': f'{group_2D}/cake_error',
        'norm'      : f'{group_2D}/norm',
        }
    
    # define key reference dictionary for old data format
    data_keys_old = {
        'I':        'I',
        'cake':     'I',
        'q':        'q',
        'tth':      '2th',
        'azi':      'phi',
        'azi_edge': 'bin_bounds',
        }


    #define range keys
    range_keys = {
        'q':'xrd',
        'tth':'xrd',
        'azi':'azi' 
        }

    # define range dictionary
    ranges = {
        'xrd': xrd_range,
        'azi': azi_range,
        }

    #define initial rois
    rois = {
        'xrd': np.s_[:], 
        'azi': np.s_[:],
        }

    meta = {}

    # read the *_pilatus_integrated.h5 file
    with h5py.File(aname,'r') as af:
        if 'entry' in af.keys():
            ### NeXus format ###
            if get_meta:
                # read meta data
                for key in af[group_meta].keys():
                    if isinstance(af[group_meta][key],h5py.Dataset):
                        # check if value is a scalar
                        if len(af[group_meta][key].shape)==0:
                            meta[key] = af[group_meta][key][()]
                        else:
                            meta[key] = af[group_meta][key][:]
                    else:
                        meta[key] = {}
                        for subkey in af[group_meta][key].keys():
                            if len(af[group_meta][key][subkey].shape)==0:
                                meta[key][subkey] = af[group_meta][key][subkey][()]
                            else:
                                meta[key][subkey] = af[group_meta][key][subkey][:]

                # check for poni filename attribute
                if 'filename' in af[group_meta]['input/poni'].attrs.keys():
                    meta['input']['poni_filename'] = af[group_meta]['input/poni'].attrs['filename']

            
            # update needed rois
            for key in range_keys.keys():
                if data_keys[key] in af:
                    data_key = key
                    key = range_keys[key]

                    # Redifine the rois to a range, hdf5 doesn't take 2 boolean lists.
                    if ranges[key] is not None:
                        rois[key] = (af[data_keys[data_key]][:] > ranges[key][0]) & (af[data_keys[data_key]][:] < ranges[key][1])
                        for i in range(len(rois[key])):
                            if rois[key][i]:
                                if i == 0:
                                    i = None
                                break
                        for j in range(-1,-len(rois[key]),-1):
                            if rois[key][j]:
                                j=j+1
                                if j == 0:
                                    j = None
                                break
                        rois[key] = np.s_[i:j]
            # read integrated data
            for key in data.keys():
                if data_keys[key] in af:
                    # check if value is a scalar
                    if len(af[data_keys[key]].shape)==0:
                        data[key] = af[data_keys[key]][()]
                    elif len(af[data_keys[key]].shape)>2:
                        data[key] = af[data_keys[key]][:,rois['azi'],rois['xrd']]
                    elif len(af[data_keys[key]].shape)>1:
                        data[key] = af[data_keys[key]][:,rois['xrd']]
                    elif  key in range_keys.keys() and  ranges[range_keys[key]] is not None: 
                       data[key] = af[data_keys[key]][rois[range_keys[key]]]
                    else:
                        data[key] = af[data_keys[key]][:]
        else:
            # update needed rois
            for key in range_keys.keys():
                if data_keys_old[key] in af:
                    data_key = key
                    key = range_keys[key]
                    if ranges[key] is not None:
                        rois[key] = (af[data_keys[data_key]] > ranges[key][0]) & (af[data_keys[data_key]] < ranges[key][1])

                        for i in range(len(rois[key])):
                            if rois[key][i]:
                                if i == 0:
                                    i = None
                                break
                        for j in range(-1,-len(rois[key]),-1):
                            if rois[key][j]:
                                j=j+1
                                if j == 0:
                                    j = None
                                break
                        rois[key] = np.s_[i:j]

            ### old format ###,
            for key in data_keys_old.keys():
                # differentiate between 1D and 2D data and update dictionary key accordingly
                if data_keys_old[key] == 'I':
                    if len(af[data_keys_old[key]].shape) > 2:
                        key = 'cake'
                    else:
                        key = 'I'
                # check if the current key is available in the file
                if data_keys_old[key] in af:
                    # check if value is a scalar
                    if len(af[data_keys_old[key]].shape)==0:
                        data[key]=af[data_keys_old[key]][()]
                    elif len(af[data_keys_old[key]].shape) > 2:
                        data[key]=af[data_keys_old[key]][:,rois['azi'],rois['xrd']]
                    elif len(af[data_keys_old[key]].shape) > 1:
                        data[key]=af[data_keys_old[key]][:,rois['xrd']]
                    elif  key in range_keys.keys() and  ranges[range_keys[key]] is not None: 
                       data[key] = af[data_keys[key]][rois[range_keys[key]]]
                    else:
                        data[key]=af[data_keys_old[key]][:]
            if get_meta:
                # read remaining entries as meta data
                for key in af.keys():
                    if key not in data_keys_old.values():
                        # check if value is a scalar
                        if len(af[key].shape)==0:
                            meta[key]=af[key][()]
                        else:
                            meta[key]=af[key][:]
    # Add bin edges if they are not included.
    for edge_key in ['q', 'tth', 'azi']:
        #Check if the slot was filled in in the first place
        if (type(data[f'{edge_key}']) != type(None)) and (type(data[f'{edge_key}_edge']) == type(None)):
            data[f'{edge_key}_edge'] = data[f'{edge_key}']

            bin_width = np.nanmean(np.abs(np.diff(data[f'{edge_key}'])))
            #print(f'Generating edges for {edge_key}, assuming an equidistant bin width of: {bin_width:.6f} unit({edge_key})')
            data[f'{edge_key}_edge'] = np.append(data[f'{edge_key}_edge'],data[f'{edge_key}_edge'][-1]+bin_width)
            data[f'{edge_key}_edge'] -= bin_width/2

    meta_length = None
    if os.path.isfile(fname):
        keys=None
        try:
            with h5py.File(fname,'r') as f:
                keys = sorted(list(f['/entry/instrument'].keys()))
                meta_length = f['/entry/instrument'][keys[0]+'/data'].shape[0]
        except KeyError:
            if keys is None:
                print(f'Unable to access {fname}::/entry/instrument/../')
            else:
                print(f'Unable to access {fname}::/entry/instrument/{keys[0]}/data')

    if data['I'][:meta_length].shape[0]<data['I'].shape[0] and not meta_length is None:
        print(f'Data size mismatch - cropped to {meta_length} frames')
    
    data['I'] = data['I'][:meta_length]
    if not data['cake'] is None:
        data['cake'] = data['cake'][:meta_length]
    
    if get_meta:
        return data, meta
    else:
        return data

       
def darkMode(use=True,style_dic={'figure.figsize':'small'}):
    """
    Toggle between dark and light mode styles for matplotlib figures.
    Use style_dic to quickly customize the styles. (see plt.rcParams for available keys)
    Use style_dic={'figure.figsize':'small'/'medium'/'large'} to quickly change figure size
    Use plt.style.use('default') to revert to the matplotlib default style
    """
    # dictionary of short hand keywords
    short_hand = {'size' :'figure.figsize',
                  'grid' : 'axes.grid'}
    # replace short hand keys with full name keywords
    keys = list(style_dic.keys())
    for key in keys:
        if key in short_hand.keys():
            style_dic[short_hand[key]]=style_dic[key]
            style_dic.pop(key)

    if 'figure.figsize' in style_dic and type(style_dic['figure.figsize']) == str:
        if style_dic['figure.figsize'].lower() == 'small':
            style_dic['figure.figsize'] = [8.533,4.8]
        elif style_dic['figure.figsize'].lower() == 'medium':
            style_dic['figure.figsize'] = [10.267, 5.775]
        elif style_dic['figure.figsize'].lower() == 'large':
            style_dic['figure.figsize'] = [12, 6.75]
    
    light_mode = {'axes.formatter.use_mathtext': True,
                 'axes.formatter.useoffset': False,
                 'axes.xmargin': 0.02,
                 'axes.ymargin': 0.02,
                 'axes.grid': True,
                 'font.family': 'DejaVu Sans Mono',
                 'xtick.minor.visible': True,
                 'ytick.minor.visible': True,
                 'lines.linewidth': 1.,
                 'lines.markersize':3.,
                 'axes.prop_cycle': cycler('color', ['#1f77b4', 
                                                     '#ff7f0e', 
                                                     '#2ca02c', 
                                                     '#d62728', 
                                                     '#9467bd', 
                                                     '#8c564b', 
                                                     '#e377c2', 
                                                     '#7f7f7f', 
                                                     '#bcbd22', 
                                                     '#17becf',
                                                     '#000000',
                                                     '#000066',
                                                     '#CC9900',
                                                     '#006600',
                                                     '#660000',
                                                    ]),
                'figure.autolayout': True,
                'figure.dpi': 100.0,
                }
    
    dark_mode = {'axes.facecolor': '#1a1a1a',
                 'axes.edgecolor': '#E0E0E0',
                 'axes.labelcolor': '#E0E0E0',
                 'text.color': '#E0E0E0',
                 'xtick.color': '#E0E0E0',
                 'ytick.color': '#E0E0E0',
                 'grid.color': '#2a2a2a',
                 'figure.facecolor': '#1a1a1a',
                 'figure.edgecolor': '#1a1a1a',
                 'savefig.facecolor': '#1a1a1a',
                 'savefig.edgecolor': '#1a1a1a',
                 'legend.edgecolor':  '#2a2a2a',
                 'axes.prop_cycle': cycler('color', ['#1f77b4', 
                                                     '#ff7f0e', 
                                                     '#2ca02c', 
                                                     '#d62728', 
                                                     '#9467bd', 
                                                     '#8c564b', 
                                                     '#e377c2', 
                                                     '#7f7f7f', 
                                                     '#bcbd22', 
                                                     '#17becf',
                                                     '#FFFFFF',
                                                     '#000066',
                                                     '#CC9900',
                                                     '#006600',
                                                     '#660000',
                                                    ])
                }
    
    light_mode.update(style_dic)
    plt.style.use('default')
    plt.style.use(light_mode)
    if use:
        plt.style.use(dark_mode)
    return plt.rcParams

def lightMode(use=True,style_dic={}):
    """
    Toggle between light and dark mode styles for matplotlib figures.
    Use style_dic to quickly customize the styles. (see plt.rcParams for available keys)
    Use style_dic={'figure.figsize':'small'/'medium'/'large'} to quickly change figure size
    Use plt.style.use('default') to revert to the matplotlib default style
    """
    return darkMode(not use,style_dic={})

#darkMode(use_dark_mode)

#if use_dark_mode:
#    print(f'DanMAX.py Version {version} - Dark mode')
#else:
#    print(f'DanMAX.py Version {version}')
print(f'DanMAX.py Version {version}')
