"""DanMAX I/O module for reading azimuthally integrated data from NeXus files."""
import numpy as np
import h5py as h5
import os
import sys
sys.path.append('../')
#from DanMAX import DM.findScan, DM.getAzintFname
import DanMAX as DM


def getAzintData_legacy(fname,
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
        fname = DM.findScan(fname,proposal=proposal,visit=visit)
    if '/raw/' in fname:
        aname = DM.getAzintFname(fname)
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
    with h5.File(aname,'r') as af:
        if 'entry' in af.keys():
            ### NeXus format ###
            if get_meta:
                # read meta data
                for key in af[group_meta].keys():
                    if isinstance(af[group_meta][key],h5.Dataset):
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
            with h5.File(fname,'r') as f:
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
    

def getAzintData(fname,
                get_meta = False,
                xrd_range = None,
                azi_range = None,
                proposal = None,
                visit = None):
    
    # get the azimuthally integrated filename from provided file name,
    if type(fname) == int:
        fname = DM.findScan(fname,proposal=proposal,visit=visit)
    if '/raw/' in fname:
        aname = DM.getAzintFname(fname)
    else:
        aname = fname
        fname = aname.replace('process/azint','raw').replace('_pilatus_integrated.h5','.h5')

    # determine the file format
    with h5.File(aname,'r') as af:
        is_nxazint = isinstance(_find_entry(af,definition='NXazint1d'),h5.Group)
    
    ### NEW NeXus FORMAT ###
    if is_nxazint:
        # read the nxazint data
        if xrd_range is None:
            xrd_range = [None,None]
        if azi_range is None:
            azi_range = [None,None]
        azint1d = Azint1d(aname,radial_range=xrd_range)
        azint2d = Azint2d(aname,radial_range=xrd_range,azi_range=azi_range)

        # assign data to dictionary
        data_keys = {
                    'I'         : 'I',
                    'cake'      : 'I',
                    'q'         : 'Q',
                    'tth'       : 'tth',
                    'azi'       : 'eta',
                    'q_edge'    : 'Q_edges',
                    'tth_edge'  : 'tth_edges',
                    'azi_edge'  : 'eta_edges',
                    'I_error'   : 'I_errors',
                    'cake_error': 'I_errors',
                    }
        data = {key:None for key in data_keys.keys()}
        for key in data_keys.keys():
            if data_keys[key] in azint1d.__dict__.keys() and not key in ['cake','cake_error']:
                data[key] = getattr(azint1d, data_keys[key])
            if data_keys[key] in azint2d.__dict__.keys() and not key in ['I','I_error'] and data[key] is None:
                data[key] = getattr(azint2d, data_keys[key])
        
        # Add bin edges if they are not included.
        for edge_key in ['q', 'tth', 'azi']:
            #Check if the slot was filled in in the first place
            if not data[f'{edge_key}'] is None and data[f'{edge_key}_edge'] is None:
                data[f'{edge_key}_edge'] = data[f'{edge_key}']
                bin_width = np.nanmean(np.abs(np.diff(data[f'{edge_key}'])))
                #print(f'Generating edges for {edge_key}, assuming an equidistant bin width of: {bin_width:.6f} unit({edge_key})')
                data[f'{edge_key}_edge'] = np.append(data[f'{edge_key}_edge'],data[f'{edge_key}_edge'][-1]+bin_width)
                data[f'{edge_key}_edge'] -= bin_width/2

        # assign meta data to dictionary - TEMPORARY, future versions should use an object oriented approach
        meta = {}
        # get all public attributes of the monochromator group
        for key in azint1d.monochromator.__dict__.keys():
            if not key.startswith('_'):
                meta[key] = getattr(azint1d.monochromator, key)
        # get all public attributes of the source group
        for key in azint1d.source.__dict__.keys():
            if not key.startswith('_'):
                meta[key] = getattr(azint1d.source, key)
        # get all public attributes of the process group
        for key in azint1d.process.__dict__.keys():
            if not key.startswith('_'):
                meta[key] = getattr(azint1d.process, key)
        # get all public attributes of the parameters group
        for key in azint1d.parameters.__dict__.keys():
            if not key.startswith('_'):
                meta[key] = getattr(azint1d.parameters, key)

        if get_meta:
            return data, meta
        return data

    ### OLD FORMAT ###
    else:
        return getAzintData_legacy(fname,
                                get_meta = get_meta,
                                xrd_range = xrd_range,
                                azi_range = azi_range,
                                proposal = proposal,
                                visit = visit)



def _read_units(dataset):
    """Read the units of an open h5 dataset."""
    unit = None
    if 'units' in dataset.attrs:
        unit = dataset.attrs['units']
    return unit

def _read_long_name(dataset):
    """Read the long_name attribute of an open h5 dataset."""
    long_name = ''
    if 'long_name' in dataset.attrs:
        long_name = dataset.attrs['long_name']
    return long_name

def _read_nxtype(dataset):
    """Read the type attribute of an open h5 dataset."""
    nx_type = ''
    if 'type' in dataset.attrs:
        nx_type = dataset.attrs['type']
    return nx_type

def _read_nxclass(group):
    """Read the NX_class attribute of an open h5 group."""
    nx_class = ''
    if not isinstance(group, h5.Group):
        return nx_class
    if 'NX_class' in group.attrs:
        nx_class = group.attrs['NX_class']
    return nx_class

def _find_group(group, name):
    """ Recursively find a group with a specified name in a group."""
    if name in group:
        return group[name]
    if isinstance(group, h5.Dataset):
        return None
    for key in group.keys():
        if isinstance(group[key], h5.Group):
            return _find_group(group[key], name)
    return None

def _find_group_from_class(group, nx_class):
    """ Recursively find a group with a specified NX_class in a group."""
    for key in group.keys():
        if isinstance(group[key], h5.Group) and _read_nxclass(group[key]) == nx_class:
            return group[key]
    if isinstance(group, h5.Dataset):
        return None
    #print(group)
    for key in group.keys():
        #print(key)
        if isinstance(group[key], h5.Group):
            return _find_group_from_class(group[key], nx_class)
    return None

def _find_entry(f,definition='NXazint1d'):
    """Find the NXentry group with a specified definition in a file. (NXazint1d or NXazint2d)"""
    # find the NXentry group
    entry = _find_group(f, 'entry')
    if entry is None:
        return None
        # raise ValueError('No NXentry group found in the file.')
    
    # check for subentries
    subentries = []
    for key in entry.keys():
        if isinstance(entry[key], h5.Group) and _read_nxclass(entry[key]) == 'NXsubentry':
            subentries.append(entry[key])
    if len(subentries) < 1:
        subentries = [entry]
    
    for subentry in subentries:
        if 'definition' in subentry and subentry['definition'][()].decode() == definition:
            return subentry
    return None
    #raise ValueError(f'No NXentry group with definition {definition} found in the file.')


class Monochromator():
    """
    Monochromator object containing attributes read from an NXmonochromator.
    Assumes energy in keV and wavelength in Angstroms. If one is given, the other
    is calculated from the given value.
    """
    def __init__(self, mono_gr):
        self.energy = None
        self.wavelength = None
        self.read_monochromator_attributes(mono_gr)

    def read_monochromator_attributes(self, mono_gr):
        # read the attributes from the monochromator group
        for key in mono_gr.keys():
            if key.lower() in ['energy', 'e']:
                self.energy = mono_gr[key][()]
                unit = _read_units(mono_gr[key])
                # check if the unit is keV, convert if eV
                if unit is not None and unit.lower() == 'ev':
                    self.energy /= 1000
            elif key.lower() in ['wavelength', 'l','lambda']:
                self.wavelength = mono_gr[key][()]
                unit = _read_units(mono_gr[key])
                # check if the unit is Angstroms, convert if nm
                if unit is not None and unit.lower() == 'nm':
                    self.wavelength *= 10
        # if either energy or wavelength is None, convert it from the other
        if self.energy is None and self.wavelength is not None:
            self.energy = 12.398 / self.wavelength
        elif self.wavelength is None and self.energy is not None:
            self.wavelength = 12.398 / self.energy
        elif self.energy is None and self.wavelength is None:
            raise ValueError('Both energy and wavelength cannot be None.')
    
    def __str__(self):
        return f'Monochromator:\n   Energy: {self.energy}\n   Wavelength: {self.wavelength}'

class Parameters():
    """
    Parameters object containing attributes read from an NXparameters group.
    """
    def __init__(self, param_gr):
        self.read_parameters(param_gr)

    def read_parameters(self, param_gr):
        # read the attributes from the parameters group
        for key in param_gr.keys():
            # check that the attribute is not a group
            if isinstance(param_gr[key], h5.Dataset):
                # decode the byte string to a string
                if isinstance(param_gr[key][()], bytes):
                    setattr(self, key, param_gr[key][()].decode())
                else:
                    setattr(self, key, param_gr[key][()])
    def __str__(self):
        return f'Parameters:\n   {self.__dict__}'

class Source():
    """
    Source object containing attributes read from an NXsource group.
    """
    def __init__(self, source_gr):
        self.read_source(source_gr)

    def read_source(self, source_gr):
        # read the attributes from the source group
        for key in source_gr.keys():
            # check that the attribute is not a group
            if isinstance(source_gr[key], h5.Dataset):
                # decode the byte string to a string
                if isinstance(source_gr[key][()], bytes):
                    setattr(self, key, source_gr[key][()].decode())
                else:
                    setattr(self, key, source_gr[key][()])
    def __str__(self):
        return f'Source:\n   {self.__dict__}'
    
class Process():
    """
    Process object containing attributes read from an NXprocess group.
    """
    def __init__(self, process_gr):
        self.read_process(process_gr)

    def read_process(self, process_gr):
        # read the attributes from the process group
        for key in process_gr.keys():
            # check that the attribute is not a group
            if isinstance(process_gr[key], h5.Dataset):
                # decode the byte string to a string
                if isinstance(process_gr[key][()], bytes):
                    setattr(self, key, process_gr[key][()].decode())
                else:
                    setattr(self, key, process_gr[key][()])
    def __str__(self):
        return f'Process:\n   {self.__dict__}'

class Data():
    """
    Data object containing attributes read from an NXdata group.
    """
    def __init__(self, data_gr=None, index_range=[None, None], radial_range=[None, None], azi_range=[None, None]):
        self.I = None           # intensity
        self.I_errors = None    # intensity errors
        self.radial_axis = None # radial axis (Q or 2theta)
        self.Q = None           # scattering vector
        self.tth = None         # 2theta
        self.eta = None         # azimuthal angle
        self.eta_edges = None   # azimuthal angle edges
        self.shape = None       # shape of the data
        self.dtype = None       # data type
        self.is_Q = False       # flag for Q axis

        # dictionary of possible names for the data variables
        # based on long names and units
        self.__variable_identifiers = {'I':{'long_name':['intensity','i'],
                                           'units':['counts','cps','intensity','arb','arb. units','a.u.']},
                                       'Q':{'long_name':['scattering vector','q'],
                                           'units':['1/A', 'A-1', '1/angstrom', '1/nm', 'nm-1']},
                                       'tth':{'long_name':['2theta', 'tth', '2θ', 'two theta'],
                                           'units':['degrees', 'degree','deg','°']},
                                       'eta':{'long_name':['eta', 'azimuthal angle', 'phi', 'azimuth', 'azi'],
                                               'units':['degrees', 'degree','deg','°']},
                                       }
        if data_gr is not None:
            dset = self.find_signal_dset(data_gr)
            self.read_axes(data_gr)
            index_slice, radial_slice, azi_slice = self.get_slices(index_range, radial_range, azi_range)
            for ax in ['radial_axis','Q','tth']:
                if hasattr(self, ax) and getattr(self, ax) is not None:
                    setattr(self, ax, getattr(self, ax)[radial_slice])
            for ax in ['eta','eta_edges']:
                if hasattr(self, ax) and getattr(self, ax) is not None:
                    setattr(self, ax, getattr(self, ax)[azi_slice])
            self.read_signal(data_gr, dset, index_slice, radial_slice, azi_slice)
            
    def find_signal_dset(self, data_gr):
        # check if the data group has a signal attribute
        if 'signal' in data_gr.attrs:
            signal = data_gr.attrs['signal']
            if signal in data_gr:
                dset = data_gr[signal]
        else:
            dset = self._find_variable(data_gr, 'I')

        self.shape = dset.shape
        self.dtype = dset.dtype
        return dset


    def read_signal(self, data_gr, dset, index_slice, radial_slice, azi_slice):
        """Read the intensity data from the NXdata group."""
        
        # read the data
        if not dset is None:
            if len(dset.shape) == 1:
                self.I = dset[index_slice]
                if dset.name+'_errors' in data_gr:
                    self.I_errors = data_gr[dset.name+'_errors'][index_slice]
            elif len(dset.shape) == 2:
                self.I = dset[index_slice,radial_slice]
                if dset.name+'_errors' in data_gr:
                    self.I_errors = data_gr[dset.name+'_errors'][index_slice,radial_slice]
            elif len(dset.shape) == 3:
                self.I = dset[index_slice,azi_slice,radial_slice]
                if dset.name+'_errors' in data_gr:
                    self.I_errors = data_gr[dset.name+'_errors'][index_slice,azi_slice,radial_slice]
            self.shape = self.I.shape
            self.dtype = self.I.dtype
            if dset.name+'_errors' in data_gr:
                self.I_errors = data_gr[dset.name+'_errors'][:]
            return

        raise ValueError('No intensity data found in the NXdata group.')

    def read_axes(self, data_gr):
        """Read the axes data from the NXdata group."""

        # check which axes to expect based on the axes attr and the number of dimensions of the intensity data
        axes_shape = list(self.shape)
        axes = []
        if 'axes' in data_gr.attrs:
            axes = data_gr.attrs['axes']
            if isinstance(axes, str):
                axes = [axes]
        # if no axes attribute is found, generate generic names for the axes
        if not len(axes) == len(axes_shape):
            axes = [f'ax{i}' for i in range(len(axes_shape))]

        # try to match the axes to the expected names
        assigned_dsets = []
        for var in self.__variable_identifiers:
            dset = self._find_variable(data_gr, var)
            if not dset is None:
                if var == 'Q':
                    self.is_Q = True
                setattr(self, var, dset[:])
                assigned_dsets.append(dset)
                axes_shape.remove(dset.shape[0])
                if dset.name+'_edges' in data_gr:
                    setattr(self, var+'_edges', data_gr[dset.name+'_edges'][:])

        # identify missing variables
        missing_vars = [var for var in self.__variable_identifiers if getattr(self, var) is None]
        # look through the remaining datasets to find the missing variables
        # based on units and shape
        for dset in data_gr.keys():
            if isinstance(data_gr[dset], h5.Dataset) and data_gr[dset] not in assigned_dsets and len(data_gr[dset].shape) == 1:
                for var in missing_vars:
                    # check both units and shape
                    units = _read_units(data_gr[dset])
                    if not units is None and units.lower() in self.__variable_identifiers[var]['units']:
                        if data_gr[dset].shape[0] in axes_shape:
                            setattr(self, var, data_gr[dset][:])
                            assigned_dsets.append(data_gr[dset])
                            missing_vars.remove(var)
                            axes_shape.remove(data_gr[dset].shape[0])
                            if var == 'Q':
                                self.is_Q = True
                            if dset+'_edges' in data_gr:
                                setattr(self, var+'_edges', data_gr[dset+'_edges'][:])
                            break
        # as a last resort, check only based on shape
        # assume that if either Q or 2theta is present, the other is not
        if not 'tth' in missing_vars:
            missing_vars.remove('Q')
        elif not 'Q' in missing_vars:
            missing_vars.remove('tth')
        
        for dset in data_gr.keys():
            if isinstance(data_gr[dset], h5.Dataset) and data_gr[dset] not in assigned_dsets and len(data_gr[dset].shape) == 1:
                for var in missing_vars:
                    if data_gr[dset].shape[0] in axes_shape:
                        setattr(self, var, data_gr[dset][:])
                        assigned_dsets.append(data_gr[dset])
                        missing_vars.remove(var)
                        axes_shape.remove(data_gr[dset].shape[0])
                        if var == 'Q':
                            self.is_Q = True
                        if dset+'_edges' in data_gr:
                            setattr(self, var+'_edges', data_gr[dset+'_edges'][:])
                        break
        if self.is_Q:
            self.radial_axis = self.Q
        else:
            self.radial_axis = self.tth

    def get_slices(self, index_range=[None, None], radial_range=[None, None], azi_range=[None, None]):
        """Get the slices for the intensity data based on the given ranges."""
        if index_range[0] is not None:
            index_range[0] = max(0, index_range[0])
        if index_range[1] is not None:
            index_range[1] = min(self.shape[0], index_range[1])
        if radial_range[0] is not None:
            radial_range[0] = np.argmax(self.radial_axis >= radial_range[0])
        if radial_range[1] is not None:
            radial_range[1] = np.where(self.radial_axis <= radial_range[1])[0][-1] + 1
        if azi_range[0] is not None:
            azi_range[0] = np.argmax(self.eta >= azi_range[0])
        if azi_range[1] is not None:
            azi_range[1] = np.where(self.eta <= azi_range[1])[0][-1] + 1

        index_slice = np.s_[index_range[0]:index_range[1]]
        rad_slice = np.s_[radial_range[0]:radial_range[1]]
        azi_slice = np.s_[azi_range[0]:azi_range[1]]
        return index_slice, rad_slice, azi_slice
    
    def _find_variable(self, data_gr, variable):
        """Find a variable in the data group based on the variable identifiers."""
        # check if the variable is already in the group
        if variable in data_gr:
            return data_gr[variable]
        # check for the variable based on the long name
        for dset in data_gr.keys():
            if isinstance(data_gr[dset], h5.Dataset):
                long_name = _read_long_name(data_gr[dset])
                if not long_name is None and long_name.lower() in self.__variable_identifiers[variable]['long_name']:
                    return data_gr[dset]
        return None

class Azint1d():
    """Azint1d object containing attributes read from an NXazint1d group."""
    definition='NXazint1d'
    def __init__(self, fname,definition=definition, index_range=[None, None], radial_range=[None, None], azi_range=[None, None]):
        self.fname = fname
        with h5.File(self.fname) as f:
            entry = _find_entry(f, definition=definition)
            if entry is None:
                for key in Data().__dict__.keys():
                    setattr(self, key, None)
                return
            instr = _find_group_from_class(entry, 'NXinstrument')
            mono = _find_group_from_class(instr, 'NXmonochromator')
            src = _find_group_from_class(instr, 'NXsource')
            proc = _find_group_from_class(entry, 'NXprocess')
            params = _find_group_from_class(proc, 'NXparameters')
            d = _find_group_from_class(entry, 'NXdata')
            # instr = _find_group(entry, 'instrument')
            # mono = _find_group(instr, 'monochromator')
            # src = _find_group(instr, 'source')
            # proc = _find_group(entry, 'process')
            # params = _find_group(proc, 'parameters')
            # d = _find_group(entry, 'data')

            self.monochromator = Monochromator(mono)
            self.source = Source(src)
            self.process = Process(proc)
            self.parameters = Parameters(params)
            data = Data(d, index_range=index_range.copy(), radial_range=radial_range.copy(), azi_range=azi_range.copy())
            # add all public attributes of the data object to the Azint1d object
            for key in data.__dict__.keys():
                if not key.startswith('_'):
                    setattr(self, key, data.__dict__[key])

class Azint2d(Azint1d):
    """Azint2d object containing attributes read from an NXazint2d group."""
    definition='NXazint2d'
    def __init__(self, fname,definition=definition,index_range=[None, None], radial_range=[None, None], azi_range=[None, None]):
        super().__init__(fname,definition=definition, index_range=index_range, radial_range=radial_range, azi_range=azi_range)

