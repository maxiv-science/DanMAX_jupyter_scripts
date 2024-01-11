"""
DanMAX Azimuthal integration functions for Jupyter notebooks

Use integrateFile() or integrateImages() in conjunction with writeAzintFile() to save the integrated data
in a .h5 file. Explicitedly specify aname in writeAzintFile() to avoid overwriting existing files.

Pass the ai_1d and/or ai_2d integrator objects to the integration call in loops to avoid re-initializing new integrators.

Example:
    import DanMAX as DM
    from integration import integrateFile, writeAzintFile
    
    scans = [1234,1235,1236]
    
    poni = '/data/visitors/danmax/PROPOSAL/VISIT/process/PONI_FILE.poni'
    mask = '/data/visitors/danmax/PROPOSAL/VISIT/process/MASK_FILE.npy'

    config = {'poni'   : poni,
              'mask'        : mask,
              'radial_bins' : 3000,
              'azimuth_bins': None,
              'n_splitting' : 15,
              'polarization_factor': 0.999997,
              'unit'        : '2th',
             }

    ai_1d, ai_2d = None, None
    for scan in scans:
        fname = DM.findScan(scan)
        aname = fname.replace('/raw/', '/process/azint/reintegrated/').split('.')[0]+'_pilatus_integrated.h5'
        data, meta, ai_1d, ai_2d = integrateFile(fname,config, ai_1d=ai_1d, ai_2d=ai_2d)
        writeAzintFile(aname,data,meta=meta)


"""

import os
import numpy as np
import azint
from azint import AzimuthalIntegrator
import fabio
from datetime import datetime
import h5py as h5
from importlib.metadata import version

def integrateFile(fname, config, ai_1d=None, ai_2d=None, im_corr=None):
    """
    DanMAX integration function for a series of .h5 files. 
    Uses the python implementation of MATFRAIA to azimuthally integrate an image
    Parameters:
        
        fname                  : File name
        config = {
            poni               : Absolute path for the poni file 
            mask               : Absolute path for the mask file in .npy format 
            radial_bins        : Number of radial bins
            azimuthal_bins     : Number of azimuthal bins - See note below 
            unit               : "q" or "2th" 
            n_splitting        : Number of sub-pixel splitting used. The actual number of sub-pixels is N^2.
            polarization_factor: Polarization factor, should be very close to (but slightly lower than) 1.
            }
        ai_1d, ai_2d (optional): 1D and 2D integrator objects. Can be provided to avoid 
                                 having to re-initialize the integrators
        im_corr      (optional): correction image, such that corrected image = im*im_corr
    
    Return:
        data  : Data dictionary following the default DanMAX data structure
        meta  : Integration meta data dictionary. NOT the experimental meta data
        ai_1d : 1D integrator object
        ai_2d : 2D integrator object
    
    
    More information about the integration can be found here:
    https://wiki.maxiv.lu.se/index.php/DanMAX:Pilatus_2M_CdTe#Live_azimuthal_integration_GUI
    
    """
    config=config.copy()
    config['error_model'] = None
    config['poni_filename'] = config['poni']
    #config['pixel_size'] = 172.0e-6
    #config['shape'] = (1679, 1475)
    
    # check if integration should be binned azimuthally
    is_binned = type(config['azimuth_bins']) != type(None)
    # get integrators
    if (ai_1d == None) or ((ai_2d == None) and is_binned):
        ai_1d, ai_2d = initializeIntegrators(config)
    # get axis values
    x = ai_1d.radial_axis
    if type(ai_2d) != type(None):
        azi = ai_2d.azimuth_axis
    else:
        azi=None

    # HDF5 dataset entry path
    dset_name = '/entry/instrument/pilatus/data'
    
    # ensure im has 3 dimensions
    #if len(im.shape)==2:
    #    im = np.expand_dims(im,axis=0)
    with h5.File(fname,'r') as f:
        n = f[dset_name].shape[0]
    print(n)
    data = initializeDictionary(config,n,x,azi)
    meta = initializeMeta(config)
    
    try:
        with h5.File(fname,'r') as f:
            for i in range(n):
                im_i = f[dset_name][i]
                if not isinstance(im_corr,type(None)):
                    im_i *= im_corr
                y , e = ai_1d.integrate(im_i)
                data['I'][i]=y
                if type(e) != type(None):
                    data['I_error'][i] = e
    
                if is_binned:
                    y , e = ai_2d.integrate(im_i)
                    data['cake'][i]=y
                    if type(e) != type(None):
                        data['cake_error'][i] = e
                print(f'progress: {(i+1)/n*100:6.2f}%',end='\r')
    except KeyboardInterrupt:
        print(f'Integration interrupted at frame {i}')
    finally:
        return data, meta, ai_1d, ai_2d

def integrateImages(im, config, ai_1d=None, ai_2d=None):
    """
    DanMAX integration function for a single dataset of one or more images. 
    Uses the python implementation of MATFRAIA to azimuthally integrate an image
    Parameters:
        
        im                     : numpy array
        config = {
            poni               : Absolute path for the poni file 
            mask               : Absolute path for the mask file in .npy format 
            radial_bins        : Number of radial bins
            azimuthal_bins     : Number of azimuthal bins - See note below 
            unit               : "q" or "2th" 
            n_splitting        : Number of sub-pixel splitting used. The actual number of sub-pixels is N^2.
            polarization_factor: Polarization factor, should be very close to (but slightly lower than) 1.
            }
        ai_1d, ai_2d (optional): 1D and 2D integrator objects. Can be provided to avoid 
                                 having to re-initialize the integrators
    
    Return:
        data  : Data dictionary following the default DanMAX data structure
        meta  : Integration meta data dictionary. NOT the experimental meta data
        ai_1d : 1D integrator object
        ai_2d : 2D integrator object
        
    
    More information about the integration can be found here:
    https://wiki.maxiv.lu.se/index.php/DanMAX:Pilatus_2M_CdTe#Live_azimuthal_integration_GUI
    
    """
    config=config.copy()
    config['error_model'] = None
    config['poni_filename'] = config['poni']
    #config['pixel_size'] = 172.0e-6
    #config['shape'] = (1679, 1475)
    
    # check if integration should be binned azimuthally
    is_binned = type(config['azimuth_bins']) != type(None)
    # get integrators
    if (ai_1d == None) or ((ai_2d == None) and is_binned):
        ai_1d, ai_2d = initializeIntegrators(config)
    # get axis values
    x = ai_1d.radial_axis
    if type(ai_2d) != type(None):
        azi = ai_2d.azimuth_axis
    else:
        azi=None
    # ensure im has 3 dimensions
    if len(im.shape)==2:
        im = np.expand_dims(im,axis=0)
    data = initializeDictionary(config,im.shape[0],x,azi)
    meta = initializeMeta(config)
    
    try:
        # iterate through images
        for i, im_i in enumerate(im):
            y , e = ai_1d.integrate(im_i)
            data['I'][i]=y
            if type(e) != type(None):
                data['I_error'][i] = e

            if is_binned:
                y , e = ai_2d.integrate(im_i)
                data['cake'][i]=y
                if type(e) != type(None):
                    data['cake_error'][i] = e
            print(f'progress: {(i+1)/im.shape[0]*100:6.2f}%',end='\r')
    except KeyboardInterrupt:
        print(f'Integration interrupted at frame {i}')
    finally:
        return data, meta, ai_1d, ai_2d

def initializeIntegrators(config):
    """
    Initialize 1D and 2D integrator objects - This is slow and 
    should only be called when configurations has been changed.
        config - dictionary of integration configurations
    return ai_1d, ai_2d
    """
    # set locked config parameters
    config=config.copy()
    config['error_model'] = None
    config.pop('poni_filename')
    #config['pixel_size'] = 172.0e-6
    #config['shape'] = (1679, 1475)
    
    # read the mask file
    if type(config['mask'])==str:
        mask_fname = config['mask']
        if mask_fname.endswith('.npy'):
            config['mask'] = np.load(mask_fname)
        else:
            config['mask'] = fabio.open(mask_fname).data
    
    # check if integration should be binned azimuthally
    is_binned = type(config['azimuth_bins']) != type(None)
    
    # initialize integrators
    if is_binned:
        print('Initializing 1D and 2D integrators')
        ai_2d = AzimuthalIntegrator(**config)
        config_1d = config.copy()
        config_1d['azimuth_bins']=None
        ai_1d = AzimuthalIntegrator(**config_1d)
    else:
        print('Initializing 1D integrator')
        ai_1d = AzimuthalIntegrator(**config)
        ai_2d = None
    return ai_1d, ai_2d
    
    
def initializeDictionary(config,n,x,azi=None):
    """
    Initialize data dictionary - This is fast, call before every new dataset
        config - dictionary of integration configurations
        n - number of images
        x - x-axis (scattering axis) values
        azi - azimuthal axis values
    return data
    """
    # check if integration should be binned azimuthally
    is_binned = type(config['azimuth_bins']) != type(None)
    
    # generate x bin edges
    bin_width = np.nanmean(np.abs(np.diff(x)))
    x_edge = np.append(x,x[-1]+bin_width)
    x_edge -= bin_width/2

    # azimuthal axis
    azi_edge = config['azimuth_bins']
    if type(azi_edge) == int:
        # convert integer to array of bin boundaries
        azi_edge = np.linspace(0,360,azi_edge+1)
    # define output dictionary
    I = np.full((n,x.shape[0]),np.nan,dtype=np.float32)
    cake = None
    if is_binned:
        cake = np.full((n,azi.shape[0],x.shape[0]),np.nan,dtype=np.float32)

    if config['unit'] == 'q':
        q = x
        q_edge = x_edge
        tth = None
        tth_edge = None
    elif config['unit'] == '2th':
        tth = x
        tth_edge = x_edge
        q = None
        q_edge = None
    if config['error_model'] == None:
        I_error = None
        cake_error = None
    else:
        I_error = I
        cake_error = np.full((n,azi.shape[0],x.shape[0]),np.nan,dtype=np.float32)

    data = {
    'I': I,
    'cake': cake,
    'q': q,
    'tth': tth,
    'azi': azi,
    'q_edge': q_edge,
    'tth_edge': tth_edge,
    'azi_edge': azi_edge,
    'I_error': I_error,
    'cake_error': cake_error,
    }
    return data

def initializeMeta(config):
    """
    Initialize meta data dictionary
        config - dictionary of integration configurations
     return meta
    """

    ver = version('azint')
    
    meta_input = {'error_model': config['error_model'],
                  'mask_file': config['mask'],
                  'n_splitting': config['n_splitting'],
                  'polarization_factor': config['polarization_factor'],
                  'poni': config['poni'],
                  'poni_filename' : config['poni_filename'],
                 }
    
    meta = {'date': datetime.now().isoformat(),
            'program': 'azint DanMAX jupyterhub',
            'version': ver,
            'input': meta_input
           }
    return meta

def writeAzintFile(aname,data,meta=None):
    """
    Write azimtuhal integration data to a .h5, following the NeXus-like format.
    Parameters:
        aname - destination file name
        data  - data dictionary as generated by DanMAX.getAzintData()
        meta (None)  - integration meta data dictionary as generated by DanMAX.getAzintData() 
    Return:
        aname - destination file name
    """
    if '/raw/' in aname:
        aname = aname.replace('/raw/', '/process/azint/').split('.')[0]+'_pilatus_integrated.h5'
    
    # prepared output path and filename
    dst = os.path.dirname(aname)
    if not os.path.isdir(dst):
        os.makedirs(dst)
    
    if type(data['q']) != type(None):
        x = data['q']
        Q = True
    else:
        x = data['tth']
        Q = False

    write_2d = not data['cake'] is None
    
    # define h5 data group names
    group_1D = 'entry/data1d'
    group_2D = 'entry/data2d'
    group_meta = 'entry/azint'
    group_meta_input = 'entry/azint/input'
    
    # define key reference dictionary
    # data_keys = {
    #   'I'         : f'{group_1D}/I',
    #   'cake'      : f'{group_2D}/cake',
    #   'q'         : f'{group_1D}/q',
    #   'tth'       : f'{group_1D}/2th',
    #   'azi'       : f'{group_2D}/azi',
    #   'q_edge'    : f'{group_1D}/q_edges',
    #   'tth_edge'  : f'{group_1D}/tth_edges',
    #   'azi_edge'  : f'{group_2D}/azi_edges',
    #   'I_error'   : f'{group_1D}/I_error',
    #   'cake_error': f'{group_2D}/cake_error',
    #   'norm'      : f'{group_2D}/norm',
    #   }
    
    # this is needed for writing arrays of utf-8 strings with h5py
    text_dtype = h5.special_dtype(vlen=str)
    
    with h5.File(aname,'w') as f:
        ## create groups
        # entry/
        entry = f.create_group('entry')
        entry.attrs['NX_class'] = 'NXentry'
        
        #entry/data1d
        data1d = f.create_group(group_1D)
        data1d.attrs['NX_class'] = 'NXdata'
        data1d.attrs['interpretation'] = 'spectrum'
        data1d.attrs['signal'] = 'I'
        data1d.attrs['axes'] = np.array(['.','2th'],dtype=text_dtype)

        if write_2d:
            #entry/data2d
            data2d = f.create_group(group_2D)
            data2d.attrs['NX_class'] = 'NXdata'
            data2d.attrs['interpretation'] = 'image'
            data2d.attrs['signal'] = 'cake'
            data2d.attrs['axes'] = np.array(['.','azi','2th'],dtype=text_dtype)
        
        #entry/azint
        azint_meta = f.create_group(group_meta)
        azint_meta.attrs['NX_class'] = 'NXprocess'
                                
        #entry/azint/input
        azint_input = f.create_group(group_meta_input)
        azint_input.attrs['NX_class'] = 'NXparameters'
        
        ## create datasets
        #entry/data1d
        if Q:
            x_1d = data1d.create_dataset('q', data=data['q'], dtype=np.float64)
            x_1d.attrs['long_name'] = np.array('q',dtype=text_dtype)
            x_1d.attrs['units'] = np.array('A-1',dtype=text_dtype)
        else:
            x_1d = data1d.create_dataset('2th', data=data['tth'], dtype=np.float64)
            x_1d.attrs['long_name'] = np.array('2theta',dtype=text_dtype)
            x_1d.attrs['units'] = np.array('degrees',dtype=text_dtype)
        
        I = data1d.create_dataset('I',data=data['I'],dtype=np.float32)
        if type(data['I_error']) != type(None):
            I_error = data1d.create_dataset('I_error',data=data['I_error'],dtype=np.float32)

        if write_2d:
            #entry/data2d
            if Q:
                x_2d = data2d.create_dataset('q', data=data['q'], dtype=np.float64)
                x_2d.attrs['long_name'] = np.array('q',dtype=text_dtype)
                x_2d.attrs['units'] = np.array('A-1',dtype=text_dtype)
                
                x_edge_2d = data2d.create_dataset('q_edges', data=data['q_edge'], dtype=np.float64)
                x_edge_2d.attrs['long_name'] = np.array('q edges',dtype=text_dtype)
                x_edge_2d.attrs['units'] = np.array('A-1',dtype=text_dtype)
            else:
                x_2d = data2d.create_dataset('2th', data=data['tth'], dtype=np.float64)
                x_2d.attrs['long_name'] = np.array('2theta',dtype=text_dtype)
                x_2d.attrs['units'] = np.array('degrees',dtype=text_dtype)
                
                x_edge_2d = data2d.create_dataset('tth_edges', data=data['tth_edge'], dtype=np.float64)
                x_edge_2d.attrs['long_name'] = np.array('2theta edges',dtype=text_dtype)
                x_edge_2d.attrs['units'] = np.array('degrees',dtype=text_dtype)
            
            azi = data2d.create_dataset('azi', data=data['azi'], dtype=np.float64)
            azi.attrs['long_name'] = np.array('Azimuthal angle',dtype=text_dtype)
            azi.attrs['units'] = np.array('degrees',dtype=text_dtype)
            
            azi_edges = data2d.create_dataset('azi_edges', data=data['azi_edge'], dtype=np.float64)
            azi_edges.attrs['long_name'] = np.array('Azimuthal angle edges',dtype=text_dtype)
            azi_edges.attrs['units'] = np.array('degrees',dtype=text_dtype)
            
            cake = data2d.create_dataset('cake',data=data['cake'],dtype=np.float32)
            if type(data['cake_error']) != type(None):
                cake_error = data2d.create_dataset('cake_error',data=data['cake_error'],dtype=np.float32)
                
        #norm = data2d.create_dataset('norm',data=data['norm'],dtype=np.float32)

        # write integration meta data
        if type(meta) != type(None):
            poni_filename = meta['input'].pop('poni_filename')
            for key in meta:
                if key != 'input':
                    azint_meta.create_dataset(key,data=meta[key])
                else:
                    for key in meta['input']:
                        #entry/azint/input
                        if meta['input'][key] is None:
                            azint_input.create_dataset(key,data="")
                        # if 'poni' in key:
                        #     if key == 'poni':
                        #         azint_input.create_dataset(key,data=meta['input'][key])
                        #         azint_input[key].attrs['filename'] = meta['input']['poni_filename']
                        else:
                            azint_input.create_dataset(key,data=meta['input'][key])
            azint_input[key].attrs['filename'] = poni_filename
    return aname