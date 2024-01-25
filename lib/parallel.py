# -*- coding: utf-8 -*-
"""DanMAX library for handling parallel scans, e.g. AMPIX or similar multi sample holder"""
import os
import h5py as h5
import numpy as np
from azint import AzimuthalIntegrator
import ipywidgets as ipyw
import IPython
#To import DanMAX from the folder above:
import sys
sys.path.append('../')
import DanMAX as DM

def parallelMetaMaster(fname):
    """
    Read the master .h5 of a parallel scan and return the scan IDs and meta data
      Return:
        scan_ids  - list of the related scan IDs
        data      - dictionary of meta data as numpy arrays
        start_pos - dictionary of starter positioners
    """
    data, start_pos = {},{}
    scan_ids = []
    with h5.File(fname,'r') as f:
        
        for entry in f.keys():
            if len(data)<1:
                data = {key:[] for key in f[f'/{entry}/instrument/'].keys() if key != 'start_positioners' and key != 'pilatus' }
            if len(start_pos)<1:
                sp_group = f[f'/{entry}/instrument/start_positioners']
                start_pos = {key:[] for key in sp_group.keys()}
            scan_ids.append(f'scan-{entry[-4:]}')
            for key in f[f'/{entry}/instrument/'].keys():
                if key != 'pilatus' and key != 'falconx' and key != 'eiger':
                    if key == 'start_positioners':
                        sp_group = f[f'/{entry}/instrument/start_positioners']
                        for pos in sp_group.keys():
                            try:
                                start_pos[pos] = np.append(start_pos[pos],sp_group[pos][()])
                            except KeyError:
                                start_pos[pos] = np.full(len(scan_ids),np.nan)
                        #start_pos = {pos:sp_group[pos][()] for pos in sp_group.keys()}
                    else:
                        for k in f[f'/{entry}/instrument/'][key].keys():
                            try:
                                try:
                                    data[key] = np.append(data[key],f[f'/{entry}/instrument/'][key][k][:][:])
                                except ValueError:
                                    print('Unable to append',key)
                                    pass
                            except KeyError:
                                data[key]= np.full(len(scan_ids),np.nan)
                                #print('Unable to append',key,' ',entry)
            # check for metadata keys that are not present in the current entry
            for key in data:
                if key not in f[f'/{entry}/instrument/'].keys():
                    #print(key,'not found in ', entry)
                    data[key] = np.append(data[key],np.nan)
            for pos in start_pos:
                if pos not in sp_group.keys():
                    #print(pos,'not found in ', entry)
                    start_pos[pos] = np.append(start_pos[pos],np.nan)
                
    for key in data:
        data[key]=np.array(data[key])
    return scan_ids, data, start_pos


def parallelMaster(fname,index=None):
    """
    Read the master .h5 of a parallel scan and return the scan IDs and meta data
      Return:
        scan_ids  - list of the related scan IDs
        data      - dictionary of iamge data as numpy arrays
        meta      - dictionary of meta data as numpy arrays
        start_pos - dictionary of starter positioners
    """
    data = []
    meta, start_pos = {},{}
    scan_ids = []
    
    with h5.File(fname,'r') as f:
        entries = list(f.keys())
        if index != None:
            entries = entries[index:index+1]
        for i,entry in enumerate(entries):
            #print(f'{(i+1)/len(f.keys())*100:.1f} - {fname:<100s}',end='\r')
            if len(meta)<1:
                meta = {key:[] for key in f[f'/{entry}/instrument/'].keys() if key != 'start_positioners' and key != 'pilatus' }
            if len(start_pos)<1:
                sp_group = f[f'/{entry}/instrument/start_positioners']
                start_pos = {key:[] for key in sp_group.keys()}
            scan_ids.append(f'scan-{entry[-4:]}')
            for key in f[f'/{entry}/instrument/'].keys():
                if key == 'pilatus':
                    if len(data)<1:
                        data = f[f'/{entry}/instrument/pilatus/data'][:]
                    else:
                        data = np.append(data,f[f'/{entry}/instrument/pilatus/data'][:],axis=0)
                elif key != 'falconx' and key != 'eiger':
                    if key == 'start_positioners':
                        sp_group = f[f'/{entry}/instrument/start_positioners']
                        for pos in sp_group.keys():
                            try:
                                start_pos[pos] = np.append(start_pos[pos],sp_group[pos][()])
                            except KeyError:
                                start_pos[pos] = np.full(len(scan_ids),np.nan)
                        #start_pos = {pos:sp_group[pos][()] for pos in sp_group.keys()}
                    else:
                        for k in f[f'/{entry}/instrument/'][key].keys():
                            try:
                                try:
                                    meta[key] = np.append(meta[key],f[f'/{entry}/instrument/'][key][k][:][:])
                                except ValueError:
                                    print('Unable to append',key)
                                    pass
                            except KeyError:
                                meta[key]= np.full(len(scan_ids),np.nan)
                                #print('Unable to append',key,' ',entry)
            # check for metadata keys that are not present in the current entry
            for key in meta:
                if key not in f[f'/{entry}/instrument/'].keys():
                    #print(key,'not found in ', entry)
                    meta[key] = np.append(meta[key],np.nan)
            for pos in start_pos:
                if pos not in sp_group.keys():
                    #print(pos,'not found in ', entry)
                    start_pos[pos] = np.append(start_pos[pos],np.nan)
        #print()            
    for key in meta:
        meta[key]=np.array(meta[key])
    
    return scan_ids, data, meta, start_pos

def getParallelAzintData(scan_list):
    """
    Read parallel azimuthally integrated .h5 files from a list of files paths
    Return azidata - dictionary of data and (unique) metadata
    """
    for i,aname in enumerate(scan_list):
        datadic, metadic = DM.getAzintData(aname,get_meta=True)
        if i<1:
            data = datadic.copy()
            meta = metadic.copy()
        else:
            for key in ['I','cake']:
                if type(datadic[key]) != type(None):
                    data[key] = np.append(data[key],datadic[key],axis=0)
            meta = {key:np.append(meta[key],metadic[key]) for key in metadic}
    
    if np.all(meta['input'][0]==meta['input']):
        meta['input']=meta['input'][0]
    for key in meta:
        if key != 'input':
            meta[key]=np.unique(meta[key])
    return data, meta
    
    #azidata = {}
    #I = []
    #for aname in scan_list:
    #    with h5.File(aname,'r') as f:
    #        if len(azidata)<1:
    #            azidata = {key:[] for key in f.keys() if key != 'I'}
    #        for key in f.keys():
    #            if key == 'I':
    #                try:
    #                    azidata['I'] = np.append(azidata['I'],f['I'][:],axis=0)
    #                except KeyError:
    #                    azidata['I'] = f['I'][:]
    #            else:
    #                try:
    #                     azidata[key].append(f[key][:])
    #                except:
    #                     azidata[key].append(f[key][()])
    #for key in azidata:
    #    if key != 'I':
    #        meta = np.unique(azidata[key])
    #        if len(meta)==1:
    #            azidata[key] = meta
    #return azidata


# DECOMMISSIONED
# def reintegrateParallelFiles(files, config, embed_meta=True):
#     """
#     Re-integrate parallel scans using specified configurations and a single master_pilatus_integrated.h5 file
#     per master.h5 file, saved in '[proposal]/[visit]/process/azint/reintegrated'. Provides an option to embed metadata in the integration .h5.
#     Tip:
#     Get a list of all master.h5 files in the current proposal/visit with
#         files = sorted(glob.glob(f'/data/visitors/danmax/[proposal]/[visit]/raw/**/master.h5', recursive=True),
#                        key = os.path.getctime,
#                        reverse=True)
                       
#     Parameters:
#         files      - list - List of master.h5 files to be re-integrated
#         config     - dic  - Azimuthal integration config dictionary. See help(DanMAX.integrateFile) for more
#         embed_meta - Bool - Toggle whether or not to embed metadata in the master_pilatus_integrated.h5 file (default=True)
    
#     Return None
#     """
    
#     # set locked config parameters
#     config=config.copy()
#     config['error_model'] = None
#     config['pixel_size'] = 172.0e-6
#     config['shape'] = (1679, 1475)
#     # read the mask file
#     if type(config['mask'])==str:
#         mask_fname = config['mask']
#         if mask_fname.endswith('.npy'):
#             config['mask'] = np.load(mask_fname)
#         else:
#             config['mask'] = fabio.open(mask_fname).data 
    
#     # initialize the AzimuthalIntegrator
#     ai = AzimuthalIntegrator(**config)
    
#     # display progress bars
#     fname_widget = ipyw.Text(value = '')
#     progress_all = ipyw.IntProgress(min=0, max=len(files)*1000,description='Total:')
#     progress = ipyw.IntProgress(min=0, max=10)
#     IPython.display.display(progress_all,display_id='prog_bar_all')
#     IPython.display.display(fname_widget)
#     IPython.display.display(progress,display_id='prog_bar')
    
#     if type(files) != list:
#         files = [files]
    
#     # loop throug the master.h5 files
#     for k,fname in enumerate(files):
#         #print(f'{(i+1)/len(files)*100:5.1f}% {fname:<150s}')#,end='\r')
#         fname_widget.value = fname.split('/raw/')[-1]

#         azi_path = os.path.dirname(fname).replace('raw','process/azint/reintegrated')
#         azi_master_path = azi_path + f'/master_pilatus_integrated.h5'
#         if not os.path.exists(azi_path):
#             os.makedirs(azi_path)

#         # write to output file
#         with h5.File(azi_master_path,'w') as fi:
#             # write the integration configuration information to the output file
#             fi.create_dataset(ai.unit, data=ai.radial_axis)
#             with open(config['poni_file'], 'r') as poni:
#                 p = fi.create_dataset('poni_file', data=poni.read())
#                 p.attrs['filename'] = config['poni_file']
#             fi.create_dataset('mask_file', data=mask_fname)
#             polarization_factor = config['polarization_factor'] 
#             data = polarization_factor if polarization_factor is not None else 0
#             fi.create_dataset('polarization_factor', data=data)

#             #read the image and meta data from the first entry in the master.h5 file
#             scan_ids, data, metadata, start_pos = parallelMaster(fname,index=0)
#             # set the max value for the progress bar
#             with h5.File(fname,'r') as f:
#                 entries = list(f.keys())
#             progress.max = len(entries)
#             # exposures per entry
#             expo_per_entry = data.shape[0]
#             shape = (len(entries)*expo_per_entry, *ai.output_shape)
            
#             # prepare the HDF5 dataset for the integrated data
#             I_dset = fi.create_dataset('I', shape=shape, dtype=np.float32)
#             if ai.error_model == 'poisson':
#                 sigma_dset = fi.create_dataset('sigma', shape=shape, dtype=np.float32)
#             if embed_meta:
#                 for key in metadata:
#                     fi.create_dataset(f'meta/{key}', 
#                                       shape=shape[0],
#                                       dtype=metadata[key].dtype)
#                 for key in start_pos:
#                     fi.create_dataset(f'meta/start_positioners/{key}',
#                                       shape=shape[0],
#                                       dtype=start_pos[key].dtype)
                    
#             # loop through each entry in the master.h5 file
#             for i, entry in enumerate(entries):
#                 # update progress bars
#                 progress.value = i
#                 progress_all.value = round((k)*1000 + i/len(entries)*1000) # update values in pro mille
                
#                 #read the image and meta data from the ith entry in the master.h5 file
#                 scan_ids, data, metadata, start_pos = parallelMaster(fname,index=i)
#                 # loop through each exposure
#                 for j,im in enumerate(data):
#                     index = i*expo_per_entry+j
#                     I, sigma = ai.integrate(im)
#                     I_dset[index] = I
#                     if sigma is not None:
#                         sigma_dset[index] = sigma
                
#                 if embed_meta:
#                     # embed metadata for all exposures
#                     for key in metadata:
#                         fi[f'meta/{key}'][i*expo_per_entry:index+1] = metadata[key]
#                     for key in start_pos:
#                         fi[f'meta/start_positioners/{key}'][i*expo_per_entry:index+1] = start_pos[key]
#             # update the progress bar one last time
#             progress.value = i+1