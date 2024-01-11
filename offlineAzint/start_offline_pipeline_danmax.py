import sys
sys.path.append('../')
import DanMAX as DM

import os
import zmq
import h5py
import asyncio
from pipeline import Pipeline
from offline_azint_config import user_config, scans

async def main():
    context = zmq.asyncio.Context()
    pipeline = Pipeline(context)

    nworkers = 8
    
    dset = f'/entry/instrument/pilatus/data'
    # dset = f'entry/instrument/pilatus/data'
    config = {
            'radial_bins': 3000,
            'azimuth_bins': None,
            'n_splitting': 3,
            'error_model': None,
            'polarization_factor': 0.999997,
            'unit': 'q'}

    config.update(user_config)
    
    for scan in scans:
        filename = DM.findScan(scan) #.replace('.h5', '_pilatus.h5')
        #filename = '/data/visitors/danmax/20220948/2023032608/raw/setup/scan-9602.h5'
        print(filename)
        try:
            with h5py.File(filename, 'r') as fh:
                fh[dset]
        except Exception as e:
            print(e)
            continue
        pipeline.start(config, 'file', nworkers, filename=filename, dset=dset)
        await pipeline.collector_task

    
# set umask to 002 to set -rw-rw-r-- permissions and allow group to write
os.umask(0o002)
asyncio.run(main())