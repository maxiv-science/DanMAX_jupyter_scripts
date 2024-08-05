import os
import zmq.asyncio
import asyncio
from azint import AzimuthalIntegrator

from dummyline import Pipeline
        
async def main():
    context = zmq.asyncio.Context()
    filename = '/data/visitors/danmax/20220948/2023032608/raw/setup/scan-9602.h5'
    filename = '/data/visitors/danmax/20220948/2023032608/raw/setup/scan-9603.h5'
    dset = f'/entry/instrument/pilatus/data'
    config = {
            'radial_bins': 3000,
            'azimuth_bins': None,
            'n_splitting': 3,
            'error_model': None,
            'polarization_factor': 0.999997,
            'unit': 'q',
        'shape': (1679, 1475),
        "pixel_size": 172e-6,
    'poni_file': '/data/visitors/danmax/20231855/2023110808/process/pxrd_cryo_LaB6_35kev_500mm.poni'}

    #ai = AzimuthalIntegrator(**config)
    #del ai
    #print("ai is", ai)
    pipeline = Pipeline(context)
    print("ai", pipeline.ai)
    pipeline.start(config, "file", 8, filename=filename, dset=dset)
    await pipeline.collector_task
    

if __name__ == '__main__':
    # set umask to 002 to set -rw-rw-r-- permissions and allow group to write
    os.umask(0o002)
    asyncio.run(main())