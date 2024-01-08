"""Backend for the azimuthal integration pipeline - To run the pipeline, see the guide in startOfflineAzint.ipynb"""

import os
import sys
import h5py
import json
import yaml
import signal
import fabio
import pickle
import asyncio
import zmq
import zmq.asyncio
import numpy as np
from tqdm import tqdm
from datetime import datetime
from multiprocessing import Process
from collections.abc import Iterable
import azint
from azint import AzimuthalIntegrator
from bitshuffle import decompress_lz4

def save_divide(a, b):
    return np.divide(a, b, out=np.zeros_like(a), where=b!=0.0)

# build lambda from zmq stream
def build_lambda_frame(full_img, header, parts):
    if full_img is None:
        full_img = np.full(header['full_shape'], -1, dtype=np.int32)
    for i in range(4):
        x, y = header['x'][i], header['y'][i]
        module = decompress_lz4(parts[i+1].buffer, header['shape'][1:], dtype=header['type'])
        module = np.rot90(module, - header['rotation'][i] // 90)
        full_img[y:y+module.shape[0], x:x+module.shape[1]] = module
    return full_img

def integrate(ai, img, mask):
    res, errors, norm = ai.integrate(img, mask=mask, normalized=False)
    data = {}
    if res.ndim == 1:
        I = save_divide(res, norm)
        if errors is not None:
            errors = save_divide(errors, norm)
            data['/entry/data1d/I_errors'] = errors
    else:
        I = save_divide(np.sum(res, axis=0), np.sum(norm, axis=0))
        cake = save_divide(res, norm)
        data['/entry/data2d/cake'] = cake
        if errors is not None:
            errors = save_divide(np.sum(errors, axis=0), np.sum(norm, axis=0))
            data['/entry/data1d/I_errors'] = errors
    data['/entry/data1d/I'] = I
    return data
    

def zmq_worker(ai: AzimuthalIntegrator, 
               host: str, 
               pull_port: int,
               push_port: int,
               mask_threshold: int):
    context = zmq.Context()
    pull_sock = context.socket(zmq.PULL)
    pull_sock.connect('tcp://%s:%d' %(host, pull_port))
    push_sock = context.socket(zmq.PUSH)
    push_sock.connect('tcp://localhost:%d' %push_port)
    full_img = None
    
    # fix problem with multiprocessing and asyncio that shutdown the server when terminating child processes
    # https://github.com/tiangolo/fastapi/issues/1487#issuecomment-1157066306
    signal.set_wakeup_fd(-1)
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    
    while True:
        parts = pull_sock.recv_multipart(copy=False)
        header = json.loads(parts[0].bytes)
        #print(header)
        if header['htype'] == 'image':
            # special case for the 4 module lambda detector
            if len(parts) == 5:
                full_img = build_lambda_frame(full_img, header, parts)
                img = full_img
            # all the other normal detectors
            else:
                if 'bslz4' in header['compression']:
                    img = decompress_lz4(parts[1].buffer, 
                                        header['shape'], 
                                        np.dtype(header['type']))
                else:
                    img = np.frombuffer(parts[1].buffer, 
                                        dtype=np.dtype(header['type'])).reshape(header['shape'])
            
            if mask_threshold:
                mask = np.zeros(img.shape, dtype=np.int8)
                mask[img > mask_threshold] = 1
            else:
                mask = None

            data = integrate(ai, img, mask)
            push_sock.send_pyobj({'htype': 'image', 
                                  'frame': header['frame'], 
                                  'msg_number': header['msg_number']},
                                 flags=zmq.SNDMORE)
            push_sock.send_pyobj(data)
        else:
            header['source'] = 'stream'
            push_sock.send_pyobj(header)
            
# build lambda from hdf5 data
def assemble_frame(modules, config):
    img = np.full(config['full_shape'], -1, dtype=np.float64)
    for i in range(4):
        module = np.rot90(modules[i], - config['rotation'][i] // 90)
        img[config['y'][i]:config['y'][i] + module.shape[0], config['x'][i]:config['x'][i]+module.shape[1]] = module
    return img
    
def hdf5_worker(ai: AzimuthalIntegrator,
                filename: str,
                dset_name: str,
                collector_port: int,
                worker_id: int, 
                nworkers: int,
                mask_threshold: int):
    context = zmq.Context()
    push_sock = context.socket(zmq.PUSH)
    push_sock.connect('tcp://localhost:%d' %collector_port)
    
    with h5py.File(filename, 'r') as fh:
        dset = fh[dset_name]
        nimages = len(dset)
        # for lambda detector
        config = None
        
        if worker_id == 0:
            header = {'htype': 'header',
                      'source': 'file',
                      'nimages': nimages,
                      'msg_number': -1,
                      'filename': filename}
            push_sock.send_pyobj(header)
        
        for i in range(worker_id, nimages, nworkers):
            img = dset[i]
            # assemble modules for the lambda
            if img.ndim == 3:
                if config is None:
                    config = {}
                    for key, value in fh['entry/instrument/lambda/'].items():
                        config[key] = value[()] 
                img = assemble_frame(img, config)
                
            if mask_threshold:
                mask = np.zeros(img.shape, dtype=np.uint8)
                mask[img > mask_threshold] = 1
            else:
                mask = None
                
            data = integrate(ai, img, mask)
            header = {'htype': 'image',
                      'frame': i,
                      'msg_number': i}
            push_sock.send_pyobj(header, flags=zmq.SNDMORE)
            push_sock.send_pyobj(data)
            
    if worker_id == 0:
        header = {'htype': 'series_end',
                  'source': 'file',
                  'msg_number': nimages}
        push_sock.send_pyobj(header)

async def ordered_recv(sock):
    cache = {}
    next_msg_number = -1
    while True:
        parts = [pickle.loads(p) for p in await sock.recv_multipart()]
        header = parts[0]
        msg_number = header['msg_number']
        if header['htype'] == 'header':
            next_msg_number = msg_number
            # clear cache and remove older leftover messages if previous series didn't properly finish
            for key in list(cache.keys()):
                if key < msg_number:
                    del cache[key]

        if msg_number == next_msg_number:
            yield parts
            next_msg_number += 1
            while next_msg_number in cache:
                entry = cache.pop(next_msg_number)
                yield entry
                next_msg_number += 1
        else:
            cache[msg_number] = parts
            
            
def write_radial_axis(group, unit, radial_axis, radial_bins):
    dset = group.create_dataset(unit, data=radial_axis)
    if unit == 'q':
        dset.attrs['units'] ='1/angstrom'
    else:
        dset.attrs['units'] = 'degrees'
        dset.attrs['long_name'] = '2theta'
    
    if isinstance(radial_bins, Iterable):
        group.create_dataset(f'{unit}_edges', data=radial_bins)
        
class Pipeline():
    def __init__(self, context, pub_port=None):
        self.context = context
        self.pull_sock = context.socket(zmq.PULL)
        self.collector_port = self.pull_sock.bind_to_random_port('tcp://*')
        
        if pub_port:
            self.pub_sock = self.context.socket(zmq.PUB)
            self.pub_sock.setsockopt(zmq.TCP_KEEPALIVE, 1)
            self.pub_sock.setsockopt(zmq.TCP_KEEPALIVE_CNT, 10)
            self.pub_sock.setsockopt(zmq.TCP_KEEPALIVE_IDLE, 60)
            self.pub_sock.setsockopt(zmq.TCP_KEEPALIVE_INTVL, 1)
            self.pub_sock.bind('tcp://*:%d' %pub_port)
        else:
            self.pub_sock = None
        
        self.collector_task = None
        self.ai = None
        self.fh = None
        self.last = None
        self.procs = []
        self.config = None
        self.status = 'Not running'
        
    def start(self, config, mode, nworkers, host=None, 
              port=None, filename=None, dset=None, mask_threshold=None):
        self.config = config
        self.last = None
        
        if self.ai:
            self.stop()
            
        self.collector_task = asyncio.create_task(self.collector())
        
        try:
            config = config.copy()
            # replace mask file with the actual mask data
            if config['mask'] == '':
                config['mask'] = None
            else:
                fname = config['mask']
                ending = os.path.splitext(fname)[1]
                if ending == '.npy':
                    config['mask'] = np.load(fname)
                else:
                    config['mask'] = fabio.open(fname).data
                    
            self.ai = AzimuthalIntegrator(**config)
            self.status = 'Ready'
        except Exception as e:
            self.status = 'Error starting azint: %s' %str(e)
            
        for i in range(nworkers):
            if mode == 'file':
                p = Process(target=hdf5_worker, args=(self.ai, filename, dset, self.collector_port, i, nworkers, mask_threshold))
            else:
                p = Process(target=zmq_worker, args=(self.ai, host, port, self.collector_port, mask_threshold))
            p.start()
            self.procs.append(p)
            
    def stop(self):
        if self.collector_task:
            self.collector_task.cancel()
         # stop old processes
        for p in self.procs:
            p.terminate()
        self.procs.clear()
        self.ai = None
        self.status = 'Not running'
        
    async def collector(self):
        pbar = None
        async for parts in ordered_recv(self.pull_sock):
            header = parts[0]
            if header['htype'] == 'image':
                self.handle_data(header, parts[1])
                if pbar is not None:
                    pbar.update()
                
            elif header['htype'] == 'header':
                self.handle_header(header)
                if header['source'] == 'file':
                    pbar = tqdm(total=header['nimages'])
                    
            elif header['htype'] == 'series_end':
                if self.fh:
                    self.fh.close()
                    if self.pub_sock:
                        self.pub_sock.send_json({'filename': self.filename})
                        
                if header['source'] == 'file':
                    return
    
    def write_header(self):
        entry = self.fh.create_group('entry')
        entry.attrs['NX_class'] = 'NXentry'
        
        reduction = entry.create_group('azint')
        reduction.attrs['NX_class'] = 'NXprocess'
        reduction.create_dataset('program', data='azint pipeline')
        reduction.create_dataset('version', data=azint.__version__)
        reduction.create_dataset('date', data=datetime.now().isoformat())
        
        # write azint parameters
        parameters = reduction.create_group('input')
        parameters.attrs['NX_class'] = 'NXparameters'
        
        poni_file = self.config['poni_file']
        with open(poni_file, 'r') as pf:
            dset = parameters.create_dataset('poni_file', data=pf.read())
        dset.attrs['filename'] = poni_file
        
        parameters.create_dataset('mask_file', data=self.config['mask'])
        parameters.create_dataset('n_splitting', data=self.config['n_splitting'])
        error_model = self.config['error_model'] if self.config['error_model'] else ''
        parameters.create_dataset('error_model', data=error_model)
        polarization = self.config.get('polarization_factor')
        polarization = polarization if polarization is not None else 0
        parameters.create_dataset('polarization_factor', data=polarization)
                
        # NXdata
        data1d = entry.create_group('data1d')
        data1d.attrs['NX_class'] = 'NXdata'
        data1d.attrs['signal'] = 'I'
        data1d.attrs['axes'] = ['.', self.ai.unit]
        data1d.attrs['interpretation'] = 'spectrum'
        write_radial_axis(data1d, self.ai.unit, self.ai.radial_axis, self.config['radial_bins'])
        
        if self.ai.azimuth_axis is not None:
            entry.attrs['default'] = 'data2d'
            data2d = entry.create_group('data2d')
            data2d.attrs['NX_class'] = 'NXdata'
            data2d.attrs['signal'] = 'cake'
            norm = self.ai.norm.reshape(self.ai.output_shape)
            data2d.create_dataset('norm', data=norm)
            write_radial_axis(data2d, self.ai.unit, self.ai.radial_axis, self.config['radial_bins'])
            dset = data2d.create_dataset('azi', data=self.ai.azimuth_axis)
            dset.attrs['units'] = 'degrees'
            dset.attrs['long_name'] = 'Azimuthal angle'
            if isinstance(self.config['azimuth_bins'], Iterable):
                data2d.create_dataset('azi_edges', data=self.config['azimuth_bins'])
            data2d.attrs['axes'] = ['.', 'azi', self.ai.unit]
            data2d.attrs['interpretation'] = 'image'
        else:
            entry.attrs['default'] = 'data1d'
            
    def handle_header(self, header):
        filename = header['filename']
        self.status = 'Processing %s  Frame ' %filename
        if filename:
            self.filename = filename
            path, fname = os.path.split(filename)
            root = os.path.splitext(fname)[0]
            fname = '%s_integrated.h5' %root
            output_folder = path.replace('raw', 'process/azint')
            if 'process/azint' in output_folder:
                output_file = os.path.join(output_folder, fname)
                if not os.path.exists(output_folder):
                    os.makedirs(output_folder)
                mode = 'w' if header['source'] == 'file' else 'a'
                try:
                    self.fh = h5py.File(output_file, mode)
                except Exception as e:
                    self.fh = None
                    self.status = 'Error openening file %s: %s' %(output_file, str(e))
                
                if 'entry' not in self.fh:
                    self.write_header()
            else:
                print('No raw folder in filepath: %s\n Not saving' %output_folder)
                self.fh = None
        else:
            self.fh = None
            
    def handle_data(self, header, data):
        for key, value in data.items():
            if self.fh:
                dset = self.fh.get(key)
                if not dset:
                    dset = self.fh.create_dataset(key, dtype=value.dtype, 
                                                shape=(0, *value.shape), 
                                                maxshape=(None, *value.shape),
                                                chunks=(1, *value.shape))
                n = dset.shape[0]
                dset.resize(n+1, axis=0)
                dset[n] = value
        
        cake = '/entry/data2d/cake'
        res = data[cake] if cake in data else data['/entry/data1d/I']
        self.last = [self.ai.radial_axis, self.ai.azimuth_axis, res]
                
        index = self.status.rfind(' ')
        self.status = self.status[:index+1] + str(header['frame'])

def make_app(pipeline, args):
    
    # default config
    config = {'poni_file': '',
              'radial_bins': 1000,
              'azimuth_bins': None,
              'n_splitting': 4,
              'unit': 'q',
              'mask': None,
              'error_model': None,
              'polarization_factor': None,
              'mask_threshold': None}
    
    app = FastAPI()
    app.state.config = config
    
    @app.get('/status')
    async def status():
        return {'value': pipeline.status}
    
    @app.get('/last', response_class=Response)
    async def last():
        last = pipeline.last
        payload = pickle.dumps(last)
        return Response(payload, media_type='image/pickle')
    
    @app.post('/start')
    async def start(request: Request):
        config = await request.json()
        app.state.config = config
        
        full_config = config.copy()
        mask_threshold = full_config.pop('mask_threshold')
        full_config['shape'] = tuple(args['shape'])
        full_config['pixel_size'] = args['pixel_size']
        print(full_config)
        pipeline.start(full_config, 'stream', args['nworkers'], args['host'], args['data_port'], mask_threshold=mask_threshold)
        return {'value': 'done'}
    
    @app.post('/stop')
    async def stop():
        pipeline.stop()
        
    @app.get('/config')
    async def get_config():
        return app.state.config
        
    return app

async def main():
    config_file = sys.argv[1]
    with open(config_file) as fh:
        args = yaml.load(fh, Loader=yaml.FullLoader)
        
    pub_port = args.get('pub_port', None)
    context = zmq.asyncio.Context()
    pipeline = Pipeline(context, pub_port)
    port = args.get('api_port', 5001)
    app = make_app(pipeline, args)
    # increase timeout_keep_alive because of the long running start command
    config = uvicorn.Config(app, host='0.0.0.0', port=port, log_level='warning', timeout_keep_alive=60)
    server = uvicorn.Server(config)
    await server.serve()

if __name__ == '__main__':
    import uvicorn
    from fastapi import FastAPI, Request, Response
    # set umask to 002 to set -rw-rw-r-- permissions and allow group to write
    os.umask(0o002)
    asyncio.run(main())

