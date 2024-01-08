import os
import sys
import h5py as h5
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

def test_worker(ai: None,
                filename: str,
                dset_name: str,
                collector_port: int,
                worker_id: int, 
                nworkers: int,
                mask_threshold: int):

    context = zmq.Context()
    push_sock = context.socket(zmq.PUSH)
    push_sock.connect('tcp://localhost:%d' %collector_port)
    nimages = 100

    fh = h5.File(filename, 'r')
    print("file handle", fh)
    dset = fh[dset_name]
    print("dataset", dset)
    try:
        first = dset[0]
        print("first", first)
    except Exception as e:
        print("failed to read", e.__repr__())
    fh.close()
    

    with h5.File(filename, 'r') as fh:
        dset = fh[dset_name]
        print("handle", fh)
        print("dataset", dset)
        print("len", len(dset))
        nimages = len(dset)
    
        if worker_id == 0:
            header = {'htype': 'header',
                      'source': 'file',
                      'nimages': nimages,
                      'msg_number': -1,
                      'filename': filename}
            push_sock.send_pyobj(header)

    
        for i in range(worker_id, nimages, nworkers):
            data = {}
            data['/entry/data1d/I'] = dset[i]
            
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
    context.destroy()
    
def worker(filename, dset_name, collector_port, worker_id):
    context = zmq.Context()
    push_sock = context.socket(zmq.PUSH)
    push_sock.connect('tcp://localhost:%d' %collector_port)
    nimages = 100
    
    print("worker", worker_id, filename, dset_name)
    fh = h5py.File(filename, 'r')
    print("file handle", fh)
    dset = fh[dset_name]
    print("dataset", dset)
    first = dset[0]
    print("first", first)
    fh.close()

    header = {'htype': 'header',
                  'source': 'file',
                  'nimages': nimages,
                  'msg_number': -1,
                  'filename': filename}
    push_sock.send_pyobj(header)

    if worker_id == 0:
        header = {'htype': 'series_end',
                  'source': 'file',
                  'msg_number': nimages}
        push_sock.send_pyobj(header)

    print("stopped worker", worker_id)
    context.destroy()

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
            
class Pipeline():
    def __init__(self, context, pub_port=None):
        self.context = context
        self.pull_sock = context.socket(zmq.PULL)
        self.collector_port = self.pull_sock.bind_to_random_port('tcp://*')
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
            
        try:
            config = config.copy()
            # replace mask file with the actual mask data
            if "mask" not in config or config['mask'] == '':
                config['mask'] = None
            else:
                fname = config['mask']
                ending = os.path.splitext(fname)[1]
                if ending == '.npy':
                    config['mask'] = np.load(fname)
                else:
                    config['mask'] = fabio.open(fname).data
            self.ai = 123        
            #self.azi = 
            #AzimuthalIntegrator(**config)
            self.status = 'Ready'
            print("status after init", self.ai, self.status)
        except Exception as e:
            msg = 'Error starting azint: %s' %str(e)
            self.status = msg
            print(msg)
            return

        self.collector_task = asyncio.create_task(self.collector())
            
        for i in range(nworkers):
            p = Process(target=test_worker, args=(None, filename, dset, self.collector_port, i, nworkers, mask_threshold))
            p.start()
            self.procs.append(p)

        
        #for i in range(5):
        #    p = Process(target=worker, args=(filename, dset, self.collector_port, i))
        #    p.start()

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
                print("done")
                break
   
            
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
                    self.fh = h5.File(output_file, mode)
                except Exception as e:
                    self.fh = None
                    self.status = 'Error openening file %s: %s' %(output_file, str(e))
                
                if 'entry' not in self.fh:
                    print("need header")
                    #self.write_header()
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
        self.last = [] # [self.ai.radial_axis, self.ai.azimuth_axis, res]
                
        index = self.status.rfind(' ')
        self.status = self.status[:index+1] + str(header['frame'])

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

