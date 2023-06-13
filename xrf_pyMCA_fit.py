########## DO THE FOLLOWING ##########
# 1. Copy this file to a sub folder in the process folder of the experiment,
# example: /data/visitors/danmax/[proposal_id]/[session_id]/process/xrf
# 2. Scans: A list of entries to be fitted, or one scan number as command line argument
# 3. Sample name: Name of the sample
# 4. Configuration file for the fitting
# 5. Select the output files you want to generate

session_path = '/data/visitors/danmax/{0}/{1}/'
sample_name = 'sample'
config_files = 'blub.cfg'
detector = 'falconx'          # 'falconx' at danmax, 'x3mini' or 'xspress3' elsewhere
channels = [0]

make_pymca_h5        = True
make_elements_h5     = True
make_elements_tif    = True
make_spectrum_fit_h5 = True

#######################################
I0_scale_factor = 1E-9 # default
#######################################

##############################################################################
# A class to read DanMAX X-ray fluorescence data, generate files suitable to
# be analysed in PyMCA, do automated fitings to generate elemental images, and
# do a fit of the average spectrum in the image stack. All output files are
# HDF5 files. Optionally TIF files can be generated with elemental maps.
#This script is based on a similar one from NanoMAX
##############################################################################
import h5py
import sys
import os
import time
import numpy
import argparse
import configparser
from PyMca5.PyMcaPhysics.xrf.FastXRFLinearFit import FastXRFLinearFit
from PyMca5.PyMcaPhysics.xrf.XRFBatchFitOutput import OutputBuffer
from PyMca5.PyMcaPhysics.xrf.ClassMcaTheory import McaTheory
from PyMca5.PyMcaIO import ConfigDict
from silx.gui import qt
import DanMAX as DM

class xrfBatch():

    # init with the most important information that is needed over and over again
    def __init__(self, session_path, sample_name, scan_list, config_file,
                 calib_file=None, detector = 'xspress3', channel = 3,):
        self.session_path = session_path
        self.sample_name  = sample_name
        self.scan_list    = scan_list
        self.config_file  = config_file
        self.calib_file   = calib_file
        self.detector     = detector
        self.channel      = channel

        if self.scan_list.len > 1:
            self.scan_name=f'scan_{self.scan_list[0]:04}_to_scan_{self.scan_list[-1]:04)}'
        else:
            self.scan_name=f'scan_{self.scan_list[0]:04}'
        self.create_out_dirs()

    # define where to save all the data
    def create_out_dirs(self):
        self.out_dir = self.session_path+'process/'+self.sample_name+f'/self.{scan_name}_xrf_pymca/'
        self.out_dir_pymca_file = self.out_dir+'data/'
        self.out_dir_elements   = self.out_dir+'elements/'
        self.out_dir_spectrum   = self.out_dir+'spectrum/'	
        os.makedirs(self.out_dir, exist_ok=True)
        os.makedirs(self.out_dir_pymca_file, exist_ok=True)
        os.makedirs(self.out_dir_elements, exist_ok=True)
        os.makedirs(self.out_dir_spectrum, exist_ok=True)

    # Reads a NanoMAX file
    def readData(self):
        self.x,self.y,self.xrf,self.energy = DM.stitchscans(self.scan_list,XRD=false,calibration = self.calibration) 
#            self.xrf_norm = ((self.xrf*I0_scale_factor)/(self.I0*self.dwell))
#            self.xrf_avg = self.xrf_norm.sum(axis=(0,1))/npixels
#            self.energy = fp['entry/snapshot/energy'][:]
#            self.x = fp['entry/measurement/pseudo/x'][0:npixels].reshape(*shape)
#            self.y = fp['entry/measurement/pseudo/y'][0:npixels].reshape(*shape)

    # Writes a PyMCA readible h5 file
    def writePymcafile(self):
        pymca_file = os.path.join(f'{self.out_dir_pymca_file}_data_pymca_{scan_name}_{self.channel}.h5')
        print("Writing pymca file: %s" % pymca_file)
        with h5py.File(pymca_file, 'w') as ofp:
            ofp.create_dataset('xrf', data=self.xrf, compression='gzip')
            #ofp.create_dataset('xrf_normalized', data=self.xrf_norm, compression='gzip')
            #ofp.create_dataset('xrf_normalized_avg', data=self.xrf_avg, compression='gzip')
            #ofp['I0'] = self.I0[:, :, 0]
            #ofp['I0_average'] = numpy.mean(self.I0)
            #ofp['dwell_time'] = self.dwell
            ofp['energy'] = self.energy
            ofp['positions'] = [self.x,self.y]
            ofp['detector'] = self.detector
            ofp['channel'] = self.channel

    # Linear fitting of xrf stack data. Generate a h5 and/or tiff files with element concentration maps
    def fitElementsToFile(self, make_elements_h5=True, make_elements_tif=False):
        elem_file_name = f'fitted_elements_{self.scan_name}_{self.channel}'
        print(f'Generating elemental maps file: {elem_file_name}.h5')
        fastFit = FastXRFLinearFit()
        fastFit.setFitConfigurationFile(self.config_file)

        outbuffer = OutputBuffer(outputDir=self.out_dir_elements,
                        outputRoot= elem_file_name,
                        fileEntry= f'{self.sample_name}_{self.scan_name}',
                        diagnostics=0,
                        tif=make_elements_tif, edf=0, csv=1,
                        h5=make_elements_h5, dat=0,
                        multipage=0,
                        overwrite=1)

        with outbuffer.saveContext():
            outbuffer = fastFit.fitMultipleSpectra(y=self.xrf,
                        weight=0,
                        refit=1,
                        concentrations=0,
                        outbuffer=outbuffer)

    # Fitting of average spectrum from image stack
    def fitAvgSpectrumToFile(self):
        spec_file = os.path.join(self.out_dir_spectrum + f'spectrum_{scan_name}_{self.channel}.h5')
        print(f'Writing average-spectrum fit file: {spec_file}')
        mca_fit = McaTheory()
        conf = ConfigDict.ConfigDict()
        conf.read(config_file)
        mca_fit.setConfiguration(conf)
        mca_fit.configure(conf)
        mca_fit.setData(self.xrf_avg)
        mca_fit.estimate()
        fit_result = mca_fit.startfit(digest=0)
        digest_file = os.path.join(self.out_dir_spectrum + f'spectrum_{self.scan_list:04}_{self.channel}.fit')
        mca_fit_result = mca_fit.digestresult(outfile=digest_file)
        #ConfigDict.prtdict(mca_fit_result)
        with h5py.File(spec_file, 'w') as sfp:
            self._recursive_save(sfp, 'mca_fit/', mca_fit_result)
            sfp['measurement/I0_average'] = numpy.mean(self.I0)
            sfp['measurement/dwell_time'] = self.dwell
            sfp['measurement/energy'] = self.energy
            sfp['measurement/detector'] = self.detector
            sfp['measurement/channel'] = self.channel

    # Internal function to save the dictionary with results of the avgerage fitting to an h5 file.
    def _recursive_save(self, h5file, path, dic):
        for key, item in dic.items():
            key = str(key)
            #print('key=' + path + key)
            if isinstance(item, list):
                item = numpy.array(item)
            # save strings, numpy.int64, and numpy.float64 types
            if isinstance(item, (numpy.int64, numpy.float64, str, numpy.float, float, numpy.float32,int)):
                h5file[path + key] = item
            # save numpy arrays
            elif isinstance(item, numpy.ndarray):
                try:
                    h5file.create_dataset(path + key, data=item, compression='gzip')
                except:
                    #print(path + key, item)
                    item = numpy.array(item).astype('|S9')
                    h5file.create_dataset(path + key, data=item)
            # save dictionaries
            elif isinstance(item, dict):
                self._recursive_save(h5file, path + key + '/', item)
            # other types cannot be saved and will result in an error
            #else:
                #print('Cannot save key=%s'%(path+key))

    def single_scan_fit(self, scan_nr):
        '''
        This funciton will run the pipeline on a single scan
        I.E. the same as running the script through terminal 
        as script scan_nr
        '''
        pass

    def stitch_scan_fit():
        '''
        This will run the pipeline on a set of scans.
        The scans will be collected usung the DM function stitchscans.
        It needs a costum saveing function, as no single scannr will work.
        But  maybe it will save to xrf_stitch_from_scan-XXXX"???
        '''
        
        pass

##############################################################################

if __name__ == "__main__":


    #Adding argument parser
    parser = argparse.ArgumentParser()

    #Adding Nonoptional arguments
    parser.add_argument('scan_list')
    #Adding noptional arguments
    parser.add_argument('--proposal','-p', type=int, help='Proposal, default get the current folder')
    parser.add_argument('--visit'   ,'-v', type=int, help='Visit, default get from the current folder')
    parser.add_argument('--config'  ,'-o', type=str, help='configuration file')
    parser.add_argument('--calib'   ,'-c', type=str, help='calibration file')
    # read which scan to work on
    scan_nr = int(sys.argv[1])

    args=parser.parse_args()

    t0 = time.time()


    proposal,visit =  DM.getCurrentProposal(proposal=args.proposal,visit=args.visit)
    config_file = sys.argv[2]
    session_path = sessionpath.format(proposal,visit)

    # create an instance of the fit class for each channel
    fits = {}
    for channel in channels:
        fits[str(channel)] = xrfBatch(session_path = session_path, 
                                      sample_name  = sample_name, 
                                      scan_list    = scan_list,
                                      config_file  = config_file,
                                      detector     = detector,
                                      channel      = channel)

    # read data  
    for channel in channels:
        fits[str(channel)].readData(I0_scale_factor)
        print("Raw data file read. Time elapsed = %s seconds" % int(time.time() - t0))

    # write  a pyMCA readable h5 file 
    if(make_pymca_h5):
        for channel in channels:
            fits[str(channel)].writePymcafile()
            print("PyMCA compatible file written. Time elapsed = %s seconds" % int(time.time() - t0))

    # fit the elements according to a given config file
    if(make_elements_h5 or make_elements_tif):
        for channel in channels:
            fits[str(channel)].fitElementsToFile(make_elements_h5, make_elements_tif)
            print("Elemental maps file written. Time elapsed = %s seconds " % int(time.time() - t0))

    # fit the average spectrum
    if(make_spectrum_fit_h5):
        for channel in channels:
            fits[str(channel)].fitAvgSpectrumToFile()
            print("Average spectrum fit file written. Time elapsed = %s seconds" % int(time.time() - t0))
