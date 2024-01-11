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
sys.path.append('../')
import DanMAX as DM

class xrfBatch():

    # init with the most important information that is needed over and over again
    def __init__(self, scan_list, session_path=None, config_file=None,
                 calib_file=None, detector = 'falconx', channel = 0,
                 save_loc=None, calibration = None, sample_name = None,
                 proposal=None,visit=None):

        #Setting values to self
        self.session_path        = session_path
        self.sample_name         = sample_name
        self.scan_list           = scan_list
        self.config_file         = config_file
        self.calib_file          = calib_file
        self.calibration         = calibration
        self.detector            = detector
        self.channel             = channel
        self.save_loc            = save_loc
        self.proposal,self.visit = DM.getCurrentProposal(proposal,visit)
        
        #Doing the following would be optimal, that way changes to this code should
        #Not break jupyter compatibility
        #Putting it off for now though
        #Note that the function doesn't currently make all values, but  does create them
        #internally
        '''
        self.sessioin_path, self.scan_name, self.sample_name, self.out_dir, self.elem_fit_name = DM.getXRFFitFilename(scan_list,proposal=proposal,visit=visit,base_folder=save_loc,channel=channel)
        '''

        #overwriting sample name if an empty was given
        if session_path == None:
            self.session_path = f'/data/visitors/danmax/{self.proposal}/{self.visit}/'

        if calib_file == None and calibration == None:
            self.calib_file = f'{self.session_path}process/pymca_calib.calib'
            if not os.path.isfile(self.calib_file):
                raise IOError('No calibration available')

        if config_file == None:
            self.config_file = f'{self.session_path}process/pymca_config.cfg'
        if not os.path.isfile(self.config_file):
            raise IOError(f'Configuration file {self.config_file} not available')

        if sample_name == None:
            fname = DM.findScan(int(scan_list[0]), proposal=self.proposal, visit=self.visit)
            idx = fname.split('/').index('danmax')
            self.sample_name = fname.split('/')[idx + 4]

        if len(self.scan_list) > 1:
            self.scan_name=f'scan_{self.scan_list[0]:04}_to_{self.scan_list[-1]:04}'
        else:
            self.scan_name=f'scan_{self.scan_list[0]:04}'
        self.create_out_dirs()
        if self.calib_file != None:
            if self.calibration != None:
                print('calibration and calibration file both provided, overwrting calibration with calibration file content')
            config = configparser.ConfigParser()
            config.read(self.calib_file)
            if not 'Stack SUM' in config.sections():
                Exception('No calibration in "Stack SUM" of the calibration file, aborting')
            self.calibration = [config['Stack SUM'].getfloat(key) for key in config['Stack SUM'].keys() if key != 'order']



    # define where to save all the data
    def create_out_dirs(self):

        if self.save_loc == None:
            self.out_dir = f'{self.session_path}process/xrf_fit/{self.sample_name}/{self.scan_name}/'
        else:
            self.out_dir = f'{self.save_loc}process/xrf_fit/{self.sample_name}/{self.scan_name}/'
        self.out_dir_pymca_file = self.out_dir+'data/'
        self.out_dir_elements   = self.out_dir+'elements/'
        self.out_dir_spectrum   = self.out_dir+'spectrum/'
        if not os.path.isdir(self.out_dir):
            os.makedirs(self.out_dir, exist_ok=True)
        if not os.path.isdir(self.out_dir_pymca_file):
            os.makedirs(self.out_dir_pymca_file, exist_ok=True)
        if not os.path.isdir(self.out_dir_elements):
            os.makedirs(self.out_dir_elements, exist_ok=True)
        if not os.path.isdir(self.out_dir_spectrum):
            os.makedirs(self.out_dir_spectrum, exist_ok=True)

    # Reads a NanoMAX file
    def readData(self):
        snitch = DM.mapping.stitchScans(self.scan_list,XRD=False,xrf_calibration = self.calibration,proposal=self.proposal,visit=self.visit)

        self.x = snitch['x_map']
        self.y = snitch['y_map']
        self.xrf = snitch['xrf_map']
        self.energy = snitch['energy'] 
        self.Emax = snitch['Emax']
        self.I0 = snitch['I0_map']   

        del snitch

        self.xrf = (self.xrf.transpose(2,0,1) /self.I0).transpose(1,2,0)
#            self.xrf_norm = ((self.xrf*I0_scale_factor)/(self.I0*self.dwell))
        self.xrf_avg = numpy.nanmean(self.xrf,axis=(0,1))
#            self.energy = fp['entry/snapshot/energy'][:]
#            self.x = fp['entry/measurement/pseudo/x'][0:npixels].reshape(*shape)
#            self.y = fp['entry/measurement/pseudo/y'][0:npixels].reshape(*shape)

    # Writes a PyMCA readible h5 file
    def writePymcafile(self,return_fname=False):
        pymca_file = os.path.join(f'{self.out_dir_pymca_file}_data_pymca_{self.scan_name}_{self.channel}.h5')
        self.pymca_file = pymca_file
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
        if return_fname:
            return pymca_file

    # Linear fitting of xrf stack data. Generate a h5 and/or tiff files with element concentration maps
    def fitElementsToFile(self, make_elements_h5=True, make_elements_tif=False,return_fname=False):

        elem_file_name = f'fitted_elements_{self.scan_name}_{self.channel}'
        self.elem_file_name = elem_file_name
        print(f'Generating elemental maps file: {elem_file_name}.h5')
        fastFit = FastXRFLinearFit()
        fastFit.setFitConfigurationFile(self.config_file)

        outbuffer = OutputBuffer(outputDir=self.out_dir_elements,
                        outputRoot= elem_file_name,
                        fileEntry= 'xrf_fits',#f'{self.sample_name}_{self.scan_name}',
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
        if return_fname:
            return self.elem_file_name

    # Fitting of average spectrum from image stack
    def fitAvgSpectrumToFile(self):
        spec_file = os.path.join(self.out_dir_spectrum + f'spectrum_{self.scan_name}_{self.channel}.h5')
        print(f'Writing average-spectrum fit file: {spec_file}')
        mca_fit = McaTheory()
        conf = ConfigDict.ConfigDict()
        conf.read(self.config_file)
        mca_fit.setConfiguration(conf)
        mca_fit.configure(conf)
        mca_fit.setData(self.xrf_avg)
        mca_fit.estimate()
        fit_result = mca_fit.startfit(digest=0)
        digest_file = os.path.join(self.out_dir_spectrum + f'spectrum_{self.scan_name}_{self.channel}.fit')
        mca_fit_result = mca_fit.digestresult(outfile=digest_file)
        #ConfigDict.prtdict(mca_fit_result)
        with h5py.File(spec_file, 'w') as sfp:
            self._recursive_save(sfp, 'mca_fit/', mca_fit_result)
            #sfp['measurement/I0_average'] = numpy.mean(self.I0)
            #sfp['measurement/dwell_time'] = self.dwell
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
    parser.add_argument('scan_list',nargs='+',type=int,help='List of scans')
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
