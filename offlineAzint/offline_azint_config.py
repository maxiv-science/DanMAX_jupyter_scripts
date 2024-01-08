# This file was generated on Mon Dec 11 10:57:23 2023 using
# /data/visitors/danmax/20231855/2023110808/process/azint/pxrd_cryo/scan-0100_pilatus_integrated.h5 
# as a template

# path to poni file [string]
poni_file = f'/data/visitors/danmax/20231855/2023110808/process/pxrd_cryo_LaB6_35kev_500mm.poni'

# path to mask file in npy format [string]
mask_file = f'/data/visitors/danmax/20231855/2023110808/process/pxrd_cryo_hot_px_mask.npy'

# settings used in the integration
user_config = {'poni': poni_file, # do not change this line
               'mask': mask_file, # do not change this line
               'radial_bins': 3000, # number of radial bins [int]
               'azimuth_bins': None, # number of azimuthal bins (or list specifying the azimuthal bins)
               'n_splitting': 15, # pixel splitting - number of subdivisions [int[]]
               'error_model': None, # Error model: use 'None' or 'poisson' [str]
               'polarization_factor': 0.999997, # Polarization factor - use 0.999997 [float]
               'unit': '2th' # Radial unit: use either 'q' or '2th' [str]
               } 

# Specification of scans to be integrated [list of int]
scans = [100]
#scans = range(firstScan, lastScan+1) # Specified as a range, remeber to end with lastScan+1
#scans = [scan1, scan2, scanN] # Manually specified list