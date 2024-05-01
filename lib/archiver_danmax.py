"""
DanMAX archiver functions - ONLY AVAILABLE ON THE MAX IV NETWORK

Based on the ArchiverTool module by Stephen Molly https://gitlab.maxiv.lu.se/stepmo/ArchiverTool
"""
import ArchiverTool.archiver_tool as at
import numpy as np
from datetime import datetime, timedelta
import sys
sys.path.append('../')
import DanMAX as DM


## DanMAX specific archiver attribute aliases
#                    alias             database                 attribute name
attr_alias = {'hdcm_x2_perp_temp'    :['danmax'     ,'b304a-oa04/dia/tco-01/temperature'],
              'hdcm_x2_pitch_temp'   :['danmax'     ,'b304a-oa04/dia/tco-02/temperature'],
              'hdcm_x2_roll_temp'    :['danmax'     ,'b304a-oa04/dia/tco-03/temperature'],
              'hdcm_x1_temp'         :['danmax'     ,'b304a-oa04/dia/tco-04/temperature'],
              'hdcm_x1_mask_temp'    :['danmax'     ,'b304a-oa04/dia/tco-05/temperature'],
              'hdcm_x2_temp'         :['danmax'     ,'b304a-oa04/dia/tco-06/temperature'],
              'hdcm_x2_mask_temp'    :['danmax'     ,'b304a-oa04/dia/tco-07/temperature'],
              'mlm_m2_bragg_temp'    :['danmax'     ,'b304a-oa06/dia/tco-01/temperature'],
              'mlm_m2_parallel_temp' :['danmax'     ,'b304a-oa06/dia/tco-02/temperature'],
              'mlm_m2_perp_temp'     :['danmax'     ,'b304a-oa06/dia/tco-03/temperature'],
              'mlm_m2_pitch_temp'    :['danmax'     ,'b304a-oa06/dia/tco-04/temperature'],
              'mlm_m2_roll_temp'     :['danmax'     ,'b304a-oa06/dia/tco-05/temperature'],
              'mlm_m1_temp'          :['danmax'     ,'b304a-oa06/dia/tco-06/temperature'],
              'mlm_m1_mask_temp'     :['danmax'     ,'b304a-oa06/dia/tco-07/temperature'],
              'mlm_m2_temp'          :['danmax'     ,'b304a-oa06/dia/tco-08/temperature'],
              'mlm_m2_mask_temp'     :['danmax'     ,'b304a-oa06/dia/tco-09/temperature'],
              'ea_hutch_temp'        :['danmax'     ,'b304a-a100433/vnt/tse-03/temperature'],
              'fe_bpx_x'             :['danmax'     ,'b304a-fe/dia/bpx-02-x-facade/position'],
              'fe_bpx_y'             :['danmax'     ,'b304a-fe/dia/bpx-02-y-facade/position'],
              'r3_current'           :['accelerator','r3-319s2/dia/dcct-01/instantaneouscurrent'],
              #'xbpm_x'           :['accelerator','b304a/dia/xbpm-01/x'],
              #'xbpm_y'           :['accelerator','b304a/dia/xbpm-01/y'],
             }

def get_attr_aliases():
    """Return a list of available archiver attribute aliases"""
    return list(attr_alias.keys())
    
def listAttributes(database,str_query):
    """
    Return list of available archiver attributes for the provided database and search query
    Parameters
        database  - str ('accelerator', 'danmax', etc.)
        str_query - str - Use '.*' as wildcard. Ex: 'b304a.*oa04.*tco-06.*temp.*'
    Return
        attr_list - list of available attributes
    """
    return at.search_for_attributes(str_query,database)

def getArchiverDictionary(database, str_query, start, end ):
    """
    Get archiver data dictionary of all attributes matching the search query within the specified time
    Parameters
        database  - str ('accelerator', 'danmax', etc.)
        str_query - str - Use '.*' as wildcard. Ex: 'b304a.*oa04.*tco-06.*temp.*'
        start     - str - start time as formatted string "%Y-%m-%dT%H:%M:%S" (yyyy-mm-ddTHH:MM:SS:)
        end       - str - end time as formatted string "%Y-%m-%dT%H:%M:%S" (yyyy-mm-ddTHH:MM:SS:)
    Return
        data_dic  - dictionary - attribute names as keys,  [time, data] as values
    """

    # database connection string
    DB_CONN_STR = f'{at.DB_TYPE}:{at.DB_USER}@{at.DB_URL}:{at.DB_PORT}/{at.DB_NAMES[database]}'
    # get data for each attribute matching the query 
    dataset = at.get_data(str_query, start, end, DB_CONN_STR)
    # convert to dictionary
    data_dic = {dset.name:[np.array(dset.time),np.array(dset.data)] for dset in dataset}
    return data_dic


def getArchiverData(fname, str_query='',database='danmax', start=None, end=None,interpolate_timestamps=True,relative_time=True):
    """
    Get archiver data of DanMAX specific archiver attribute alias or of all attributes matching the search query 
    within the absolute time of the provided scan or in the time range provided by start-end. Interpolate the archive
    attribute data to line up with the scan absolute timestamps.
    Parameters
        fname                  - str  - Full master .h5 file path
        str_query              - str  - Looks up the query in the attr_alias dictionary, otherwise search for all 
                                        matching queries. Use '.*' as wildcard. Ex: 'b304a.*oa04.*tco-06.*temp.*'
        database               - str  - ('accelerator', 'danmax', etc.). Automatically set if the str_query is in 
                                        the alias dictionary
        start                  - str  - Start time. If None (default) use the master .h5 scan timestamps, otherwise
                                        use formatted string "%Y-%m-%dT%H:%M:%S" (yyyy-mm-ddTHH:MM:SS:)
        end                    - str  - End time. Same as start
        interpolate_timestamps - bool - (True) Toggle interpolation to the scan absolute timestamps. If False, 
                                        timestamps are returned as datetime objects
        relative_time          - bool - (True) Relative timestamps (t0=0)
    Return
        data_dic  - dictionary - attribute names as keys,  [time, data] as values
    """
    # read the .h5 meta data as absolute
    meta = DM.getMetaData(fname,relative=False)
    t = meta['time']
    # convert to timestamp strings
    t_str = DM.getTimeStamps(t)
    t_str = [ts[:-7].replace(' ','T') for ts in t_str]
    
    # set t-zero
    if relative_time:
        t0=t[0]
    else:
        t0=0.
        
    if start is None:
        start = t_str[0]
    if end is None:
        end = t_str[-1]
    
    # check if the str_query is in the alias dictionary
    attr_name = None
    if str_query.lower() in attr_alias:
        attr_name = str_query.lower()
        database, str_query = attr_alias[str_query]
    
    data_dic = getArchiverDictionary(database, str_query, start, end )
    
    # rename keys
    if attr_name is None:
        keys = list(data_dic.keys())
        data_dic = {key.split('10000/')[1]:data_dic[key] for key in keys}
    else:
        key = list(data_dic.keys())[0]
        data_dic = {attr_name:data_dic[key]}


    for key in data_dic:
        x,y = data_dic[key]
        # convert to timestamp
        x =  np.array([_x.timestamp() for _x in x])-t0
        if interpolate_timestamps:
            #dx = np.array([(_x-x[0]).total_seconds() for _x in x])
            y_interp = np.interp(t-t0, x, y)
            data_dic[key] = [t-t0,y_interp]
        else:
            data_dic[key][0] = x
    return data_dic
















