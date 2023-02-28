# -*- coding: utf-8 -*- 
from pathlib import Path
from timeit import repeat
from typing import Callable, Dict, List, Optional, Union, IO
from unicodedata import name
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.stats import moment
from scipy.signal import find_peaks
from scipy.ndimage import shift
from datetime import date
import pickle
import os
import xlsxwriter as xls
from scipy.ndimage.morphology import binary_dilation, binary_erosion
from read_roi import read_roi_zip
import pandas as pd

import eyepy as ep

from heyex_tools import vol_reader
from grade_ml_segmentation import macustar_segmentation_analysis

from relEZIquantification.relEZIquantification_os import *
from relEZIquantification.relEZIquantification_utils import *


class OCTmap(object):

    date_of_origin = None

    name = ""

    scan_size = None

    scan_field = None



    def __init__(
            self,
            name: Optional[str] = None,
            date_of_origin: Optional[date] = None,
            scan_size: Optional[tuple] = None,
            scan_field: Optional[tuple] = None,
            ):
        self.name = name
        self.date_of_origin = date_of_origin
        self.scan_size = scan_size
        self.scanfield = scan_field

class Distance_map(OCTmap):

    distance_array = None # 2d array containing the site specific distance values
    std_array = None # 2d array containing the site specific standard diviation

    def __init__(self,
        name: Optional[str] = None, 
        date_of_origin: Optional[date] = None, 
        scan_size: Optional[tuple] = None, 
        scan_field: Optional[tuple] = None,
        distance_array: Optional[np.ndarray] = None,
        std_array: Optional[np.ndarray] = None
        ):
        super().__init__(name, date_of_origin, scan_size, scan_field)
        self.distance_array = distance_array
        self.std_array = std_array
   
class SSDmap:
    
    name = None 

    ez_ssd_map = None

    elm_ssd_map = None 

    file_location = None

    def __new__(
            cls,
            name = None,
            stackwidth = None,
            ez_ssd_map = None,
            elm_ssd_map = None,
            file_location = None
        ):
        return object.__new__(cls)

    def __init__(self, 
            name: Optional[str] = None, 
            ez_ssd_map: Optional[Distance_map] = None, 
            elm_ssd_map: Optional[Distance_map] = None,
            file_location: Union[str, Path, IO] = None
    ):
        self.name = name
        self.ez_ssd_map = ez_ssd_map
        self.elm_ssd_map = elm_ssd_map 
        self.file_location = file_location     

    
    def interpolate_map(array,method): #method to fill the gaps of the distance and std maps

        x = np.arange(0, array.shape[1])
        y = np.arange(0, array.shape[0])
        #mask invalid values
        array = np.ma.masked_invalid(array)
        xx, yy = np.meshgrid(x, y)
        #get only the valid values
        x1 = xx[~array.mask]
        y1 = yy[~array.mask]
        newarr = array[~array.mask]

        from scipy import interpolate
        GD1 = interpolate.griddata((x1, y1), newarr.ravel(),
                          (xx, yy),
                             method=method)
        return GD1

    @classmethod
    def create_ssd_maps(
        cls,
        project_name: Optional[str] = None,
        data_list: Optional[Dict] = None,
        fovea_coords: Optional[Dict] = None,
        scan_size: Optional[tuple] = None,
        scan_field:Optional[tuple] = None,
        stackwidth: Optional[int] = None,
        ref_layer: Optional[str] = None,
    ) -> None:

        """
        Args:
            project (Optional[str] = None): project name
            data_list (Optional[Dict]): file dict where files are stored
            fovea_coords (Optional[Dict]): location of fovea
                !!! B-scan number counted from bottom to top like HEYEX !!! -> easier handling for physicians
                bscan (int): Number of B-scan including fovea
                ascan (int): Number of A-scan including fovea
            scan_size (Optional[tuple]): scan field size in x and y direction
                x (int): Number of B-scans
                y (int): Number of A-scans
            stackwidth (Optional[int]): number of columns for a single profile
            ref_layer (Optional[str]): layer to flatten the image 
        """
        if not project_name:
            raise ValueError("Project name not given")

        if not data_list:
            raise ValueError("Dictionary of data_list not given")

        if not fovea_coords:
            raise ValueError("Dictionary of fovea coords not given")

        if not scan_size:
            raise ValueError("Scan_size (tuple) of recording not given")

        if not scan_field:
            raise ValueError("Scan_size (tuple) of recording not given")

        if not stackwidth:
            raise ValueError("Stackwidth (tuple) of recording not given")
        else: 
            stackwidth_fix = np.copy(stackwidth)
            factor = 1
        
        if not ref_layer:
            ref_layer = 11 # BM by default
        else:
            if ref_layer == "RPE" or ref_layer == 10:
                ref_layer = 10
            elif ref_layer == "BM" or ref_layer == 11:
                ref_layer = 11
            else:
                raise ValueError("Layer name for reference layer not vaild")


        # central bscan/ascan, number of stacks (nos)
        c_bscan = scan_size[0] // 2 + scan_size[0] % 2
        c_ascan = scan_size[1] // 2 + scan_size[1] % 2
        nos = scan_size[1] // stackwidth # number of stacks

        ez_distance = np.empty(shape=[0, scan_size[0], nos])
        elm_distance = np.empty_like(ez_distance)

        
        # iterate  over .vol-list
        for vol_id in data_list:

            # current distance map
            curr_ez_distance = np.empty((1, scan_size[0], nos))
            curr_ez_distance[:] = np.nan
            curr_elm_distance = np.full_like(curr_ez_distance, np.nan)
            

            # d_bscan (int): delta_bscan = [central bscan (number of bscans // 2)] - [current bscan]
            try:
                fovea_bscan, fovea_ascan = fovea_coords[vol_id]
            except:
                print("ID %s is missing in Fovea List " % vol_id)
                continue
            
            # change orientation from top down, subtract on from coords to keep 0-indexing of python            
            fovea_bscan = scan_size[0] - fovea_bscan

            
            # get data
            ms_analysis = macustar_segmentation_analysis.MacustarSegmentationAnalysis(
                vol_file_path=data_list[vol_id],
                model_file_path=None,
                use_gpu=True,
                cuda_device=0,
                normalize_mean=0.5,
                normalize_std=0.25,
                cache_segmentation=True
)

            # laterality 
            lat = ms_analysis._vol_file.header.scan_position

            if lat == "OS": # if left eye is processed
                fovea_ascan = ms_analysis._vol_file.header.size_x - fovea_ascan + 1
            else:
                fovea_ascan = fovea_ascan -1

            d_bscan  = c_bscan - fovea_bscan


            # check if given number of b scans match with pre-defined number 
            if ms_analysis._vol_file.header.num_bscans != scan_size[0]:
                print("ID: %s has different number of bscans (%i) than expected (%i)" % (vol_id, ms_analysis._vol_file.header.num_bscans, scan_size[0]))
                continue

            # check if given number of a scans match with pre-defined number 
            if ms_analysis._vol_file.header.size_x != scan_size[1]:
                print("ID: %s has different number of ascans (%i) than expected (%i)" % (vol_id, ms_analysis._vol_file.header.size_x, scan_size[1]))
                factor = ms_analysis._vol_file.header.size_x / scan_size[1]
                if factor * stackwidth_fix >= 1 and  factor % 1 == 0:
                    stackwidth = int(factor * stackwidth_fix)
                
            
            for bscan, seg_mask, ez, elm in zip(
                ms_analysis._vol_file.oct_volume_raw[::-1][max([-d_bscan, 0]): scan_size[0] + min([-d_bscan, 0])], # read raw data
                ms_analysis.classes[::-1,:,:][max([-d_bscan, 0]): scan_size[0] + min([-d_bscan, 0])], # read seg mask
                curr_ez_distance[0, max([d_bscan, 0]): scan_size[0] + min([d_bscan, 0]), :], # write
                curr_elm_distance[0, max([d_bscan, 0]): scan_size[0] + min([d_bscan, 0]), :], # write
                ):
                
                if lat == "OS":
                    bscan = np.flip(bscan,1)
                    seg_mask = np.flip(seg_mask,1)
                
                # get start position to read data
                d_ascan = int((factor * c_ascan) - fovea_ascan)
                shift = min([d_ascan, 0])
                start_r = int(- shift + ((factor * c_ascan) - (stackwidth//2) + shift) % stackwidth) # start reading
                start_w = int(max([(((factor * c_ascan) - (stackwidth//2)) // stackwidth) - ((fovea_ascan) - (stackwidth//2)) // stackwidth, 0]))
                n_st = int((ms_analysis._vol_file.header.size_x - start_r - max([d_ascan,0])) // stackwidth) # possible number of stacks 
                

                # get rois
                raw_roi, seg_mask_roi = get_roi_masks(bscan, ref_layer, ms_analysis._vol_file.header.size_x, seg_mask)
                

                for i in range(n_st):

                    rpe_roi = np.copy(raw_roi[:,start_r + i * stackwidth: start_r + (i + 1) * stackwidth])
                    rpe_roi[seg_mask_roi[:,start_r + i * stackwidth: start_r + (i + 1) * stackwidth] != 10] = np.nan

                    rpe_peak = find_peaks(np.nanmean(rpe_roi,1))[0]
                    if len(rpe_peak) >= 1:
                        rpe_peak = rpe_peak[-1]
                    else:
                        rpe_peak = None


                    ez_roi = np.copy(raw_roi[:,start_r + i * stackwidth: start_r + (i + 1) * stackwidth])
                    ez_roi[seg_mask_roi[:,start_r + i * stackwidth: start_r + (i + 1) * stackwidth] != 8] = np.nan

                    ez_peak = find_peaks(np.nanmean(ez_roi,1))[0]
                    if len(ez_peak) == 1:
                        ez_peak = ez_peak[0]
                    elif len(ez_peak) == 2:
                        if ez_peak[0] == np.max(ez_peak):
                            ez_peak = ez_peak[0]
                        else:
                            ez_peak = None
                    else:
                        ez_peak = None
                    
        
                    elm_roi = np.copy(raw_roi[:,start_r + i * stackwidth: start_r + (i + 1) * stackwidth])
                    elm_roi[np.roll(seg_mask_roi[:,start_r + i * stackwidth: start_r + (i + 1) * stackwidth] != 7, -3)] = np.nan

                    elm_peak = find_peaks(np.nanmean(elm_roi,1))[0]
                    if len(elm_peak) >= 1:
                        elm_peak = elm_peak[0]
                    else:
                        elm_peak = None                    
                            
                        #plt.plot(np.arange(len(i_profile)), i_profile,
                        #             rpe_peak, i_profile[rpe_peak], "x",
                        #             ez_peak, i_profile[ez_peak], "x",
                        #             elm_peak, i_profile[elm_peak], "x")
                        

                    # set distances
                    if rpe_peak:
                        if ez_peak:
                            ez[start_w + i] = rpe_peak - ez_peak
                        if elm_peak:
                            elm[start_w + i] = rpe_peak - elm_peak


            ez_distance = np.append(ez_distance, curr_ez_distance, axis=0)
            elm_distance = np.append(elm_distance, curr_elm_distance, axis=0)

            # set stackwith and factor to default
            stackwidth = stackwidth_fix
            factor = 1
            
        # set all zeros to nan
        ez_std = np.nanstd(ez_distance, axis=0)
        ez_std[ez_std == np.nan] = 0.
        ez_dist = np.nanmean(ez_distance, axis=0)
        ez_dist[ez_dist == np.nan] = 0.

        elm_std = np.nanstd(elm_distance, axis=0)
        elm_std[elm_std == np.nan] = 0.
        elm_dist = np.nanmean(elm_distance, axis=0)
        elm_dist[elm_dist == np.nan] = 0.    


        # create ssd map containing the created maps
        return cls(
                name = "ssd_map_" + project_name,
                ez_ssd_map = Distance_map("ez_ssd", date.today(), scan_size, scan_field, cls.interpolate_map(ez_dist,"cubic"), cls.interpolate_map(ez_std,"cubic")),
                elm_ssd_map = Distance_map("elm_ssd", date.today(), scan_size, scan_field, cls.interpolate_map(elm_dist,"cubic"), cls.interpolate_map(elm_std,"cubic")),
                file_location = None
        )

        
    def save_ssd(
            self,
            directory: Union[str, Path, IO, None] = None
            ):
        
        """
        Args:
            directory (Union[str, Path, IO, None]): directory where ssd maps should be stored
            save OCTMap Object in pkl-file with the order ez before elm
        
        """
        
        if not directory:
            directory = ""
            
        sdd_file_path = os.path.join(
                directory, self.name + "_" +
                            self.ez_ssd_map.date_of_origin.strftime("%Y-%m-%d") 
                            + ".pkl")
        
        with open(sdd_file_path, "wb") as outp:
            pickle.dump(self, outp, pickle.HIGHEST_PROTOCOL)
            
        self.file_location = sdd_file_path

    @staticmethod
    def load_ssd(
            directory: Union[str, Path, IO, None] = None,
            filename: Optional[str] = None
            ):
        
        if not filename:
            filename = "ssd"
            
            
        if directory:
            obj_list = get_list_by_format(directory, [".pkl"])
            for file in obj_list[".pkl"]:
                if filename in file:
                    with open(file, 'rb') as inp:
                        tmp_obj = pickle.load(inp)
                        if type(tmp_obj) is SSDmap:
                            return tmp_obj # Initialization based on the loaded instance         
        else:
            raise ValueError("No directory to load ssd maps is given\n Try to create ssd first by method 'create_ssd_maps()' and than save it by save_ssd_maps()")

class Mean_rpedc_map(Distance_map):

    file_location = None

    def __new__(
            cls,
            name = None,
            date_of_origin = None,
            scan_size = None,
            scan_field = None,
            distance_array = None,
            std_array = None,
            file_location = None
        ):
        return object.__new__(cls)

    
    def __init__(self, 
    name: Optional[str] = None, 
    date_of_origin: Optional[date] = None, 
    scan_size: Optional[tuple] = None, 
    scan_field: Optional[tuple] = None, 
    distance_array: Optional[np.ndarray] = None, 
    std_array: Optional[np.ndarray] = None,
    file_location: Union[str, Path, IO] = None
    ):
        super().__init__(name, date_of_origin, scan_size, scan_field, distance_array, std_array)  
        self.file_location = file_location 

    @classmethod
    def create_mean_rpedc_map(
            cls,
            project_name: Optional[str] = None,
            data_list: Optional[Dict] = None,
            folder_path: Union[str, Path, IO] = None, # path to get rpedc maps 
            fovea_coords: Optional[Dict] = None,
            scan_size: Optional[tuple] = None,
            scan_field: Optional[tuple] = None
            ):
        """
        Args:
            folder_path (Union[str, Path, IO]): folder path where files are storedB
            fovea_coords (Optional[Dict]): location of fovea
                !!! B-scan number counted from bottom to top like HEYEX !!! -> easier handling for physicians
                bscan (int): Number of B-scan including fovea
                ascan (int): Number of A-scan including fovea
            scan_size (Optional[tuple]): scan field size in x and y direction
                x (int): Number of B-scans
                y (int): Number of A-scans
            stackwidth (Optional[int]): number of columns for a single profile
            ref_layer (Optional[str]): layer to flatten the image 
            base_layer (Optional[str]): pre segmented layer 
                "masks": if segmentation mask is used
                "vol": if segmentation by .vol-file is used
            *args: file formats that contain the data
        """
        if not project_name:
            raise ValueError("Project name not given")

        if not data_list:
            raise ValueError("Dictionary of data_list not given")
                   
        if not fovea_coords:
            raise ValueError("Dictionary of fovea coords not given")

        if not scan_size:
            raise ValueError("Scan_size (tuple) of recording not given")

        if not scan_field:
            raise ValueError("Scan_size (tuple) of recording not given")

            
            
        # central bscan/ascan, number of stacks (nos)
        c_bscan = scan_size[0] // 2 + scan_size[0] % 2
        c_ascan = scan_size[1] // 2 + scan_size[1] % 2
            
        # get list of all rpedc maps
        rpedc_list = ut.get_rpedc_list(folder_path)
        
        
        rpedc_thickness = np.empty(shape=[0, 640, scan_size[1]])
        
        for ids in rpedc_list.keys():
            if ids in fovea_coords.keys():

                # load map
                maps = cv2.imread(rpedc_list[ids], flags=(cv2.IMREAD_GRAYSCALE | cv2.IMREAD_ANYDEPTH)).astype(float)
                                
                
                
                # d_bscan (int): delta_bscan = [central bscan (number of bscans // 2)] - [current bscan]
                fovea_bscan, fovea_ascan = fovea_coords[ids]
                    
                # change orientation from top down, subtract on from coords to keep 0-indexing of python            
                fovea_bscan = scan_size[0] - (fovea_bscan +1) 

                # laterality 
                vol_data = ep.Oct.from_heyex_vol(data_list[ids])
                lat = vol_data._meta["ScanPosition"]

                if lat == "OS": # if left eye is processed
                    fovea_ascan = scan_size[1] - (fovea_ascan +1)
                    maps = np.flip(maps, 1)
                else:
                    fovea_ascan = fovea_ascan -1

                d_bscan  = c_bscan - fovea_bscan
                d_ascan = c_ascan - fovea_ascan
                

                rpedc_thickness = np.append(rpedc_thickness, shift(maps, (int(640./241.)*d_bscan, d_ascan))[None, ...], axis=0) # add dimension to append stack
        
      
        rpedc_thickness[rpedc_thickness <= 0.1] = np.nan # invalid values are set to nan
        
        # create and return instance
        return cls(
                name = "mean_rpedc_map_" + project_name,
                date_of_origin = date.today(),
                scan_size = scan_size,
                scan_field = scan_field,
                distance_array = np.nanmean(rpedc_thickness, axis=0),
                std_array = np.nanstd(rpedc_thickness, axis=0),
                file_location = None            
        )

    def save_mean_rpedc_map(            
            self,
            directory: Union[str, Path, IO, None] = None
            ):
            
            """
            Args:
                directory (Union[str, Path, IO, None]): directory where mean_rpedc_map should be stored
            
            save OCTMap Object in pkl-file 
            """
            
            if not directory:
                directory = ""
                
            mrpedc_file_path = os.path.join(
                    directory, self.name + "_" + 
                                self.date_of_origin.strftime("%Y-%m-%d") 
                                + ".pkl")
            
            with open(mrpedc_file_path, "wb") as outp:
                pickle.dump(self, outp, pickle.HIGHEST_PROTOCOL)

                
            self.file_location = mrpedc_file_path

    @staticmethod
    def load_mean_rpedc_map(
            directory: Union[str, Path, IO, None] = None,
            filename: Optional[str] = None
            ):
        
        if not filename:
            filename = "mean_rpedc"
        
        if directory:
            obj_list = ut.get_list_by_format(directory, [".pkl"])
            for file in obj_list[".pkl"]:
                if filename in file:
                    with open(file, 'rb') as inp:
                        tmp_obj = pickle.load(inp)
                        if type(tmp_obj) is Mean_rpedc_map:
                            return tmp_obj # Initialization based on the loaded instance                 
        else:
            raise ValueError("No directory to load mean_rpedc_map maps is given\n Try to create mean_rpedc_map first by method 'create_mean_rpedc_map()'")            

class RelEZI_map(OCTmap):

    _series_uid = None

    _volfile_path = None

    _stackwidth = None

    _scan_area = None 
    
    _laterality = None

    _fovea_coordinates = None

    _ezi_map = None

    _elmi_map = None

    _excluded_maps = {}


    def __init__(
        self,
        name: Optional[str] = None,
        date_of_origin: Optional[date] = None, 
        scan_size: Optional[tuple] = None, 
        scan_field: Optional[tuple] = None,
        scan_area: Optional[tuple] = None,
        stackwidth = None,
        series_uid: Optional[str] = None,
        volfile_path: Union[str, Path, IO] = None,
        laterality: Optional[str] = None,
        fovea_coordinates: Optional[tuple] = None,
        ezi_map: Optional[np.ndarray] = None,
        elmi_map: Optional[np.ndarray] = None,
        excluded_maps: Optional[Dict] = None
        ):
        super().__init__(name, date_of_origin, scan_size, scan_field)
        self._stackwidth = stackwidth
        self._scan_area = scan_area
        self._series_uid = series_uid
        self._volfile_path = volfile_path
        self._laterality = laterality
        self._fovea_coordinates = fovea_coordinates
        self._ezi_map = ezi_map
        self._elmi_map = elmi_map
        self._excluded_maps = excluded_maps

    @property
    def laterality(self):
        return self._laterality

    @property
    def series_uid(self):
        return self._series_uid

    @property
    def fovea_coordinates(self):
        return self._fovea_coordinates

    @property
    def ezi_map(self):
        return self._ezi_map

    @property
    def elmi_map(self):
        return self._elmi_map

    @property
    def excluded_maps(self):
        return self._excluded_maps

class Visit:

    vid = None

    date_of_recording = None

    relEZI_map_OD = None

    relEZI_map_OS = None

    def __init__(
        self,
        vid: Optional[str] = None,
        date_of_recording: Optional[date] = None,
        relEZI_map_OD: Optional[RelEZI_map] = None,
        relEZI_map_OS: Optional[RelEZI_map] = None
        ):
        self.vid = vid
        self.date_of_recording = date_of_recording
        self.relEZI_map_OD = relEZI_map_OD
        self.relEZI_map_OS = relEZI_map_OS

    def add(self, map: Optional[RelEZI_map] = None):

        if map.laterality == "OD":
            if not self.relEZI_map_OD:
                self.relEZI_map_OD = map
                return False 
            else:
                return True 
        elif map.laterality == "OS":
            if not self.relEZI_map_OS:
                self.relEZI_map_OS = map
                return False 
            else:
                return True 
        else:
            raise ValueError("Map attribute 'laterality' is not correct") 

    def get_maps(self):

        maps_list = []
        if self.relEZI_map_OD:
            maps_list.append(self.relEZI_map_OD)
        if self.relEZI_map_OS:
            maps_list.append(self.relEZI_map_OS)
        
        return maps_list

class Patient:

    pid = None

    day_of_birth = None 

    visits = None

    def __init__(
        self,
        pid: Optional[str] = None,
        day_of_birth: Optional[date] = None,
        visits: Optional[List[Visit]] = None
        ):
        self.pid = pid
        self.day_of_birth = day_of_birth
        self.visits = [visits]

    def add(self, map: Optional[RelEZI_map] = None, visitdate: Optional[date] = None, vid: Optional[str] = None):

        for i, visit in enumerate(self.visits):
            
            if visit.date_of_recording == visitdate: # same visit
                if visit.add(map):
                    print("Visit already exists")
                break
            
            if visit.date_of_recording > visitdate:
                if  i < len(self.visits) -1: 
                    continue
                else:
                    if map.laterality == "OD":
                        self.visits.append(Visit(vid, visitdate, map, None))
                        break
                    else: # "OS"
                        self.visits.append(Visit(vid, visitdate, None, map))
                        break

            if visit.date_of_recording < visitdate:
                if map.laterality == "OD":
                    self.visits.insert(i, Visit(vid, visitdate, map, None))
                else: # "OS"
                    self.visits.insert(i, Visit(vid, visitdate, None, map))
                break 
            

if __name__ == "__main__":
    pass


