# -*- coding: utf-8 -*- 
from pathlib import Path
from timeit import repeat
from typing import Callable, Dict, List, Optional, Union, IO
from unicodedata import name
from weakref import ref
import numpy as np
import sys
import matplotlib.pyplot as plt
from PIL import Image
from scipy.stats import moment
from scipy.signal import find_peaks
from scipy.ndimage import shift
from datetime import date
import pickle
import os
import cv2
import xlsxwriter as xls
from scipy.ndimage.morphology import binary_dilation
from read_roi import read_roi_zip
import pandas as pd

import eyepy as ep

from heyex_tools import vol_reader
from grade_ml_segmentation import macustar_segmentation_analysis

from rel_ez_intensity.getAdjacencyMatrix import plot_layers
from rel_ez_intensity.seg_core import get_retinal_layers
from rel_ez_intensity import utils as ut



class OCTMap:

    def __init__(
            self,
            vid: Optional[int] = None,
            name: str = None,
            volfile_path: Union[str, Path, IO] = None, 
            date_of_origin: Optional[date] = None, # if REZI-Map the day of recording is stored
            scan_size: Optional[tuple] = None,
            stackwidth: Optional[int] = None,
            laterality: Optional[str] = None,
            fovea_coords: Optional[tuple] = None,
            octmap: Optional[Dict] = None,
            ) -> None:
        self.vid = vid
        self.name = name
        self.volfile_path = volfile_path
        self.date_of_origin = date_of_origin
        self.scan_size = scan_size
        self.stackwidth = stackwidth
        self.laterality = laterality
        self.fovea_coords = fovea_coords
        self.octmap = octmap

    

class Patient:

    def __init__(
        self,
        pid: Optional[str] = None,
        dob: Optional[str] = None,
        visits: Optional[List[OCTMap]] = None,

    ) -> None:
        self.pid = pid
        self.dob = dob
        self.visits = visits 

    

class RelEZIntensity:

    
    """
    CREATE RELEZI-MAPS


    1. Create List of .vol, .xml or .tif files 
        file-format depends on the study
        e.g.
            MacTel
                - .vol: Raw Data
                - .xml: Segmentation
                - .png: EZ_loss Maps

    2. Create Patient object by meta data

    3. Analyze rel EZ Intensity
        !!! mechanism to exclude region by:
            - Condition like Thickness of RPEDC
            - EZ-loss or Thickness Map
            - RPD 

        3.1 Iteration over .vol
            - Shift by fovea coords
            - exclusion
            - Plot 
            - Rpe or Bm coords
            - ssd-maps to define search area


    CREATE SSD-MAPS

    1. like CREATE RELEZI-MAPS
    
    2. Iteration over .vols
        2.1 search peaks
            - Old method by maximal peak decay between EZ and ELM
            - !!! Segmentation Mask !!!
            - Graph-Cut
        2.2 Stack all results in 3d-data cube 
            - mean
            - std
        

    """
    
    def __init__(
            self,
            project: Optional[str] = None,
            fovea_coords: Optional[Dict] = None,
            ez_distance_map: Optional[OCTMap] = None, # [mean distance, standard diviation]
            elm_distance_map: Optional[OCTMap] = None, # [mean distance, standard diviation]
            scan_size: Optional[tuple] = None,
            stackwidth: Optional[int] = None,
            patients: Optional[Dict] = {},
            base_layer: Optional[str] = None
            
            ) -> None:
        self.project = project
        self.fovea_coords = fovea_coords
        self.ez_distance_map = ez_distance_map
        self.elm_distance_map = elm_distance_map
        self.scan_size = scan_size
        self.stackwidth = stackwidth
        self.patients = patients
        self.base_layer = base_layer 
        self.ssd_dir = None
        self.area_exclusion = None
        self.mean_rpedc_map = None 
        self.mrpedc_dir = None
        
 
 

    def get_ez_elm_peak(self, i_profile, rpe_peak, ez_mean, ez_std, elm_mean, elm_std):
        
        
        # What to do if 2 peaks are found in the search 
            # Suggested to set the value to invalid. !!!

        if np.isnan([ez_mean, ez_std, elm_mean, elm_std]).any():
            return 0, 0

        # find ez and elm peaks
        left_border = int(np.round(rpe_peak - elm_mean - 2. * elm_std))
        peaks = left_border + find_peaks(i_profile[left_border : int(rpe_peak)])[0]
                        
        if 2 * ez_std < 1:
            ez_left = int(np.round(rpe_peak - ez_mean - 1))
            ez_right = int(np.round(rpe_peak - ez_mean + 1))
        else:
            ez_left = int(np.round(rpe_peak - ez_mean - 2. * ez_std)) 
            ez_right = int(np.round(rpe_peak - ez_mean + 2. * ez_std)) 
                        
        
        ez_peaks = peaks[np.logical_and(peaks >= ez_left, peaks <= ez_right)]


        # 3 possible cases 
        # first: no peak was found
        if len(ez_peaks) == 0:
            return 0, 0
        # second: the estimated valid case where only a single peaks was found 
        elif len(ez_peaks) == 1:
            ez_peak = ez_peaks[0]
        # third: invalid case if more then one peak was found
        else:
            return 0, 0
        
        

        if 2 * elm_std < 1:
            elm_left = int(np.round(rpe_peak - elm_mean - 1))
            elm_right = int(np.round(rpe_peak - elm_mean + 1))
        else:
            elm_left = int(np.round(rpe_peak - elm_mean - 2. * elm_std)) 
            if ez_peak <= int(np.round(rpe_peak - elm_mean + 2. * elm_std)):
                elm_right = ez_peaks -1 
            else:
                elm_right = int(np.round(rpe_peak - elm_mean + 2. * elm_std)) 
        
                        
        elm_peaks = peaks[np.logical_and(peaks >= elm_left, peaks <= elm_right)]
         
        # 3 possible cases 
        # first: no peak was found
        if len(elm_peaks) == 0:
            return ez_peak, 0
        # second: the estimated valid case where only one peaks was found 
        elif len(elm_peaks) == 1:
            elm_peak = elm_peaks[0]
        # third: invalid case if more then one peak was found
        else:
            return ez_peak, 0
            
        return ez_peak, elm_peak

    def get_rpe_peak(self, i_profile):
        peaks = find_peaks(i_profile[35:45])[0]
        if len(peaks) > 0:
            return 35 + peaks[np.where(i_profile[35 + peaks] == max(i_profile[35 + peaks]))[0]][-1]
        else:
            return 38
    
    def get_rpedc_map(
        self,
        file_path: Union[str, Path, IO] = None,
        scan_size: Optional[tuple] = None,
        mean_rpedc: Optional[OCTMap] = None,
        laterality: Optional[str] = None,
        translation: Optional[tuple] = None
        ) -> np.ndarray:
        
        maps = cv2.imread(file_path, flags=(cv2.IMREAD_GRAYSCALE | cv2.IMREAD_ANYDEPTH))
        
        if laterality == "OS":
            maps = np.flip(maps, 1)

        atrophy = maps  == 0.
        
        maps_shifted = shift(maps, translation)
        atrophy_shifted = shift(atrophy, translation).astype(np.uint8)
        
        # substract mean thickness of rpedc plus 3 times std (Duke/AREDS Definition) 
        sub = maps_shifted - (mean_rpedc.octmap["mean"] + (3. * mean_rpedc.octmap["std"])) 


        sub = np.logical_or(sub > 0., maps_shifted <= 0.01).astype(np.uint8) # rpedc area
        
        sub_resized = cv2.resize(sub, scan_size[::-1], cv2.INTER_LINEAR) # cv2.resize get fx argument befor fy, so  the tuple "scan_size" must be inverted
        atrophy_resized = cv2.resize(atrophy_shifted, scan_size[::-1], cv2.INTER_LINEAR)
    
        # structure element should have a radius of 100 um (Macustar-format (
        # ~30 um per bscan => ~ 3 px in y-direction and 2 * 3 px to get size of rectangle in y-dir. 
        # ~ 10 um per ascan => ~ 10 px in x-direction and 2 * 10 px to get size of rectangle in x-dir. 
        struct = np.ones((4, 10), dtype=bool)
        sub_dilation = binary_dilation(sub_resized, structure = struct).astype(int) * 2
        atrophy_dilation = binary_dilation(atrophy_resized, structure = struct)   
        
        sub_dilation[atrophy_dilation] = 1 # atrophy

        return sub_dilation  
    
    def get_rpd_map(
        self,
        file_path: Union[str, Path, IO] = None,
        scan_size: Optional[tuple] = None,
        laterality: Optional[str] = None,
        translation: Optional[tuple] = None
        ) -> np.ndarray:

        roi = read_roi_zip(file_path)
        roi = list(roi.values())[0]
        
        def get_annotated_mask(roi, shape):
            x = np.array(roi['x'])
            y = np.array(roi['y'])
    
            x -= 1
            y -= 1
    
            mask = np.zeros(shape, dtype = int)
    
            mask = cv2.fillPoly(mask, pts=[np.array([x,y]).transpose()], color = 1)
    
            return mask.astype(bool)

        
        # get mask by annotation
        SHAPE = (640, 768) # !!!!! Should be dependent on the recording format of oct
        mask = get_annotated_mask(roi, SHAPE)

        # if left eye the map is flipped
        if laterality == "OS":
            mask = np.flip(mask, 1)
        
        # shift the fovea in the center
        mask_shifted = shift(mask, translation).astype(np.uint8)

        # resize to scan conditions
        mask_resized = cv2.resize(mask_shifted, scan_size[::-1], cv2.INTER_LINEAR) # cv2.resize get fx argument befor fy, so  the tuple "scan_size" must be inverted

        return mask_resized

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
                    directory, "mean_rpedc_" +
                                self.mean_rpedc_map.date_of_origin.strftime("%Y-%m-%d") 
                                + ".pkl")
            
            with open(mrpedc_file_path, "wb") as outp:
                pickle.dump(self.mean_rpedc_map, outp, pickle.HIGHEST_PROTOCOL)
                pickle.dump(self.mean_rpedc_map, outp, pickle.HIGHEST_PROTOCOL)
                
            self.mrpedc_dir = mrpedc_file_path

    def load_mean_rpedc_map(
            self,
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
                        if type(tmp_obj) is OCTMap:
                            self.mean_rpedc_map = tmp_obj
                     
        elif self.mrpedc_dir:
            self.mean_rpedc_map = pickle.load(open(self.mrpedc_dir, "r"))
        
        else:
            raise ValueError("No directory to load mean_rpedc_map maps is given\n Try to create mean_rpedc_map first by method 'create_mean_rpedc_map()'")
    
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
                directory, "ssd_" +
                            self.ez_distance_map.date_of_origin.strftime("%Y-%m-%d") 
                            + ".pkl")
        
        with open(sdd_file_path, "wb") as outp:
            pickle.dump(self.ez_distance_map, outp, pickle.HIGHEST_PROTOCOL)
            pickle.dump(self.elm_distance_map, outp, pickle.HIGHEST_PROTOCOL)
            
        self.ssd_dir = sdd_file_path
   
    def load_ssd(
            self,
            directory: Union[str, Path, IO, None] = None,
            filename: Optional[str] = None
            ):
        
        if not filename:
            filename = "ssd"
            
            
        if directory:
            obj_list = ut.get_list_by_format(directory, [".pkl"])
            for file in obj_list[".pkl"]:
                if filename in file:
                    with open(file, 'rb') as inp:
                        tmp_obj = pickle.load(inp)
                        if type(tmp_obj) is OCTMap:
                            self.ez_distance_map = tmp_obj
                            self.elm_distance_map = pickle.load(inp) # load next object from file

        elif self.ssd_dir:
            self.ez_distance_map = pickle.load(open(self.ssd_dir["ez"], "r"))
            self.elm_distance_map = pickle.load(open(self.ssd_dir["ez"], "r"))
        
        else:
            raise ValueError("No directory to load ssd maps is given\n Try to create ssd first by method 'create_ssd_maps()'")
        
    def calculate_relEZI_maps(
        self,
        folder_path: Union[str, Path, IO] = None,
        project: Optional[str] = None,
        fovea_coords: Optional[Dict] = None,
        scan_size: Optional[tuple] = None,
        stackwidth: Optional[int] = None,
        ref_layer: Optional[str] = None,
        base_layer: Optional[str] = None,
        area_exclusion: Optional[Dict] = None,
        *args
    ) -> None:

        """
        Args:
            folder_path (Union[str, Path, IO]): folder path where files are stored
            project (Optional[str]): project name 
            fovea_coords (Optional[Dict]): location of fovea
                !!! B-scan number counted from bottom to top like HEYEX !!! -> easier handling for physicians
                bscan (int): Number of B-scan including fovea
                ascan (int): Number of A-scan including fovea
            scan_size (Optional[tuple]): scan field size in x and y direction
                x (int): Number of B-scans
                y (int): Number of A-scans
            stackwidth (Optional[int]): number of columns for a single profile
            ref_layer (Optional[str]): layer to flatten the image 
            base_layer (Optional[str]): "vol" (default) if the layer segmentation of the vol-file ist used and "mask" if the segmentation mask of extern semantic segmentation method is used 
            area_exclusion ( Optional[Dict]): Method to determine area of exclusion 
                                            # if values (boolean) are True the area should not be analysed.
            *args: file formats that contain the data
 
 
        """
        if not project:
            project = self.project
        else:
            self.project = project

        if not fovea_coords:
            fovea_coords = self.fovea_coords
        else:
            self.fovea_coords = fovea_coords
        
        if not scan_size:
            scan_size = self.scan_size
        else:
            self.scan_size = scan_size
        
        if not stackwidth:
            stackwidth = self.stackwidth
        else:
            self.stackwidth = stackwidth
        
        if not ref_layer:
            ref_layer = "BM"
        
        if not base_layer:
            if not self.base_layer:
                base_layer = self.base_layer = "vol"
            else:
                base_layer = self.base_layer
        else:
            self.base_layer = base_layer

        if not area_exclusion:
            if not self.area_exclusion:
                area_exclusion = self.area_exclusion = "default"
            else:
                area_exclusion = self.area_exclusion
        else:
            self.area_exclusion = area_exclusion


        # data directories
        if args:
            if self.project == "macustar":
                data_dict, _ = ut.get_vol_list(folder_path, self.project)
            elif self.project == "mactel":
                data_dict, vids = ut.get_vol_list(folder_path, self.project)
        else:
            raise ValueError("no file format given")

        if base_layer == "masks":
            mask_dict = ut.get_mask_list(folder_path)

        if len(self.area_exclusion) == 1:
            if "rpedc" in self.area_exclusion.keys():
                ae_dict_1 = ut.get_rpedc_list(folder_path)
            if "rpd" in self.area_exclusion.keys():
                ae_dict_2 = ut.get_rpd_list(folder_path)
            if "ez_loss" in self.area_exclusion.keys():
                pass
            ###########################
        else: 
            for area_ex in self.area_exclusion.keys():
                if area_ex == "rpedc":
                    ae_dict_1 = ut.get_rpedc_list(folder_path)
                if area_ex == "rpd":
                    ae_dict_2 = ut.get_rpd_list(folder_path)

        
        # central bscan/ascan, number of stacks (nos)
        c_bscan = scan_size[0] // 2 + scan_size[0] % 2
        c_ascan = scan_size[1] // 2 + scan_size[1] % 2
        nos = scan_size[1] // stackwidth # number of stacks


        # iterate  over .vol-list
        for vol_id in data_dict:

            # current distance map
            curr_ez_intensity = np.zeros((scan_size[0], nos))
            curr_elm_intensity = np.zeros_like(curr_ez_intensity)
            curr_excluded = np.zeros_like(curr_ez_intensity)
            
            
            # get vol data from file
            vol_data = ep.Oct.from_heyex_vol(data_dict[vol_id])

            
            # check if given number of b scans match with pre-defined number 
            if vol_data._meta["NumBScans"] != scan_size[0]:
                print("ID: %s has different number of bscans (%i) than expected (%i)" % (ut.get_id_by_file_path(data_dict[vol_id]), vol_data._meta["NumBScans"], scan_size[0]))
                continue
            
            
            # check if mask dict contains vol id
            if base_layer == "masks":
                if vol_id in mask_dict.keys():
                    mask_list = mask_dict[vol_id]
                else:
                    print("ID: %s considered segmentation masks not exist" % vol_id)
                    continue

            
            # d_bscan (int): delta_bscan = [central bscan (number of bscans // 2)] - [current bscan]
            try:
                fovea_bscan, fovea_ascan = fovea_coords[vol_id]
            except:
                print("ID %s is missing in Fovea List " % vol_id)
                continue
            
            
            # change orientation from top down, subtract on from coords to keep 0-indexing of python            
            fovea_bscan = scan_size[0] - fovea_bscan
            
    
            # laterality 
            lat = vol_data._meta["ScanPosition"]

            if lat == "OS": # if left eye is processed
                fovea_ascan = scan_size[1] - fovea_ascan
            else:
                fovea_ascan = fovea_ascan -1


            # delta between real fovea centre and current fovea bscan position 
            d_bscan  = c_bscan - fovea_bscan
            # get start position to read data
            d_ascan = c_ascan - fovea_ascan

            if not self.elm_distance_map or not self.ez_distance_map:
                raise ValueError("Site specific distance maps not given")
                
                
            # if area_exception is "rpedc" get list of thickness maps 
            if "rpedc" in self.area_exclusion.keys():
                if vol_id in ae_dict_1.keys():
                    rpedc_map = self.get_rpedc_map(ae_dict_1[vol_id], self.scan_size, self.mean_rpedc_map, lat, (int(640./241.)*d_bscan, d_ascan))
                else:
                    print("ID: %s considered segmentation masks not exist" % vol_id)
                    continue
            
            # if area_exception is "rpedc" get list of thickness maps 
            if "rpd" in self.area_exclusion.keys():
                if vol_id in ae_dict_2.keys():
                    rpd_map = self.get_rpd_map(ae_dict_2[vol_id], self.scan_size, lat, (int(640./241.)*d_bscan, d_ascan))
                else:
                    rpd_map = np.zeros(self.scan_size).astype(bool)            

            
            for bscan, ez, elm, excl, ez_ssd_mean, ez_ssd_std, elm_ssd_mean, elm_ssd_std, idx_r, idx_w in zip(
                vol_data[::-1][max([-d_bscan, 0]): scan_size[0] + min([-d_bscan, 0])], # read
                curr_ez_intensity[max([d_bscan, 0]): scan_size[0] + min([d_bscan, 0]), :], # write
                curr_elm_intensity[max([d_bscan, 0]): scan_size[0] + min([d_bscan, 0]), :], # write
                curr_excluded[max([d_bscan, 0]): scan_size[0] + min([d_bscan, 0]), :], # write
                self.ez_distance_map.octmap["distance"][max([d_bscan, 0]): scan_size[0] + min([d_bscan, 0]), :], # write
                self.ez_distance_map.octmap["std"][max([d_bscan, 0]): scan_size[0] + min([d_bscan, 0]), :], # write                
                self.elm_distance_map.octmap["distance"][max([d_bscan, 0]): scan_size[0] + min([d_bscan, 0]), :], # write
                self.elm_distance_map.octmap["std"][max([d_bscan, 0]): scan_size[0] + min([d_bscan, 0]), :], # write
                range(max([-d_bscan, 0]), scan_size[0] + min([-d_bscan, 0])), # read
                range(max([d_bscan, 0]), scan_size[0] + min([d_bscan, 0]))
                ):

                if self.base_layer == None or self.base_layer == "vol":
                    try:
                        layer = bscan.layers[ref_layer].astype(np.uint16)
                    except:
                        continue
                if self.base_layer == "masks":
                    layer = ut.get_seg_by_mask(mask_list[idx_r], 10) # the last argument depends on the number of layer classes of the segmentation mask

                bscan_data = bscan._scan_raw
                if lat == "OS":
                    bscan_data = np.flip(bscan_data,1)
                    layer = np.flip(layer)
                
                

                shift = min([d_ascan, 0])
                start_r = - shift + (c_ascan - (stackwidth//2) + shift) % stackwidth # start reading
                start_w = max([((c_ascan - (stackwidth//2)) // stackwidth) - (fovea_ascan - (stackwidth//2)) // stackwidth, 0])
                n_st = (scan_size[1] - start_r - max([d_ascan,0])) // stackwidth # possible number of stacks 
                
                
                
                # create region of interest image 
                roi = np.zeros((50,scan_size[1])).astype(np.float32)
                
                for i, l in enumerate(layer):
                    if l < 496 and l > 0:
                        roi[:,i] = bscan_data[l-40:l+10,i]
                        
            
# =============================================================================
#                 # get ez and rpe boundary 
#                 imglayers = get_retinal_layers(roi) 
#                 
#                 plot_layers(roi, imglayers)                
# =============================================================================
                
                # iterate over bscans
                for i in range(n_st):
                    
                    
                    if not any(np.isnan(layer[start_r + i * stackwidth: start_r + (i + 1) * stackwidth])):
                        
                        
                        # excluding section
                        # excluding condition can be 
                            
                        # a thickness map of rpedc
                        if "rpedc" in self.area_exclusion.keys():
                            if any(rpedc_map[idx_w, start_r + i * stackwidth: start_r + (i + 1) * stackwidth] == 1):
                                excl[start_w + i] = 1
                            if not any(rpedc_map[idx_w, start_r + i * stackwidth: start_r + (i + 1) * stackwidth] == 1) and any(rpedc_map[idx_w, start_r + i * stackwidth: start_r + (i + 1) * stackwidth] == 2):
                                    excl[start_w + i] = 2


                        if "rpd" in self.area_exclusion.keys():
                            if any(rpd_map[idx_w, start_r + i * stackwidth: start_r + (i + 1) * stackwidth]):
                                if excl[start_w + i] == 2:
                                    excl[start_w + i] = 4 # if area contains rpedc and rpd
                                    if self.area_exclusion["rpedc"]:
                                        continue
                                
                                if excl[start_w + i] == 0:
                                    excl[start_w + i] = 3
                                
                                if self.area_exclusion["rpd"]:
                                        continue    
                            else:
                                if  excl[start_w + i] > 1 and self.area_exclusion["rpedc"]:
                                    continue
                        
                        if excl[start_w + i] == 1: # atrophy condition
                            continue


                        # a ez-loss map like in mactel project
                        #...
                        # a thickness determine by the distance between bm and rpe based on segmentation layer
                        #...
                            
                        
                        
                        i_profile =  np.zeros((50, 1)).astype(np.float32)[:,0] # intensity profile
                        for idxs, l in enumerate(layer[start_r + i * stackwidth: start_r + (i + 1) * stackwidth]):
                            if l < 496 and l > 0:
                                i_profile = i_profile + bscan_data[l-40:l+10,start_r + i * stackwidth + idxs]
                        i_profile/= self.stackwidth
                        
                        
                        # get rpe peak
                        rpe_peak = self.get_rpe_peak(i_profile)
                        
                        ez_peak, elm_peak = self.get_ez_elm_peak(i_profile,
                                                            float(rpe_peak),
                                                            ez_ssd_mean[start_w + i],
                                                            ez_ssd_std[start_w + i],
                                                            elm_ssd_mean[start_w + i],
                                                            elm_ssd_std[start_w + i])
                        
                        if ez_peak != 0:
                            ez[start_w + i] = i_profile[ez_peak]
                        if elm_peak != 0:
                            elm[start_w + i] = i_profile[elm_peak]
                        
                        
# =============================================================================
#                         if ez_peak != 0 and elm_peak !=0:
#                             plt.plot(np.arange(len(i_profile)), i_profile,
#                                      rpe_peak, i_profile[rpe_peak], "x",
#                                      ez_peak, i_profile[ez_peak], "x",
#                                      elm_peak, i_profile[elm_peak], "x")
# =============================================================================
                        
            maps_data = {
                "ez" : curr_ez_intensity,
                "elm": curr_elm_intensity
                }
            
            if "rpedc" in self.area_exclusion.keys():
                maps_data["atrophy"] = curr_excluded == 1
                maps_data["rpedc"] = np.logical_or(curr_excluded == 2, curr_excluded == 4)
                
            
            if "rpd" in self.area_exclusion.keys():
                maps_data["rpd"] = np.logical_or(curr_excluded == 3, curr_excluded == 4)
          
            
            # create Map Objects containing the created maps
            if self.project == "macustar": 
                current_map = OCTMap(
                    None,
                    "REZI-Map",
                    data_dict[vol_id],
                    vol_data._meta["VisitDate"],
                    self.scan_size,
                    self.stackwidth,
                    lat,
                    (fovea_ascan, fovea_bscan), # (x,y)
                    maps_data
                )            
            elif self.project == "mactel": 
                current_map = OCTMap(
                    vids[vol_id],
                    "REZI-Map",
                    data_dict[vol_id],
                    vol_data._meta["VisitDate"],
                    self.scan_size,
                    self.stackwidth,
                    lat,
                    (fovea_ascan, fovea_bscan), # (x,y)
                    maps_data
                )       
        
            if vol_id in self.patients.keys():
                
                # not yet tested
                for i, visit in enumerate(self.patients[vol_id].visits):
                    if visit.date_of_origin < current_map.date_of_origin:
                        continue
                    if visit.date_of_origin > current_map.date_of_origin:
                        self.patients[vol_id].visits.insert(current_map, i)
                        break
            else:
                self.patients[vol_id] = Patient(
                                            vol_id,
                                            vol_data._meta["DOB"],
                                            [current_map])
                        
    def get_microperimetry_grid_field(self, micro_data_path, micro_ir_path, visit, radius, use_gpu):

        if len(self.patients) == 0:
            raise Exception("So far, no patient has been analyzed, please first use calculate_relEZI_maps()")

        ir_list_m, ir_list_s = ut.get_microperimetry_IR_image_list(micro_ir_path)

        df = pd.read_excel(micro_data_path)

        key_list_num = np.copy(len(self.patients.keys()))

        for i in range(key_list_num):

            keys = list(self.patients.keys())[i]
            # read vol by macustarpredicter
            analysis_obj = macustar_segmentation_analysis.MacustarSegmentationAnalysis(
            vol_file_path=self.patients[keys].visits[visit -2].volfile_path,
            cache_segmentation=True,
            use_gpu = use_gpu
            )


            vol = analysis_obj.vol_file

            # get slo image 
            slo_img = vol.slo_image
            h_slo, w_slo = slo_img.shape


            # laterality 
            lat = self.patients[keys].visits[visit -2].laterality



            stimuli_s = ut.get_microperimetry(
                df,
                self.patients[keys].pid,
                visit,
                lat,
                "S")

            stimuli_m = ut.get_microperimetry(
                df,
                self.patients[keys].pid,
                visit,
                lat,
                "M")

        

            # create grid coords
            px_deg_y = px_deg_x = slo_img.shape[0] / 30 # pixel per degree
            ecc = np.array([items[0] for items in ut.grid_iamd.values()]) * px_deg_y
            ang = np.array([items[1] for items in ut.grid_iamd.values()]) * np.pi / 180

            x = (np.sin(ang) * ecc) + slo_img.shape[0]/2
            y = (np.cos(ang) * ecc) + slo_img.shape[1]/2

            # get slo_coordinates
            grid = np.array(vol.grid)
        
            # expected coordinates of scan field
            p = np.array([
                [0, 768],
                [64, 64]
                ])

            # actual coordinates of scan field
            if lat == "OD":
                q = np.array([
                    [grid[-1,0], grid[-1,2]],
                    [grid[-1,1], grid[-1,3]]
                    ])
            else:
                # flip slo_img
                slo_img = np.flip(slo_img,1) 
                
                # Coordinates are mirrored on the x-axis
                q = np.array([
                    [768 - grid[-1,2],768 - grid[-1,0]],
                    [grid[-1,3], grid[-1,1]]
                    ])                


            # calculate rigid transfromation matrix R in oct scan filed coordinate system "vol"
            vol_R = ut.get2DRigidTransformationMatrix(q, p)

            # coordinates of fovea center expected and patient
            vol_p_fovea = np.array([self.scan_size[1]/2, (self.scan_size[0])//2]).T
            vol_p_pat = np.array(self.patients[keys].visits[visit-2].fovea_coords).T
        
            # translation in oct scan filed coordinate system
            vol_t_F = (vol_p_fovea - vol_p_pat)
            vol_t_F[1] = (vol_t_F[1] * (640/241)).astype(int) # bscan number in pixel [(640/241) => pixel/bscan]

            # Matrix including complete transformation
            vol_R_t_F = vol_R + np.append(np.zeros((2,2)), vol_t_F[:,None], axis=1)

            # transform slo_img
            slo_img = cv2.warpAffine(slo_img, vol_R_t_F, (768, 768))

            # get microperimetry IR image m and s
            img1_raw_m = cv2.imread(ir_list_m[self.patients[keys].pid],0)
            img1_raw_s = cv2.imread(ir_list_s[self.patients[keys].pid],0)
            (h_micro, w_micro) = img1_raw_m.shape[:2]

            # rotated IR image 
            (cX, cY) = (w_micro // 2, h_micro // 2)
            M = cv2.getRotationMatrix2D((cX, cY), 90, 1.0)
            img1_raw_m = cv2.warpAffine(img1_raw_m, M, (w_micro, h_micro))
            img1_raw_s = cv2.warpAffine(img1_raw_s, M, (w_micro, h_micro))

            # flip microperimetry-IR image
            if lat == "OS":
                img1_raw_m = np.flip(img1_raw_m, 1)
                img1_raw_s = np.flip(img1_raw_s, 1)

            # crop image to 30° x 30° around centered fovea 3.25° offset on each side
            offset = int((h_micro/36) * 3)
            img1_m = img1_raw_m[offset:-offset,offset:-offset]
            img1_m = cv2.resize(img1_m,(h_slo,w_slo))
            img1_s = img1_raw_s[offset:-offset,offset:-offset]
            img1_s = cv2.resize(img1_s,(h_slo,w_slo))


            # calculate affine transformation matrix A
            H_m = ut.get2DProjectiveTransformationMartix_by_SuperRetina(img1_m, slo_img)
            H_s = ut.get2DProjectiveTransformationMartix_by_SuperRetina(img1_s, slo_img)
        

            # transform grid
            grid_coords = np.zeros((3,len(x)))
            grid_coords[0,:] = x
            grid_coords[1,:] = y
            grid_coords[2,:] = np.ones((1,len(x)))

            grid_coords_transf_m = H_m @ grid_coords
            grid_coords_transf_s = H_s @ grid_coords

            x_new_m = (grid_coords_transf_m[0,:] * (30/ 768)) 
            y_new_m = ((grid_coords_transf_m[1,:] - 64) * (25 / 640))

            x_new_s = (grid_coords_transf_s[0,:] * (30/ 768))
            y_new_s = ((grid_coords_transf_s[1,:] - 64) * (25 / 640))

            # create binary image with iamd grid 
            mask_iamd_m = np.zeros((self.scan_size[0],self.scan_size[1] // self.stackwidth))
            mask_iamd_s = np.zeros_like(mask_iamd_m)
            stimuli_m_map = np.zeros_like(mask_iamd_m)
            stimuli_s_map = np.zeros_like(mask_iamd_m)

            
            yy,xx = np.mgrid[:241,:(768 // self.stackwidth)]

            xx = xx * (30/(768 // self.stackwidth))
            yy = yy * (25/241
            )
            for num, stil_s, stil_m, y_cur_m, x_cur_m, y_cur_s, x_cur_s in zip(np.arange(1,34,1), stimuli_s, stimuli_m, y_new_m // self.scan_size[0], x_new_m , y_new_s // self.scan_size[0], x_new_s // self.stackwidth):
                mask_iamd_m[((yy - y_cur_m) ** 2) + ((xx - x_cur_m)**2) < radius ** 2] = num 
                mask_iamd_s[((yy - y_cur_s) ** 2) + ((xx - x_cur_s)**2) < radius ** 2] = num 
                stimuli_m_map[((yy - y_cur_m) ** 2) + ((xx - x_cur_m)**2) < radius ** 2] = stil_m
                stimuli_s_map[((yy - y_cur_s) ** 2) + ((xx - x_cur_s)**2) < radius ** 2] = stil_s
                

            
            self.patients[keys].visits[visit -2].octmap["micro_mask_m"] = mask_iamd_m
            self.patients[keys].visits[visit -2].octmap["micro_mask_s"] = mask_iamd_s
            self.patients[keys].visits[visit -2].octmap["micro_stim_m"] = stimuli_m_map
            self.patients[keys].visits[visit -2].octmap["micro_stim_s"] = stimuli_s_map


    def get_microperimetry_grid_field_show(self, micro_data_path, micro_ir_path, target_path, visit, use_gpu):
        if len(self.patients) == 0:
            raise Exception("So far, no patients have been analyzed, please first use calculate_relEZI_maps()")

        ir_list_m, ir_list_s = ut.get_microperimetry_IR_image_list(micro_ir_path)

        df = pd.read_excel(micro_data_path)

        key_list_num = np.copy(len(self.patients.keys()))

        for i in range(key_list_num):

            keys = list(self.patients.keys())[i]

            # read vol by macustarpredicter
            analysis_obj = macustar_segmentation_analysis.MacustarSegmentationAnalysis(
            vol_file_path=self.patients[keys].visits[visit -2].volfile_path,
            cache_segmentation=True,
            use_gpu = use_gpu
            )


            # calculate rel_ez_i_map
            ez_i_map = cv2.resize(self.patients[keys].visits[visit -2].octmap["ez"], (768, int(768*(25/30))))
            elm_i_map = cv2.resize(self.patients[keys].visits[visit -2].octmap["elm"], (768, int(768*(25/30))))
            ez_i_map[np.isnan(ez_i_map)] = 0
            elm_i_map[np.isnan(elm_i_map)] = 0

            rel_ez_i_map = np.full((ez_i_map.shape[0], ez_i_map.shape[1]), np.NaN)
            not_zeror = np.logical_and(ez_i_map != 0, elm_i_map != 0)
            rel_ez_i_map[not_zeror] = ez_i_map[not_zeror] / elm_i_map[not_zeror]    


            vol = analysis_obj.vol_file

            # get slo image 
            slo_img = vol.slo_image
            h_slo, w_slo = slo_img.shape


            # laterality 
            lat = self.patients[keys].visits[visit -2].laterality



            stimuli_s = ut.get_microperimetry(
                df,
                self.patients[keys].pid,
                visit,
                lat,
                "S")

            stimuli_m = ut.get_microperimetry(
                df,
                self.patients[keys].pid,
                visit,
                lat,
                "M")

        

            # create grid coords
            px_deg_y = px_deg_x = slo_img.shape[0] / 30 # pixel per degree
            ecc = np.array([items[0] for items in ut.grid_iamd.values()]) * px_deg_y
            ang = np.array([items[1] for items in ut.grid_iamd.values()]) * np.pi / 180

            x = (np.sin(ang) * ecc) + slo_img.shape[0]/2
            y = (np.cos(ang) * ecc) + slo_img.shape[1]/2

            # get slo_coordinates
            grid = np.array(vol.grid)
        
            # expected coordinates of scan field
            p = np.array([
                [0, 768],
                [64, 64]
                ])

            # actual coordinates of scan field
            if lat == "OD":
                q = np.array([
                    [grid[-1,0], grid[-1,2]],
                    [grid[-1,1], grid[-1,3]]
                    ])
            else:
                # flip slo_img
                slo_img = np.flip(slo_img,1) 
                
                # Coordinates are mirrored on the x-axis
                q = np.array([
                    [768 - grid[-1,2],768 - grid[-1,0]],
                    [grid[-1,3], grid[-1,1]]
                    ])                


            # calculate rigid transfromation matrix R in oct scan filed coordinate system "vol"
            vol_R = ut.get2DRigidTransformationMatrix(q, p)

            # coordinates of fovea center expected and patient
            vol_p_fovea = np.array([self.scan_size[1]/2, (self.scan_size[0])//2]).T
            vol_p_pat = np.array(self.patients[keys].visits[visit-2].fovea_coords).T
        
            # translation in oct scan filed coordinate system
            vol_t_F = (vol_p_fovea - vol_p_pat)
            vol_t_F[1] = (vol_t_F[1] * (640/241)).astype(int) # bscan number in pixel [(640/241) => pixel/bscan]

            # Matrix including complete transformation
            vol_R_t_F = vol_R + np.append(np.zeros((2,2)), vol_t_F[:,None], axis=1)

            # transform slo_img
            slo_img = cv2.warpAffine(slo_img, vol_R_t_F, (768, 768))

            # get microperimetry IR image m and s
            img1_raw_m = cv2.imread(ir_list_m[self.patients[keys].pid],0)
            img1_raw_s = cv2.imread(ir_list_s[self.patients[keys].pid],0)
            (h_micro, w_micro) = img1_raw_m.shape[:2]

            # rotated IR image 
            (cX, cY) = (w_micro // 2, h_micro // 2)
            M = cv2.getRotationMatrix2D((cX, cY), 90, 1.0)
            img1_raw_m = cv2.warpAffine(img1_raw_m, M, (w_micro, h_micro))
            img1_raw_s = cv2.warpAffine(img1_raw_s, M, (w_micro, h_micro))

            # flip microperimetry-IR image
            if lat == "OS":
                img1_raw_m = np.flip(img1_raw_m, 1)
                img1_raw_s = np.flip(img1_raw_s, 1)

            # crop image to 30° x 30° around centered fovea 3.25° offset on each side
            offset = int((h_micro/36) * 3)
            img1_m = img1_raw_m[offset:-offset,offset:-offset]
            img1_m = cv2.resize(img1_m,(h_slo,w_slo))
            img1_s = img1_raw_s[offset:-offset,offset:-offset]
            img1_s = cv2.resize(img1_s,(h_slo,w_slo))


            # calculate affine transformation matrix A
            H_m = ut.get2DProjectiveTransformationMartix_by_SuperRetina(slo_img, img1_m)
            H_s = ut.get2DProjectiveTransformationMartix_by_SuperRetina(slo_img, img1_s)
        

            # transform grid
            grid_coords = np.zeros((3,len(x)))
            grid_coords[0,:] = x
            grid_coords[1,:] = y
            grid_coords[2,:] = np.ones((1,len(x)))

            grid_coords_transf_m = H_m @ grid_coords
            grid_coords_transf_s = H_s @ grid_coords

            x_new_m = grid_coords_transf_m[0,:]
            y_new_m = grid_coords_transf_m[1,:]

            x_new_s = grid_coords_transf_s[0,:]
            y_new_s = grid_coords_transf_s[1,:]

            ut.show_grid_over_relEZIMap(
                slo_img,
                rel_ez_i_map,
                x_new_m,
                y_new_m,
                (slo_img.shape[0]//2,slo_img.shape[1]//2),
                stimuli_m,
                H_m,
                8,
                slo_img.shape[0] / 30, # pixel per degree
                True,
                self.patients[keys].pid + "_M",
                True,
                target_path
                )

            ut.show_grid_over_relEZIMap(
                slo_img,
                rel_ez_i_map,
                x_new_s,
                y_new_s,
                (slo_img.shape[0]//2,slo_img.shape[1]//2),
                stimuli_s,
                H_s,
                8,
                slo_img.shape[0] / 30, # pixel per degree
                True,
                self.patients[keys].pid + "_S",
                True,
                target_path
                )

    def create_ssd_maps(
        self,
        folder_path: Union[str, Path, IO] = None,
        fovea_coords: Optional[Dict] = None,
        scan_size: Optional[tuple] = None,
        stackwidth: Optional[int] = None,
        ref_layer: Optional[str] = None,
        base_layer: Optional[str] = None,
        *args
    ) -> None:

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
        
        if not fovea_coords:
            fovea_coords = self.fovea_coords
        else:
            self.fovea_coords = fovea_coords
        if not scan_size:
            scan_size = self.scan_size
        else:
            self.scan_size = scan_size
        if not stackwidth:
            stackwidth = self.stackwidth
        else:
            self.stackwidth = stackwidth
        if not ref_layer:
            ref_layer = "BM"
        if not base_layer:
            if not self.base_layer:
                base_layer = self.base_layer = "vol"
            else:
                base_layer = self.base_layer
        else:
            self.base_layer = base_layer


        # data directories
        if args:
            data_dict = ut.get_list_by_format(folder_path, args)
        else:
            raise ValueError("no file format given")

        if base_layer == "masks":
            mask_dict = ut.get_mask_list(folder_path)

        
        # central bscan/ascan, number of stacks (nos)
        c_bscan = scan_size[0] // 2 + scan_size[0] % 2
        c_ascan = scan_size[1] // 2 + scan_size[1] % 2
        nos = scan_size[1] // stackwidth # number of stacks

        ez_distance = np.empty(shape=[0, scan_size[0], nos])
        elm_distance = np.empty_like(ez_distance)

        
        # iterate  over .vol-list
        for vol_id in data_dict[".vol"]:

            # current distance map
            curr_ez_distance = np.empty((1, scan_size[0], nos))
            curr_ez_distance[:] = np.nan
            curr_elm_distance = np.full_like(curr_ez_distance, np.nan)
            
            # get vol data from file
            vol_data = ep.Oct.from_heyex_vol(data_dict[".vol"][vol_id])


            # check if given number of b scans match with pre-defined number 
            if vol_data._meta["NumBScans"] != scan_size[0]:
                print("ID: %s has different number of bscans (%i) than expected (%i)" % (ut.get_id_by_file_path(data_dict[".vol"][vol_id]), vol_data._meta["NumBScans"], scan_size[0]))
                continue

            # check if mask dict contains vol id
            if base_layer == "masks":
                if vol_id in mask_dict.keys():
                    mask_list = mask_dict[vol_id]
                else:
                    print("ID: %s considered segmentation mask not exist" % vol_id)

            # d_bscan (int): delta_bscan = [central bscan (number of bscans // 2)] - [current bscan]
            fovea_bscan, fovea_ascan = fovea_coords[ut.get_id_by_file_path(data_dict[".vol"][vol_id])]
            
            # change orientation from top down, subtract on from coords to keep 0-indexing of python            
            fovea_bscan = scan_size[0] - fovea_bscan

            # laterality 
            lat = vol_data._meta["ScanPosition"]

            if lat == "OS": # if left eye is processed
                fovea_ascan = scan_size[1] - fovea_ascan
            else:
                fovea_ascan = fovea_ascan -1

            d_bscan  = c_bscan - fovea_bscan


            for bscan, ez, elm, idx in zip(
                vol_data[::-1][max([-d_bscan, 0]): scan_size[0] + min([-d_bscan, 0])], # read
                curr_ez_distance[0, max([d_bscan, 0]): scan_size[0] + min([d_bscan, 0]), :], # write
                curr_elm_distance[0, max([d_bscan, 0]): scan_size[0] + min([d_bscan, 0]), :], # write
                range(max([-d_bscan, 0]), scan_size[0] + min([-d_bscan, 0]))
                ):
                

                if self.base_layer == None or self.base_layer == "vol":
                    try:
                        layer = bscan.layers[ref_layer].astype(np.uint16)
                    except:
                        continue
                if self.base_layer == "masks":
                    layer = ut.get_seg_by_mask(mask_list[idx], 10)
                
                bscan_data = bscan._scan_raw
                if lat == "OS":
                    bscan_data = np.flip(bscan_data,1)
                    layer = np.flip(layer)
                
                # get start position to read data
                d_ascan = c_ascan - fovea_ascan
                shift = min([d_ascan, 0])
                start_r = - shift + (c_ascan - (stackwidth//2) + shift) % stackwidth # start reading
                start_w = max([((c_ascan - (stackwidth//2)) // stackwidth) - (fovea_ascan - (stackwidth//2)) // stackwidth, 0])
                n_st = (scan_size[1] - start_r - max([d_ascan,0])) // stackwidth # possible number of stacks 
                
                
                # create region of interest image 
                roi = np.zeros((50,scan_size[1])).astype(np.float32)
                
                for i, l in enumerate(layer):
                    if l < 496 and l > 0:
                        roi[:,i] = bscan_data[l-40:l+10,i]
                        
            
                # get ez and rpe boundary 
                imglayers = get_retinal_layers(roi) 
                
                #if idx == 123:
                 #   plot_layers(roi, imglayers)
                
                
                for i in range(n_st):
                    
                    if not any(np.isnan(layer[start_r + i * stackwidth: start_r + (i + 1) * stackwidth])):

                        i_profile = np.mean(roi[:,start_r + i * stackwidth: start_r + (i + 1) * stackwidth],1)


                        rpe_grad = int(
                            np.round(
                                np.max(imglayers["rpe"].layerY[start_r + i * stackwidth: start_r + (i + 1) * stackwidth])))

                        ez_grad = int(
                            np.round(
                                np.min(imglayers["isos"].layerY[start_r + i * stackwidth: start_r + (i + 1) * stackwidth])))
                        
                        ez_peak  = ez_grad - 2 + np.where(i_profile[ez_grad - 2:ez_grad + 5] == np.max(i_profile[ez_grad - 2:ez_grad + 5]))[0][0]

                        rpe_peak = find_peaks(i_profile[rpe_grad - 5:rpe_grad + 2], height=0)[0]
                        if len(rpe_peak) > 0:
                            rpe_peak = rpe_grad - 5 + rpe_peak[-1]
                        else:
                            rpe_peak = None                        
                        
                        
                        elm_peak = find_peaks(i_profile[ez_peak - 10:ez_peak], height=0)[0]
                        if len(elm_peak) > 0:
                            elm_peak = ez_peak - 10 + elm_peak[-1]
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
            
        
            
        # set all zeros to nan
        ez_std = np.nanstd(ez_distance, axis=0)
        ez_std[ez_std == np.nan] = 0.
        ez_dist = np.nanmean(ez_distance, axis=0)
        ez_dist[ez_dist == np.nan] = 0.

        elm_std = np.nanstd(elm_distance, axis=0)
        elm_std[elm_std == np.nan] = 0.
        elm_dist = np.nanmean(elm_distance, axis=0)
        elm_dist[elm_dist == np.nan] = 0.       
        
        # create Map Objects containing the created maps 
        self.ez_distance_map = OCTMap(
            "rpe_ez",
            date.today(),
            self.scan_size,
            self.stackwidth,
            None, 
            {
            "distance" : ez_dist,
            "std"      : ez_std
            }
            )
        self.elm_distance_map = OCTMap(
            "rpe_elm",
            date.today(),
            self.scan_size,
            self.stackwidth,
            None, 
            {
            "distance" : elm_dist,
            "std"      : elm_std
            }
            )
            
    def create_mean_rpedc_map(
            self,
            folder_path: Union[str, Path, IO] = None,
            fovea_coords: Optional[Dict] = None,
            scan_size: Optional[tuple] = None
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
        
        if not scan_size:
            scan_size = self.scan_size
        else:
            self.scan_size = scan_size
            
            
        # central bscan/ascan, number of stacks (nos)
        c_bscan = scan_size[0] // 2 + scan_size[0] % 2
        c_ascan = scan_size[1] // 2 + scan_size[1] % 2
            
        # get all rpedc maps
        rpedc_dict = ut.get_rpedc_list(folder_path)
        
        # get vol_data to determin laterality
        data_dict = ut.get_vol_list(folder_path)
        
        
        rpedc_thickness = np.empty(shape=[0, 640, scan_size[1]])
        
        for ids in rpedc_dict.keys():
            if ids in fovea_coords.keys():

                # load map
                maps = cv2.imread(rpedc_dict[ids], flags=(cv2.IMREAD_GRAYSCALE | cv2.IMREAD_ANYDEPTH)).astype(float)
                                
                
                
                # d_bscan (int): delta_bscan = [central bscan (number of bscans // 2)] - [current bscan]
                fovea_bscan, fovea_ascan = fovea_coords[ids]
                    
                # change orientation from top down, subtract on from coords to keep 0-indexing of python            
                fovea_bscan = scan_size[0] - (fovea_bscan -1) 

                # laterality 
                vol_data = ep.Oct.from_heyex_vol(data_dict[ids])
                lat = vol_data._meta["ScanPosition"]

                if lat == "OS": # if left eye is processed
                    fovea_ascan = scan_size[1] - (fovea_ascan -1)
                    maps = np.flip(maps, 1)
                else:
                    fovea_ascan = fovea_ascan -1

                d_bscan  = c_bscan - fovea_bscan
                d_ascan = c_ascan - fovea_ascan
                

                rpedc_thickness = np.append(rpedc_thickness, shift(maps, (int(640./241.)*d_bscan, d_ascan))[None, ...], axis=0) # add dimension to append stack
        
      
        rpedc_thickness[rpedc_thickness <= 0.1] = np.nan # invalid values are set to nan
        
        self.mean_rpedc_map = OCTMap(
            "mean_rpedc",
            date.today(),
            self.scan_size,
            self.stackwidth,
            None, 
            {
            "mean" : np.nanmean(rpedc_thickness, axis=0),
            "std"      : np.nanstd(rpedc_thickness, axis=0)
            }
            )

    def create_excel_sheets(
            self, 
            folder_path,
            n,
            scan_field,
            project,
            ):
        
        """
        Args:
            folder_path (str): Target folder for data
            n (int): Number of visits per sheet
            scan_field (int, int): Scan field in degree
            project (str): project name like Macustar
            
        """
        nos = self.scan_size[1] // self.stackwidth
        d_a_scan = scan_field[1] / nos
        d_b_scan = scan_field[0] / self.scan_size[0]
        a_scan_mesh, b_scan_mesh = np.meshgrid(
                    np.arange(-scan_field[1]/2, scan_field[1]/2,d_a_scan),
                    np.arange(-scan_field[0]/2, scan_field[0]/2,d_b_scan))
        a_scan_mesh = a_scan_mesh.flatten()
        b_scan_mesh = b_scan_mesh.flatten()
        
        b_scan_n = (np.ones((nos, self.scan_size[0])) * np.arange(1, self.scan_size[0] + 1,1)).T.flatten()
    
        if project == "macustar micro":
            header = ["ID", "eye", "b-scan", "visit date", "A-Scan [°]", "B-Scan [°]",
             "druse(y/n)", "rpd(y/n)", "atrophy", "m stimulus grid", "m stimulus", "m stimulus grid", "m stimulus", "ez", "elm"]
        elif project == "macustar":
            header = ["ID", "eye", "b-scan", "visit date", "A-Scan [°]", "B-Scan [°]", "druse(y/n)", "ez", "elm"]

        if os.path.isdir(folder_path):
            workbook = xls.Workbook(os.path.join(folder_path, project + "_0.xlsx"))
            worksheet = workbook.add_worksheet()
            worksheet.write_row(0, 0, header)
            
        else:
            os.path.mkdir(folder_path)
            workbook = xls.Workbook(os.path.join(folder_path, project + "_0.xlsx"))
            worksheet = workbook.add_worksheet()            
            worksheet.write_row(0, 0, header)
            
        row = 1
        
        if project == "macustar micro":
            for i, ids in enumerate(self.patients.keys()):
            
                for j, visit in enumerate(self.patients[ids].visits): # if more than one visit is given, the sheet is extended to the right
                
                    worksheet.write(row, j * len(header), ids) # ID
                    worksheet.write_column(row, j * len(header) + 1, nos * self.scan_size[0] * [visit.laterality]) # Eye
                    worksheet.write_column(row, j * len(header) + 2, b_scan_n) # bscan
                    worksheet.write(row, j * len(header) + 3, visit.date_of_origin.strftime("%Y-%m-%d")) # Visit Date
                    worksheet.write_column(row, j * len(header) + 4, a_scan_mesh) # A-scan
                    worksheet.write_column(row, j * len(header) + 5, b_scan_mesh) # B-scan
                    worksheet.write_column(row, j * len(header) + 6, visit.octmap["rpedc"].flatten()) # Druse
                    worksheet.write_column(row, j * len(header) + 7, visit.octmap["rpd"].flatten()) # RPD
                    worksheet.write_column(row, j * len(header) + 8, visit.octmap["atrophy"].flatten()) # Atrophy
                    worksheet.write_column(row, j * len(header) + 9, visit.octmap["micro_mask_m"].flatten()) # Mask micro m
                    worksheet.write_column(row, j * len(header) + 10, visit.octmap["micro_stim_m"].flatten()) # Stimulus micro m
                    worksheet.write_column(row, j * len(header) + 11, visit.octmap["micro_mask_s"].flatten()) # Mask micro s
                    worksheet.write_column(row, j * len(header) + 12, visit.octmap["micro_stim_s"].flatten()) # Stimulus micro s
                    worksheet.write_column(row, j * len(header) + 13, visit.octmap["ez"].flatten())
                    worksheet.write_column(row, j * len(header) + 14, visit.octmap["elm"].flatten())
                   
                row += nos * self.scan_size[0]
            
                if (i +1) % n == 0 and i < len(self.patients.keys()) -1:
                    workbook.close()
                    workbook = xls.Workbook(os.path.join(folder_path, project + "_" + str(int((i +1) / n)) + ".xlsx"))
                    worksheet = workbook.add_worksheet()            
                    worksheet.write_row(0, 0, header)   
                    row = 1


        if project == "macutar":
            for i, ids in enumerate(self.patients.keys()):
            
                for j, visit in enumerate(self.patients[ids].visits): # if more than one visit is given, the sheet is extended to the right
                
                    worksheet.write(row, j * len(header), ids)
                    worksheet.write_column(row, j * len(header) + 1, nos * self.scan_size[0] * [visit.laterality])
                    worksheet.write_column(row, j * len(header) + 2, b_scan_n)
                    worksheet.write(row, j * len(header) + 3, visit.date_of_origin.strftime("%Y-%m-%d"))
                    worksheet.write_column(row, j * len(header) + 4, a_scan_mesh)
                    worksheet.write_column(row, j * len(header) + 5, b_scan_mesh)
                    worksheet.write_column(row, j * len(header) + 6, visit.octmap["exc"].flatten())
                    worksheet.write_column(row, j * len(header) + 7, visit.octmap["ez"].flatten())
                    worksheet.write_column(row, j * len(header) + 8, visit.octmap["elm"].flatten())
                   
                row += nos * self.scan_size[0]
            
                if (i +1) % n == 0 and i < len(self.patients.keys()) -1:
                    workbook.close()
                    workbook = xls.Workbook(os.path.join(folder_path, project + "_" + str(int((i +1) / n)) + ".xlsx"))
                    worksheet = workbook.add_worksheet()            
                    worksheet.write_row(0, 0, header)   
                    row = 1
                
        workbook.close()        
                
            
            

if __name__ == '__main__':
    pass

    








        

        


        







