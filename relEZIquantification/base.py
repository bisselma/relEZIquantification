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
import cv2
import xlsxwriter as xls
from scipy.ndimage.morphology import binary_dilation, binary_erosion
from read_roi import read_roi_zip
import pandas as pd

import eyepy as ep

from heyex_tools import vol_reader
from grade_ml_segmentation import macustar_segmentation_analysis

from rel_ez_intensity.getAdjacencyMatrix import plot_layers
from rel_ez_intensity.seg_core import get_retinal_layers
from rel_ez_intensity import utils as ut

from relEZIquantification_os import *



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
        visits_OD: Optional[List[OCTMap]] = None,
        visits_OS: Optional[List[OCTMap]] = None

    ) -> None:
        self.pid = pid
        self.dob = dob
        self.visits_OD = visits_OD
        self.visits_OS = visits_OS

    

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
            base_layer: Optional[str] = None,
            scope_factor: Optional[int] = None,
            
            ) -> None:
        self.project = project
        self.fovea_coords = fovea_coords
        self.ez_distance_map = ez_distance_map
        self.elm_distance_map = elm_distance_map
        self.scan_size = scan_size
        self.stackwidth = stackwidth
        self.patients = patients
        self.base_layer = base_layer 
        self.scope_factor = scope_factor
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

    def get_rpe_peak(self, raw_roi, seg_mask_roi, start_r, i, stackwidth):
        rpe_roi = np.copy(raw_roi[:,start_r + i * stackwidth: start_r + (i + 1) * stackwidth])
        rpe_roi[np.logical_and(seg_mask_roi[:,start_r + i * stackwidth: start_r + (i + 1) * stackwidth] != 9,
        seg_mask_roi[:,start_r + i * stackwidth: start_r + (i + 1) * stackwidth] != 10)] = np.nan

        rpe_peak = find_peaks(np.nanmean(rpe_roi,1))[0]
        if len(rpe_peak) >= 1:
            return rpe_peak[-1]
        else:
            return None


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
        area_exclusion: Optional[Dict] = None,
        **kwargs
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
            area_exclusion ( Optional[Dict]): Method to determine area of exclusion 
                                            # if values (boolean) are True the area should not be analysed.
            *args: project
 
 
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
            ref_layer = 11
        else:
            if ref_layer == "RPE":
                ref_layer = 10
            elif ref_layer == "BM":
                ref_layer = 11
            else:
                raise ValueError("layer name for reference layer not vaild")

        if not area_exclusion:
            if not self.area_exclusion:
                area_exclusion = self.area_exclusion = "default"
            else:
                area_exclusion = self.area_exclusion
        else:
            self.area_exclusion = area_exclusion

        if "scope_factor" not in kwargs.keys():
            self.scope_factor = 2
        else:
            self.scope_factor = kwargs["scope_factor"] 




        # data directories
        if self.project:
            if self.project == "macustar":
                data_dict, _ = ut.get_vol_list(folder_path, self.project)
                if "micro_ir_path" in kwargs.keys(): # check weather ID of the macustar cohort exist also in microperimetry cohort. Only ids existing in both cohorts are considered 
                    ir_list_keys = ut.get_microperimetry_IR_image_list(kwargs["micro_ir_path"])[0].keys()
                    for keys in list(data_dict.keys()):
                        if keys not in ir_list_keys:
                            del data_dict[keys]
            elif self.project == "mactel":
                data_dict, pids = ut.get_vol_list(folder_path, self.project)
        else:
            raise ValueError("no project name is given")

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
                  
            
            # d_bscan (int): delta_bscan = [central bscan (number of bscans // 2)] - [current bscan]
            try:
                fovea_bscan, fovea_ascan = fovea_coords[vol_id]
            except:
                print("ID %s is missing in Fovea List " % vol_id)
                continue
            
            # change orientation from top down, subtract on from coords to keep 0-indexing of python            
            fovea_bscan = scan_size[0] - fovea_bscan +1

            # delta between real fovea centre and current fovea bscan position 
            d_bscan  = c_bscan - fovea_bscan
            # get start position to read data
            d_ascan = c_ascan - fovea_ascan

            if not self.elm_distance_map or not self.ez_distance_map:
                raise ValueError("Site specific distance maps not given")
                           

            # get data
            ms_analysis = macustar_segmentation_analysis.MacustarSegmentationAnalysis(
                vol_file_path=data_dict[vol_id],
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
                fovea_ascan = scan_size[1] - fovea_ascan +1


            # if area_exception is "rpedc" get list of thickness maps 
            if "rpedc" in self.area_exclusion.keys():
                if vol_id in ae_dict_1.keys():
                    rpedc_map = self.get_rpedc_map(ae_dict_1[vol_id], self.scan_size, self.mean_rpedc_map, lat, (int(640./241.)*d_bscan, d_ascan))
                else:
                    print("ID: %s considered rpedc map not exist" % vol_id)
                    continue
            
            # if area_exception is "rpedc" get list of thickness maps 
            if "rpd" in self.area_exclusion.keys():
                if vol_id in ae_dict_2.keys():
                    rpd_map = self.get_rpd_map(ae_dict_2[vol_id], self.scan_size, lat, (int(640./241.)*d_bscan, d_ascan))
                else:
                    rpd_map = np.zeros(self.scan_size).astype(bool) 

            # check if given number of b scans match with pre-defined number 
            if ms_analysis._vol_file.header.num_bscans != scan_size[0]:
                print("ID: %s has different number of bscans (%i) than expected (%i)" % (vol_id, ms_analysis._vol_file.header.num_bscans, scan_size[0]))
                continue

            # check if given number of a scans match with pre-defined number 
            if ms_analysis._vol_file.header.size_x != scan_size[1]:
                print("ID: %s has different number of ascans (%i) than expected (%i)" % (ut.get_id_by_file_path(data_dict[vol_id]), ms_analysis._vol_file.header.size_x, scan_size[1]))
                continue  

            
            for bscan, seg_mask, ez, elm, excl, ez_ssd_mean, ez_ssd_std, elm_ssd_mean, elm_ssd_std, idx_r, idx_w in zip(
                ms_analysis._vol_file.oct_volume_raw[::-1][max([-d_bscan, 0]): scan_size[0] + min([-d_bscan, 0])], # read raw data
                ms_analysis.classes[::-1,:,:][max([-d_bscan, 0]): scan_size[0] + min([-d_bscan, 0])], # read seg mask
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


                #if self.base_layer == None or self.base_layer == "vol":
                #    try:
                #        layer = bscan.layers[ref_layer].astype(np.uint16)
                #    except:
                #        continue


                
                if lat == "OS":
                    bscan = np.flip(bscan,1)
                    seg_mask = np.flip(seg_mask,1)
                
                

                shift = min([d_ascan, 0])
                start_r = - shift + (c_ascan - (stackwidth//2) + shift) % stackwidth # start reading
                start_w = max([((c_ascan - (stackwidth//2)) // stackwidth) - (fovea_ascan - (stackwidth//2)) // stackwidth, 0])
                n_st = (scan_size[1] - start_r - max([d_ascan,0])) // stackwidth # possible number of stacks 
                
                
                        
            
# =============================================================================
#                 # get ez and rpe boundary 
#                 imglayers = get_retinal_layers(roi) 
#                 
#                 plot_layers(roi, imglayers)                
# =============================================================================
                
                
                # get rois
                raw_roi, seg_mask_roi = ut.get_roi_masks(bscan, ref_layer, scan_size, seg_mask)
                
                # iterate over bscans
                for i in range(n_st):                    
                                           
                        
                        # excluding section
                        # excluding condition can be:
                            
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
                        if "ezloss" in self.area_exclusion.keys():
                            pass 
                        #...
                        # a thickness determine by the distance between bm and rpe based on segmentation layer
                        #...
                        
                        
                        # get rpe peak
                        rpe_peak = self.get_rpe_peak(raw_roi, seg_mask_roi, start_r, i, stackwidth)

                        if not rpe_peak:
                            continue
                        
                        i_profile = np.nanmean(raw_roi[:,start_r + i * stackwidth: start_r + (i + 1) * stackwidth],1)

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
                    ms_analysis._vol_file.header.visit_date,
                    self.scan_size,
                    self.stackwidth,
                    lat,
                    (fovea_ascan, fovea_bscan), # (x,y)
                    maps_data
                )            
            elif self.project == "mactel": 
                current_map = OCTMap(
                    vol_id,
                    "REZI-Map",
                    data_dict[vol_id],
                    ms_analysis._vol_file.header.visit_date,
                    self.scan_size,
                    self.stackwidth,
                    lat,
                    (fovea_ascan, fovea_bscan), # (x,y)
                    maps_data
                )

            if self.project == "macustar":    
        
                if vol_id in self.patients.keys():
                
                    if current_map.laterality == "OD":
                        if self.patients[vol_id].visits_OD:
                            for i, visit in enumerate(self.patients[vol_id].visits_OD):
                                if visit.date_of_origin >= current_map.date_of_origin:
                                    self.patients[vol_id].visits_OD.insert(i, current_map)
                                    break
                                else:
                                    self.patients[vol_id].visits_OD.insert(i+1, current_map)
                                    break                                    
                        else:
                            self.patients[vol_id].visits_OD = [current_map]
                    else:
                        if self.patients[vol_id].visits_OS:
                            for i, visit in enumerate(self.patients[vol_id].visits_OS):
                                if visit.date_of_origin >= current_map.date_of_origin:
                                    self.patients[vol_id].visits_OS.insert(i, current_map)
                                    break
                                else:
                                    self.patients[vol_id].visits_OS.insert(i+1, current_map)
                                    break  
                        else:
                            self.patients[vol_id].visits_OS = [current_map]

                else:
                    if current_map.laterality == "OD":
                        self.patients[vol_id] = Patient(
                                            vol_id,
                                            ms_analysis._vol_file.header.birthdate,
                                            [current_map], # visit OD
                                            None)
                    else:
                        self.patients[vol_id] = Patient(
                                            vol_id,
                                            ms_analysis._vol_file.header.birthdate,
                                            None,
                                            [current_map] # visit OS
                                            )

            if self.project == "mactel":    
        
                if pids[vol_id] in self.patients.keys():

                    if current_map.laterality == "OD":
                        if self.patients[pids[vol_id]].visits_OD:
                            for i, visit in enumerate(self.patients[pids[vol_id]].visits_OD):
                                if visit.date_of_origin >= current_map.date_of_origin:
                                    self.patients[pids[vol_id]].visits_OD.insert(i, current_map)
                                    break
                                else:
                                    self.patients[pids[vol_id]].visits_OD.insert(i+1, current_map)
                                    break  
                        else:
                            self.patients[pids[vol_id]].visits_OD = [current_map]
                    else:
                        if self.patients[pids[vol_id]].visits_OS:
                            for i, visit in enumerate(self.patients[pids[vol_id]].visits_OS):
                                if visit.date_of_origin >= current_map.date_of_origin:
                                    self.patients[pids[vol_id]].visits_OS.insert(i, current_map)
                                    break
                                else:
                                    self.patients[pids[vol_id]].visits_OS.insert(i+1, current_map)
                                    break  
                        else:
                            self.patients[pids[vol_id]].visits_OS = [current_map]

                else:
                    if current_map.laterality == "OD":
                        self.patients[pids[vol_id]] = Patient(
                                            vol_id,
                                            ms_analysis._vol_file.header.birthdate,
                                            [current_map], # visit OD
                                            None)
                    else:
                        self.patients[pids[vol_id]] = Patient(
                                            vol_id,
                                            ms_analysis._vol_file.header.birthdate,
                                            None,
                                            [current_map] # visit OS
                                            )
                
            
                        
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

            mask_iamd_m, stimuli_m_map = ut.get_microperimetry_maps(
                    ir_list_m[self.patients[keys].pid],
                    lat,
                    radius,
                    slo_img,  
                    self.scan_size,
                    self.stackwidth,
                    stimuli_m,
                    x,y)

            mask_iamd_s, stimuli_s_map = ut.get_microperimetry_maps(
                    ir_list_s[self.patients[keys].pid],
                    lat,
                    radius,
                    slo_img,  
                    self.scan_size,
                    self.stackwidth,
                    stimuli_s,
                    x,y)

    
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
        

            ut.show_grid_over_relEZIMap(
                img1_m,
                rel_ez_i_map,
                x,
                y,
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
                img1_s,
                rel_ez_i_map,
                x,
                y,
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
        project: Optional[str] = None,
        fovea_coords: Optional[Dict] = None,
        scan_size: Optional[tuple] = None,
        stackwidth: Optional[int] = None,
        ref_layer: Optional[str] = None,
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
            *args: file formats that contain the data
        """
        
        if not fovea_coords:
            fovea_coords = self.fovea_coords
        else:
            self.fovea_coords = fovea_coords
        if not project:
            project = self.project
        else:
            self.project = project
        if not scan_size:
            scan_size = self.scan_size
        else:
            self.scan_size = scan_size
        if not stackwidth:
            stackwidth = self.stackwidth
        else:
            self.stackwidth = stackwidth
        if not ref_layer:
            ref_layer = 11
        else:
            if ref_layer == "RPE":
                ref_layer = 10
            elif ref_layer == "BM":
                ref_layer = 11
            else:
                raise ValueError("layer name for reference layer not vaild")



        # data directories
        if self.project:
            if self.project == "macustar":
                data_dict, _ = ut.get_vol_list(folder_path, self.project)
            elif self.project == "mactel":
                data_dict, pids = ut.get_vol_list(folder_path, self.project)
        else:
            raise ValueError("no project name is given")


        # central bscan/ascan, number of stacks (nos)
        c_bscan = scan_size[0] // 2 + scan_size[0] % 2
        c_ascan = scan_size[1] // 2 + scan_size[1] % 2
        nos = scan_size[1] // stackwidth # number of stacks

        ez_distance = np.empty(shape=[0, scan_size[0], nos])
        elm_distance = np.empty_like(ez_distance)

        
        # iterate  over .vol-list
        for vol_id in data_dict:

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
                vol_file_path=data_dict[vol_id],
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
                fovea_ascan = scan_size[1] - fovea_ascan
            else:
                fovea_ascan = fovea_ascan -1

            d_bscan  = c_bscan - fovea_bscan


            # check if given number of b scans match with pre-defined number 
            if ms_analysis._vol_file.header.num_bscans != scan_size[0]:
                print("ID: %s has different number of bscans (%i) than expected (%i)" % (vol_id, ms_analysis._vol_file.header.num_bscans, scan_size[0]))
                continue
            
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
                d_ascan = c_ascan - fovea_ascan
                shift = min([d_ascan, 0])
                start_r = - shift + (c_ascan - (stackwidth//2) + shift) % stackwidth # start reading
                start_w = max([((c_ascan - (stackwidth//2)) // stackwidth) - (fovea_ascan - (stackwidth//2)) // stackwidth, 0])
                n_st = (scan_size[1] - start_r - max([d_ascan,0])) // stackwidth # possible number of stacks 
                

                # get rois
                raw_roi, seg_mask_roi = ut.get_roi_masks(bscan, ref_layer, scan_size, seg_mask)
                

                for i in range(n_st):

                    rpe_roi = np.copy(raw_roi[:,start_r + i * stackwidth: start_r + (i + 1) * stackwidth])
                    rpe_roi[np.logical_and(seg_mask_roi[:,start_r + i * stackwidth: start_r + (i + 1) * stackwidth] != 9,
                    seg_mask_roi[:,start_r + i * stackwidth: start_r + (i + 1) * stackwidth] != 10)] = np.nan

                    rpe_peak = find_peaks(np.nanmean(rpe_roi,1))[0]
                    if len(rpe_peak) >= 1:
                        rpe_peak = rpe_peak[-1]
                    else:
                        rpe_peak = None


                    ez_roi = np.copy(raw_roi[:,start_r + i * stackwidth: start_r + (i + 1) * stackwidth])
                    ez_roi[np.roll( # use erosion to expand the search area by one on both sides
                        seg_mask_roi[:,start_r + i * stackwidth: start_r + (i + 1) * stackwidth] != 8,
                        -2)] = np.nan

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
                    elm_roi[seg_mask_roi[:,start_r + i * stackwidth: start_r + (i + 1) * stackwidth] != 7 ] = np.nan

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
            None, # no visit id
            "rpe_ez",
            None, # path
            date.today(),
            self.scan_size,
            self.stackwidth,
            None,
            None,
            {
            "distance" : ez_dist,
            "std"      : ez_std
            }
            )
        self.elm_distance_map = OCTMap(
            None, # no visit id
            "rpe_elm",
            None, # path
            date.today(),
            self.scan_size,
            self.stackwidth,
            None, 
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
            None, # no visit id 
            "mean_rpedc",
            None, # path
            date.today(),
            self.scan_size,
            self.stackwidth,
            None, 
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
        nos = self.scan_size[1] // self.stackwidth # number of stacks
        d_a_scan = scan_field[1] / nos # distance between stacks in degree
        d_b_scan = scan_field[0] / self.scan_size[0] # distance between b-scans in degree
        a_scan_mesh, b_scan_mesh = np.meshgrid(
                    np.arange(-scan_field[1]/2, scan_field[1]/2,d_a_scan),
                    np.arange(-scan_field[0]/2, scan_field[0]/2,d_b_scan))
        a_scan_mesh = a_scan_mesh.flatten() # serialized a-scan mesh
        b_scan_mesh = b_scan_mesh.flatten() # serialized a-scan mesh
        
        b_scan_n = (np.ones((nos, self.scan_size[0])) * np.arange(1, self.scan_size[0] + 1,1)).T.flatten() # b-scan number
    
        if project == "macustar micro":
            header = ["ID", "eye", "b-scan", "visit date", "A-Scan [°]", "B-Scan [°]",
             "druse(y/n)", "rpd(y/n)", "atrophy", "m stimulus grid", "s stimulus", "s stimulus grid", "m stimulus", "ez", "elm"]
        elif project == "macustar":
            header = ["ID", "eye", "b-scan", "visit date", "A-Scan [°]", "B-Scan [°]", "druse(y/n)", "ez", "elm"]
        elif project == "mactel":
            header = ["ID", "eye", "b-scan", "visit date", "A-Scan [°]", "B-Scan [°]", "ezloss(y/n)", "ez", "elm"]

        if os.path.isdir(folder_path):
            workbook = xls.Workbook(os.path.join(folder_path, project + "_0.xlsx"),  {'nan_inf_to_errors': True})
            worksheet = workbook.add_worksheet()
            worksheet.write_row(0, 0, header)
            
        else:
            os.path.mkdir(folder_path)
            workbook = xls.Workbook(os.path.join(folder_path, project + "_0.xlsx"),  {'nan_inf_to_errors': True})
            worksheet = workbook.add_worksheet()            
            worksheet.write_row(0, 0, header)
            
        row = 1
        
        if project == "macustar micro":
            for i, ids in enumerate(self.patients.keys()):

                if len(self.patients[ids].visits_OD) > 0:
            
                    for j, visit in enumerate(self.patients[ids].visits_OD): # if more than one visit is given, the sheet is extended to the right
                
                        worksheet.write(row, j * len(header), "313" + "".join(i for i in ids.split("-"))) # ID
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

                if len(self.patients[ids].visits_OS) > 0:
            
                    for j, visit in enumerate(self.patients[ids].visits_OS): # if more than one visit is given, the sheet is extended to the right
                
                        worksheet.write(row, j * len(header), "313" + "".join(i for i in ids.split("-"))) # ID
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
                    workbook = xls.Workbook(os.path.join(folder_path, project + "_" + str(int((i +1) / n)) + ".xlsx"), {'nan_inf_to_errors': True})
                    worksheet = workbook.add_worksheet()            
                    worksheet.write_row(0, 0, header)   
                    row = 1


        if project == "macutar" or project == "mactel":
            for i, ids in enumerate(self.patients.keys()):

                if len(self.patients[ids].visits_OD) > 0:
            
                    for j, visit in enumerate(self.patients[ids].visits_OD): # if more than one visit is given, the sheet is extended to the right
                        
                        if project == "macutar":
                            worksheet.write(row, j * len(header), ids)
                        else:
                            worksheet.write(row, j * len(header), visit.vid)
                        worksheet.write_column(row, j * len(header) + 1, nos * self.scan_size[0] * [visit.laterality])
                        worksheet.write_column(row, j * len(header) + 2, b_scan_n)
                        worksheet.write(row, j * len(header) + 3, visit.date_of_origin.strftime("%Y-%m-%d"))
                        worksheet.write_column(row, j * len(header) + 4, a_scan_mesh)
                        worksheet.write_column(row, j * len(header) + 5, b_scan_mesh)
                        worksheet.write_column(row, j * len(header) + 6, visit.octmap["exc"].flatten())
                        worksheet.write_column(row, j * len(header) + 7, visit.octmap["ez"].flatten())
                        worksheet.write_column(row, j * len(header) + 8, visit.octmap["elm"].flatten())

                if len(self.patients[ids].visits_OS) > 0:
            
                    for j, visit in enumerate(self.patients[ids].visits_OS): # if more than one visit is given, the sheet is extended to the right
                
                        
                        if project == "macutar":
                            worksheet.write(row, j * len(header), ids)
                        else:
                            worksheet.write(row, j * len(header), visit.vid)
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

    








        

        


        






