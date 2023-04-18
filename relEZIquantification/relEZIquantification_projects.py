# -*- coding: utf-8 -*- 
from pathlib import Path
from typing import Dict, List, Optional, Union, IO
import numpy as np
from PIL import Image
from scipy.ndimage import shift
from datetime import date
import os
import xlsxwriter as xls
from scipy.ndimage.morphology import binary_dilation
from  skimage.morphology import disk
from PIL import Image
import pandas as pd
from skimage.measure import label

from heyex_tools import vol_reader
from grade_ml_segmentation import macustar_segmentation_analysis


from relEZIquantification.relEZIquantification_structure import *


class RelEZIQuantificationBase:

    _data_folder = None # folder with project data c 

    _project_name = None
    
    _fovea_coords = {} # Dict with IDs and fovea_coords 
    
    _scan_size = None
    
    _scan_field = None # field size in degree
    
    _stackwidth = None
    
    _scope_factor = 2 # the scope factor is 2 by default 
    
    _ref_layer = "BM" # layer to which the stack is to flatten 
    
    _ssd_maps = None # [mean distance, standard diviation]
    
    _mean_rpedc_map = None # mean thickness and sd map of rpe druse complex after DUKE/AREDS 

    global exclusion_dict
    exclusion_dict = {} # tmp dict with all exclusion types considered 

    _parameter = None

    _patients = {} # list with patient objects

    _header = ["ID", "eye", "b-scan", "visit date", "a-Scan [°]", "b-Scan [°]","ez", "elm"] # standard header of excel sheets

    

    def __init__(
        self,
        project_name: Optional[str] = None,
        data_folder: Optional[Path] = None,
        fovea_coords: Optional[Dict] = None,
        scan_size: Optional[tuple] = None,
        scan_field: Optional[tuple] = None,
        stackwidth: Optional[int] = None,
        ssd_maps: Optional[SSDmap] = None, 
        mean_rpedc_map: Optional[Mean_rpedc_map] = None,
        parameter: Optional[List] = None,
        patients: Optional[Dict] = {},
        
 
            
        ):
        self._project_name = project_name
        self._data_folder = data_folder
        self._fovea_coords = fovea_coords
        self._scan_size = scan_size
        self._scan_field = scan_field
        self._stackwidth = stackwidth
        self._ssd_maps = ssd_maps
        self._mean_rpedc_map = mean_rpedc_map
        self._parameter = parameter
        self._patients = patients

    @property
    def project_name(self):
        return self._project_name        

    @property
    def data_folder(self):
        return self._data_folder
    
    @property
    def fovea_coords(self):
        return self._fovea_coords

    @property
    def scan_size(self):
        return self._scan_size

    @property
    def scan_field(self):
        return self._scan_field

    @property
    def stackwidth(self):
        return self._stackwidth

    @property
    def scope_factor(self):
        return self._scope_factor

    @scope_factor.setter
    def scope_factor(self, value):
        self._scope_factor = value 

    @property
    def ref_layer(self):
        return self._ref_layer

    @ref_layer.setter
    def ref_layer(self, value):
        self._ref_layer = value

    @property
    def ssd_maps(self):
        return self._ssd_maps 

    @ssd_maps.setter
    def ssd_maps(self, value):
        self._ssd_maps = value 

    @property
    def mean_rpedc_map(self):
        return self._mean_rpedc_map 

    @mean_rpedc_map.setter
    def mean_rpedc_map(self, value):
        self._mean_rpedc_map = value 

    @property
    def parameter(self):
        return self._parameter

    @property
    def patients(self):
        return self._patients 

    @property
    def header(self):
        return self._header

    def get_ezloss_map(self, filepath, laterality):

        roi = np.array(Image.open(filepath),dtype=float)
        
        if self.scan_field == (25,30):
            if roi.shape[0] == 989:
                roi = roi[0:886,:,:]
            if self.scan_size[1] == 768:
                crop = int((roi.shape[1] - ((roi.shape[0]) * (25/20))) / 2) + 26
                roi = roi[:,crop:-crop]
    
        if self.scan_field == (10,15):
            if roi.shape[0] == 972:
                roi = roi[0:867,:,:]



        grey = np.logical_and(roi[:,:,0] == roi[:,:,1], roi[:,:,0] == roi[:,:,2])

        blue = np.logical_and(roi[:,:,0] != 0, np.logical_and(roi[:,:,1] == 0, roi[:,:,2] == 0))
        green = np.logical_and(roi[:,:,0] == 0, np.logical_and(roi[:,:,1] != 0, roi[:,:,2] == 0))
        red = np.logical_and(roi[:,:,0] == 0, np.logical_and(roi[:,:,1] == 0, roi[:,:,2] != 0))

        blue_less = np.logical_and(roi[:,:,0] != 0, np.logical_and(roi[:,:,1] != 0, np.logical_and(roi[:,:,2] != 0 ,roi[:,:,1] == roi[:,:,2])))
        green_less = np.logical_and(roi[:,:,0] != 0, np.logical_and(roi[:,:,1] != 0, np.logical_and(roi[:,:,0] != 0 ,roi[:,:,0] == roi[:,:,2])))
        red_less = np.logical_and(roi[:,:,0] != 0, np.logical_and(roi[:,:,1] != 0, np.logical_and(roi[:,:,2] != 0 ,roi[:,:,0] == roi[:,:,1])))

        mask = ~np.logical_or(red_less, np.logical_or(green_less,np.logical_or(blue_less, np.logical_or(grey, np.logical_or(blue, np.logical_or(green, red))))))


        mask[1:50,1:200] = False
        mask[-50:-1,1:100] = False


        if self.scan_field == (10,15):   
            crop = int((mask.shape[1] - ((mask.shape[0]) * (15/10))) / 2)
            mask = mask[:,crop-4:-crop-4]
            roi = roi[:,crop-4:-crop-4]


        struct = disk(2)
        mask = (binary_dilation(mask, structure = struct) * 255).astype(np.uint8)

        # get bounding box coordinates from the one filled external contour
        ezloss_map = np.zeros_like(mask)
        contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours[0]
        for cont in contours:
            x,y,w,h = cv2.boundingRect(cont)
            if h <= 10 or w <=10:
                continue
            cv2.drawContours(ezloss_map, [cont], 0, 255, -1)      

        if laterality == "OS":
            ezloss_map = np.flip(ezloss_map, 1)

        # label map
        ezloss_map = label(ezloss_map)

        ezloss_map = cv2.resize(ezloss_map.astype(np.uint8), self.scan_size[::-1], cv2.INTER_NEAREST)

        return ezloss_map

    def get_edtrs_grid_map(self, scan_area):

        # create mesh
        a_scan_mesh, b_scan_mesh = np.meshgrid(
                    np.arange( -scan_area[1] / 2, scan_area[1] / 2, scan_area[1] / (self.scan_size[1] // self.stackwidth)),
                    np.arange( -scan_area[0] / 2, scan_area[0] / 2, scan_area[0] / self.scan_size[0]),
            )

        # create degree map
        degree_map = 180 + np.arctan2(b_scan_mesh, -a_scan_mesh) * 180 / np.pi

        # radius map
        radius_map = (a_scan_mesh**2 + b_scan_mesh**2)**0.5

        # initilize edtrs_grid_map with zeros valued array of the same size as degree_map
        edtrs_grid_map = np.zeros_like(degree_map).astype(str)


        # nasal pericentral
        edtrs_grid_map[
            np.logical_and(
                np.logical_and(radius_map > 0.5, radius_map <= 1.5),
                       np.logical_or(degree_map <= 45, degree_map > 315))] = "nasal_pericentral"
    
        # nasal peripheral
        edtrs_grid_map[
            np.logical_and(radius_map > 1.5,
                       np.logical_or(degree_map <= 45, degree_map > 315))] = "nasal_peripheral"
    
    
        # superior pericentral
        edtrs_grid_map[
            np.logical_and(
                np.logical_and(radius_map > 0.5, radius_map <= 1.5),
                       np.logical_and(degree_map > 45, degree_map <= 135))] = "superior_pericentral"
    
        # superior peripheral
        edtrs_grid_map[
            np.logical_and(radius_map > 1.5,
                       np.logical_and(degree_map > 45, degree_map <= 135))] = "superior_peripheral"

        # temporal pericentral
        edtrs_grid_map[
            np.logical_and(
                np.logical_and(radius_map > 0.5, radius_map <= 1.5),
                       np.logical_and(degree_map > 135, degree_map <= 225))] = "temporal_pericentral"
    
        # temporal peripheral
        edtrs_grid_map[
            np.logical_and(radius_map > 1.5,
                       np.logical_and(degree_map > 135, degree_map <= 225))] = "temporal_peripheral"

        # inferior pericentral
        edtrs_grid_map[
            np.logical_and(
                np.logical_and(radius_map > 0.5, radius_map <= 1.5),
                       np.logical_and(degree_map > 225, degree_map <= 315))] = "inferior_pericentral"
    
        # inferior peripheral
        edtrs_grid_map[
            np.logical_and(radius_map > 1.5,
                       np.logical_and(degree_map > 225, degree_map <= 315))] = "inferior_peripheral"

        # center
        edtrs_grid_map[radius_map <= 0.5] =  "center"
    
        return edtrs_grid_map
        
    def get_rpedc_map(
        self,
        file_path: Union[str, Path, IO] = None,
        scan_size: Optional[tuple] = None,
        mean_rpedc: Optional[Mean_rpedc_map] = None,
        laterality: Optional[str] = None,
        translation: Optional[tuple] = None
        ) -> np.ndarray:

        translation = (int(640./scan_size[0]) * translation[0], translation[1])  
        
        maps = cv2.imread(file_path, flags=(cv2.IMREAD_GRAYSCALE | cv2.IMREAD_ANYDEPTH))
        
        if laterality == "OS":
            maps = np.flip(maps, 1)

        atrophy = maps  == 0.
        
        maps_shifted = shift(maps, translation)
        atrophy_shifted = shift(atrophy, translation).astype(np.uint8)
        
        # substract mean thickness of rpedc plus 3 times std (Duke/AREDS Definition) 
        sub = maps_shifted - (mean_rpedc.distance_array + (3. * mean_rpedc.std_array)) 


        sub = np.logical_or(sub > 0., maps_shifted <= 0.01).astype(np.uint8) # rpedc area
        
        sub_resized = cv2.resize(sub, scan_size[::-1], cv2.INTER_LINEAR) # cv2.resize get fx argument befor fy, so  the tuple "scan_size" must be inverted
        atrophy_resized = cv2.resize(atrophy_shifted, scan_size[::-1], cv2.INTER_LINEAR)
    
        # structure element should have a radius of 100 um (Macustar-format (
        # ~30 um per bscan => ~ 3 px in y-direction and 2 * 3 px to get size of rectangle in y-dir. 
        # ~ 10 um per ascan => ~ 10 px in x-direction and 2 * 10 px to get size of rectangle in x-dir. 
        struct = np.ones((4, 10), dtype=bool)
        sub_dilation = binary_dilation(sub_resized, structure = struct).astype(int) * 2 # rpedc condition =2 
        atrophy_dilation = binary_dilation(atrophy_resized, structure = struct)   
        
        sub_dilation[atrophy_dilation] = 1 # atrophy condition =1

        return sub_dilation      
    
    def update_header(self, idx, value):
        tmp = self._header.copy()
        tmp.insert(idx, value)
        self._header = tmp

    def check_args(
        self,
        data_folder: Union[str, Path, IO] = None,
        fovea_coords: Optional[Dict] = None,
        scan_size: Optional[tuple] = None,
        scan_field: Optional[tuple] = None,
        stackwidth: Optional[int] = None,
        ref_layer: Optional[str] = None,
        area_exclusion: Optional[Dict] = None,
        **kwargs
        ):
        # argument control section
        if not data_folder: # project_name
            if not self._data_folder:
                raise ValueError("No data folder. Directory where project data are stored is requiered") # should be updated if new project is available
        else:
            self._data_folder = data_folder

        if not self._project_name: # project_name
            print("No project_name. \nAdd one of the following project names:\macustar\nmicro\nmactel\n'macustar' is used as default")
            self._project_name = "macustar"

        if not fovea_coords: # fovea_coords
            if not self._fovea_coords:
                raise ValueError("No fovea_coords. Dict of the shape <'ID': (bscan, ascan)> was expected")
        else:
            self._fovea_coords = fovea_coords   

        if not scan_size: # scan_size
            if not self._scan_size:
                raise ValueError("No scan_size. Tuple of shap <(number of bscans, number of ascans)> was expected")
        else:
            self._scan_size =  scan_size           

        if not scan_field: # scan_field
            if not self._scan_field:
                raise ValueError("No scan_field. Tuple of shap <( y-direction in degree (bscan), x-direction in degree (ascan))> was expected")
        else:
            self._scan_field = scan_field 

        if not stackwidth: # stackwidth
            if not self._stackwidth:
                raise ValueError("No stackwidth. Integer higher 0 was expected")
        else:
            self._stackwidth = stackwidth                   
   
        if ref_layer:
            if ref_layer == "RPE" or ref_layer == 10:
                self.ref_layer = 10
            elif ref_layer == "BM" or ref_layer == 11:
                pass # the value 11 is given by default 
            else:
                raise ValueError("layer name or number for reference layer not valid ")

        if not area_exclusion: # exclusion_dict
            print("No area_exclusion. Dict of the shape <{'exclusion type': boolean}> was expected. Exclusion  types are:\nrpedc, atrophy, rpd, ezloss\nThe default condition is used")
            self._exclusion_dict({"default": True})
        else:
            self._parameter = area_exclusion
 
        if not self._ssd_maps:
            raise ValueError("Site specific distance maps not given")

    def get_exclusion_value(
        self,
        idx_w,
        start_r,
        i,
        ):

        """

        e.g.
        if the order of the exclusion dict would be like <rpedc, atrophy, rpd, ezloss>
        rpedc = 2^0
        atrophy = 2^1
        rpd = 2^2
        ezloss = 2^3

        """
        
        binary_number = 0

        for idx, exclusion_type in zip(range(len(exclusion_dict)-1,-1,-1), exclusion_dict):
            if any(exclusion_dict[exclusion_type][idx_w, start_r + i * self.stackwidth: start_r + (i + 1) * self.stackwidth] >= 1):
                    binary_number += 2**idx

        return binary_number

    def add(self, map, pid, visitdate, vid:Optional[str] = None, *args): # if patient allready exists
        if args: # new patient
            if map.laterality == "OD":
                self.patients[pid] = Patient(pid, args[0], Visit(vid, visitdate, map, None))
            elif  map.laterality == "OS": 
                self.patients[pid] = Patient(pid, args[0], Visit(vid, visitdate, None, map))
            else:
                raise ValueError("Map attribute 'laterality' is not correct") 
        else: # add visit to patient object
            self.patients[pid].add(map, visitdate, vid)


    def get_list(self, *args):
        pass

    def create_relEZI_maps(self):
        pass

    def create_excel_sheets(self):
	    pass

class RelEZIQuantificationMactel(RelEZIQuantificationBase):
    
    def __init__(
        self, 
        _project_name: Optional[str] = None,
        _data_folder: Optional[Path] = None, 
        _fovea_coords: Optional[Dict] = None, 
        _scan_size: Optional[tuple] = None, 
        _scan_field: Optional[tuple] = None, 
        _stackwidth: Optional[int] = None, 
        _ssd_maps: Optional[SSDmap] = None, 
        _mean_rpedc_map: Optional[Mean_rpedc_map] = None,
        _parameter: Optional[List] = None, 
        _patients: Optional[Dict] = {}
        ):
        super().__init__(_project_name, _data_folder,  _fovea_coords, _scan_size, _scan_field, _stackwidth, _ssd_maps, _mean_rpedc_map, _parameter, _patients)


    def get_list(self, *args):

        if not args:
            data_folder = self.data_folder
        else:
            data_folder = args[0]

        if not os.path.exists(data_folder):
            raise NotADirectoryError("directory: " +  data_folder + " not exist")

        path_list = {}

        dir_list = os.listdir(data_folder)
        for dir in dir_list:
            full_path = os.path.join(data_folder, dir)
            if os.path.isdir(full_path):
                dir_list.extend(os.path.join(dir, subfolder) for subfolder in os.listdir(full_path))
            if os.path.isfile(full_path) and full_path.endswith(".vol"):
                sid = full_path.split("\\")[-1].split("_")[-1].split(".")[0] # series uid 
                path_list[sid] = full_path  
         
        return  path_list   

    def create_relEZI_maps(        
        self,
        data_folder: Union[str, Path, IO] = None,
        fovea_coords: Optional[Dict] = None,
        scan_size: Optional[tuple] = None,
        scan_field: Optional[tuple] = None,
        stackwidth: Optional[int] = None,
        ref_layer: Optional[str] = None,
        area_exclusion: Optional[List] = None,
        **kwargs
        ):
        """
        Args:
            folder_path (Union[str, Path, IO]): folder path where files are stored
            fovea_coords (Optional[Dict]): location of fovea
                !!! B-scan number counted from bottom to top like HEYEX !!! -> easier handling for physicians
                bscan (int): Number of B-scan including fovea
                ascan (int): Number of A-scan including fovea
            scan_size (Optional[tuple]): scan field size in x and y direction
                x (int): Number of B-scans
                y (int): Number of A-scans
            scan_field (Optional[tuple]): scan field size in x and y direction in degree
            stackwidth (Optional[int]): number of columns for a single profile
            ref_layer (Optional[str]): layer to flatten the image 
            area_exclusion ( Optional[Dict]): Method to determine area of exclusion 
                                            # if values (boolean) are True the area should not be analysed.
        """


        # create copy of stackwidth to handle different scan sizes than expected
        stackwidth_fix = np.copy(stackwidth)
        factor = 1


        # raise expection if at least on argument is incorrect. Set instance variables.
        self.check_args(data_folder, fovea_coords, scan_size, scan_field, stackwidth, ref_layer, area_exclusion)

        # get a dict structure containing the data in the shape <"ID":"path + .format">
        data_list = self.get_list()

        # get lists of exclusion data 
        for exclusion_type in area_exclusion:
            if exclusion_type == "ezloss":
                ae_dict_1 = get_ezloss_list(self.data_folder, "roi")
                self.update_header(-2, "ezloss(y/n)") 
        if "etdrs" in self.parameter:
            self.update_header(-2, "etdrs_grid") 


        # central bscan/ascan, number of stacks (nos)
        c_bscan = self.scan_size[0] // 2 + self.scan_size[0] % 2
        c_ascan = self.scan_size[1] // 2 + self.scan_size[1] % 2
        nos = self.scan_size[1] // self.stackwidth # number of stacks

        for sid in data_list:

            # current distance map/ exclusion map
            curr_ez_intensity = np.zeros((scan_size[0], nos))
            curr_elm_intensity = np.zeros_like(curr_ez_intensity)
            curr_excluded = np.zeros_like(curr_ez_intensity)
                  
            
            # d_bscan (int): delta_bscan = [central bscan (number of bscans // 2)] - [current bscan]
            try:
                fovea_bscan, fovea_ascan = fovea_coords[sid]
            except:
                print("ID %s is missing in Fovea List " % sid)
                continue
            
            # change orientation from top down, subtract on from coords to keep 0-indexing of python            
            fovea_bscan = scan_size[0] - fovea_bscan +1

            # delta between real fovea centre and current fovea bscan position 
            d_bscan  = c_bscan - fovea_bscan
                

            # get data from vol-file
            ms_analysis = macustar_segmentation_analysis.MacustarSegmentationAnalysis(
                vol_file_path=data_list[sid],
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


            # get ezloss map if ez_loss exclusion is considered
            if "ezloss" in area_exclusion:
                if sid in ae_dict_1.keys():
                    exclusion_dict["ezloss"] = self.get_ezloss_map(ae_dict_1[sid], lat) 

            
            # check if given number of b scans match with pre-defined number 
            if ms_analysis._vol_file.header.num_bscans != scan_size[0]:
                print("ID: %s has different number of bscans (%i) than expected (%i)" % (sid, ms_analysis._vol_file.header.num_bscans, scan_size[0]))
                continue

            # check if given number of a scans match with pre-defined number 
            if ms_analysis._vol_file.header.size_x != scan_size[1]:
                print("ID: %s has different number of ascans (%i) than expected (%i)" % (sid, ms_analysis._vol_file.header.size_x, scan_size[1]))
                factor = ms_analysis._vol_file.header.size_x / scan_size[1]
                if factor * stackwidth_fix >= 1 and  factor % 1 == 0:
                    stackwidth = int(factor * stackwidth_fix) # change stackwidth temporarily to adjust to different scan sizes

            
            for bscan, seg_mask, ez, elm, excl, ez_ssd_mean, ez_ssd_std, elm_ssd_mean, elm_ssd_std, idx_r, idx_w in zip(
                ms_analysis._vol_file.oct_volume_raw[::-1][max([-d_bscan, 0]): scan_size[0] + min([-d_bscan, 0])], # read raw data
                ms_analysis.classes[::-1,:,:][max([-d_bscan, 0]): scan_size[0] + min([-d_bscan, 0])], # read seg mask
                curr_ez_intensity[max([d_bscan, 0]): scan_size[0] + min([d_bscan, 0]), :], # write
                curr_elm_intensity[max([d_bscan, 0]): scan_size[0] + min([d_bscan, 0]), :], # write
                curr_excluded[max([d_bscan, 0]): scan_size[0] + min([d_bscan, 0]), :], # write
                self.ssd_maps.ez_ssd_map.distance_array[max([d_bscan, 0]): scan_size[0] + min([d_bscan, 0]), :], # write
                self.ssd_maps.ez_ssd_map.std_array[max([d_bscan, 0]): scan_size[0] + min([d_bscan, 0]), :], # write
                self.ssd_maps.elm_ssd_map.distance_array[max([d_bscan, 0]): scan_size[0] + min([d_bscan, 0]), :], # write                
                self.ssd_maps.elm_ssd_map.std_array[max([d_bscan, 0]): scan_size[0] + min([d_bscan, 0]), :], # write 
                range(max([-d_bscan, 0]), scan_size[0] + min([-d_bscan, 0])), # read
                range(max([d_bscan, 0]), scan_size[0] + min([d_bscan, 0]))
                ):

                
                # flip maps if laterality is left (OS)
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
                raw_roi, seg_mask_roi = get_roi_masks(bscan, self.ref_layer, ms_analysis._vol_file.header.size_x, seg_mask)
                
                # iterate over bscans
                for i in range(n_st):  



                    if "default" in exclusion_dict:
                        pass 
                        ############### to be implemented ##################
                        # condition seg_mask_area 10 and 11 thickness over stackwidth not higher 15

                    else:
                        excl[start_w + i] = self.get_exclusion_value(idx_w, start_r, i)

  
                    # get rpe peak
                    rpe_peak = get_rpe_peak(raw_roi, seg_mask_roi, start_r, i, stackwidth)

                    if not rpe_peak:
                        continue
                        
                    i_profile = np.nanmean(raw_roi[:,start_r + i * stackwidth: start_r + (i + 1) * stackwidth],1)

                    ez_peak, elm_peak = get_ez_elm_peak(i_profile,
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

            # set stackwith and factor to default
            stackwidth = stackwidth_fix
            factor = 1

            tmp_excluded_dict = {}

            for idx, exclusion_type in zip(range(len(exclusion_dict)-1,-1,-1), exclusion_dict):
                tmp_excluded_dict[exclusion_type] = curr_excluded // 2**idx
                curr_excluded = curr_excluded % 2**idx


            # calculate scan_area
            scan_area = tuple(
                [ms_analysis._vol_file.bscan_headers[0].start_y - ms_analysis._vol_file.bscan_headers[-1].start_y, # enface y-direction in mm
                ms_analysis._vol_file.bscan_headers[0].end_x - ms_analysis._vol_file.bscan_headers[0].start_x] # enface x-direction in mm
                )  
            
            # create Map Objects containing the created maps
            current_map = RelEZI_map(
                "relEZI-Map",
                date.today(),
                self.scan_size,
                self.scan_field,
                scan_area,
                self.stackwidth,
                sid,
                data_list[sid],
                lat,
                (fovea_bscan, fovea_ascan), # (x,y)
                curr_ez_intensity,
                curr_elm_intensity,
                tmp_excluded_dict
                )            

            pid = ms_analysis._vol_file.header.pid

            if pid in self.patients.keys():
                self.add(current_map, pid, ms_analysis._vol_file.header.visit_date, None)
            else:
                self.add(current_map, pid, ms_analysis._vol_file.header.visit_date, None, ms_analysis._vol_file.header.birthdate)

    def create_excel_sheets(
        self,
        folder_path,
        n
        ):
        """
        Args:
            folder_path (str): Target folder for data
            n (int): Number of visits per sheet
            project (str): project name like Macustar
            
        """

        nos = self.scan_size[1] // self.stackwidth # number of stacks
        d_a_scan = self.scan_field[1] / nos # distance between stacks in degree
        d_b_scan = self.scan_field[0] / self.scan_size[0] # distance between b-scans in degree
        a_scan_mesh, b_scan_mesh = np.meshgrid(
                    np.arange(-self.scan_field[1]/2, self.scan_field[1]/2,d_a_scan),
                    np.arange(-self.scan_field[0]/2, self.scan_field[0]/2,d_b_scan))
        a_scan_mesh = a_scan_mesh.flatten() # serialized a-scan mesh
        b_scan_mesh = b_scan_mesh.flatten() # serialized a-scan mesh 

        b_scan_n = (np.ones((nos, self.scan_size[0])) * np.arange(1, self.scan_size[0] + 1,1)).T.flatten() # b-scan number       

        if os.path.isdir(folder_path):
            workbook = xls.Workbook(os.path.join(folder_path, self.project_name + "_0.xlsx"),  {'nan_inf_to_errors': True})
            worksheet = workbook.add_worksheet()
            worksheet.write_row(0, 0, self.header)
        else:
            os.path.mkdir(folder_path)
            workbook = xls.Workbook(os.path.join(folder_path, self.project_name + "_0.xlsx"),  {'nan_inf_to_errors': True})
            worksheet = workbook.add_worksheet()            
            worksheet.write_row(0, 0, self.header)
            
        row = 1
        header_length = len(self.header)
        file_num = 1

        for i, ids in enumerate(self.patients.keys()):
            for j in range(2): # first all OD than all OS
                for vi, visit in enumerate(self.patients[ids].visits): 
                    if j == 0:
                        if visit.relEZI_map_OD:
                            map = visit.relEZI_map_OD
                        else:
                            continue
                    else:
                        if visit.relEZI_map_OS:
                            map = visit.relEZI_map_OS
                        else:
                            continue
                        
                    # standard entries
                    worksheet.write(row,         0, "SeriesUID: " + str(map._series_uid) + " (PID: " + str(ids) + ")") # ID
                    worksheet.write_column(row,  1, nos * self.scan_size[0] * [map.laterality]) # Eye
                    worksheet.write_column(row,  2, b_scan_n) # bscan
                    worksheet.write(row,         3, visit.date_of_recording.strftime("%Y-%m-%d")) # Visit Date
                    worksheet.write_column(row,  4, a_scan_mesh) # A-scan
                    worksheet.write_column(row,  5, b_scan_mesh) # B-scan
                    worksheet.write_column(row, header_length -2, map.ezi_map.flatten())
                    worksheet.write_column(row, header_length -1, map.elmi_map.flatten())

                    # additional entries
                    for idx, ex_type in enumerate(map.excluded_maps.values()):
                         worksheet.write_column(row, 6 + idx, ex_type.flatten()) # exclusion type is added to the sheet

                    if "etdrs" in self.parameter:
                        worksheet.write_column(row, header_length -3, self.get_edtrs_grid_map(map._scan_area).flatten())

                    row += nos * self.scan_size[0]

                    if row == ((n * nos * self.scan_size[0]) +1)  and not (i == len(self.patients.keys()) -1 and vi == len(self.patients[ids].visits) -1 and  j == 1):
                        workbook.close()
                        workbook = xls.Workbook(os.path.join(folder_path, self.project_name + "_" + str(file_num) + ".xlsx"), {'nan_inf_to_errors': True})
                        worksheet = workbook.add_worksheet()            
                        worksheet.write_row(0, 0, self.header)   
                        row = 1
                        file_num += 1


        workbook.close()

class RelEZIQuantificationMactel2(RelEZIQuantificationMactel):

    '''
    Inherits from RelEZIQuantificationMactel
    Instead of making fovea shift for each visit the first visit of a particular pation is used as base and all follow up visits are registrated 
    on its SLO-Image

    Changes compared to RelEZIQuantificationMactel.create_relEZI_maps()
    -> No fovea shift is made
    -> After the maps are created there will be a loop over all follow up visits
        1. Registration on SLO-Image of first visit
        2. Shift all visits by the same shift parameter of the first visit
    '''
    
    def __init__(
        self, 
        _project_name: Optional[str] = None,
        _data_folder: Optional[Path] = None, 
        _fovea_coords: Optional[Dict] = None, 
        _scan_size: Optional[tuple] = None, 
        _scan_field: Optional[tuple] = None, 
        _stackwidth: Optional[int] = None, 
        _ssd_maps: Optional[SSDmap] = None, 
        _mean_rpedc_map: Optional[Mean_rpedc_map] = None,
        _parameter: Optional[List] = None, 
        _patients: Optional[Dict] = {}
        ):
        super().__init__(_project_name, _data_folder,  _fovea_coords, _scan_size, _scan_field, _stackwidth, _ssd_maps, _mean_rpedc_map, _parameter, _patients)


    def get_list(self, *args):

        if not args:
            data_folder = self.data_folder
        else:
            data_folder = args[0]

        if not os.path.exists(data_folder):
            raise NotADirectoryError("directory: " +  data_folder + " not exist")

        path_list = {}

        dir_list = os.listdir(data_folder)
        for dir in dir_list:
            full_path = os.path.join(data_folder, dir)
            if os.path.isdir(full_path):
                dir_list.extend(os.path.join(dir, subfolder) for subfolder in os.listdir(full_path))
            if os.path.isfile(full_path) and full_path.endswith(".vol"):
                sid = full_path.split("\\")[-1].split("_")[-1].split(".")[0] # series uid 
                path_list[sid] = full_path  
         
        return  path_list   

    def create_relEZI_maps(        
        self,
        data_folder: Union[str, Path, IO] = None,
        fovea_coords: Optional[Dict] = None,
        scan_size: Optional[tuple] = None,
        scan_field: Optional[tuple] = None,
        stackwidth: Optional[int] = None,
        ref_layer: Optional[str] = None,
        area_exclusion: Optional[List] = None,
        **kwargs
        ):
        """
        Args:
            folder_path (Union[str, Path, IO]): folder path where files are stored
            fovea_coords (Optional[Dict]): location of fovea
                !!! B-scan number counted from bottom to top like HEYEX !!! -> easier handling for physicians
                bscan (int): Number of B-scan including fovea
                ascan (int): Number of A-scan including fovea
            scan_size (Optional[tuple]): scan field size in x and y direction
                y (int): Number of B-scans
                x (int): Number of A-scans
            scan_field (Optional[tuple]): scan field size in x and y direction in degree
            stackwidth (Optional[int]): number of columns for a single profile
            ref_layer (Optional[str]): layer to flatten the image 
            area_exclusion ( Optional[Dict]): Method to determine area of exclusion 
                                            # if values (boolean) are True the area should not be analysed.


            Loop analyse data with respect to logithudinal data
            
            first visits is reference for all following visits 
            before creating relEZImaps the voxel ist registrated by using the SLO-Image between SLO[0] and SLO[n-1] where n ist the number of visits

        """


        # create copy of stackwidth to handle different scan sizes than expected
        stackwidth_fix = np.copy(stackwidth)
        factor = 1


        # raise expection if at least on argument is incorrect. Set instance variables.
        self.check_args(data_folder, fovea_coords, scan_size, scan_field, stackwidth, ref_layer, area_exclusion)

        # create the patient list structure without data
        self.create_patient_list()

        # get a dict structure containing the data in the shape <"ID":"path + .format">
        data_list = self.get_list()

        # get lists of exclusion data 
        for exclusion_type in area_exclusion:
            if exclusion_type == "ezloss":
                ae_dict_1 = get_ezloss_list(self.data_folder, "roi")
                self.update_header(-2, "ezloss(y/n)") 
        if "etdrs" in self.parameter:
            self.update_header(-2, "etdrs_grid") 


        # central bscan/ascan, number of stacks (nos)
        c_bscan = self.scan_size[0] // 2 + self.scan_size[0] % 2
        c_ascan = self.scan_size[1] // 2 + self.scan_size[1] % 2
        nos = self.scan_size[1] // self.stackwidth # number of stacks



        for i, ids in enumerate(self.patients.keys()):
            for j in range(2): # first all OD than all OS
                slo0 = None
                for vi, visit in enumerate(self.patients[ids].visits): 
                    if j == 0:
                        if visit.relEZI_map_OD:
                            map = visit.relEZI_map_OD
                        else:
                            continue
                    else:
                        if visit.relEZI_map_OS:
                            map = visit.relEZI_map_OS
                        else:
                            continue

                    # get data from vol-file
                    ms_analysis = macustar_segmentation_analysis.MacustarSegmentationAnalysis(
                        vol_file_path=map._volfile_path,
                        model_file_path=None,
                        use_gpu=True,
                        cuda_device=0,
                        normalize_mean=0.5,
                        normalize_std=0.25,
                        cache_segmentation=True
                        )

                    if vi == 0 or vi > 0 and not slo0:
                        # only if patient has more than one visit
                        if len(self.patients[ids].visits) > 1:
                            slo0 = ms_analysis.vol_file.slo_image # first SLO image

                            # get slo_coordinates
                            grid = np.array(ms_analysis.vol_file.grid)

                            slo0 = rotate_slo(slo0, grid, self.scan_field)


                        fovea_bscan, fovea_ascan = map._fovea_coordinates
                        factor = ms_analysis._vol_file.header.size_x / scan_size[1]
                        if factor * stackwidth_fix <= 1 and  factor % 1 == 0:
                            fovea_ascan = fovea_ascan * factor
                            stackwidth = int((1/factor) * stackwidth_fix)


                        # delta between real fovea centre and current fovea bscan position 
                        d_bscan  = c_bscan - fovea_bscan

                        # get start position to read data
                        d_ascan = int((factor * c_ascan) - fovea_ascan)
                        shift = min([d_ascan, 0])
                        start_r = int(- shift + ((factor * c_ascan) - (stackwidth//2) + shift) % stackwidth) # start reading
                        start_w = int(max([(((factor * c_ascan) - (stackwidth//2)) // stackwidth) - ((fovea_ascan) - (stackwidth//2)) // stackwidth, 0]))
                        n_st = int((ms_analysis._vol_file.header.size_x - start_r - max([d_ascan,0])) // stackwidth) # possible number of stacks 
                        
                        raw_voxel = ms_analysis._vol_file.oct_volume_raw[::-1]
                        seg_voxel = ms_analysis.classes[::-1,:,:]


                    else:
                        # registrate voxel based on slo0
                        # vol data
                        vol_raw = ms_analysis._vol_file.oct_volume_raw # default z y x -> x y z
                        vol_seg = ms_analysis.classes

                        # convert voxel data to sitk images
                        vol_raw_img = sitk.GetImageFromArray(vol_raw) # y x z -> z x y (sitk order)
                        vol_seg_img = sitk.GetImageFromArray(vol_seg) # y x z -> z x y (sitk order)
    
                        # set metrical spacing in each direction based on vol-header information 
                        z_scale = ms_analysis._vol_file.header.distance
                        x_scale = ms_analysis._vol_file.header.scale_x
                        y_scale = ms_analysis._vol_file.header.scale_z
                        vol_raw_img.SetSpacing((x_scale, y_scale, z_scale))
                        vol_seg_img.SetSpacing((x_scale, y_scale, z_scale))

                        # get orientation between voxel and slo 
                        grid = np.array(ms_analysis.vol_file.grid)
                        slon = ms_analysis.vol_file.slo_image
                        slon = rotate_slo(slon, grid, scan_field) 

                        # Matrix H
                        H = get2DProjectiveTransformationMartix_by_SuperRetina(slon, slo0)
                        if len(H) == 0:
                            print("%s: Registration failed" % (map.series_uid))
                            continue

                        # setup transformation

                        # traslation vector
                        translation = (-x_scale * H[0,-1], 0., z_scale * H[1,-1]  *(self.scan_size[0]/slon.shape[0])) # x y z
                        rotation_center = (0,0,z_scale*97) # x y z
                        affine= sitk.AffineTransform(3)

                        # rotation Matrix R
                        R = np.eye(3)
                        R[0,0] = H[0,0]
                        R[-1,-1] = H[1,1]
                        R[0,2] = -H[1,0]
                        R[2,0] = -H[0,1]

                        affine.SetMatrix(R.flatten())
                        affine.SetTranslation(translation)
                        affine.SetCenter(rotation_center)
    
                        resampled_raw = resample(vol_raw_img, affine, "Linear")
                        resampled_seg = resample(vol_seg_img, affine, "Linear")

                        raw_voxel = sitk.GetArrayFromImage(resampled_raw)[::-1] # x y z -> z y x
                        seg_voxel = sitk.GetArrayFromImage(resampled_seg)[::-1] # x y z -> z y x



                    # laterality 
                    lat = map.laterality

                    #sid
                    sid = map.series_uid


                    # current distance map/ exclusion map
                    curr_ez_intensity = np.zeros((scan_size[0], nos))
                    curr_elm_intensity = np.zeros_like(curr_ez_intensity)
                    curr_excluded = np.zeros_like(curr_ez_intensity)
                  

                    # get ezloss map if ez_loss exclusion is considered
                    if "ezloss" in area_exclusion:
                        if sid in ae_dict_1.keys():
                            ezloss_map = self.get_ezloss_map(ae_dict_1[sid], lat) 
                            exc_ezloss_map = np.zeros_like(curr_excluded)

            
                    # check if given number of b scans match with pre-defined number 
                    if ms_analysis._vol_file.header.num_bscans != scan_size[0]:
                        print("ID: %s has different number of bscans (%i) than expected (%i)" % (sid, ms_analysis._vol_file.header.num_bscans, scan_size[0]))
                        continue

                    # check if given number of a scans match with pre-defined number 
                    if ms_analysis._vol_file.header.size_x != scan_size[1]:
                        print("ID: %s has different number of ascans (%i) than expected (%i)" % (sid, ms_analysis._vol_file.header.size_x, scan_size[1]))
                        factor = ms_analysis._vol_file.header.size_x / scan_size[1]
                        if factor * stackwidth_fix <= 1 and  factor % 1 == 0:
                            stackwidth = int((1/factor) * stackwidth_fix) # change stackwidth temporarily to adjust to different scan sizes
            
            
                    for bscan, seg_mask, ez, elm, ezloss_excl, excl, ez_ssd_mean, ez_ssd_std, elm_ssd_mean, elm_ssd_std, idx_r, idx_w in zip(
                        raw_voxel[max([-d_bscan, 0]): scan_size[0] + min([-d_bscan, 0])], # read raw data
                        seg_voxel[max([-d_bscan, 0]): scan_size[0] + min([-d_bscan, 0])], # read seg mask
                        curr_ez_intensity[max([d_bscan, 0]): scan_size[0] + min([d_bscan, 0]), :], # write
                        curr_elm_intensity[max([d_bscan, 0]): scan_size[0] + min([d_bscan, 0]), :], # write
                        exc_ezloss_map[max([d_bscan, 0]): scan_size[0] + min([d_bscan, 0]), :], # write
                        curr_excluded[max([d_bscan, 0]): scan_size[0] + min([d_bscan, 0]), :], # write
                        self.ssd_maps.ez_ssd_map.distance_array[max([d_bscan, 0]): scan_size[0] + min([d_bscan, 0]), :], # write
                        self.ssd_maps.ez_ssd_map.std_array[max([d_bscan, 0]): scan_size[0] + min([d_bscan, 0]), :], # write
                        self.ssd_maps.elm_ssd_map.distance_array[max([d_bscan, 0]): scan_size[0] + min([d_bscan, 0]), :], # write                
                        self.ssd_maps.elm_ssd_map.std_array[max([d_bscan, 0]): scan_size[0] + min([d_bscan, 0]), :], # write 
                        range(max([-d_bscan, 0]), scan_size[0] + min([-d_bscan, 0])), # read
                        range(max([d_bscan, 0]), scan_size[0] + min([d_bscan, 0]))
                        ):

                
                        # flip maps if laterality is left (OS)
                        if lat == "OS":
                            bscan = np.flip(bscan,1)
                            seg_mask = np.flip(seg_mask,1)
            
                
                        # get rois
                        raw_roi, seg_mask_roi = get_roi_masks(bscan, self.ref_layer, ms_analysis._vol_file.header.size_x, seg_mask)
                
                        # iterate over bscans
                        for i in range(n_st):  



                            if "default" in exclusion_dict:
                                pass 
                                ############### to be implemented ##################
                                # condition seg_mask_area 10 and 11 thickness over stackwidth not higher 15

                            else:
                                if "ezloss" in area_exclusion:
                                    if any(ezloss_map[idx_w, start_r + i * self.stackwidth: start_r + (i + 1) * self.stackwidth] >= 1):
                                        ezloss_excl[start_w + i] = np.nanmax(ezloss_map[idx_w, start_r + i * self.stackwidth: start_r + (i + 1) * self.stackwidth]) 

                                excl[start_w + i] = self.get_exclusion_value(idx_w, start_r, i)

  
                            # get rpe peak
                            rpe_peak = get_rpe_peak(raw_roi, seg_mask_roi, start_r, i, stackwidth)

                            if not rpe_peak:
                                continue
                        
                            i_profile = np.nanmean(raw_roi[:,start_r + i * stackwidth: start_r + (i + 1) * stackwidth],1)

                            ez_peak, elm_peak = get_ez_elm_peak(i_profile,
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

                    # set stackwith and factor to default
                    stackwidth = stackwidth_fix
                    factor = 1

                    tmp_excluded_dict = {}

                    for idx, exclusion_type in zip(range(len(exclusion_dict)-1,-1,-1), exclusion_dict):
                        tmp_excluded_dict[exclusion_type] = curr_excluded // 2**idx
                        curr_excluded = curr_excluded % 2**idx

                    if "ezloss" in area_exclusion:
                        tmp_excluded_dict["ezloss"] = exc_ezloss_map


                    # add data to map object
                    map._ezi_map = curr_ez_intensity
                    map._elmi_map = curr_elm_intensity
                    map._excluded_maps = tmp_excluded_dict


    def create_patient_list(self):

        data_list = self.get_list()

        for sid in data_list:

            vol_file = vol_reader.VolFile(data_list[sid])

            lat = vol_file.header.scan_position

            # d_bscan (int): delta_bscan = [central bscan (number of bscans // 2)] - [current bscan]
            try:
                fovea_bscan, fovea_ascan = self.fovea_coords[sid]
            except:
                print("ID %s is missing in Fovea List " % sid)
                continue
            
            # change orientation from top down, subtract on from coords to keep 0-indexing of python            
            fovea_bscan = self.scan_size[0] - fovea_bscan +1

            if lat == "OS": # if left eye is processed
                fovea_ascan = self.scan_size[1] - fovea_ascan +1   

            # calculate scan_area
            scan_area = tuple(
                    [vol_file.bscan_headers[0].start_y - vol_file.bscan_headers[-1].start_y, # enface y-direction in mm
                    vol_file.bscan_headers[0].end_x - vol_file.bscan_headers[0].start_x] # enface x-direction in mm
                     )          
            
            # create Map Objects containing without maps
            current_map = RelEZI_map(
                "relEZI-Map",
                date.today(),
                self.scan_size,
                self.scan_field,
                scan_area,
                self.stackwidth,
                sid,
                data_list[sid],
                lat,
                (fovea_bscan, fovea_ascan), # (x,y)
                None, # curr_ez_intensity
                None, # curr_elm_intensity
                None, # tmp_excluded_dict
                )            

            pid = vol_file.header.pid

            if pid in self.patients.keys():
                self.add(current_map, pid, vol_file.header.visit_date, None)
            else:
                self.add(current_map, pid, vol_file.header.visit_date, None, vol_file.header.birthdate)

                    

    def create_excel_sheets(
        self,
        folder_path,
        n
        ):
        """
        Args:
            folder_path (str): Target folder for data
            n (int): Number of visits per sheet
            project (str): project name like Macustar
            
        """

        nos = self.scan_size[1] // self.stackwidth # number of stacks
        d_a_scan = self.scan_field[1] / nos # distance between stacks in degree
        d_b_scan = self.scan_field[0] / self.scan_size[0] # distance between b-scans in degree
        a_scan_mesh, b_scan_mesh = np.meshgrid(
                    np.arange(-self.scan_field[1]/2, self.scan_field[1]/2,d_a_scan),
                    np.arange(-self.scan_field[0]/2, self.scan_field[0]/2,d_b_scan))
        a_scan_mesh = a_scan_mesh.flatten() # serialized a-scan mesh
        b_scan_mesh = b_scan_mesh.flatten() # serialized a-scan mesh 

        b_scan_n = (np.ones((nos, self.scan_size[0])) * np.arange(1, self.scan_size[0] + 1,1)).T.flatten() # b-scan number       

        if os.path.isdir(folder_path):
            workbook = xls.Workbook(os.path.join(folder_path, self.project_name + "_0.xlsx"),  {'nan_inf_to_errors': True})
            worksheet = workbook.add_worksheet()
            worksheet.write_row(0, 0, self.header)
        else:
            os.path.mkdir(folder_path)
            workbook = xls.Workbook(os.path.join(folder_path, self.project_name + "_0.xlsx"),  {'nan_inf_to_errors': True})
            worksheet = workbook.add_worksheet()            
            worksheet.write_row(0, 0, self.header)
            
        row = 1
        header_length = len(self.header)
        file_num = 1

        for i, ids in enumerate(self.patients.keys()):
            for j in range(2): # first all OD than all OS
                for vi, visit in enumerate(self.patients[ids].visits): 
                    if j == 0:
                        if visit.relEZI_map_OD:
                            map = visit.relEZI_map_OD
                        else:
                            continue
                    else:
                        if visit.relEZI_map_OS:
                            map = visit.relEZI_map_OS
                        else:
                            continue
                        
                    # standard entries
                    worksheet.write_column(row,  0, np.array(["SeriesUID: " + str(map._series_uid) + " (PID: " + str(ids) + ")", 
                                                            "y-direction in mm:" + str(np.around(map._scan_area[0], decimals=2)),"x-direction in mm:" + str(np.around(map._scan_area[1], decimals=2))])) # ID and scan area
                    worksheet.write_column(row,  1, nos * self.scan_size[0] * [map.laterality]) # Eye
                    worksheet.write_column(row,  2, b_scan_n) # bscan
                    worksheet.write(row,         3, visit.date_of_recording.strftime("%Y-%m-%d")) # Visit Date
                    worksheet.write_column(row,  4, a_scan_mesh) # A-scan
                    worksheet.write_column(row,  5, b_scan_mesh) # B-scan

                    # get arrays as string type
                    ez_map =np.copy(map.ezi_map).astype(str).flatten()
                    ez_map[ez_map == "0.0"] = "nan"
                    elm_map =np.copy(map.elmi_map).astype(str).flatten()
                    elm_map[elm_map == "0.0"] = "nan"

                    worksheet.write_column(row, header_length -2, ez_map)
                    worksheet.write_column(row, header_length -1, elm_map)

                    # additional entries

                    if map.excluded_maps is not None:
                        for idx, ex_type in enumerate(map.excluded_maps.values()):
                            worksheet.write_column(row, 6 + idx, ex_type.flatten()) # exclusion type is added to the sheet

                    if "etdrs" in self.parameter:
                        worksheet.write_column(row, header_length -3, self.get_edtrs_grid_map(map._scan_area).flatten())

                    row += nos * self.scan_size[0]

                    if row == ((n * nos * self.scan_size[0]) +1)  and not (i == len(self.patients.keys()) -1 and vi == len(self.patients[ids].visits) -1 and  j == 1):
                        workbook.close()
                        workbook = xls.Workbook(os.path.join(folder_path, self.project_name + "_" + str(file_num) + ".xlsx"), {'nan_inf_to_errors': True})
                        worksheet = workbook.add_worksheet()            
                        worksheet.write_row(0, 0, self.header)   
                        row = 1
                        file_num += 1


        workbook.close()

class RelEZIQuantificationMacustar(RelEZIQuantificationBase):

   

    def __init__(
        self, 
        _project_name: Optional[str] = None,
        _data_folder: Optional[Path] = None, 
        _fovea_coords: Optional[Dict] = None, 
        _scan_size: Optional[tuple] = None, 
        _scan_field: Optional[tuple] = None, 
        _stackwidth: Optional[int] = None, 
        _ssd_maps: Optional[SSDmap] = None, 
        _mean_rpedc_map: Optional[Mean_rpedc_map] = None, 
        _parameter: Optional[List] = None,
        _patients: Optional[Dict] = {}
        ):
        super().__init__(_project_name, _data_folder,  _fovea_coords, _scan_size, _scan_field, _stackwidth, _ssd_maps, _mean_rpedc_map, _parameter, _patients)


    def get_list(self, *args):

        if not args:
            data_folder = self.data_folder
        else:
            data_folder = args[0]

        if not os.path.exists(data_folder):
            raise NotADirectoryError("directory: " +  data_folder + " not exist")

        path_list = {}

        dir_list = os.listdir(data_folder)
        for dir in dir_list:
            full_path = os.path.join(data_folder, dir)
            if os.path.isdir(full_path):
                dir_list.extend(os.path.join(dir, subfolder) for subfolder in os.listdir(full_path))
            if os.path.isfile(full_path) and full_path.endswith(".vol"):
                pid = full_path.split("\\")[-2].split("_")[1][4:]
                path_list[pid] = full_path        

        return path_list


    def create_relEZI_maps(        
        self,
        data_folder: Union[str, Path, IO] = None,
        fovea_coords: Optional[Dict] = None,
        scan_size: Optional[tuple] = None,
        scan_field: Optional[tuple] = None,
        stackwidth: Optional[int] = None,
        ref_layer: Optional[str] = None,
        area_exclusion: Optional[List] = None,
        **kwargs
        ):
        """
        Args:
            folder_path (Union[str, Path, IO]): folder path where files are stored
            fovea_coords (Optional[Dict]): location of fovea
                !!! B-scan number counted from bottom to top like HEYEX !!! -> easier handling for physicians
                bscan (int): Number of B-scan including fovea
                ascan (int): Number of A-scan including fovea
            scan_size (Optional[tuple]): scan field size in x and y direction
                x (int): Number of B-scans
                y (int): Number of A-scans
            scan_field (Optional[tuple]): scan field size in x and y direction in degree
                x (int): Number of B-scans
                y (int): Number of A-scans
            stackwidth (Optional[int]): number of columns for a single profile
            ref_layer (Optional[str]): layer to flatten the image 
            area_exclusion ( Optional[Dict]): Method to determine area of exclusion 
                                            # if values (boolean) are True the area should not be analysed.
        """

        # create copy of stackwidth to handle different scan sizes than expected
        stackwidth_fix = np.copy(stackwidth)
        factor = 1

        # raise expection if at least on argument is incorrect. Set instance variables.
        self.check_args(data_folder, fovea_coords, scan_size, scan_field, stackwidth, ref_layer, area_exclusion)

        # get a dict structure containing the data in the shape <"ID":"path + .format">
        data_list = self.get_list()

        # get lists of exclusion data 
        for exclusion_type in area_exclusion:
            if exclusion_type == "rpedc":
                ae_dict_1 = get_rpedc_list(self.data_folder)
                self.update_header(-2, "druse(y/n)") 
                if "atrophy" in area_exclusion:
                    self.update_header(-2, "atrophy(y/n)")
                if len(ae_dict_1.keys()) == 0:
                    raise ValueError("If rpedc maps should be considered the data must be in the same folder as the other data")
            if exclusion_type == "rpd":
                ae_dict_2 = get_rpd_list(self.data_folder)
                self.update_header(-2, "rpd(y/n)")


        # central bscan/ascan, number of stacks (nos)
        c_bscan = self.scan_size[0] // 2 + self.scan_size[0] % 2
        c_ascan = self.scan_size[1] // 2 + self.scan_size[1] % 2
        nos = self.scan_size[1] // self.stackwidth # number of stacks

        for vol_id in data_list:

            # current distance map/ exclusion map
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

            # get data from vol-file
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
                fovea_ascan = scan_size[1] - fovea_ascan +1


            # get rpedc map if rpedc exclusion is considered
            if "rpedc" in area_exclusion:
                if vol_id in ae_dict_1.keys():
                    rpedc_map = self.get_rpedc_map(ae_dict_1[vol_id], self.scan_size, self.mean_rpedc_map, lat, (d_bscan, d_ascan))
                    if "atrophy" in area_exclusion:
                        exclusion_dict["atrophy"] = rpedc_map == 1
                    exclusion_dict["rpedc"] = rpedc_map == 2
                else:
                    print("ID: %s considered rpedc map not exist" % vol_id)
                    continue
            
            # get rpd map if rpd exclusion is considered
            if "rpd" in area_exclusion:
                if vol_id in ae_dict_2.keys():
                    exclusion_dict["rpd"] = get_rpd_map(ae_dict_2[vol_id], self.scan_size, lat, (d_bscan, d_ascan))
                else:
                    print("ID: %s considered rpd map not exist" % vol_id)
                    exclusion_dict["rpd"] = np.zeros(self.scan_size).astype(bool)

            
            # check if given number of b scans match with pre-defined number 
            if ms_analysis._vol_file.header.num_bscans != scan_size[0]:
                print("ID: %s has different number of bscans (%i) than expected (%i)" % (vol_id, ms_analysis._vol_file.header.num_bscans, scan_size[0]))
                continue

            # check if given number of a scans match with pre-defined number 
            if ms_analysis._vol_file.header.size_x != scan_size[1]:
                print("ID: %s has different number of ascans (%i) than expected (%i)" % (vol_id, ms_analysis._vol_file.header.size_x, scan_size[1]))
                factor = ms_analysis._vol_file.header.size_x / scan_size[1]
                if factor * stackwidth_fix >= 1 and  factor % 1 == 0:
                    stackwidth = int(factor * stackwidth_fix) # change stackwidth temporarily to adjust to different scan sizes

            
            for bscan, seg_mask, ez, elm, excl, ez_ssd_mean, ez_ssd_std, elm_ssd_mean, elm_ssd_std, idx_r, idx_w in zip(
                ms_analysis._vol_file.oct_volume_raw[::-1][max([-d_bscan, 0]): scan_size[0] + min([-d_bscan, 0])], # read raw data
                ms_analysis.classes[::-1,:,:][max([-d_bscan, 0]): scan_size[0] + min([-d_bscan, 0])], # read seg mask
                curr_ez_intensity[max([d_bscan, 0]): scan_size[0] + min([d_bscan, 0]), :], # write
                curr_elm_intensity[max([d_bscan, 0]): scan_size[0] + min([d_bscan, 0]), :], # write
                curr_excluded[max([d_bscan, 0]): scan_size[0] + min([d_bscan, 0]), :], # write
                self.ssd_maps.ez_ssd_map.distance_array[max([d_bscan, 0]): scan_size[0] + min([d_bscan, 0]), :], # write
                self.ssd_maps.ez_ssd_map.std_array[max([d_bscan, 0]): scan_size[0] + min([d_bscan, 0]), :], # write
                self.ssd_maps.elm_ssd_map.distance_array[max([d_bscan, 0]): scan_size[0] + min([d_bscan, 0]), :], # write                
                self.ssd_maps.elm_ssd_map.std_array[max([d_bscan, 0]): scan_size[0] + min([d_bscan, 0]), :], # write 
                range(max([-d_bscan, 0]), scan_size[0] + min([-d_bscan, 0])), # read
                range(max([d_bscan, 0]), scan_size[0] + min([d_bscan, 0]))
                ):

                
                # flip maps if laterality is left (OS)
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
                raw_roi, seg_mask_roi = get_roi_masks(bscan, self.ref_layer, ms_analysis._vol_file.header.size_x, seg_mask)
                
                # iterate over bscans
                for i in range(n_st):  



                    if "default" in exclusion_dict:
                        pass 
                        ############### to be implemented ##################
                        # condition seg_mask_area 10 and 11 thickness over stackwidth not higher 15

                    else:
                        excl[start_w + i] = self.get_exclusion_value(idx_w, start_r, i)

  
                        # get rpe peak
                        rpe_peak = get_rpe_peak(raw_roi, seg_mask_roi, start_r, i, self.stackwidth)

                        if not rpe_peak:
                            continue
                        
                        i_profile = np.nanmean(raw_roi[:,start_r + i * self.stackwidth: start_r + (i + 1) * self.stackwidth],1)

                        ez_peak, elm_peak = get_ez_elm_peak(i_profile,
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

            # set stackwith and factor to default
            stackwidth = stackwidth_fix
            factor = 1

            tmp_excluded_dict = {}

            for idx, exclusion_type in zip(range(len(exclusion_dict)-1,-1,-1), exclusion_dict):
                tmp_excluded_dict[exclusion_type] = curr_excluded // 2**idx
                curr_excluded = curr_excluded % 2**idx

            # calculate scan_area
            scan_area = tuple(
                [ms_analysis._vol_file.bscan_headers[0].start_y - ms_analysis._vol_file.bscan_headers[-1].start_y, # enface y-direction in mm
                ms_analysis._vol_file.bscan_headers[0].end_x - ms_analysis._vol_file.bscan_headers[0].start_x] # enface x-direction in mm
                )                          
            
            # create Map Objects containing the created maps
            current_map = RelEZI_map(
                "relEZI-Map",
                date.today(),
                self.scan_size,
                self.scan_field,
                scan_area,
                self.stackwidth,
                None,
                data_list[vol_id],
                lat,
                (fovea_bscan, fovea_ascan), # (x,y)
                curr_ez_intensity,
                curr_elm_intensity,
                tmp_excluded_dict
                )            


            if vol_id in self.patients.keys():
                self.add(current_map, vol_id, ms_analysis._vol_file.header.visit_date, vol_id)
            else:
                self.add(current_map, vol_id, ms_analysis._vol_file.header.visit_date, vol_id, ms_analysis._vol_file.header.birthdate)

    def create_excel_sheets(
        self,
        folder_path,
        n
        ):
        """
        Args:
            folder_path (str): Target folder for data
            n (int): Number of visits per sheet
            project (str): project name like Macustar
            
        """

        nos = self.scan_size[1] // self.stackwidth # number of stacks
        d_a_scan = self.scan_field[1] / nos # distance between stacks in degree
        d_b_scan = self.scan_field[0] / self.scan_size[0] # distance between b-scans in degree
        a_scan_mesh, b_scan_mesh = np.meshgrid(
                    np.arange(-self.scan_field[1]/2, self.scan_field[1]/2,d_a_scan),
                    np.arange(-self.scan_field[0]/2, self.scan_field[0]/2,d_b_scan))
        a_scan_mesh = a_scan_mesh.flatten() # serialized a-scan mesh
        b_scan_mesh = b_scan_mesh.flatten() # serialized a-scan mesh 

        b_scan_n = (np.ones((nos, self.scan_size[0])) * np.arange(1, self.scan_size[0] + 1,1)).T.flatten() # b-scan number       


        if os.path.isdir(folder_path):
            workbook = xls.Workbook(os.path.join(folder_path, self.project_name + "_0.xlsx"),  {'nan_inf_to_errors': True})
            worksheet = workbook.add_worksheet()
            worksheet.write_row(0, 0, self.header)
        else:
            os.path.mkdir(folder_path)
            workbook = xls.Workbook(os.path.join(folder_path, self.project_name + "_0.xlsx"),  {'nan_inf_to_errors': True})
            worksheet = workbook.add_worksheet()            
            worksheet.write_row(0, 0, self.header)
            
        row = 1
        header_length = len(self.header)
        file_num = 1

        for i, ids in enumerate(self.patients.keys()):
            for j in range(2): # first all OD than all OS
                for vi, visit in enumerate(self.patients[ids].visits): 
                    if j == 0:
                        if visit.relEZI_map_OD:
                            map = visit.relEZI_map_OD
                        else:
                            continue
                    else:
                        if visit.relEZI_map_OS:
                            map = visit.relEZI_map_OS
                        else:
                            continue
                        
                    # standard entries
                    worksheet.write(row,         0,  "313" + "".join(n for n in ids.split("-"))) # ID
                    worksheet.write_column(row,  1, nos * self.scan_size[0] * [map.laterality]) # Eye
                    worksheet.write_column(row,  2, b_scan_n) # bscan
                    worksheet.write(row,         3, visit.date_of_recording.strftime("%Y-%m-%d")) # Visit Date
                    worksheet.write_column(row,  4, a_scan_mesh) # A-scan
                    worksheet.write_column(row,  5, b_scan_mesh) # B-scan
                    worksheet.write_column(row, header_length -2, map.ezi_map.flatten())
                    worksheet.write_column(row, header_length -1, map.elmi_map.flatten())

                    # additional entries
                    for idx, ex_type in enumerate(map.excluded_maps.values()):
                         worksheet.write_column(row, 6 + idx, ex_type.flatten()) # exclusion type is added to the sheet

                    if "etdrs" in self.parameter:
                        worksheet.write_column(row, header_length -3, self.get_edtrs_grid_map(map._scan_area).flatten())

                    row += nos * self.scan_size[0]

                    if row == ((n * nos * self.scan_size[0]) +1)  and not (i == len(self.patients.keys()) -1 and vi == len(self.patients[ids].visits) -1 and  j == 1):
                        workbook.close()
                        workbook = xls.Workbook(os.path.join(folder_path, self.project_name + "_" + str(int((i +1) / n)) + ".xlsx"), {'nan_inf_to_errors': True})
                        worksheet = workbook.add_worksheet()            
                        worksheet.write_row(0, 0, self.header)   
                        row = 1
                        file_num += 1

        workbook.close()

class RelEZIQuantificationMicro(RelEZIQuantificationMacustar):

    def __init__(
        self, 
        _project_name: Optional[str] = None,
        _data_folder: Optional[Path] = None, 
        _fovea_coords: Optional[Dict] = None, 
        _scan_size: Optional[tuple] = None, 
        _scan_field: Optional[tuple] = None, 
        _stackwidth: Optional[int] = None, 
        _ssd_maps: Optional[SSDmap] = None, 
        _mean_rpedc_map: Optional[Mean_rpedc_map] = None, 
        _parameter: Optional[List] = None,
        _patients: Optional[Dict] = {}
        ):
        super().__init__(_project_name, _data_folder,  _fovea_coords, _scan_size, _scan_field, _stackwidth, _ssd_maps, _mean_rpedc_map, _parameter, _patients)


    def get_microperimetry_grid_field(self, micro_data_path, micro_ir_path, visit, radius, use_gpu):

        if len(self.patients) == 0:
            raise Exception("So far, no patient has been analyzed, please first use calculate_relEZI_maps()")

        ir_list_m, ir_list_s = get_microperimetry_IR_image_list(micro_ir_path)

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

            stimuli_s = get_microperimetry(
                df,
                self.patients[keys].pid,
                visit,
                lat,
                "S")

            stimuli_m = get_microperimetry(
                df,
                self.patients[keys].pid,
                visit,
                lat,
                "M")

           # create grid coords
            px_deg_y = px_deg_x = slo_img.shape[0] / 30 # pixel per degree
            ecc = np.array([items[0] for items in grid_iamd.values()]) * px_deg_y
            ang = np.array([items[1] for items in grid_iamd.values()]) * np.pi / 180

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
            vol_R = get2DRigidTransformationMatrix(q, p)

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

            mask_iamd_m, stimuli_m_map = get_microperimetry_maps(
                    ir_list_m[self.patients[keys].pid],
                    lat,
                    radius,
                    slo_img,  
                    self.scan_size,
                    self.stackwidth,
                    stimuli_m,
                    x,y)

            mask_iamd_s, stimuli_s_map = get_microperimetry_maps(
                    ir_list_s[self.patients[keys].pid],
                    lat,
                    radius,
                    slo_img,  
                    self.scan_size,
                    self.stackwidth,
                    stimuli_s,
                    x,y)

            if lat == "OD":
                self.update_header(-2, "micro_mask_m")
                self.patients[keys].visits[visit -2].relEZI_map_OD._excluded_maps["micro_mask_m"] = mask_iamd_m
                self.update_header(-2, "micro_mask_s")
                self.patients[keys].visits[visit -2].relEZI_map_OD._excluded_maps["micro_mask_s"] = mask_iamd_s
                self.update_header(-2, "micro_stim_m")
                self.patients[keys].visits[visit -2].relEZI_map_OD._excluded_maps["micro_stim_m"] = stimuli_m_map
                self.update_header(-2, "micro_stim_s")
                self.patients[keys].visits[visit -2].relEZI_map_OD._excluded_maps["micro_stim_s"] = stimuli_s_map
            else:
                self.update_header(-2, "micro_mask_m")
                self.patients[keys].visits[visit -2].relEZI_map_OS._excluded_maps["micro_mask_m"] = mask_iamd_m
                self.update_header(-2, "micro_mask_s")
                self.patients[keys].visits[visit -2].relEZI_map_OS._excluded_maps["micro_mask_s"] = mask_iamd_s
                self.update_header(-2, "micro_stim_m")
                self.patients[keys].visits[visit -2].relEZI_map_OS._excluded_maps["micro_stim_m"] = stimuli_m_map
                self.update_header(-2, "micro_stim_s")
                self.patients[keys].visits[visit -2].relEZI_map_OS._excluded_maps["micro_stim_s"] = stimuli_s_map


    def get_microperimetry_grid_field_show(self, micro_data_path, micro_ir_path, target_path, visit, use_gpu):
        if len(self.patients) == 0:
            raise Exception("So far, no patients have been analyzed, please first use calculate_relEZI_maps()")

        ir_list_m, ir_list_s = get_microperimetry_IR_image_list(micro_ir_path)

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



            stimuli_s = get_microperimetry(
                df,
                self.patients[keys].pid,
                visit,
                lat,
                "S")

            stimuli_m = get_microperimetry(
                df,
                self.patients[keys].pid,
                visit,
                lat,
                "M")

        

            # create grid coords
            px_deg_y = px_deg_x = slo_img.shape[0] / 30 # pixel per degree
            ecc = np.array([items[0] for items in grid_iamd.values()]) * px_deg_y
            ang = np.array([items[1] for items in grid_iamd.values()]) * np.pi / 180

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
            vol_R = get2DRigidTransformationMatrix(q, p)

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
            H_m = get2DProjectiveTransformationMartix_by_SuperRetina(slo_img, img1_m)
            H_s = get2DProjectiveTransformationMartix_by_SuperRetina(slo_img, img1_s)
        

            show_grid_over_relEZIMap(
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

            show_grid_over_relEZIMap(
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
if __name__ == "__main__":
    pass