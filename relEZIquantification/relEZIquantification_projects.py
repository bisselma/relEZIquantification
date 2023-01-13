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

from relEZIquantification.relEZIquantification_structure import *
from relEZIquantification.relEZIquantification_os import *
from relEZIquantification.relEZIquantification_utils import *



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
    def patients(self):
        return self._patients 

    @property
    def header(self):
        return self._header

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
            if any(exclusion_dict[exclusion_type][idx_w, start_r + i * self.stackwidth: start_r + (i + 1) * self.stackwidth] == 1):
                    binary_number += 2**idx

        return binary_number

    def add(self, map, pid, visitdate, *args): # if patient allready exists
        if args: # new patient
            if map.laterality == "OD":
                self.patients[pid] = Patient(pid, args[0], Visit(None, visitdate, map, None))
            elif  map.laterality == "OS": 
                self.patients[pid] = Patient(pid, args[0], Visit(None, visitdate, None, map))
            else:
                raise ValueError("Map attribute 'laterality' is not correct") 
        else: # add visit to patient object
            self.patients[pid].add(map, visitdate)


    def get_list(self, *args):
        pass

    def create_relEZI_maps(self):
        pass

    def create_excel_sheets(self):
	    pass

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
        _patients: Optional[Dict] = {}
        ):
        super().__init__(_project_name, _data_folder,  _fovea_coords, _scan_size, _scan_field, _stackwidth, _ssd_maps, _mean_rpedc_map, _patients)


    def get_list(self, *args):

        if args:
            data_folder = args[0]
        else:
            data_folder = self.data_folder

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
        area_exclusion: Optional[Dict] = None,
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

        # raise expection if at least on argument is incorrect. Set instance variables.
        self.check_args(data_folder, fovea_coords, scan_size, scan_field, stackwidth, ref_layer, area_exclusion)

        # get a dict structure containing the data in the shape <"ID":"path + .format">
        data_list = self.get_list()

        # get lists of exclusion data 
        for exclusion_type in area_exclusion.keys():
            if exclusion_type == "rpedc":
                ae_dict_1 = ut.get_rpedc_list(self.data_folder)
                self.update_header(-2, "druse(y/n)") 
                if "atrophy" in area_exclusion.keys():
                    self.update_header(-2, "atrophy(y/n)")
                if len(ae_dict_1.keys()) == 0:
                    raise ValueError("If rpedc maps should be considered the data must be in the same folder as the other data")
            if exclusion_type == "rpd":
                ae_dict_2 = ut.get_rpd_list(self.data_folder)
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
            if "rpedc" in area_exclusion.keys():
                if vol_id in ae_dict_1.keys():
                    rpedc_map = get_rpedc_map(ae_dict_1[vol_id], self.scan_size, self.mean_rpedc_map, lat, (d_bscan, d_ascan))
                    if "atrophy" in area_exclusion.keys():
                        exclusion_dict["atrophy"] = rpedc_map == 1
                    exclusion_dict["rpedc"] = rpedc_map == 2
                else:
                    print("ID: %s considered rpedc map not exist" % vol_id)
                    continue
            
            # get rpd map if rpd exclusion is considered
            if "rpd" in area_exclusion.keys():
                if vol_id in ae_dict_2.keys():
                    exclusion_dict["rpd"] = self.get_rpd_map(ae_dict_2[vol_id], self.scan_size, lat, (d_bscan, d_ascan))
                else:
                    print("ID: %s considered rpd map not exist" % vol_id)
                    exclusion_dict["rpd"] = np.zeros(self.scan_size).astype(bool)

            
            # check if given number of b scans match with pre-defined number 
            if ms_analysis._vol_file.header.num_bscans != scan_size[0]:
                print("ID: %s has different number of bscans (%i) than expected (%i)" % (vol_id, ms_analysis._vol_file.header.num_bscans, scan_size[0]))
                continue

            # check if given number of a scans match with pre-defined number 
            if ms_analysis._vol_file.header.size_x != scan_size[1]:
                print("ID: %s has different number of ascans (%i) than expected (%i)" % (ut.get_id_by_file_path(data_list[vol_id]), ms_analysis._vol_file.header.size_x, scan_size[1]))
                continue  

            
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
                
                
                # initilized start indices to read and write data
                shift = min([d_ascan, 0])
                start_r = - shift + (c_ascan - (stackwidth//2) + shift) % stackwidth # start reading
                start_w = max([((c_ascan - (stackwidth//2)) // stackwidth) - (fovea_ascan - (stackwidth//2)) // stackwidth, 0])
                n_st = (scan_size[1] - start_r - max([d_ascan,0])) // stackwidth # possible number of stacks 
                
                
                # get rois
                raw_roi, seg_mask_roi = get_roi_masks(bscan, self.ref_layer, self.scan_size, seg_mask)
                
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


            tmp_excluded_dict = {}

            for idx, exclusion_type in zip(range(len(exclusion_dict)-1,-1,-1), exclusion_dict):
                tmp_excluded_dict[exclusion_type] = curr_excluded // 2**idx
                curr_excluded = curr_excluded % 2**idx
                         
            
            # create Map Objects containing the created maps
            current_map = RelEZI_map(
                "relEZI-Map",
                date.today(),
                self.scan_size,
                self.scan_field,
                self.stackwidth,
                data_list[vol_id],
                lat,
                (fovea_ascan, fovea_bscan), # (x,y)
                curr_ez_intensity,
                curr_elm_intensity,
                tmp_excluded_dict
                )            


            if vol_id in self.patients.keys():
                self.add(current_map, vol_id, ms_analysis._vol_file.header.visit_date)
            else:
                self.add(current_map, vol_id, ms_analysis._vol_file.header.visit_date, ms_analysis._vol_file.header.birthdate)

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

        for i, ids in enumerate(self.patients.keys()):
            for visit in self.patients[ids].visits: 
                for k, map in enumerate(visit.get_maps()): # if OD and OS, the sheet is extended to the right

                        # standard entries
                        worksheet.write(row, k * header_length, "313" + "".join(i for i in ids.split("-"))) # ID
                        worksheet.write_column(row, k * header_length + 1, nos * self.scan_size[0] * [map.laterality]) # Eye
                        worksheet.write_column(row, k * header_length + 2, b_scan_n) # bscan
                        worksheet.write(row, k * header_length + 3, visit.date_of_recording.strftime("%Y-%m-%d")) # Visit Date
                        worksheet.write_column(row, k * header_length + 4, a_scan_mesh) # A-scan
                        worksheet.write_column(row, k * header_length + 5, b_scan_mesh) # B-scan
                        worksheet.write_column(row, k * header_length + header_length -2, map.ezi_map.flatten())
                        worksheet.write_column(row, k * header_length + header_length -1, map.elmi_map.flatten())

                        # additional entries
                        for idx, ex_type in enumerate(map.excluded_maps.values()):
                            worksheet.write_column(row, k * header_length + 6 + idx, ex_type.flatten()) # exclusion type is added to the sheet


                row += nos * self.scan_size[0]

                if (i +1) % n == 0 and i < len(self.patients.keys()) -1:
                    workbook.close()
                    workbook = xls.Workbook(os.path.join(folder_path, self.project_name + "_" + str(int((i +1) / n)) + ".xlsx"), {'nan_inf_to_errors': True})
                    worksheet = workbook.add_worksheet()            
                    worksheet.write_row(0, 0, self.header)   
                    row = 1

        workbook.close()

class RelEZIQuantificationMicro(RelEZIQuantificationMacustar):
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
        _patients: Optional[Dict] = {}
        ):
        super().__init__(_project_name, _data_folder,  _fovea_coords, _scan_size, _scan_field, _stackwidth, _ssd_maps, _mean_rpedc_map, _patients)


    def get_list(self, *args):

        if not data_folder:
            data_folder = self.data_folder

        if not os.path.exists(data_folder):
            raise NotADirectoryError("directory: " +  data_folder + " not exist")

        path_list = {}

        dir_list = os.listdir(data_folder)
        for dir in dir_list:
            full_path = os.path.join(data_folder, dir)
            if os.path.isdir(full_path):
                dir_list.extend(os.path.join(dir, subfolder) for subfolder in os.listdir(full_path))
            if os.path.isfile(full_path) and full_path.endswith(".vol"):
                vid = full_path.split("\\")[-1].split("_")[-1].split(".")[0]   
                path_list[vid] = full_path  
         
        return  path_list   


    def create_relEZI_maps(        
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

        # raise expection if at least on argument is incorrect. Set instance variables.
        self.check_args(data_folder, fovea_coords, scan_size, scan_field, stackwidth, ref_layer, area_exclusion)

        # get a dict structure containing the data in the shape <"ID":"path + .format">
        data_list = self.get_list()

        # get lists of exclusion data 
        for exclusion_type in area_exclusion.keys():
            if exclusion_type == "rpedc":
                ae_dict_1 = ut.get_rpedc_list(self.data_folder)
                self.update_header(-2, "druse(y/n)") 
                if "atrophy" in area_exclusion.keys():
                    self.update_header(-2, "atrophy(y/n)")
                if len(ae_dict_1.keys()) == 0:
                    raise ValueError("If rpedc maps should be considered the data must be in the same folder as the other data")
            if exclusion_type == "rpd":
                ae_dict_2 = ut.get_rpd_list(self.data_folder)
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
            if "rpedc" in area_exclusion.keys():
                if vol_id in ae_dict_1.keys():
                    rpedc_map = get_rpedc_map(ae_dict_1[vol_id], self.scan_size, self.mean_rpedc_map, lat, (d_bscan, d_ascan))
                    if "atrophy" in area_exclusion.keys():
                        exclusion_dict["atrophy"] = rpedc_map == 1
                    exclusion_dict["rpedc"] = rpedc_map == 2
                else:
                    print("ID: %s considered rpedc map not exist" % vol_id)
                    continue
            
            # get rpd map if rpd exclusion is considered
            if "rpd" in area_exclusion.keys():
                if vol_id in ae_dict_2.keys():
                    exclusion_dict["rpd"] = self.get_rpd_map(ae_dict_2[vol_id], self.scan_size, lat, (d_bscan, d_ascan))
                else:
                    print("ID: %s considered rpd map not exist" % vol_id)
                    exclusion_dict["rpd"] = np.zeros(self.scan_size).astype(bool)

            
            # check if given number of b scans match with pre-defined number 
            if ms_analysis._vol_file.header.num_bscans != scan_size[0]:
                print("ID: %s has different number of bscans (%i) than expected (%i)" % (vol_id, ms_analysis._vol_file.header.num_bscans, scan_size[0]))
                continue

            # check if given number of a scans match with pre-defined number 
            if ms_analysis._vol_file.header.size_x != scan_size[1]:
                print("ID: %s has different number of ascans (%i) than expected (%i)" % (ut.get_id_by_file_path(data_list[vol_id]), ms_analysis._vol_file.header.size_x, scan_size[1]))
                continue  

            
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
                
                
                # initilized start indices to read and write data
                shift = min([d_ascan, 0])
                start_r = - shift + (c_ascan - (stackwidth//2) + shift) % stackwidth # start reading
                start_w = max([((c_ascan - (stackwidth//2)) // stackwidth) - (fovea_ascan - (stackwidth//2)) // stackwidth, 0])
                n_st = (scan_size[1] - start_r - max([d_ascan,0])) // stackwidth # possible number of stacks 
                
                
                # get rois
                raw_roi, seg_mask_roi = get_roi_masks(bscan, self.ref_layer, self.scan_size, seg_mask)
                
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


            tmp_excluded_dict = {}

            for idx, exclusion_type in zip(range(len(exclusion_dict)-1,-1,-1), exclusion_dict):
                tmp_excluded_dict[exclusion_type] = curr_excluded // 2**idx
                curr_excluded = curr_excluded % 2**idx
                         
            
            # create Map Objects containing the created maps
            current_map = RelEZI_map(
                "relEZI-Map",
                date.today(),
                self.scan_size,
                self.scan_field,
                self.stackwidth,
                data_list[vol_id],
                lat,
                (fovea_ascan, fovea_bscan), # (x,y)
                curr_ez_intensity,
                curr_elm_intensity,
                tmp_excluded_dict
                )            


            if vol_id in self.patients.keys():
                self.add(current_map, vol_id, ms_analysis._vol_file.header.visit_date)
            else:
                self.add(current_map, vol_id, ms_analysis._vol_file.header.visit_date, ms_analysis._vol_file.header.birthdate)

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

        for i, ids in enumerate(self.patients.keys()):
            for visit in self.patients[ids].visits: 
                for k, map in enumerate(visit.get_maps()): # if OD and OS, the sheet is extended to the right

                        # standard entries
                        worksheet.write(row, k * header_length, "313" + "".join(i for i in ids.split("-"))) # ID
                        worksheet.write_column(row, k * header_length + 1, nos * self.scan_size[0] * [map.laterality]) # Eye
                        worksheet.write_column(row, k * header_length + 2, b_scan_n) # bscan
                        worksheet.write(row, k * header_length + 3, visit.date_of_recording.strftime("%Y-%m-%d")) # Visit Date
                        worksheet.write_column(row, k * header_length + 4, a_scan_mesh) # A-scan
                        worksheet.write_column(row, k * header_length + 5, b_scan_mesh) # B-scan
                        worksheet.write_column(row, k * header_length + header_length -2, map.ezi_map.flatten())
                        worksheet.write_column(row, k * header_length + header_length -1, map.elmi_map.flatten())

                        # additional entries
                        for idx, ex_type in enumerate(map.excluded_maps.values()):
                            worksheet.write_column(row, k * header_length + 6 + idx, ex_type.flatten()) # exclusion type is added to the sheet


                row += nos * self.scan_size[0]

                if (i +1) % n == 0 and i < len(self.patients.keys()) -1:
                    workbook.close()
                    workbook = xls.Workbook(os.path.join(folder_path, self.project_name + "_" + str(int((i +1) / n)) + ".xlsx"), {'nan_inf_to_errors': True})
                    worksheet = workbook.add_worksheet()            
                    worksheet.write_row(0, 0, self.header)   
                    row = 1

        workbook.close()








