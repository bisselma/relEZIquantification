# -*- coding: utf-8 -*- 
from pathlib import Path
from timeit import repeat
from typing import Callable, Dict, List, Optional, Union, IO
from unicodedata import name
from weakref import ref
import numpy as np
import utils as ut
import eyepy as ep
import matplotlib.pyplot as plt
from PIL import Image
from seg_core import get_retinal_layers
from getAdjacencyMatrix import plot_layers


class OCTMap:

    def __init__(
            self,
            name: str,
            date_of_origin: Optional[str] = None,
            scan_size: Optional[tuple] = None,
            stackwidth: Optional[int] = None,
            laterality: Optional[str] = None,
            octmap: Optional[np.ndarray] = None,
            ) -> None:
        self.name = name
        self.date_of_origin = date_of_origin
        self.scan_size = scan_size
        self.stackwidth = stackwidth
        self.laterality = laterality
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
            fovea_coords: Optional[Dict] = None,
            ez_distance_map: Optional[List[OCTMap]] = None, # [mean distance, standard diviation]
            elm_distance_map: Optional[List[OCTMap]] = None, # [mean distance, standard diviation]
            scan_size: Optional[tuple] = None,
            stackwidth: Optional[int] = None,
            patients: Optional[List[Patient]] = None
            ) -> None:
        self.fovea_coords = fovea_coords
        self.ez_distance_map = ez_distance_map
        self.elm_distance_map = elm_distance_map
        self.scan_size = scan_size
        self.stackwidth = stackwidth
        self.patients = patients

    """
    standard methods
    #
    #
    #
    #
    #
    #
    """
    def create_intensity_profile(self, stack, layer) -> Optional[np.ndarray]:
        
        sum = np.zeros((200,))
        for col, l in zip(stack.T, 496 - np.round(layer).astype(int)):
                a_scan = col[l[0] - 100: l[0] + 100]
                if not any(a_scan > 468):
                    sum += a_scan
        
        return sum / self.stackwidth


    def create_ssd_maps(
        self,
        folder_path: Union[str, Path, IO] = None,
        fovea_coords: Optional[Dict] = None,
        scan_size: Optional[tuple] = None,
        stackwidth: Optional[int] = None,
        ref_layer: Optional[str] = None,
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
            ref_layer (Optional[str): layer to flatten the image 
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


        # data directories
        if args:
            data_dict = ut.get_list_by_format(folder_path, args)
        else:
            raise ValueError("no file format given")

        
        # central bscan/ascan, number of stacks (nos)
        c_bscan = scan_size[0] // 2 + scan_size[0] % 2
        c_ascan = scan_size[1] // 2 + scan_size[1] % 2
        nos = scan_size[1] // stackwidth # number of stacks

        ez_distance = np.empty(shape=[scan_size[0], nos, 0])
        elm_distance = np.empty_like(ez_distance)

        
        # iterate  over .vol-list
        for vol_path in data_dict[".vol"]:

            # current distance map
            curr_ez_distance = np.zeros((scan_size[0], nos))
            curr_elm_distance = np.zeros_like(curr_ez_distance)
            
            # get vol data from file
            vol_data = ep.Oct.from_heyex_vol(vol_path)

            # check if given number of b scans match with pre-defined number 
            if vol_data._meta["NumBScans"] != scan_size[0]:
                print("ID: %s has different number of bscans (%i) than expected (%i)" % (ut.get_id_by_file_path(vol_path), vol_data._meta["NumBScans"], scan_size[0]))
                continue

            # d_bscan (int): delta_bscan = [central bscan (number of bscans // 2)] - [current bscan]
            fovea_bscan, fovea_ascan = fovea_coords[ut.get_id_by_file_path(vol_path)]
            
            # change orientation from top down, subtract on from coords to keep 0-indexing of python            
            fovea_bscan = scan_size[0] - (fovea_bscan -1) 

            # laterality 
            lat = vol_data._meta["ScanPosition"]

            if lat == "OS": # if left eye is processed
                fovea_ascan = scan_size[1] - (fovea_ascan -1)
            else:
                fovea_ascan = fovea_ascan -1

            d_bscan  = c_bscan - fovea_bscan


            for bscan, ez, elm in zip(
                vol_data[::-1][max([-d_bscan, 0]): scan_size[0] + min([-d_bscan, 0])], # read
                curr_ez_distance[max([d_bscan, 0]): scan_size[0] + min([d_bscan, 0]), :], # write
                curr_elm_distance[max([d_bscan, 0]): scan_size[0] + min([d_bscan, 0]), :] # write
                ):

                bscan_data = bscan._scan_raw
                
                
                if lat == "OS":
                    np.flip(bscan_data)

                # layer to caculate distance. E.g. RPE (default) or BM  !!!!!!!!!!! TO BE ADDED
                
                layer = bscan.layers[ref_layer].astype(np.uint16)

                # get start position to read data
                d_ascan = c_ascan - fovea_ascan
                shift = min([d_ascan, 0])
                start = - shift + (fovea_ascan - (stackwidth//2) + shift) % stackwidth

                n_st = (scan_size[1] - start - max([d_ascan,0])) // stackwidth # possible number of stacks 
                
                
                # create region of interest image 
                roi = np.zeros((50,scan_size[1])).astype(np.float32)
                
                for i, l in enumerate(layer):
                    if l < 496 and l > 0:
                        roi[:,i] = bscan_data[l-40:l+10,i]
                        
            
                # get ez and rpe boundary 
                imglayers = get_retinal_layers(roi) 
                
                
                for i in range(n_st):
                    
                    if not any(np.isnan(layer[start + i * stackwidth: start + (i + 1) * stackwidth])):
                        i_profile = self.create_intensity_profile(
                            bscan_data[:,start + i * stackwidth: start + (i + 1) * stackwidth],
                            layer[start + i * stackwidth: start + (i + 1) * stackwidth][:,None]
                            )

        


                        

if __name__ == '__main__':
    fovea_coords = {"017-0064": (123,768//2 +1)}
    path = "E:\\benis\\Documents\\Arbeit\\Arbeit\\Augenklinik\\GitLab\\test_data\\vols"
    
    data = RelEZIntensity()
    data.create_ssd_maps(path, fovea_coords, (241, 768), 10, None, ".vol")
    








        

        


        







