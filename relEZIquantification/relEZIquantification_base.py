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

from relEZIquantification.getAdjacencyMatrix import plot_layers
from relEZIquantification.seg_core import get_retinal_layers
from relEZIquantification import utils as ut

from relEZIquantification.relEZIquantification_projects import *

class RelEZIQuantification:

    project_name = ""

    file_location = None

    _rel_EZI_data = None # return of calculate_relEZI_maps() is stored here

    def __init__(self, project_name: Optional[str] = None):
        self.project_name = project_name

        if self.project_name == "macustar":
            self._rel_EZI_data = RelEZIQuantificationMacustar(self.project_name)
        elif self.project_name == "micro":
                self._rel_EZI_data = RelEZIQuantificationMicro(self.project_name)
        elif self.project_name == "mactel":
                self._rel_EZI_data = RelEZIQuantificationMactel(self.project_name)
        elif self.project_name == "mactel2":
                self._rel_EZI_data = RelEZIQuantificationMactel2(self.project_name)
        else:
            raise ValueError("The project name is no correct or not yet implemented\nThe existing project names are:\nmacustar\nmicro\nmactel")
    

    # get fovea coords by excel file of shape (first_column = "Patient ID", second column = "Fovea B-Scan", third column = "Fovea A-Achse")
    @staticmethod
    def get_fovea_coords(path):
        df = pd.read_excel(path)
        fovea_coords = {}
        for ids, bscan, ascan in zip(df["Patient ID"],df["Fovea B-Scan"],df["Fovea A-Achse"]):
            fovea_coords[str(ids)] = (bscan, ascan)
        
        return fovea_coords

    # managing ssd maps 
    def create_ssd_maps(self, data_folder, *args):
        """
        Args:
            data_folder (Union[str, Path, IO]): folder path where files are stored
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
        self._rel_EZI_data.ssd_maps = SSDmap.create_ssd_maps(self.project_name, self._rel_EZI_data.get_list(data_folder), *args)

    def save_ssd(self, *args):
        self._rel_EZI_data.ssd_maps.save_ssd(*args)

    def load_ssd(self, *args):
        self._rel_EZI_data.ssd_maps = SSDmap.load_ssd(*args)


    # managing mean_rpedc maps 
    def create_mean_rpedc_map(self, data_folder, *args):
        self._rel_EZI_data.mean_rpedc_map = Mean_rpedc_map.create_mean_rpedc_map(self.project_name, self._rel_EZI_data.get_list(data_folder), data_folder, *args)

    def save_mean_rpedc_map(self, *args):
        self._rel_EZI_data.mean_rpedc_map.save_mean_rpedc_map(*args)

    def load_mean_rpedc_map(self, *args):
        self._rel_EZI_data.mean_rpedc_map = Mean_rpedc_map.load_mean_rpedc_map(*args)


    # analyse project data to get relEZI_maps
    def create_relEZI_maps(self, *args):
        self._rel_EZI_data.create_relEZI_maps(*args)


    def save_relEZI_maps(
            self,
            directory: Union[str, Path, IO, None] = None
            ):
        
        """
        Args:
            directory (Union[str, Path, IO, None]): directory where RelEZIQuantification object should be stored        
        """
        
        if not self._rel_EZI_data:
            ValueError("RelEZIQuantification object no exist. First use create_relEZI_maps()")


        if not directory:
            directory = ""
            
        relEZI_maps_file_path = os.path.join(
                directory, "relEZI_maps_" + self.project_name + "_" +
                            date.today().strftime("%Y-%m-%d") 
                            + ".pkl")
        
        with open(relEZI_maps_file_path, "wb") as outp:
            pickle.dump(self, outp, pickle.HIGHEST_PROTOCOL)
            
        self.file_location = directory


    def load_relEZI_maps(
            self,
            directory: Union[str, Path, IO, None] = None,
            filename: Optional[str] = None
            ):
        
        if not filename:
            filename = "relEZI_maps"

        if self.file_location:
            directory = self.file_location
            
        if directory:
            obj_list = ut.get_list_by_format(directory, [".pkl"])
            for file in obj_list[".pkl"]:
                if filename in file:
                    with open(file, 'rb') as inp:
                        tmp_obj = pickle.load(inp)
                        if type(tmp_obj) is RelEZIQuantification:
                            return tmp_obj # Initialization based on the loaded instance         
        else:
            raise ValueError("No directory to load relEZI_maps is given\n Try to create relEZI_maps first by method 'create_relEZI_maps()' and than save it by save_relEZI_maps()")
    


    @property
    def relEZI_maps(self):
        """The relEZI_maps property."""
        return self._rel_EZI_data

    @relEZI_maps.setter
    def relEZI_maps(self, value):
        self._rel_EZI_data = value

    @relEZI_maps.deleter
    def relEZI_maps(self):
        del self._rel_EZI_data


    # microperimetry grid
    def create_micro_grid_field_data(self, *args):
        self._rel_EZI_data.create_micro_grid_field_data(*args)

    def show_micro_grid_field_data(self, *args):
        self._rel_EZI_data.show_micro_grid_field_data(*args)
        

    # create excel sheets
    def create_excel_sheets(self, *args):
        self._rel_EZI_data.create_excel_sheets(*args)

if __name__ == "__main__":
    pass