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

from relEZIquantification.getAdjacencyMatrix import plot_layers
from relEZIquantification.seg_core import get_retinal_layers
from relEZIquantification import utils as ut

from relEZIquantification.relEZIquantification_projects import *

class RelEZIQuantification:

    project_name = ""

    _rel_EZI_data = None # return of calculate_relEZI_maps() is stored here

    def __init__(self, project_name: Optional[str] = None):
        self.project_name = project_name

        if self.project_name == "macustar":
            self._rel_EZI_data = RelEZIQuantificationMacustar()
        elif self._project_name == "micro":
                self._rel_EZI_data = RelEZIQuantificationMicro()
        elif self.project_name == "mactel":
                self._rel_EZI_data = RelEZIQuantificationMactel()
        else:
            raise ValueError("The project name is no correct or not yet implemented\nThe existing project names are:\nmacustar\nmicro\nmactel")
    
    
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
        self._rel_EZI_data.ssd_maps = SSDmap.create_ssd_maps(self._rel_EZI_data.get_list(data_folder), *args)

    def save_ssd(self, *args):
        self._rel_EZI_data.ssd_maps.save_ssd(*args)

    def load_ssd(self, *args):
        self._rel_EZI_data.ssd_maps(SSDmap.load_ssd(*args))


    # managing mean_rpedc maps 
    def create_mean_rpedc_map(self, *args):
        self._rel_EZI_data.mean_rpedc_map = Mean_rpedc_map.create_mean_rpedc_map(*args)

    def save_mean_rpedc_map(self, *args):
        self._rel_EZI_data.mean_rpedc_map.save_mean_rpedc_map(*args)

    def load_mean_rpedc_map(self, *args):
        self._rel_EZI_data.mean_rpedc_map(Mean_rpedc_map.load_mean_rpedc_map(*args))


    # analyse project data to get relEZI_maps
    def create_relEZI_maps(self, *args):
        self._rel_EZI_data.create_relEZI_maps(*args)

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

