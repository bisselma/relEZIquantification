import rel_ez_intensity.utils as ut
from heyex_tools import vol_reader
from grade_ml_segmentation import macustar_segmentation_analysis
import cv2 
import numpy as np
import matplotlib.pyplot as plt
import os 

path_micro_ir = "X:\\Projects\\Macustar\\Microperimetrie\\mp_ir_images"
res = "X:\\Projects\\Macustar\\Microperimetrie\\Prework\\res\\"

ir_m, ir_s = ut.get_microperimetry_IR_image_list(path_micro_ir)
ir_m_keys = ir_m.keys()
ir_s_keys = ir_s.keys()

path_macustar = "Y:\\V1\\1_sorted"

# read in all folder 
dir_list = os.listdir(path_macustar)
vol_path = ""

for micro_ids in ir_m_keys:

    for name in dir_list:
        if micro_ids in name:
            vol_path_dir = path_macustar + "\\" + name
            folder_list = os.listdir(vol_path_dir)
            for item in folder_list:
                if item.endswith(".vol"):
                    vol_path = vol_path_dir + "\\" + item
                    break

    if len(vol_path) > 0:
        analysis_obj = macustar_segmentation_analysis.MacustarSegmentationAnalysis(
            vol_file_path=vol_path,
            cache_segmentation=True,
            use_gpu = False
            )
        vol_path = ""
    else:
        continue


    vol = analysis_obj.vol_file

    # get slo image 
    slo_img = vol.slo_image
    h_slo, w_slo = slo_img.shape
    
    


    micro_ir_m = cv2.imread(ir_m[micro_ids],0)
    if micro_ids in ir_s_keys:
        micro_ir_s = cv2.imread(ir_s[micro_ids],0)
    else:
        continue
    w_micro, h_micro = micro_ir_m.shape
    (cX, cY) = (w_micro // 2, h_micro // 2)


    
    M = cv2.getRotationMatrix2D((cX, cY),90, 1.0)
    img1_raw_m = cv2.warpAffine(micro_ir_m, M, (w_micro, h_micro))
    img1_raw_s = cv2.warpAffine(micro_ir_s, M, (w_micro, h_micro))
    
    # crop image to 30° x 30° around centered fovea 3.25° offset on each side
    offset = int((h_micro/36) * 3)
    img1_m = img1_raw_m[offset:-offset,offset:-offset]
    img1_m = cv2.resize(img1_m,(h_slo,w_slo))    
    img1_s = img1_raw_s[offset:-offset,offset:-offset]
    img1_s = cv2.resize(img1_s,(h_slo,w_slo))



    # calculate homographic transformation matrix A
    H_m = ut.get2DProjectiveTransformationMartix_by_SuperRetina(img1_m, slo_img)
    H_s = ut.et2DProjectiveTransformationMartix_by_SuperRetina(img1_s, slo_img)
    
    rows, cols = img1_m.shape
    img_output_m = cv2.warpPerspective(img1_m, H_m, (rows, cols), borderMode=cv2.BORDER_CONSTANT, borderValue=(0))
    img_output_s = cv2.warpPerspective(img1_s, H_s, (rows, cols), borderMode=cv2.BORDER_CONSTANT, borderValue=(0))

    fig, ax = plt.subplots(2,2)
    fig.set_size_inches((12,12))
    ax[0,0].imshow(img1_m, cmap="gray")
    ax[0,0].set_title("Microperimetry IR-Image",fontsize=15)
    ax[0,1].imshow(slo_img, cmap="gray")
    ax[0,1].set_title("Slo Image", color="white",fontsize=15)
    ax[1,0].imshow(img_output_m, cmap="gray")
    ax[1,0].set_title("After Transformation",fontsize=15)
    ax[1,1].imshow(slo_img, cmap="gray")
    ax[1,1].imshow(img_output_m, cmap="gray",alpha=0.4)
    ax[1,1].set_title("Overlay of Slo and Trasformated Image",fontsize=15)
    
    plt.savefig(res + micro_ids + "M.png")
    #plt.close(fig)

    fig, ax = plt.subplots(2,2)
    fig.set_size_inches((12,12))
    ax[0,0].imshow(img1_s, cmap="gray")
    ax[0,0].set_title("Microperimetry IR-Image",fontsize=15)
    ax[0,1].imshow(slo_img, cmap="gray")
    ax[0,1].set_title("Slo Image", color="white",fontsize=15)
    ax[1,0].imshow(img_output_s, cmap="gray")
    ax[1,0].set_title("After Transformation",fontsize=15)
    ax[1,1].imshow(slo_img, cmap="gray")
    ax[1,1].imshow(img_output_s, cmap="gray",alpha=0.4)
    ax[1,1].set_title("Overlay of Slo and Trasformated Image",fontsize=15)
    
    plt.savefig(res + micro_ids + "S.png")
    #plt.close(fig)
    
    


            
            


