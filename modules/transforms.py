from fcntl import F_SETFL
from scipy import ndimage
import nibabel as nib
import numpy as np
import SimpleITK as sitk
import os
import subprocess



def convert_dcm_to_nii(dicom_series_path, nii_outpath):
    reader = sitk.ImageSeriesReader()

    dicom_names = reader.GetGDCMSeriesFileNames(dicom_series_path)
    reader.SetFileNames(dicom_names)

    image = reader.Execute()

    size = image.GetSize()
    
    image = sitk.Flip(image, [False, True, False])

    print("Converted .dcm series to .nii file" + "\nImage size:", size[0], size[1], size[2])

    sitk.WriteImage(image, nii_outpath)


def register_imgs(static_img_path, moving_img_path, registered_output_path):
    subprocess.run(["bash",  "../modules/reg_resample.sh",  static_img_path, moving_img_path, registered_output_path], shell=False)
   