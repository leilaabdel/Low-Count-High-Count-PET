from scipy import ndimage
import nibabel as nib
import numpy as np
import SimpleITK as sitk
import os
from pathlib import Path



def convert_dcm_to_nii(dicom_series_path, nii_outpath):
    reader = sitk.ImageSeriesReader()

    dicom_names = reader.GetGDCMSeriesFileNames(dicom_series_path)
    reader.SetFileNames(dicom_names)

    image = reader.Execute()

    size = image.GetSize()
    
    image = sitk.Flip(image, [False, True, False])

    print("Converted .dcm series to .nii file" + "\nImage size:", size[0], size[1], size[2])

    sitk.WriteImage(image, nii_outpath)
