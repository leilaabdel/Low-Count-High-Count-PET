import glob
import shutil
import os
from tqdm import tqdm

def move_reconstructed_files(starting_folder_path, final_data_path, ground_truth_path=None, recon_alg = "OP"):
    all_patient_paths = glob.glob(f"{starting_folder_path}")

    recon_with_gt = None

    if ground_truth_path != None:
       
        # Check if there is a known reconstruction
        for recon in tqdm(all_patient_paths):
            pat_id = recon.split("/")[-2]
            sub_files = glob.glob(f"{recon}/{recon_alg}*.nii")
            if len(sub_files) > 0:

                # Move the PET files

                ## Only move those files that have nifty reconstructions
                pet_destination = f"{final_data_path}/{pat_id}/pet_nifti/"
                mri_destination = f"{final_data_path}/{pat_id}/mr_nifti/"
                
                os.makedirs(pet_destination, exist_ok=True)
                os.makedirs(mri_destination, exist_ok=True)

                ## Copy the reconstructed PETs 
                [shutil.copyfile(file, os.path.join(pet_destination, os.path.basename(file))) for file in sub_files]

                ## Copy the GT PET
                gt_pet_path = f"{ground_truth_path}/{pat_id}/PET/PET_60-90_SUV.nii.gz" 
                shutil.copyfile(gt_pet_path, os.path.join(pet_destination, "gt_recon.nii.gz"))

                ## Copy the T1 mask files
                mask_file_path = f"{ground_truth_path}/{pat_id}/PET/SUVR_2mm-processing/aparc+aseg_BIN.nii.gz"
                shutil.copyfile(mask_file_path, os.path.join(mri_destination, "gt_t1_head_mask.nii.gz"))

                ## Copy the T1 image files 
                t1_file_path = f"{ground_truth_path}/{pat_id}/PET/SUVR_2mm-processing/T1.nii.gz"
                shutil.copyfile(t1_file_path, os.path.join(mri_destination, "t1_original.nii.gz"))

            