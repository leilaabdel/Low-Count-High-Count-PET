from codecs import ascii_encode
import glob
import shutil
import os
from tqdm import tqdm
from modules import transforms
import subprocess
import pickle


def move_reconstructed_files(starting_folder_path, final_data_path, recon_trial_name, ground_truth_path=None, recon_alg = "OP"):
    all_patient_paths = glob.glob(f"{starting_folder_path}")

    if ground_truth_path != None:
       
        # Check if there is a known reconstruction
        for path in tqdm(all_patient_paths):
            pat_id = path.split("/")[-2]


            dcm_sub_files = glob.glob(f"{path}/**/**/{recon_trial_name}**{recon_alg}**/")

            # Source the reg_resample tool
            subprocess.call("source /autofs/space/celer_001/users/software_2/build_sirf/INSTALL/bin/env_sirf.sh", shell=True)

            if len(dcm_sub_files) > 0:
                print("*** MOVING FILES ***")

                for dcm_path in dcm_sub_files:
                    low_count_old_path = f"{path}/{recon_trial_name}_{recon_alg}.nii.gz"
                    transforms.convert_dcm_to_nii(dcm_path, low_count_old_path)
                
                # Move the PET files

                ## Only move those files that have nifty reconstructions
                pet_destination = f"{final_data_path}/{pat_id}/pet_nifti/"
                mri_destination = f"{final_data_path}/{pat_id}/mr_nifti/"
                
                os.makedirs(pet_destination, exist_ok=True)
                os.makedirs(mri_destination, exist_ok=True)

                ## Copy the reconstructed PETs 
                low_count_input_path = os.path.join(pet_destination, os.path.basename(low_count_old_path))
                shutil.copyfile(low_count_old_path, low_count_input_path)

                ## Copy the GT PET
                gt_pet_path = f"{ground_truth_path}/{pat_id}/PET/PET_60-90_SUV.nii.gz" 
                gt_pet_input_path = os.path.join(pet_destination, "gt_recon.nii.gz")
                shutil.copyfile(gt_pet_path, gt_pet_input_path)

                ## Copy the T1 mask files
                t1_mask_source_path = f"{ground_truth_path}/{pat_id}/PET/SUVR_2mm-processing/aparc+aseg_BIN.nii.gz"
                t1_mask_input_path = os.path.join(mri_destination, "gt_t1_head_mask.nii.gz")
                shutil.copyfile(t1_mask_source_path, t1_mask_input_path)

                ## Copy the T1 image files 
                t1_source_path = f"{ground_truth_path}/{pat_id}/PET/SUVR_2mm-processing/T1.nii.gz"
                t1_original_input_path = os.path.join(mri_destination, "t1_original.nii.gz")
                shutil.copyfile(t1_source_path, t1_original_input_path)

                ## Register the T1 Mask and Original T1 to the low count PET Img 
                t1_mask_registered_path =  os.path.join(mri_destination, "t1_mask_registered.nii.gz")
                t1_original_registered_path = os.path.join(mri_destination, "t1_img_registered.nii.gz")
                
                transforms.register_imgs(static_img_path=low_count_input_path, 
                    moving_img_path=t1_mask_input_path, 
                    registered_output_path=t1_mask_registered_path)
                transforms.register_imgs(static_img_path=low_count_input_path, 
                    moving_img_path=t1_original_input_path, 
                    registered_output_path=t1_original_registered_path)   


                