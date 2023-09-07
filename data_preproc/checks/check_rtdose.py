"""
Description:
    - For each patient:
        - For each RTSTRUCT file:
            - For each structure:
                - If it is relevant: save voxel counts
                - If the voxel count is larger than current voxel count (of previous RTSTRUCT files), then replace
                    the lower value by this larger value.

Note: RTSTRUCTs are not resampled to CT here, so the voxel counts in the data preprocessing step could differ. However,
    the initial count from this file could give us an indication of the existence of
"""
import os
import time
import pandas as pd
import SimpleITK as sitk
import data_preproc_config as cfg

from tqdm import tqdm
from glob import glob
from data_preproc_functions import get_all_folders, Logger, sort_human
from data_preproc_ct_rtdose import load_rtdose_file


def main():
    """
    If there are multiple RTDOSE files, then make sure that their contents are identical, because then we can use any
    of them.

    Returns:

    """
    # Initialize variables
    # From data collection
    mode_list = cfg.mode_list
    data_dir_mode_list = cfg.save_dir_mode_list
    patient_id_length = cfg.patient_id_length

    save_root_dir = cfg.save_root_dir
    logger = Logger()
    start = time.time()

    mode_list = [mode_list[0]]
    data_dir_mode_list = [data_dir_mode_list[0]]
    for m_i, d_i in zip(mode_list, data_dir_mode_list):
        data_dir = cfg.data_dir

        # Get all patient_id
        patients_list = os.listdir(data_dir)
        patients_list = sort_human(patients_list)

        df = pd.DataFrame(columns=['Patient_id', 'Comments'])
        for patient_id in tqdm(patients_list):
            logger.my_print('Patient_id: {}'.format(patient_id))
            # List all folders of the patient
            all_folders = get_all_folders(os.path.join(data_dir, patient_id))
            # Get relevant path
            path_rtdose = [x for x in all_folders if x.endswith('\\RTDOSE')]

            # Initialize variables
            contains_different_files = False
            if len(path_rtdose) == 1:
                path_rtdose = path_rtdose[0]
                # Get all files in path_rtdose
                files_rtdose = os.listdir(path_rtdose)

                arr_rtdose_0 = None
                arr_rtdose_i = None

                for f in files_rtdose:
                    if arr_rtdose_0 is None:
                        arr_rtdose_0 = sitk.GetArrayFromImage(load_rtdose_file(os.path.join(path_rtdose, f)))
                    else:
                        arr_rtdose_i = sitk.GetArrayFromImage(load_rtdose_file(os.path.join(path_rtdose, f)))

                    # Make sure that arr_rtdose_0 and arr_rtdose_i are identical
                    if (arr_rtdose_0 is not None) and (arr_rtdose_i is not None):
                        try:
                            # Use Try because we may get "AttributeError: 'bool' object has no attribute 'all'"
                            if not (arr_rtdose_0 == arr_rtdose_i).all():
                                contains_different_files = True
                        except:
                            logger.my_print('The RTDOSE array is invalid.', level='warning')

                if contains_different_files:
                    logger.my_print('The RTDOSE files are different.'.format(patient_id), level='warning')
                elif (len(files_rtdose) > 1) and (not contains_different_files):
                    logger.my_print('There are multiple RTDOSE files, but they are all the same.'.format(patient_id))
            else:
                logger.my_print('There are multiple RTDOSE folders.'.format(patient_id), level='warning')

    end = time.time()
    logger.my_print('Elapsed time: {} seconds'.format(round(end - start, 3)))
    logger.my_print('DONE!')


if __name__ == '__main__':
    main()



