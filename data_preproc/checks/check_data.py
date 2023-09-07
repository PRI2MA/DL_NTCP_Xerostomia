"""
This script:
    - Save own consistent segmentation value map (config.structures_values) as json.
    - List all patient_ids in our main data folder (for which we have CT and RTDOSE), and compares this list to
        RTSTRUCTS_CITOR and RTSTRUCTS_DLC.
"""
import os
import time
import json
from tqdm import tqdm

import data_preproc_config as cfg
from data_preproc_functions import copy_folder, Logger, set_default, sort_human


def main_save_structures_values():
    # Save structures_values
    with open(os.path.join(cfg.save_root_dir, cfg.filename_structures_values_json), 'w') as f:
        json.dump(cfg.structures_values, f, default=set_default)


def main_citor_dlc():
    # Initialize variables
    save_root_dir = cfg.save_root_dir
    data_dir = cfg.data_dir
    data_dir_citor = cfg.data_dir_citor
    data_dir_dlc = cfg.data_dir_dlc
    save_dir_citor = cfg.save_dir_citor
    logger = Logger(os.path.join(save_root_dir, cfg.filename_check_data_logging_txt))
    start = time.time()

    # Get all files
    patients_list = os.listdir(data_dir)
    patients_citor_list = os.listdir(data_dir_citor)
    patients_dlc_list = os.listdir(data_dir_dlc)

    patients_list = sort_human(patients_list)
    patients_citor_list = sort_human(patients_citor_list)
    patients_dlc_list = sort_human(patients_dlc_list)

    # Compare the different patient lists to get an overview of available RTSTRUCTs
    in_citor_list, in_dlc_list, not_available_list = list(), list(), list()
    for patient_id in patients_list:
        logger.my_print('Patient_id: {id}'.format(id=patient_id))
        if patient_id in patients_citor_list:
            logger.my_print('\tIs in {}'.format(data_dir_citor))
            in_citor_list.append(patient_id)
        elif patient_id in patients_dlc_list:
            logger.my_print('\tIs in {}'.format(data_dir_dlc))
            in_dlc_list.append(patient_id)
        else:
            logger.my_print('\tNo RTSTRUCT')
            not_available_list.append(patient_id)

    with open(os.path.join(save_root_dir, cfg.filename_patients_in_citor_and_dlc_txt), 'w') as f:
        f.write('RTSTRUCT available in CITOR for patient_id:\n')
        for row in in_citor_list:
            f.write(str(row) + '\n')
        f.write('\nRTSTRUCT not available in CITOR, but available in DLC for patient_id:\n')
        for row in in_dlc_list:
            f.write(str(row) + '\n')
        f.write('\nRTSTRUCT not available for patient_id:\n')
        for row in not_available_list:
            f.write(str(row) + '\n')

    # Copy CITOR RTSTRUCTs to local folder
    logger.my_print('Copying CITOR RTSTRUCTs from {src} to local folder {dst}'.format(src=data_dir_citor,
                                                                                      dst=save_dir_citor))
    for patient_id in tqdm(in_citor_list):
        logger.my_print('Patient_id: {id}'.format(id=patient_id))
        src = os.path.join(data_dir_citor, patient_id)
        dst = os.path.join(save_dir_citor, patient_id)
        copy_folder(src=src, dst=dst)

    end = time.time()
    logger.my_print('Elapsed time: {time} seconds'.format(time=round(end - start, 3)))
    logger.my_print('DONE!')


def main():
    main_save_structures_values()
    main_citor_dlc()


if __name__ == '__main__':
    main()
