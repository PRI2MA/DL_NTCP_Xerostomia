"""
Determine which folder type (e.g. 'with_contrast' and/or 'no_contrast') should be considered:
    - For each patient_id:
        - If 'with_contrast' is available: select 'with_contrast'.
        - Else: select 'no_contrast'.
"""
import os
import time
import pandas as pd
from tqdm import tqdm

import data_preproc_config as cfg
from data_preproc_functions import Logger, sort_human


def main():
    # Initialize variables
    mode_list = cfg.mode_list
    data_dir_mode_list = cfg.save_dir_mode_list
    patient_id_length = cfg.patient_id_length

    save_root_dir = cfg.save_root_dir
    patient_folder_types = cfg.patient_folder_types
    logger = Logger(output_filename=None)
    start = time.time()

    logger.my_print('Patient folder types (highest priority first): {}'.format(patient_folder_types))
    for _, d_i in zip(mode_list, data_dir_mode_list):
        data_dir = cfg.data_dir.format(d_i)

        # Get all patient_ids
        patients_list = os.listdir(data_dir)
        patients_list = sort_human(patients_list)
        n_patients = len(patients_list)
        logger.my_print('Total number of patients: {n}'.format(n=n_patients))

        # TODO: temporary: consider all patients, for exclude_patients.csv
        # Testing for a small number of patients
        if cfg.test_patients_list is not None:
            test_patients_list = cfg.test_patients_list
            patients_list = [x for x in test_patients_list if x in patients_list]

        df = pd.DataFrame(columns=['Patient_id', 'Folder', 'Comments'])
        for patient_id in tqdm(patients_list):
            logger.my_print('Patient_id: {}'.format(patient_id))

            # Get all folders
            all_folders_i = os.listdir(os.path.join(data_dir, patient_id))

            for f_t in patient_folder_types:
                if f_t in all_folders_i:
                    folder_type_i = f_t
                    break
                else:
                    folder_type_i = None

            if len(all_folders_i) > 1:
                logger.my_print('\tAll folders: {}'.format(all_folders_i))
                logger.my_print('\tFolder type: {}'.format(folder_type_i))

            df_i = pd.DataFrame({'Patient_id': patient_id, 'Folder': folder_type_i}, index=[0])
            df = pd.concat([df, df_i], axis=0)

        # Save to csv
        df = df.sort_values(by=['Patient_id', 'Folder'])
        df.to_csv(os.path.join(save_root_dir, cfg.filename_patient_folder_types_csv), sep=';', index=False)

    end = time.time()
    logger.my_print('Elapsed time: {time} seconds'.format(time=round(end - start, 3)))
    logger.my_print('DONE!')


if __name__ == '__main__':
    main()




