"""
This contains checks/prints for data_preproc_ct_rtdose.py:
    - Read JSON files and check/print content.
    - For all patients: load ct.npy and rtdose.npy, and create descriptive statistics of their values.
"""
import os
import time
import json
import numpy as np
import pandas as pd
from tqdm import tqdm

import data_preproc_config as cfg
from data_preproc_functions import Logger, sort_human


def main():
    # Initialize variables
    use_umcg = cfg.use_umcg
    mode_list = cfg.mode_list
    data_dir_mode_list = cfg.save_dir_mode_list
    patient_id_length = cfg.patient_id_length

    save_root_dir = cfg.save_root_dir
    save_dir = cfg.save_dir_ct
    ct_json_filename = cfg.filename_ct_metadata_json
    rtdose_json_filename = cfg.filename_rtdose_metadata_json
    logger = Logger(os.path.join(save_root_dir, cfg.filename_check_data_preproc_ct_rtdose_logging_txt))
    nr_of_decimals = cfg.nr_of_decimals
    start = time.time()

    for _, d_i in zip(mode_list, data_dir_mode_list):
        data_dir = cfg.data_dir.format(d_i)

        # TODO: currently redundant
        # Load file from data_folder_types.py
        # TODO: temporary, for MDACC
        if use_umcg:
            df_data_folder_types = pd.read_csv(os.path.join(save_root_dir, cfg.filename_patient_folder_types_csv), sep=';',
                                               index_col=0)
            df_data_folder_types.index = ['%0.{}d'.format(patient_id_length) % int(x)
                                          for x in df_data_folder_types.index.values]

        patients_list = os.listdir(data_dir)
        patients_list = sort_human(patients_list)
        n_patients = len(patients_list)
        logger.my_print('Total number of patients: {n}'.format(n=n_patients))

        # Check JSON metadata
        ct_metadata_df = pd.DataFrame()
        ct_df = pd.DataFrame()
        rtdose_df = pd.DataFrame()

        # TODO: temporary: consider all patients, for exclude_patients.csv
        # Testing for a small number of patients
        if cfg.test_patients_list is not None:
            test_patients_list = cfg.test_patients_list
            patients_list = [x for x in test_patients_list if x in patients_list]

        for patient_id in tqdm(patients_list):
            logger.my_print('Patient_id: {id}'.format(id=patient_id))

            # Determine folder type of interest ('with_contrast'/'no_contrast')
            # TODO: temporary, for MDACC
            if use_umcg:
                folder_type_i = df_data_folder_types.loc[patient_id]['Folder']
            else:
                folder_type_i = ''

            patient_dir = os.path.join(save_dir, patient_id)

            # Load CT JSON
            ct_json_file = open(os.path.join(patient_dir, ct_json_filename))
            ct_metadata_dict = json.load(ct_json_file)

            # Convert metadata to DataFrame
            ct_metadata_df_i = pd.DataFrame(list(ct_metadata_dict.values())).T
            ct_metadata_df_i.columns = ct_metadata_dict.keys()
            ct_metadata_df = pd.concat([ct_metadata_df, ct_metadata_df_i])

            # Close JSON
            ct_json_file.close()

            # Load RTDOSE JSON
            rtdose_json_file = open(os.path.join(patient_dir, rtdose_json_filename))
            rtdose_metadata_dict = json.load(rtdose_json_file)

            # Get Path_rtdose and Path_ct
            logger.my_print('\tPath_rtdose: {f}'.format(f=rtdose_metadata_dict['Path_rtdose']))
            logger.my_print('\tPath_ct: {f}'.format(f=rtdose_metadata_dict['Path_ct']))

            # Close JSON
            rtdose_json_file.close()

            # Load CT
            ct = np.load(os.path.join(save_dir, patient_id, cfg.filename_ct_npy))
            ct_df_i = pd.DataFrame([np.min(ct), np.mean(ct), np.median(ct),
                                    np.percentile(ct, 90), np.max(ct)],
                                    index=['min', 'mean', 'median', '90th_perc', 'max'],
                                    columns=[patient_id]).T
            ct_df = pd.concat([ct_df, ct_df_i], axis=0)

            # Load RTDOSE
            rtdose = np.load(os.path.join(save_dir, patient_id, cfg.filename_rtdose_npy))
            rtdose_df_i = pd.DataFrame([np.min(rtdose), np.mean(rtdose), np.median(rtdose),
                                        np.percentile(rtdose, 90), np.max(rtdose)],
                                       index=['min', 'mean', 'median', '90th_perc', 'max'],
                                       columns=[patient_id]).T
            rtdose_df = pd.concat([rtdose_df, rtdose_df_i], axis=0)

        ct_df = ct_df.round(nr_of_decimals)
        rtdose_df = rtdose_df.round(nr_of_decimals)

        # Save df to csv file
        ct_metadata_df.to_csv(os.path.join(save_root_dir, cfg.filename_overview_ct_metadata_csv), sep=';')
        ct_df.to_csv(os.path.join(save_root_dir, cfg.filename_overview_ct_csv), sep=';')
        rtdose_df.to_csv(os.path.join(save_root_dir, cfg.filename_overview_rtdose_csv), sep=';')

    end = time.time()
    logger.my_print('Elapsed time: {time} seconds'.format(time=round(end - start, 3)))
    logger.my_print('DONE!')


if __name__ == '__main__':
    main()



