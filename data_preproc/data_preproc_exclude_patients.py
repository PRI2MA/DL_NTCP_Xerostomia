"""
This script:
    - Load overview_rtdose.csv and check which patients have invalid dose values
    - Load segmentation_fraction_outside_ct.csv and check which patients have too large segmentation fraction outside CT
    - Load endpoints.csv and check which patients have no endpoint (i.e. for which df_endpoints[endpoint].isna()).

We do not check the UID match between CT and RTSTRUCT, because those (should) have DLC segmentation
"""
import os
import time
import numpy as np
import pandas as pd
import data_preproc_config as cfg
from data_preproc_functions import Logger


def main():
    """
    Tip: disable the usage of cfg.test_patients_list in the following files to make sure that exclude_patients.csv
        will be created for all patients:
        - data_folder_types.py (function: main())
        - data_collection.py (function: main_match_ct_rtdose_rtstruct())
        - check_data_preproc_ct_rtdose.py (function: main())
        - check_data_preproc.py (function: main_segmentation_outside_ct())

    Search/Add `# TODO: temporary: consider all patients, for exclude_patients.csv` in those files above.

    Returns:

    """
    # Initialize variables
    use_umcg = cfg.use_umcg
    mode_list = cfg.mode_list
    data_dir_mode_list = cfg.save_dir_mode_list
    patient_id_length = cfg.patient_id_length

    save_root_dir = cfg.save_root_dir
    logger = Logger(output_filename=None)
    rtdose_lower_limit = cfg.rtdose_lower_limit
    rtdose_upper_limit = cfg.rtdose_upper_limit
    max_outside_fraction = cfg.max_outside_fraction
    start = time.time()

    for _, d_i in zip(mode_list, data_dir_mode_list):
        # TODO: currently redundant
        # Load file from data_folder_types.py (data_folder_types.py)
        # TODO: temporary, for MDACC
        if use_umcg:
            df_data_folder_types = pd.read_csv(os.path.join(save_root_dir, cfg.filename_patient_folder_types_csv), sep=';',
                                               index_col=0)
            # df_data_folder_types.index = ['%0.{}d'.format(patient_id_length) % int(x)
            #                               for x in df_data_folder_types.index.values]

        # Check which patients have no match between CT UID and RTDOSE UID (data_collection.py)
        # Note: We do not consider patients without matching UID CT and RTSTRUCT, because those (should) have DLC
        # segmentation
        df_invalid_uid = pd.read_csv(os.path.join(save_root_dir, cfg.filename_invalid_uid_csv), sep=';', index_col=0)
        if len(df_invalid_uid) > 0:
            df_invalid_uid = df_invalid_uid[df_invalid_uid['No Frame of Reference UID match CT RTDOSE']][
                'No Frame of Reference UID match CT RTDOSE']

        # Check which patients have invalid dose values (check_data_preproc_ct_rtdose.py)
        rtdose = pd.read_csv(os.path.join(save_root_dir, cfg.filename_overview_rtdose_csv), sep=';', index_col=0)
        # Check which patients have too large segmentation fraction outside CT (check_data_preproc.py)
        df_fraction_outside = pd.read_csv(os.path.join(save_root_dir, cfg.filename_segmentation_fraction_outside_ct_csv),
                                          sep=';', index_col=0)

        logger.my_print('RTDOSE in [{}, {}] allowed'.format(rtdose_lower_limit, rtdose_upper_limit))
        logger.my_print('Maximum segmentation fraction outside CT allowed: {}'.format(max_outside_fraction))

        # List of patients
        df_endpoints = pd.read_csv(os.path.join(save_root_dir, cfg.filename_endpoints_csv), sep=';')
        patients_list = df_endpoints[cfg.patient_id_col]

        df = pd.DataFrame(columns=['Patient_id'])
        for patient_id in patients_list:
            # Determine folder type of interest ('with_contrast'/'no_contrast')
            # TODO: temporary, for MDACC
            if use_umcg:
                folder_type_i = df_data_folder_types.loc[patient_id]['Folder']
            else:
                folder_type_i = ''

            invalid_dict_i = dict()

            # Too low or too large max RTDOSE value
            try:
                rtdose_i = rtdose.loc[[patient_id]]
                if not (rtdose_lower_limit < rtdose_i['max'].values < rtdose_upper_limit):
                    invalid_dict_i['Patient_id'] = patient_id
                    invalid_dict_i['Max RTDOSE'] = rtdose_i['max']
            except:
                pass

            # Segmentation fraction outside CT
            fraction_outside_i = df_fraction_outside.loc[[patient_id]]
            for structure in cfg.parotis_structures + cfg.submand_structures:
                if fraction_outside_i[structure].values > max_outside_fraction:
                    invalid_dict_i['Patient_id'] = patient_id
                    invalid_dict_i['Fraction {} outside CT'.format(structure)] = fraction_outside_i[structure]

            # Add patients to be excluded and the reason
            if len(invalid_dict_i) > 0:
                df_i = pd.DataFrame(invalid_dict_i, index=[patient_id])
                df = pd.concat([df, df_i], axis=0)

        # Set column = 'patient_id' as index
        df = df.set_index(df.iloc[:, 0].values)
        del df['Patient_id']

        # Add patients with no match between CT and RTDOSE UID
        if len(df_invalid_uid) > 0:
            df = pd.concat([df, df_invalid_uid], axis=1)

        # Sort columns based on their names, and save df to csv file
        logger.my_print('Number of patients to be excluded: {}'.format(len(df)))
        df = df.sort_index(axis=1)
        df.to_csv(os.path.join(save_root_dir, cfg.filename_exclude_patients_csv), sep=';')

    end = time.time()
    logger.my_print('Elapsed time: {time} seconds'.format(time=round(end - start, 3)))
    logger.my_print('DONE!')


if __name__ == '__main__':
    main()


