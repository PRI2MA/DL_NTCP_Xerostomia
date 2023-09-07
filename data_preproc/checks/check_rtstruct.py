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
from tqdm import tqdm
from glob import glob

import data_preproc_config as cfg
from data_preproc_functions import get_all_folders, Logger, sort_human
from data_preproc_ct_segmentation_map_citor import dcmread, read_ds


def main_all_structures_raw():
    # Initialize variables
    save_root_dir = cfg.save_root_dir
    mode_list = cfg.mode_list
    data_dir_mode_list = cfg.save_dir_mode_list
    patient_id_length = cfg.patient_id_length

    logger = Logger()
    start = time.time()

    for _, d_i in zip(mode_list, data_dir_mode_list):
        data_dir = cfg.data_dir.format(d_i)

        # Load file from data_folder_types.py
        df_data_folder_types = pd.read_csv(os.path.join(save_root_dir, cfg.filename_patient_folder_types_csv), sep=';',
                                           index_col=0)
        df_data_folder_types.index = ['%0.{}d'.format(patient_id_length) % int(x)
                                      for x in df_data_folder_types.index.values]

        # Get all patient_id
        patients_list = os.listdir(data_dir)
        patients_list = sort_human(patients_list)

        df = pd.DataFrame(columns=['Patient_id'])

        # Testing for a small number of patients
        if cfg.test_patients_list is not None:
            test_patients_list = cfg.test_patients_list
            patients_list = [x for x in test_patients_list if x in patients_list]

        for patient_id in tqdm(patients_list):
            # Determine folder type of interest ('with_contrast'/'no_contrast')
            folder_type_i = df_data_folder_types.loc[patient_id]['Folder']

            # List all folders of the patient
            all_folders = get_all_folders(os.path.join(data_dir, patient_id, folder_type_i))
            # Get relevant path
            path_rtstruct = [x for x in all_folders if x.endswith('\\RTSTRUCT')]

            # Initialize patient's variables
            dict_i = {'Patient_id': patient_id}

            # TODO: temporary: remove this if-statement when every patient has one RTSTRUCT folder
            #  (see output of check_folder_structure.py).
            path_rtstruct = path_rtstruct[0]
            files_rtss = glob(path_rtstruct + '/*')

            # Normally patients should have one RTSTRUCT file, but sometimes they have multiple. Consider all of them
            for f_rtss in files_rtss:
                logger.my_print('Loading RTSTRUCT file: {f}'.format(f=f_rtss))

                # Load RTSTRUCT
                ds = dcmread(f_rtss)

                # Extract contour information from structures (if available)
                # See the description of read_ds() function for details about its output
                contours = read_ds(ds)

                for c in contours:
                    # Get the contour name
                    contour_name = c['name']

                    # Count number of coordinate values for this contour
                    nr_coordinates = 0
                    if 'contours' in c.keys():
                        for slice in c['contours']:
                            if slice is not None:
                                nr_coordinates += (len(slice) // 3)

                    # Add counts if contour_name is not in dict_i yet, or if it is larger than the count so far
                    if (contour_name not in dict_i.keys()) or ((contour_name in dict_i.keys()) and
                                                               (nr_coordinates > dict_i[contour_name])):
                        dict_i[contour_name] = nr_coordinates

            df_i = pd.DataFrame(dict_i, index=[0])
            df = pd.concat([df, df_i])

        # Save as csv
        df = df.set_index('Patient_id')
        df = df.fillna(0)
        df = df[sorted(df.columns)]
        df.to_csv(os.path.join(save_root_dir, cfg.filename_overview_all_structures_csv), sep=';')

    end = time.time()
    logger.my_print('Elapsed time: {time} seconds'.format(time=round(end - start, 3)))
    logger.my_print('DONE!')


def main_rtstruct():
    # Initialize variables
    # From data collection
    mode_list = cfg.mode_list
    data_dir_mode_list = cfg.save_dir_mode_list
    patient_id_length = cfg.patient_id_length

    save_root_dir = cfg.save_root_dir
    filename_patient_ids = cfg.filename_patient_ids
    logger = Logger()
    start = time.time()

    for _, d_i in zip(mode_list, data_dir_mode_list):
        data_dir = cfg.data_dir.format(d_i)

        # Get all patient_id
        patients_list = os.listdir(data_dir)
        patients_list = sort_human(patients_list)

        df = pd.DataFrame(columns=['Patient_id', 'Reason', 'Source path', 'Comments'])
        for patient_id in tqdm(patients_list):
            # List all folders of the patient
            all_folders = get_all_folders(os.path.join(data_dir, patient_id))
            # Get relevant path
            path_rtstruct = [x for x in all_folders if x.endswith('\\RTSTRUCT')]


def main():
    main_all_structures_raw()
    # main_rtstruct()


if __name__ == '__main__':
    main()



