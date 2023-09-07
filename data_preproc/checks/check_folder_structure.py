"""
Checks:
    - CT: check that:
        - There is one and only one folder named 'CT'.
        - That it contains at least one DICOM file.
        - That it only contains DICOM files (and no other file types).
    - RTDOSE: check that:
        - There is one and only one RTDOSE file.
    - RTSTRUCT: check that
        - There is one and only one folder named 'RTSTRUCT'.
        - That it contains at least one DICOM file.
        - That it only contains DICOM files (and no other file types).
"""
import os
import time

import pandas as pd
from tqdm import tqdm
from glob import glob

import data_preproc_config as cfg
from data_preproc_functions import get_all_folders, Logger, sort_human


def main_ct(df, data_dir, patient_id, folder_type, src, logger):
    # Initialize variables
    has_incorrect = 0
    patient_folder_types = cfg.patient_folder_types

    # List all folders of the patient
    all_folders = get_all_folders(os.path.join(data_dir, patient_id, folder_type))

    if len(all_folders) > 0:
        # Get relevant path
        path_ct = [x for x in all_folders if x.endswith('\\CT')]
        # 'with_contrast' has highest priority: if it does not contain one or more of the required folders,
        # then search in 'no_contrast'
        # TODO: temporary, for MDACC
        """
        if len(path_ct) == 0:
            all_folders_2 = get_all_folders(os.path.join(data_dir, patient_id, patient_folder_types[1]))
            path_ct = [x for x in all_folders_2 if x.endswith('\\CT')]
        """

        # Check that there is one and only one folder named 'CT'
        if len(path_ct) == 1:
            # Check that it contains at least 50 DICOM files (i.e. slices)
            path_ct = path_ct[0]
            files_dcm_ct = glob(path_ct + '/*{}'.format(cfg.ct_file_ext))
            if not len(files_dcm_ct) >= 50:
                logger.my_print('Patient_id: {}'.format(patient_id))
                logger.my_print('len(files_dcm_ct): {}'.format(len(files_dcm_ct)), level='warning')
                df_i = pd.DataFrame({'Patient_id': patient_id,
                                     'Folder': folder_type,
                                     'Reason': '(CT) Number of DICOM files: {}'.format(len(files_dcm_ct)),
                                     'Source path': src}, index=[0])
                df = pd.concat([df, df_i])
                has_incorrect = 1

            # Check that it only contains DICOM files (and no other file types).
            files_ct = glob(path_ct + '/*')
            if not len(files_ct) == len(files_dcm_ct):
                logger.my_print('Patient_id: {}'.format(patient_id))
                logger.my_print('files_ct: {}'.format(files_ct), level='warning')
                logger.my_print('files_dcm_ct: {}'.format(files_dcm_ct), level='warning')
                df_i = pd.DataFrame({'Patient_id': patient_id,
                                     'Folder': folder_type,
                                     'Reason': '(CT) Contains filetype(s) other than DICOM',
                                     'Source path': src},
                                    index=[0])
                df = pd.concat([df, df_i])
                has_incorrect = 1

            # Check that it does not contain "... (1).dcm", because this could imply that we
            # copied the files twice.
            duplicates = [x for x in files_ct if '(1)' in x]
            if len(duplicates) > 0:
                logger.my_print('(CT) Possibly duplicated files')
                df_i = pd.DataFrame({'Patient_id': patient_id,
                                     'Folder': folder_type,
                                     'Reason': '(CT) Possibly duplicated files',
                                     'Source path': src},
                                    index=[0])
                df = pd.concat([df, df_i])
                has_incorrect = 1
        else:
            logger.my_print('Patient_id: {}'.format(patient_id))
            logger.my_print('path_ct: {}'.format(path_ct), level='warning')
            df_i = pd.DataFrame({'Patient_id': patient_id,
                                 'Folder': folder_type,
                                 'Reason': '(CT) Number of folders: {}'.format(len(path_ct)),
                                 'Source path': src}, index=[0],)
            df = pd.concat([df, df_i])
            has_incorrect = 1

    return df, has_incorrect


def main_rtdose(df, data_dir, patient_id, folder_type, src, logger):
    # Initialize variables
    use_umcg = cfg.use_umcg
    # filename_rtdose = cfg.filename_rtdose_dcm
    has_incorrect = 0
    patient_folder_types = cfg.patient_folder_types

    # List all folders of the patient
    all_folders = get_all_folders(os.path.join(data_dir, patient_id, folder_type))

    if len(all_folders) > 0:
        # Get relevant path
        path_rtdose = [x for x in all_folders if x.endswith('\\RTDOSE')]

        if use_umcg:
            # 'with_contrast' has highest priority: if it does not contain one or more of the required folders,
            # then search in 'no_contrast'
            if len(path_rtdose) == 0:
                all_folders_2 = get_all_folders(os.path.join(data_dir, patient_id, patient_folder_types[1]))
                path_rtdose = [x for x in all_folders_2 if x.endswith('\\RTDOSE')]

        # Check that there is one and only one RTDOSE file
        all_rtdose_files = []
        for p in path_rtdose:
            all_rtdose_files += glob(p + '/*')
        if not len(all_rtdose_files) == 1:
            logger.my_print('Patient_id: {}'.format(patient_id))
            logger.my_print('all_rtdose_files: {}'.format(all_rtdose_files), level='warning')
            df_i = pd.DataFrame({'Patient_id': patient_id,
                                 'Folder': folder_type,
                                 'Reason': '(RTDOSE) Number of files: {}'.format(len(all_rtdose_files)),
                                 'Source path': src}, index=[0])
            df = pd.concat([df, df_i])
            has_incorrect = 1

    return df, has_incorrect


def main_rtstruct(df, data_dir, patient_id, folder_type, src, logger):
    # Initialize variables
    use_umcg = cfg.use_umcg
    has_incorrect = 0
    patient_folder_types = cfg.patient_folder_types

    # List all folders of the patient
    all_folders = get_all_folders(os.path.join(data_dir, patient_id, folder_type))

    if len(all_folders) > 0:
        # Get relevant path
        path_rtstruct = [x for x in all_folders if x.endswith('\\RTSTRUCT')]
        # 'with_contrast' has highest priority: if it does not contain one or more of the required folders,
        # then search in 'no_contrast'
        # TODO: temporary, for MDACC
        if use_umcg:
            if len(path_rtstruct) == 0:
                all_folders_2 = get_all_folders(os.path.join(data_dir, patient_id, patient_folder_types[1]))
                path_rtstruct = [x for x in all_folders_2 if x.endswith('\\RTSTRUCT')]

        # Check that there is one and only one folder named 'RTSTRUCT'
        if not len(path_rtstruct) == 1:
            logger.my_print('Patient_id: {}'.format(patient_id))
            logger.my_print('path_rtstruct: {}'.format(path_rtstruct), level='warning')
            df_i = pd.DataFrame({'Patient_id': patient_id,
                                 'Folder': folder_type,
                                 'Reason': '(RTSTRUCT) Number of folders: {}'.format(len(path_rtstruct)),
                                 'Source path': src}, index=[0])
            df = pd.concat([df, df_i])
            has_incorrect = 1
        else:
            # Check that it contains at least one DICOM file
            path_rtstruct = path_rtstruct[0]
            # Note glob(): '*.dcm' also considers '*.DCM'
            files_dcm_rtstruct = glob(path_rtstruct + '/*{}'.format(cfg.rtstruct_file_ext))
            if not len(files_dcm_rtstruct) >= 1:
                logger.my_print('Patient_id: {}'.format(patient_id))
                logger.my_print('files_dcm_rtstruct: {}'.format(files_dcm_rtstruct), level='warning')
                df_i = pd.DataFrame({'Patient_id': patient_id,
                                     'Folder': folder_type,
                                     'Reason': '(RTSTRUCT) Number of DICOM files: {}'.format(len(files_dcm_rtstruct)),
                                     'Source path': src},
                                    index=[0])
                df = pd.concat([df, df_i])
                has_incorrect = 1

            # Check that it only contains DICOM files (and no other file types).
            files_rtstruct = glob(path_rtstruct + '/*')
            if not len(files_rtstruct) == len(files_dcm_rtstruct):
                logger.my_print('Patient_id: {}'.format(patient_id))
                logger.my_print('files_rtstruct: {}'.format(files_rtstruct), level='warning')
                logger.my_print('files_dcm_rtstruct: {}'.format(files_dcm_rtstruct), level='warning')
                df_i = pd.DataFrame({'Patient_id': patient_id,
                                     'Folder': folder_type,
                                     'Reason': '(RTSTRUCT) Contains filetype(s) other than DICOM',
                                     'Source path': src}, index=[0])
                df = pd.concat([df, df_i])
                has_incorrect = 1

    return df, has_incorrect


def main():
    # Initialize variables
    # From data collection
    use_umcg = cfg.use_umcg
    mode_list = cfg.mode_list
    data_dir_mode_list = cfg.save_dir_mode_list
    patient_id_length = cfg.patient_id_length

    save_root_dir = cfg.save_root_dir
    data_collection_dir = cfg.data_collection_dir
    filename_patient_ids = cfg.filename_patient_ids
    patients_incorrect_folder_struct = list()
    logger = Logger(os.path.join(save_root_dir, cfg.filename_check_folder_structure_logging_txt))
    start = time.time()

    for m_i, d_i in zip(mode_list, data_dir_mode_list):
        # TODO: temporary, for MDACC
        filename = filename_patient_ids.format(m_i)
        data_dir = cfg.data_dir.format(d_i)

        # Load file from data_folder_types.py
        # TODO: temporary, for MDACC
        if use_umcg:
            df_data_folder_types = pd.read_csv(os.path.join(save_root_dir, cfg.filename_patient_folder_types_csv), sep=';',
                                               index_col=0)
            df_data_folder_types.index = ['%0.{}d'.format(patient_id_length) % int(x)
                                          for x in df_data_folder_types.index.values]

        # Load file from data collection
        # df_data_collect = pd.read_csv(os.path.join(data_collection_dir, filename), sep=';', index_col=0)
        # df_data_collect.index = ['%0.{}d'.format(patient_id_length) % int(x) for x in df_data_collect.index.values]

        # Get all patient_id
        logger.my_print('Listing all patient ids and their files...')
        patients_list = os.listdir(data_dir)
        patients_list = sort_human(patients_list)

        # Testing for a small number of patients
        if cfg.test_patients_list is not None:
            test_patients_list = cfg.test_patients_list
            patients_list = [x for x in test_patients_list if x in patients_list]
        n_patients = len(patients_list)
        logger.my_print('Total number of patients: {n}'.format(n=n_patients))

        df = pd.DataFrame(columns=['Patient_id', 'Folder', 'Reason', 'Source path', 'Comments'])
        for patient_id in tqdm(patients_list):
            # Determine folder type of interest ('with_contrast'/'no_contrast')
            # TODO: temporary, for MDACC
            if use_umcg:
                folder_type_i = df_data_folder_types.loc[patient_id]['Folder']
            else:
                folder_type_i = ''

            # TODO: temporary: so that we do not need df_data_collect / filename_patient_ids
            src_i = 'Unknown'
            # # Get patient's src folder
            # try:
            #     df_data_collect_i = df_data_collect.loc[patient_id]
            #     assert df_data_collect_i['Exists']
            #     # Fetch first path (i.e. highest hierarchy) with True
            #     first_path = df_data_collect_i.index[df_data_collect_i == True][0]
            #     src_i = os.path.join(first_path, patient_id)
            # except:
            #     src_i = 'OUD PACS / RT Archief'

            df, has_incorrect_ct = main_ct(df=df, data_dir=data_dir, patient_id=patient_id,
                                           folder_type=folder_type_i, src=src_i, logger=logger)
            df, has_incorrect_rtdose = main_rtdose(df=df, data_dir=data_dir, patient_id=patient_id,
                                                   folder_type=folder_type_i, src=src_i, logger=logger)
            df, has_incorrect_rtstruct = main_rtstruct(df=df, data_dir=data_dir, patient_id=patient_id,
                                                       folder_type=folder_type_i, src=src_i, logger=logger)
            if max(has_incorrect_ct, has_incorrect_rtdose, has_incorrect_rtstruct) > 0:
                patients_incorrect_folder_struct += [patient_id]
        logger.my_print('Patients with incorrect folder structure: {}.'.format(patients_incorrect_folder_struct))
        logger.my_print('Number of patients with incorrect folder structure or files: {} / {}.'
                        .format(len(patients_incorrect_folder_struct), len(patients_list)))

        # Save to csv
        df = df.sort_values(by=['Patient_id', 'Folder', 'Reason'])
        df.to_csv(os.path.join(save_root_dir, cfg.filename_patients_incorrect_folder_structure_csv.format(m_i)),
                  sep=';', index=False)

        end = time.time()
        logger.my_print('Elapsed time: {time} seconds'.format(time=round(end - start, 3)))
        logger.my_print('DONE!')


if __name__ == '__main__':
    main()



