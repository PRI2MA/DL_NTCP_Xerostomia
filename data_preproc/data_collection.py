"""
Data collection preparation.

- Check that the expected patient_id (i.e. folder name) is identical to the patient_id according to the DICOM header
    of CT, RTDOSE and RTSTRUCT.
- Match UID between CT and RTDOSE and RTSTRUCTs.


MDACC:
WARNING: Patient_id = 1910821180: CT and RTDOSE Frame of Reference UID do NOT match!
WARNING: Patient_id = 3514675268: CT and RTDOSE Frame of Reference UID do NOT match!
WARNING: Patient_id = 9878102359: CT and RTDOSE Frame of Reference UID do NOT match!
WARNING: Patient_id = 1910821180: CT and RTSTRUCT Frame of Reference UID do NOT match!
WARNING: Patient_id = 3514675268: CT and RTSTRUCT Frame of Reference UID do NOT match!
"""
import os
import time
import json
import pandas as pd
import pydicom as pdcm
from glob import glob
from tqdm import tqdm
from pydicom import dcmread

import data_preproc_config as cfg
from data_preproc_functions import (copy_folder, create_folder_if_not_exists, get_all_folders,
                                    get_folders_diff_or_intersection, Logger, sort_human)


def get_metadata_rtdose(path_rtdose_file):
    # Load RTDOSE file
    ds = dcmread(path_rtdose_file)

    # Construct metadata_rtdose
    metadata_rtdose = json.loads(ds.to_json())

    return metadata_rtdose


def get_metadata_ct(path_ct):
    # Load CT slice
    slices_path = glob(path_ct + '/*')
    # slices_path = sort_human(slices_path)
    slice = pdcm.read_file(slices_path[0])

    # Construct metadata_ct
    metadata_ct = json.loads(slice.to_json())

    return metadata_ct


def get_metadata_rtstruct(path_rtstruct_file):
    # Load RTSTRUCT file
    ds = dcmread(path_rtstruct_file)

    # Construct metadata_rtdose
    metadata_rtstruct = json.loads(ds.to_json())

    return metadata_rtstruct


def main_folder_intersection():
    logger = Logger(output_filename=None)
    start = time.time()

    in_path_1 = '//zkh/appdata/RTDicom/OPC_data/DICOM_data'
    in_path_2 = '//zkh/appdata/RTDicom/HungChu/A1_CT_IBM/All_Correct_Selection/DICOM'
    out_path_diff = '//zkh/appdata/RTDicom/HungChu/diff.txt'
    out_path_intersection = '//zkh/appdata/RTDicom/HungChu/intersection.txt'

    # Get list of folders
    list_1 = os.listdir(in_path_1)
    list_2 = os.listdir(in_path_2)

    # Get difference and intersection of folders between list_1 and list_2
    diff = get_folders_diff_or_intersection(list_1=list_1, list_2=list_2, mode='diff')
    intersection = get_folders_diff_or_intersection(list_1=list_1, list_2=list_2, mode='intersection')

    # Save output (list) to txt
    with open(out_path_diff, 'w') as f:
        f.write('Folders in {} but not in {}:\n'.format(in_path_1, in_path_2))
        for row in diff:
            f.write(str(row) + '\n')

    with open(out_path_intersection, 'w') as f:
        f.write('Folders in both {} and {}:\n'.format(in_path_1, in_path_2))
        for row in intersection:
            f.write(str(row) + '\n')

    end = time.time()
    logger.my_print('Elapsed time: {time} seconds'.format(time=round(end - start, 3)))
    logger.my_print('DONE!')


def main_copy():
    """
    IMPORTANT REQUIREMENTS:
        - Input csv file contains a column 'Exists' (= True: if at least one path contains True, else: False),
            which should be the most-right column! Otherwise we may get src_i = True and therefore an error in
            copy_folder().
        - For each column in the input csv file holds: the columns to its left have higher hierarchy, while
            the columns to its right have lower hierarchy. Moreover, for each patient_id we only copy the folder
            from the path with highest available hierarchy (if any).

    Returns:

    """
    # Initialize variables
    mode_list = cfg.mode_list
    save_dir_mode_list = cfg.save_dir_mode_list
    patient_id_length = cfg.patient_id_length

    data_collection_dir = cfg.data_collection_dir
    filename_patient_ids = cfg.filename_patient_ids
    logger = Logger()
    start = time.time()

    for m_i, d_i in zip(mode_list, save_dir_mode_list):
        logger.my_print('Mode: {}'.format(m_i))
        filename = filename_patient_ids.format(m_i)
        dst = cfg.data_dir.format(d_i)

        # Load patient_ids
        df = pd.read_csv(os.path.join(data_collection_dir, filename), sep=';', index_col=0)
        patient_ids = df.index.values
        patient_ids = ['%0.{}d'.format(patient_id_length) % int(x) for x in patient_ids]
        df.index = patient_ids

        logger.my_print('Len(patient_ids): {}'.format(len(patient_ids)))
        for patient_id in tqdm(patient_ids):
            logger.my_print('Patient_id: {}'.format(patient_id))

            # Select patient's row
            df_i = df.loc[patient_id]
            if df_i['Exists']:
                # Get first path (i.e. highest hierarchy) with True
                first_path = df_i.index[df_i == True][0]
                src_i = os.path.join(first_path, patient_id)

                # OPC_data/DICOM_data_organized
                if 'RTDicom' in src_i and 'OPC_data' in src_i and 'DICOM_data_organized' in src_i:
                    relevant_folder_i = None
                    relevant_folder_list_i = [x for x in os.listdir(src_i) if x.endswith('_CT_DOSE')]
                    # Determine correct relevant_folder, for determining correct date
                    if len(relevant_folder_list_i) == 1:
                        relevant_folder_i = relevant_folder_list_i[0]
                    # # Sometimes there are multiple '_CT_DOSE' folders
                    # elif len(relevant_folder_list_i) > 1:
                    #     logger.my_print('There are multiple _CT_DOSE folders.', level='warning')
                    #     relevant_folder_i_in = list()
                    #     relevant_folder_i_out = list()
                    #
                    #     for f in relevant_folder_list_i:
                    #         if 'CT' in os.listdir(src_i, f):
                    #             relevant_folder_i_in.append(f)
                    #     if len(relevant_folder_i_in) == 1:
                    #         relevant_folder_i = relevant_folder_i_in[0]
                    #     else:
                    #         logger.my_print('There are multiple _CT_DOSE folders with a CT folder.', level='warning')
                    #         assert False
                    #
                    #     for f in relevant_folder_list_i:
                    #         if relevant_folder_i is None:
                    #             date_f = f.split('_')[0]
                    #             if date_f + '_CT' in os.listdir(src_i):
                    #                 relevant_folder_i_out.append(f)
                    #     if len(relevant_folder_i_out) == 1:
                    #         relevant_folder_i = relevant_folder_i_out[0]
                    #     # else: assert False

                        # Determine date
                        date_i = relevant_folder_i.split('_')[0]

                        # Initialize patient folder in dst
                        dst_i = os.path.join(dst, patient_id, date_i)
                        create_folder_if_not_exists(dst_i)

                        # Count number of folders: should be less than or equal to 1 for each modality
                        nr_ct_folders, nr_rtdose_folders, nr_rtstruct_folders = 0, 0, 0

                        relevant_sub_folders_i = [x for x in os.listdir(os.path.join(src_i, relevant_folder_i)) if
                                                  (x == 'CT' or x == 'RTDOSE' or x == 'RTSTRUCT')]
                        for f_i in relevant_sub_folders_i:
                            src_i_i = os.path.join(src_i, relevant_folder_i, f_i)
                            dst_i_i = os.path.join(dst_i, f_i)
                            copy_folder(src=src_i_i, dst=dst_i_i)
                            if f_i == 'CT':
                                nr_ct_folders += 1
                            elif f_i == 'RTDOSE':
                                nr_rtdose_folders += 1
                            elif f_i == 'RTSTRUCT':
                                nr_rtstruct_folders += 1

                        # CT folder may be outside '_CT_DOSE' folder
                        if nr_ct_folders == 0:
                            try:
                                src_i_i = os.path.join(src_i, date_i + '_CT', 'CT')
                                dst_i_i = os.path.join(dst_i, 'CT')
                                copy_folder(src=src_i_i, dst=dst_i_i)
                                nr_ct_folders += 1
                            except:
                                logger.my_print('No correct CT folder found.')

                        # RTSTRUCT folder may be outside '_CT_DOSE' folder
                        if nr_rtstruct_folders == 0:
                            try:
                                src_i_i = os.path.join(src_i, date_i + '_RTSTRUCT', 'RTSTRUCT')
                                dst_i_i = os.path.join(dst_i, 'RTSTRUCT')
                                copy_folder(src=src_i_i, dst=dst_i_i)
                                nr_rtstruct_folders += 1
                            except:
                                logger.my_print('No correct RTSTRUCT folder found.')
                    else:
                        # Create empty folder
                        create_folder_if_not_exists(os.path.join(dst, patient_id))

                    # Checks
                    assert nr_ct_folders <= 1
                    assert nr_rtdose_folders <= 1
                    assert nr_rtstruct_folders <= 1

                # OPC_data/DICOM_data_organized
                elif ('RTDicom' in src_i and 'OPC_data' in src_i and 'DICOM_data' in src_i and
                      'DICOM_data_organized' not in src_i):
                    logger.my_print('OPC_data/DICOM_data will be skipped', level='warning')
                else:
                    dst_i = os.path.join(dst, patient_id)
                    copy_folder(src=src_i, dst=dst_i)

                logger.my_print('src_i: {}'.format(src_i))
            else:
                logger.my_print('Patient_id {} has no data so far.'.format(patient_id))

    end = time.time()
    logger.my_print('Elapsed time: {time} seconds'.format(time=round(end - start, 3)))
    logger.my_print('DONE!')


def main_frame_of_reference_uid():
    # Initialize variables
    use_umcg = cfg.use_umcg
    mode_list = cfg.mode_list
    data_dir_mode_list = cfg.save_dir_mode_list
    patient_id_length = cfg.patient_id_length

    save_root_dir = cfg.save_root_dir
    patient_folder_types = cfg.patient_folder_types
    frame_of_ref_uid_tag = cfg.frame_of_ref_uid_tag

    logger = Logger(output_filename=None)
    start = time.time()

    for _, d_i in zip(mode_list, data_dir_mode_list):
        data_dir = cfg.data_dir.format(d_i)
        data_dir_citor = data_dir

        # Load file from data_folder_types.py
        # TODO: temporary, for MDACC
        if use_umcg:
            df_data_folder_types = pd.read_csv(os.path.join(save_root_dir, cfg.filename_patient_folder_types_csv),
                                               sep=';',
                                               index_col=0)
            df_data_folder_types.index = ['%0.{}d'.format(patient_id_length) % int(x)
                                          for x in df_data_folder_types.index.values]

        # Get all patient_ids
        logger.my_print('Listing all patient ids and their files...')
        patients_list = os.listdir(data_dir)
        patients_list = sort_human(patients_list)

        # TODO: temporary: consider all patients, for exclude_patients.csv
        # Testing for a small number of patients
        if cfg.test_patients_list is not None:
            test_patients_list = cfg.test_patients_list
            patients_list = [x for x in test_patients_list if x in patients_list]
        n_patients = len(patients_list)
        logger.my_print('Total number of patients: {n}'.format(n=n_patients))

        ct_for_uid_list = list()
        rtdose_for_uid_list = list()
        is_matching_uid_list = list()
        for patient_id in tqdm(patients_list):
            logger.my_print('Patient_id: {}'.format(patient_id))

            # Determine folder type of interest ('with_contrast'/'no_contrast')
            # TODO: temporary, for MDACC
            if use_umcg:
                folder_type_i = df_data_folder_types.loc[patient_id]['Folder']
            else:
                folder_type_i = ''

            ##### CT vs RTDOSE
            # List all folders of the patient
            all_folders = get_all_folders(os.path.join(data_dir, patient_id, folder_type_i))
            # Get relevant path
            path_ct = [x for x in all_folders if x.endswith('\\CT')]
            path_rtdose = [x for x in all_folders if x.endswith('\\RTDOSE')]

            # 'with_contrast' has highest priority: if it does not contain one or more of the required folders,
            # then search in 'no_contrast'
            # TODO: temporary, for MDACC
            if use_umcg:
                if len(path_ct) == 0:
                    all_folders_2 = get_all_folders(os.path.join(data_dir, patient_id, patient_folder_types[1]))
                    path_ct = [x for x in all_folders_2 if x.endswith('\\CT')]
                if len(path_rtdose) == 0:
                    all_folders_2 = get_all_folders(os.path.join(data_dir, patient_id, patient_folder_types[1]))
                    path_rtdose = [x for x in all_folders_2 if x.endswith('\\RTDOSE')]

            # Checks
            if (len(path_rtdose) != 1) or (len(path_ct) != 1):
                logger.my_print('path_rtdose: {}'.format(path_rtdose))
                logger.my_print('path_ct: {}'.format(path_ct))
                assert False
            path_rtdose = path_rtdose[0]
            path_ct = path_ct[0]

            rtdose_file = os.listdir(path_rtdose)
            assert len(rtdose_file) == 1
            path_rtdose_file = os.path.join(path_rtdose, rtdose_file[0])

            # Get metadata
            metadata_rtdose = get_metadata_rtdose(path_rtdose_file=path_rtdose_file)
            metadata_ct = get_metadata_ct(path_ct=path_ct)

            # Extract Frame Of Reference UID
            frame_of_ref_uid_ct = metadata_ct[frame_of_ref_uid_tag]['Value'][0]
            frame_of_ref_uid_rtdose = metadata_rtdose[frame_of_ref_uid_tag]['Value'][0]
            is_matching = (frame_of_ref_uid_ct == frame_of_ref_uid_rtdose)

            ct_for_uid_list.append(frame_of_ref_uid_ct)
            rtdose_for_uid_list.append(frame_of_ref_uid_rtdose)
            is_matching_uid_list.append(is_matching)

        # Create dataframe
        df = pd.DataFrame(
            {'PatientID': patients_list,
             'CT_FoR_UID': ct_for_uid_list,
             'RTDOSE_FoR_UID': rtdose_for_uid_list,
             'Matching_UID': is_matching_uid_list}
        )

        # Save to csv
        df.to_csv(os.path.join(save_root_dir, cfg.filename_frame_of_reference_uid_csv), sep=';', index=False)

    end = time.time()
    logger.my_print('Elapsed time: {time} seconds'.format(time=round(end - start, 3)))
    logger.my_print('DONE!')


def main_match_ct_rtdose_rtstruct():
    # Initialize variables
    use_umcg = cfg.use_umcg
    mode_list = cfg.mode_list
    data_dir_mode_list = cfg.save_dir_mode_list
    patient_id_length = cfg.patient_id_length

    save_root_dir = cfg.save_root_dir
    patient_folder_types = cfg.patient_folder_types
    frame_of_ref_uid_tag = cfg.frame_of_ref_uid_tag
    patient_id_tag = cfg.patient_id_tag

    logger = Logger(output_filename=None)
    start = time.time()

    for _, d_i in zip(mode_list, data_dir_mode_list):
        data_dir = cfg.data_dir.format(d_i)
        data_dir_citor = data_dir

        # Load file from data_folder_types.py
        # TODO: temporary, for MDACC
        if use_umcg:
            df_data_folder_types = pd.read_csv(os.path.join(save_root_dir, cfg.filename_patient_folder_types_csv), sep=';',
                                               index_col=0)
            df_data_folder_types.index = ['%0.{}d'.format(patient_id_length) % int(x)
                                          for x in df_data_folder_types.index.values]

        # Get all patient_ids
        logger.my_print('Listing all patient ids and their files...')
        patients_list = os.listdir(data_dir)
        patients_list = sort_human(patients_list)

        # TODO: temporary: consider all patients, for exclude_patients.csv
        # Testing for a small number of patients
        if cfg.test_patients_list is not None:
            test_patients_list = cfg.test_patients_list
            patients_list = [x for x in test_patients_list if x in patients_list]
        n_patients = len(patients_list)
        logger.my_print('Total number of patients: {n}'.format(n=n_patients))

        # List of patient_ids for with the patient_id in DICOM file is not identical to the expected patient_id, and
        # patient_ids for which CT UID and RTDOSE/RTSTRUCT UID do NOT match
        wrong_patient_id_ct_list = set()
        wrong_patient_id_rtdose_list = set()
        wrong_patient_id_rtstruct_list = set()
        no_frame_of_ref_uid_match_rtdose_list = set()
        no_frame_of_ref_uid_match_rtstruct_list = set()
        non_matching_frame_of_ref_rtstructs_dict = dict()

        for patient_id in tqdm(patients_list):
            logger.my_print('Patient_id: {}'.format(patient_id))

            # Determine folder type of interest ('with_contrast'/'no_contrast')
            # TODO: temporary, for MDACC
            if use_umcg:
                folder_type_i = df_data_folder_types.loc[patient_id]['Folder']
            else:
                folder_type_i = ''

            ##### CT vs RTDOSE
            # List all folders of the patient
            all_folders = get_all_folders(os.path.join(data_dir, patient_id, folder_type_i))
            # Get relevant path
            path_ct = [x for x in all_folders if x.endswith('\\CT')]
            path_rtdose = [x for x in all_folders if x.endswith('\\RTDOSE')]

            # 'with_contrast' has highest priority: if it does not contain one or more of the required folders,
            # then search in 'no_contrast'
            # TODO: temporary, for MDACC
            if use_umcg:
                if len(path_ct) == 0:
                    all_folders_2 = get_all_folders(os.path.join(data_dir, patient_id, patient_folder_types[1]))
                    path_ct = [x for x in all_folders_2 if x.endswith('\\CT')]
                if len(path_rtdose) == 0:
                    all_folders_2 = get_all_folders(os.path.join(data_dir, patient_id, patient_folder_types[1]))
                    path_rtdose = [x for x in all_folders_2 if x.endswith('\\RTDOSE')]

            # Checks
            if (len(path_rtdose) != 1) or (len(path_ct) != 1):
                logger.my_print('path_rtdose: {}'.format(path_rtdose))
                logger.my_print('path_ct: {}'.format(path_ct))
                assert False
            path_rtdose = path_rtdose[0]
            path_ct = path_ct[0]

            # Make sure that there is only one RTDOSE folder per patient
            # assert len(path_rtdose) == 1
            # path_rtdose = path_rtdose[0]
            # all_rtdose_files = []
            # for p in path_rtdose:
            #     all_rtdose_files += glob(p + '/*')
            # assert len(all_rtdose_files) == 1
            # path_rtdose_file = [x for x in all_rtdose_files if x.endswith(filename_rtdose)]
            # path_rtdose = [x.replace('\\' + filename_rtdose, '') for x in path_rtdose_file]

            rtdose_file = os.listdir(path_rtdose)
            assert len(rtdose_file) == 1
            path_rtdose_file = os.path.join(path_rtdose, rtdose_file[0])

            # Get metadata
            metadata_rtdose = get_metadata_rtdose(path_rtdose_file=path_rtdose_file)
            metadata_ct = get_metadata_ct(path_ct=path_ct)

            # Get Patient_id
            if metadata_ct[patient_id_tag]['Value'] != [patient_id]:
                wrong_patient_id_ct_list.add(patient_id)
                logger.my_print("metadata_ct[patient_id_tag]['Value']: {}".format(metadata_ct[patient_id_tag]['Value']))
            if metadata_rtdose[patient_id_tag]['Value'] != [patient_id]:
                wrong_patient_id_rtdose_list.add(patient_id)
                logger.my_print("metadata_rtdose[patient_id_tag]['Value']: {}".format(metadata_rtdose[patient_id_tag]['Value']))

            # Get UID
            frame_of_ref_uid_rtdose = metadata_rtdose[frame_of_ref_uid_tag]['Value'][0]
            frame_of_ref_uid_ct = metadata_ct[frame_of_ref_uid_tag]['Value'][0]
            if frame_of_ref_uid_rtdose != frame_of_ref_uid_ct:
                no_frame_of_ref_uid_match_rtdose_list.add(patient_id)
                logger.my_print('frame_of_ref_uid_rtdose: {}'.format(frame_of_ref_uid_tag, frame_of_ref_uid_rtdose))
                logger.my_print('frame_of_ref_uid_ct: {}'.format(frame_of_ref_uid_tag, frame_of_ref_uid_ct))

            ##### CT vs RTSTRUCT
            all_folders_citor = get_all_folders(os.path.join(data_dir_citor, patient_id, folder_type_i))

            # Normally patients should have one RTSTRUCT folder and one file, but sometimes they have multiple. Consider
            # all of them
            path_rtss = [x for x in all_folders_citor if x.endswith('\\RTSTRUCT')]
            n_path_rtss = len(path_rtss)

            # 'with_contrast' has highest priority: if it does not contain one or more of the required folders,
            # then search in 'no_contrast'
            # TODO: temporary, for MDACC
            if use_umcg:
                if n_path_rtss == 0:
                    all_folders_citor_2 = get_all_folders(
                        os.path.join(data_dir_citor, patient_id, patient_folder_types[1]))
                    path_rtss = [x for x in all_folders_citor_2 if x.endswith('\\RTSTRUCT')]
                    n_path_rtss = len(path_rtss)
                elif n_path_rtss > 1:
                    logger.my_print('Patient {id} has {n} RTSTRUCT folders!'.format(id=patient_id, n=n_path_rtss),
                                    level='warning')

            # Patients could have multiple RTSTRUCT folders
            non_matching_frame_of_ref_rtstructs = []
            for p in path_rtss:
                files_rtss = glob(p + '/*')

                # Patients can have multiple RTSTRUCT files. Consider all of them
                for f_rtss in files_rtss:
                    # Get metadata
                    metadata_rtstruct = get_metadata_rtstruct(path_rtstruct_file=f_rtss)

                    # Get Patient_id
                    if metadata_rtstruct[patient_id_tag]['Value'] != [patient_id]:
                        wrong_patient_id_rtstruct_list.add(patient_id)
                        logger.my_print("metadata_rtstruct[patient_id_tag]['Value']: {}".format(metadata_rtstruct[patient_id_tag]['Value']))

                    # Get UID
                    frame_of_ref_uid_rtstruct = metadata_rtstruct['30060010']['Value'][0][frame_of_ref_uid_tag]['Value'][0]
                    if frame_of_ref_uid_rtstruct != frame_of_ref_uid_ct:
                        no_frame_of_ref_uid_match_rtstruct_list.add(patient_id)
                        non_matching_frame_of_ref_rtstructs += [f_rtss]
                        logger.my_print('f_rtss: {}'.format(f_rtss))
                        logger.my_print('frame_of_ref_uid_rtstruct: {}'.format(frame_of_ref_uid_rtstruct))
                        logger.my_print('frame_of_ref_uid_ct: {}'.format(frame_of_ref_uid_ct))

            non_matching_frame_of_ref_rtstructs_dict[patient_id] = ', '.join(non_matching_frame_of_ref_rtstructs) if len(non_matching_frame_of_ref_rtstructs) > 0 else None

        # Convert set to list
        wrong_patient_id_ct_list = sort_human(list(set(wrong_patient_id_ct_list)))
        wrong_patient_id_rtdose_list = sort_human(list(set(wrong_patient_id_rtdose_list)))
        wrong_patient_id_rtstruct_list = sort_human(list(set(wrong_patient_id_rtstruct_list)))
        no_frame_of_ref_uid_match_rtdose_list = sort_human(list(set(no_frame_of_ref_uid_match_rtdose_list)))
        no_frame_of_ref_uid_match_rtstruct_list = sort_human(list(set(no_frame_of_ref_uid_match_rtstruct_list)))

        if len(no_frame_of_ref_uid_match_rtdose_list) == 0:
            logger.my_print('Every patient_id has matching CT and RTDOSE Frame of Reference UID.')
        else:
            for patient_id in no_frame_of_ref_uid_match_rtdose_list:
                logger.my_print('Patient_id = {}: CT and RTDOSE Frame of Reference UID do NOT match!'.format(patient_id), level='warning')

        if len(no_frame_of_ref_uid_match_rtstruct_list) == 0:
            logger.my_print('Every patient_id has matching CT and RTSTRUCT Frame of Reference UID.')
        else:
            for patient_id in no_frame_of_ref_uid_match_rtstruct_list:
                logger.my_print('Patient_id = {}: CT and RTSTRUCT Frame of Reference UID do NOT match!'.format(patient_id), level='warning')

        df = pd.DataFrame(columns=[
            'Patient_id', 'Folder', 'Wrong patient_id (CT)', 'Wrong patient_id (RTDOSE)',
            'Wrong patient_id (RTSTRUCT)', 'No Frame of Reference UID match CT RTDOSE',
            'No Frame of Reference UID match CT RTSTRUCT', 'Non-matching Frame of Reference UID RTSTRUCT files'])
        issues_patient_ids = sort_human(list(set(wrong_patient_id_ct_list + wrong_patient_id_rtdose_list + wrong_patient_id_rtstruct_list +
                                                 no_frame_of_ref_uid_match_rtdose_list + no_frame_of_ref_uid_match_rtstruct_list)))

        # issues_patient_ids = ['1910821180', '3514675268', '9878102359']
        for patient_id in issues_patient_ids:
            # Determine folder type of interest ('with_contrast'/'no_contrast')
            # TODO: temporary, for MDACC
            if use_umcg:
                folder_type_i = df_data_folder_types.loc[patient_id]['Folder']
            else:
                folder_type_i = ''

            wrong_patient_id_ct = True if patient_id in wrong_patient_id_ct_list else False
            wrong_patient_id_rtdose = True if patient_id in wrong_patient_id_rtdose_list else False
            wrong_patient_id_rtstruct = True if patient_id in wrong_patient_id_rtstruct_list else False
            no_frame_of_ref_uid_match_ct_rtdose = True if patient_id in no_frame_of_ref_uid_match_rtdose_list else False
            no_frame_of_ref_uid_match_ct_rtstruct = True if patient_id in no_frame_of_ref_uid_match_rtstruct_list else False
            non_matching_frame_of_ref_rtstructs = non_matching_frame_of_ref_rtstructs_dict[patient_id]

            df_i = pd.DataFrame(
                {'Patient_id': patient_id,
                 'Folder': folder_type_i,
                 'Wrong patient_id (CT)': wrong_patient_id_ct,
                 'Wrong patient_id (RTDOSE)': wrong_patient_id_rtdose,
                 'Wrong patient_id (RTSTRUCT)': wrong_patient_id_rtstruct,
                 'No Frame of Reference UID match CT RTDOSE': no_frame_of_ref_uid_match_ct_rtdose,
                 'No Frame of Reference UID match CT RTSTRUCT': no_frame_of_ref_uid_match_ct_rtstruct,
                 'Non-matching Frame of Reference UID RTSTRUCT files': non_matching_frame_of_ref_rtstructs},
                index=[0])
            df = pd.concat([df, df_i], axis=0)

        # Save to csv
        df.to_csv(os.path.join(save_root_dir, cfg.filename_invalid_uid_csv), sep=';', index=False)

    end = time.time()
    logger.my_print('Elapsed time: {time} seconds'.format(time=round(end - start, 3)))
    logger.my_print('DONE!')


def main():
    # main_folder_intersection()
    # main_copy()
    main_frame_of_reference_uid()
    main_match_ct_rtdose_rtstruct()
    

if __name__ == '__main__':
    main()




